"""Weight-gradient GEMMs on ROCm GPUs (docs/benchmarks.md「GPU 学习器」).

Every weight gradient of the token-level layers is ``a.T @ b`` over all tokens of a minibatch: a ``[K, m]`` by
``[K, n]`` product with ``K`` in the tens of thousands and ``m, n`` about 64. rocBLAS ships no tuned fp32 GEMM
kernels for gfx1150 / gfx1151 (ROCm 7.2 and 7.14: one generic 32 x 32 tile per contraction type, versus hundreds for
gfx1100), and that tile does not split ``K``: 2.5-3.7 ms for ``K`` = 51,200, where the same sum split into chunks and
reduced with ``bmm`` takes 0.13 ms. Even gfx1100's tuned kernels (``HSA_OVERRIDE_GFX_VERSION=11.0.0``) take 0.9 ms.
:func:`tn_mm` does that split on ROCm and is the plain product elsewhere; :func:`use_split_k` switches a model's
``nn.Linear`` layers to a backward that uses it. Values equal the plain product up to float summation order.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

SPLIT_MIN_ROWS = 4096  # below this the plain product is fast enough
CHUNK_ROWS = 1024  # rows per chunk of the reduction
MAX_CHUNKS = 64


def _split_k(t: Tensor) -> bool:
    return t.is_cuda and torch.version.hip is not None and t.shape[0] >= SPLIT_MIN_ROWS


def tn_mm(a: Tensor, b: Tensor) -> Tensor:
    """``a.T @ b`` for ``a`` ``[K, m]`` and ``b`` ``[K, n]``; split over ``K`` on ROCm when ``K`` is large."""
    if not _split_k(a):
        return a.t() @ b
    k = a.shape[0]
    s = min(MAX_CHUNKS, k // CHUNK_ROWS)
    body = s * (k // s)
    out = torch.bmm(a[:body].view(s, -1, a.shape[1]).transpose(1, 2), b[:body].view(s, -1, b.shape[1])).sum(0)
    return out if body == k else out + a[body:].t() @ b[body:]


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad: Tensor):
        x, weight = ctx.saved_tensors
        g = grad.reshape(-1, grad.shape[-1])
        gx = grad @ weight if ctx.needs_input_grad[0] else None
        gw = tn_mm(g, x.reshape(-1, x.shape[-1])) if ctx.needs_input_grad[1] else None
        gb = g.sum(0) if ctx.has_bias and ctx.needs_input_grad[2] else None
        return gx, gw, gb


class SplitKLinear(nn.Linear):
    """``nn.Linear`` whose weight gradient goes through :func:`tn_mm` (same parameters and state dict)."""

    def forward(self, x: Tensor) -> Tensor:
        if torch.is_grad_enabled() and _split_k(x.reshape(-1, x.shape[-1])):
            return _Linear.apply(x, self.weight, self.bias)
        return super().forward(x)


def use_split_k(model: nn.Module) -> nn.Module:
    """Switch every plain ``nn.Linear`` of ``model`` to :class:`SplitKLinear` in place (a no-op off ROCm GPUs)."""
    for m in model.modules():
        if type(m) is nn.Linear:
            m.__class__ = SplitKLinear
    return model


__all__ = ["SplitKLinear", "tn_mm", "use_split_k"]
