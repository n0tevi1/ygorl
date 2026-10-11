"""Training-only ridge residual probe for frozen Q/V predictions."""

import math

import torch


def fit_residual(features, prediction, target, weight, ridge):
    """Equal-game sample weights are supplied by the caller; intercept is unpenalized.

    Fit the unconstrained squared residual objective, select ridge on validation
    with clipped predictions. Zero feature columns give an intercept-only control.
    No network parameters or feature tensors are changed.
    """
    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be finite and positive")
    x, p, y, w = (v.double() for v in (features, prediction, target, weight))
    assert x.ndim == p.ndim == 2 and y.ndim == w.ndim == 1
    assert len(x) == len(p) == len(y) == len(w) > 0
    assert all(bool(torch.isfinite(v).all()) for v in [x, p, y, w]) and bool((w > 0).all())
    w = w / w.sum()
    mean = (x * w[:, None]).sum(0)
    scale = ((x - mean).square() * w[:, None]).sum(0).sqrt()
    scale = torch.where(scale > 1e-5, scale, torch.ones_like(scale))
    design = torch.cat([(x - mean) / scale, x.new_ones((len(x), 1))], -1)
    penalty = torch.eye(design.shape[1], dtype=x.dtype, device=x.device)
    penalty[-1, -1] = 0
    matrix = design.T @ (w[:, None] * design) + ridge * penalty
    rhs = design.T @ (w[:, None] * (y[:, None] - p))
    beta = torch.linalg.solve(matrix, rhs)
    torch.testing.assert_close(matrix @ beta, rhs, rtol=1e-8, atol=1e-10)
    return dict(mean=mean, scale=scale, beta=beta, ridge=ridge)


def predict_residual(state, features, prediction):
    x = features.double()
    assert x.ndim == 2 and x.shape[1] == len(state["mean"])
    design = torch.cat([(x - state["mean"]) / state["scale"], x.new_ones((len(x), 1))], -1)
    return (prediction.double() + design @ state["beta"]).clamp(-1, 1)
