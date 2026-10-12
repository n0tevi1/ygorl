"""PCGrad across training roots, computed in their gradient span.

Yu et al., Gradient Surgery for Multi-Task Learning, arXiv:2001.06782.
Each row visits all other ORIGINAL gradients in its supplied random order.
This is a root-level research adaptation; it does not project the retention anchor.
"""

import torch


@torch.no_grad()
def project_root_gradients(gradients, orders):
    """Return float64 projected rows and diagnostics; inputs are never modified.

    Caller supplies reproducible per-row permutations and applies its own reduction,
    anchoring, clipping and optimizer. Sequential projections and Adam do not guarantee
    a common descent direction. This helper is not enabled in production training.
    """
    if (
        gradients.ndim != 2
        or not all(gradients.shape)
        or not gradients.is_floating_point()
        or not torch.isfinite(gradients).all()
    ):
        raise ValueError("expected finite [roots, parameters] gradients")
    n = gradients.shape[0]
    if orders.shape != (n, n) or orders.dtype != torch.long or orders.device != gradients.device:
        raise ValueError("one integer permutation per root is required")
    expected = torch.arange(n, device=orders.device).expand(n, n)
    if not torch.equal(orders.sort(1).values, expected):
        raise ValueError("orders must be permutations")
    g = gradients.double()
    gram = g @ g.T
    coefficients = torch.eye(n, dtype=g.dtype, device=g.device)
    rows = torch.arange(n, device=g.device)
    projections = 0
    for position in range(n):
        other = orders[:, position]
        dot = (coefficients * gram[:, other].T).sum(1)
        norm = gram[other, other]
        active = (other != rows) & (norm > 0) & (dot < 0)
        factor = torch.where(active, dot / norm.clamp_min(torch.finfo(g.dtype).tiny), 0.0)
        coefficients[rows, other] -= factor
        projections += int(active.sum())
    projected = coefficients @ g
    assert torch.isfinite(projected).all()
    return projected, dict(
        projections=projections,
        conflicting_ordered_pairs=int((gram < 0).sum()),
        original_norm=float(g.sum(0).norm()),
        projected_norm=float(projected.sum(0).norm()),
    )
