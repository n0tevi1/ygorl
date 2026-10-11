"""Experimental game-weighted group-MSE objectives for a frozen-actor critic."""

import math

import torch


def objective(pred, target, weight, groups, temperature=None):
    """Ordinary MSE or a smooth maximum of target/role group MSEs.

    Smaller positive temperatures emphasize high-error groups. Group labels are
    training-only metadata. Subtracting log(group count) normalizes the smooth
    maximum without changing its derivative.
    """
    if temperature is not None and (not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("temperature must be finite and positive")
    totals = torch.zeros(int(groups.max()) + 1, dtype=pred.dtype, device=pred.device).index_add_(0, groups, weight)
    assert bool((totals > 0).all())
    error = pred - target[:, None]
    sums = torch.zeros_like(totals).index_add_(0, groups, weight * error.square().mean(-1))
    losses = sums / totals
    if temperature is None:
        shares = totals / totals.sum()
        value = (shares * losses).sum()
    else:
        shares = (losses / temperature).softmax(0)
        value = temperature * (torch.logsumexp(losses / temperature, 0) - math.log(len(losses)))
    return value, losses, shares, totals


def coefficients(pred, target, weight, groups, temperature=None):
    """Exact two-pass derivative; parameters stay fixed until all batches finish."""
    with torch.no_grad():
        _, _, shares, totals = objective(pred, target, weight, groups, temperature)
        return 2 / pred.shape[-1] * (pred - target[:, None]) * (shares[groups] * weight / totals[groups])[:, None]
