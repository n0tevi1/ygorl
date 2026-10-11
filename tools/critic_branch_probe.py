"""Research-only counterfactual Q supervision, equally weighted by root state."""

from __future__ import annotations

import torch


def branch_squared_error(pred, target, mask, *, selected=None):
    """Return one loss per root, averaging only supported action targets.

    Targets are frozen Monte Carlo branch means. Raw calibrated Q is used so
    saturation cannot remove the recovery gradient. ``selected`` restricts the
    control arm to its original action; the all-action arm uses the whole mask.
    Callers weight roots equally, regardless of the number of legal actions.
    """
    if pred.ndim != 2 or target.shape != pred.shape or mask.shape != pred.shape:
        raise ValueError("predictions, targets and masks must share [roots, actions] shape")
    if mask.dtype != torch.bool or not bool(mask.any(-1).all()):
        raise ValueError("each root requires at least one supported action")
    if selected is not None:
        if selected.shape != pred.shape[:1] or selected.dtype != torch.long:
            raise ValueError("one integer selected action per root is required")
        if bool(((selected < 0) | (selected >= pred.shape[1])).any()):
            raise ValueError("selected action out of range")
        if not bool(mask.gather(1, selected[:, None]).all()):
            raise ValueError("selected action must be supported")
        mask = torch.zeros_like(mask).scatter_(1, selected[:, None], True)
    error = pred.double() - target.detach().double()
    # Mask before squaring: ignored padding may contain nonfinite sentinels.
    return error.masked_fill(~mask, 0).square().sum(-1) / mask.sum(-1)


def relative_branch_squared_error(pred, target, mask):
    """Supervise action differences, without rewarding a shared value shift.

    Centered MSE is half the mean squared error over all ordered action-pair
    differences, including equal-action pairs. The normalization prevents roots
    with more actions from dominating. A common state correction cancels out.
    """
    # Reuse validation before centering; ignore unsupported padding in the mean.
    branch_squared_error(pred, target, mask)
    count = mask.sum(-1, keepdim=True)
    p = pred.double().masked_fill(~mask, 0)
    z = target.detach().double().masked_fill(~mask, 0)
    return branch_squared_error(p - p.sum(-1, keepdim=True) / count, z - z.sum(-1, keepdim=True) / count, mask)
