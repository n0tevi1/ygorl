"""Research losses for comparing clipped and unclipped calibrated critic fitting.

Reporting may still bound predictions; training on a clamped prediction removes
its gradient outside the interval. This helper does not change production PPO.
"""


def calibrated_squared_error(raw, target, offset, *, clip_prediction):
    """Per-row mean Q/V error; caller supplies fixed game weights separately."""
    if raw.ndim != 2 or raw.shape[1] != 2 or offset.shape != raw.shape or target.shape != raw.shape[:1]:
        raise ValueError("expected raw/offset [N,2] and target [N]")
    prediction = raw.double() + offset.detach().double()
    if clip_prediction:
        prediction = prediction.clamp(-1, 1)
    return (prediction - target[:, None]).square().mean(-1)
