"""Bounded deck-exhaustion features for critic diagnostics, not reward shaping."""

import torch


def deckrace_features(resources):
    """Both deck pressures and their interaction, independent of card/role labels.

    Resource columns 3 and 8 store own/opponent counts divided by 40. Pressure
    is zero at >=6 cards, rises through counts 5..0, and is one at zero. A zero
    deck does not itself mean defeat: these are inputs, never forced values.
    """
    counts = resources[:, [3, 8]] * 40
    if not bool(torch.isfinite(counts).all()) or bool((counts < 0).any()):
        raise ValueError("deck counts must be finite and nonnegative")
    pressure = ((6 - counts) / 6).clamp(0, 1)
    return torch.cat((pressure, pressure[:, :1] * pressure[:, 1:]), dim=-1)
