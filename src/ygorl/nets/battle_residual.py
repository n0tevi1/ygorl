"""Experimental battle-outcome correction; no actor or action filtering."""

from __future__ import annotations

import copy

import torch
from torch import Tensor, nn


class BattleResidualHead(nn.Module):
    """Keep a numeric predictor fixed and learn a gated contextual correction.

    Callers fit both input normalizations on training data only. The gate is a
    deviation score, not a calibrated probability or an action-quality label.
    """

    def __init__(self, numeric: nn.Module, width: int = 396, hidden: int = 64):
        super().__init__()
        self.numeric = copy.deepcopy(numeric).requires_grad_(False).eval()
        self.correction = nn.Sequential(nn.Linear(width, hidden), nn.ReLU(), nn.Linear(hidden, 9))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.numeric.eval()
        return self

    def forward(self, numeric_inputs: Tensor, context_inputs: Tensor) -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            baseline = self.numeric(numeric_inputs)
        change = self.correction(context_inputs)
        gate = change[..., 8]
        return baseline + gate.sigmoid().unsqueeze(-1) * change[..., :8], gate
