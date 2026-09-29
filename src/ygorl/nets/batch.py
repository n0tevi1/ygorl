"""Observation dicts (numpy, docs/encoding.md) -> batched torch tensors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

INDEX_KEYS = ("cards", "globals", "actions", "events")  # int32 feature arrays -> int64
MASK_KEYS = ("action_mask", "event_mask")  # 0/1 -> bool
OBS_KEYS = INDEX_KEYS + MASK_KEYS

Batch = dict[str, torch.Tensor]


def to_tensors(obs: Mapping[str, np.ndarray], device: torch.device | str | None = None) -> Batch:
    """Already-batched arrays (leading batch dimension) -> tensors; unknown keys are dropped.

    ``events`` / ``event_mask`` are optional (``EncodedVecEnv(event_length=0)``).
    """
    out: Batch = {}
    for key in OBS_KEYS:
        if key not in obs:
            continue
        t = torch.as_tensor(np.asarray(obs[key]))
        out[key] = (t != 0 if key in MASK_KEYS else t.long()).to(device)
    return out


def collate(observations: Sequence[Mapping[str, np.ndarray]], device: torch.device | str | None = None) -> Batch:
    """Stack a list of per-decision obs dicts (``EncodedEvent.obs``) into one batch."""
    if not observations:
        raise ValueError("empty batch")
    keys = [k for k in OBS_KEYS if k in observations[0]]
    return to_tensors({k: np.stack([o[k] for o in observations]) for k in keys}, device)


def policy_logits(module: nn.Module, batch: Batch) -> Tensor:
    """The actor's action logits of any policy module: ``module.policy_logits(batch)`` when it has one (an actor-critic
    skips its critic; snapshot opponents, KL references, batched evaluation), else ``module(batch).logits``."""
    fn = getattr(module, "policy_logits", None)
    return fn(batch) if fn is not None else module(batch).logits
