"""Observation dicts (numpy, docs/encoding.md) -> batched torch tensors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

INDEX_KEYS = (
    "cards",
    "globals",
    "actions",
    "events",
    "selection_history",
    "attack_history",
)  # int32 feature arrays -> int64
MASK_KEYS = ("action_mask", "event_mask")  # 0/1 -> bool
OBS_KEYS = INDEX_KEYS + MASK_KEYS

Lengths = tuple[int, int, int | None]  # (n_cards, n_actions, n_events), see Batch


class Batch(dict):
    """Collated observation tensors, ``str -> Tensor``: a dict with one extra attribute, ``lengths``.

    ``lengths = (n_cards, n_actions, n_events)`` is the longest valid prefix of the padded card, action and event
    axes over the batch (``n_events`` None without events; the attribute None when unknown), found on the CPU side
    from the numpy arrays by :func:`to_tensors` / :func:`collate`. :func:`ygorl.nets.policy.trim_padding` cuts to
    these lengths without reading the device tensors back (no synchronization); a plain dict (an indexed minibatch,
    another model's observations) is trimmed by scanning its masks, to the same widths. ``dict(batch)`` drops the
    attribute.
    """

    lengths: Lengths | None

    def __init__(self, *args, lengths: Lengths | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lengths = lengths


def _prefix(mask: np.ndarray) -> int:
    """Longest valid prefix of ``mask`` ``[..., N]`` over all rows (valid entries first), at least 1; 0 when N = 0."""
    n = mask.shape[-1]
    if n == 0:
        return 0
    cols = np.flatnonzero(mask.reshape(-1, n).any(0))
    return int(cols[-1]) + 1 if len(cols) else 1


def prefix_lengths(obs: Mapping[str, np.ndarray]) -> Lengths | None:
    """``(n_cards, n_actions, n_events)`` of batched numpy observations (:class:`Batch`); None without the card and
    action tables (other models' observations)."""
    if "cards" not in obs or "actions" not in obs:
        return None
    n_cards = _prefix(np.asarray(obs["cards"])[..., 0] > 0)
    n_actions = _prefix(np.asarray(obs["action_mask"]))
    n_events = None
    if obs.get("events") is not None and "event_mask" in obs:
        n_events = _prefix(np.asarray(obs["event_mask"]))
    return n_cards, n_actions, n_events


def to_tensors(obs: Mapping[str, np.ndarray], device: torch.device | str | None = None) -> Batch:
    """Already-batched arrays (leading batch dimension) -> tensors; unknown keys are dropped.

    ``events`` / ``event_mask`` are optional (``EncodedVecEnv(event_length=0)``). Index arrays are uploaded as they
    are and widened to int64 on the device (half the bytes of a host-side cast, the same values).
    """
    arrays = {key: np.asarray(obs[key]) for key in OBS_KEYS if key in obs}
    out = Batch(lengths=prefix_lengths(arrays))
    for key, a in arrays.items():
        t = torch.as_tensor(a).to(device)
        out[key] = t != 0 if key in MASK_KEYS else t.long()
    return out


def stack_arrays(items: Sequence[Mapping[str, np.ndarray]], keys: Sequence[str],
                 device: torch.device | str | None = None) -> tuple[dict[str, np.ndarray], dict[str, Tensor]]:  # fmt: skip
    """``{key: stack([item[key] for item in items])}`` as numpy arrays on the host and as tensors of the same dtype
    on ``device``. On a CUDA device all keys are stacked into one pinned staging buffer, uploaded with one copy and
    split into views (the arrays must be int32 of one shape per key, what the C++ encoder emits); elsewhere one
    stack per key."""
    first = items[0]
    if device is None or torch.device(device).type != "cuda" or any(np.asarray(first[k]).dtype != np.int32 for k in keys):  # fmt: skip
        host = {k: np.stack([it[k] for it in items]) for k in keys}
        return host, {k: torch.as_tensor(a).to(device) for k, a in host.items()}
    n = len(items)
    shapes = {k: (n, *np.shape(first[k])) for k in keys}
    sizes = {k: int(np.prod(shapes[k])) for k in keys}
    staging = torch.empty(sum(sizes.values()), dtype=torch.int32, pin_memory=True)
    flat, host, offset = staging.numpy(), {}, 0
    for k in keys:
        host[k] = flat[offset : offset + sizes[k]].reshape(shapes[k])
        np.stack([it[k] for it in items], out=host[k])
        offset += sizes[k]
    # the caching host allocator keeps the pinned block until this copy has completed (the event is recorded)
    uploaded, dev, offset = staging.to(device, non_blocking=True), {}, 0
    for k in keys:
        dev[k] = uploaded[offset : offset + sizes[k]].view(shapes[k])
        offset += sizes[k]
    return host, dev


def collate(observations: Sequence[Mapping[str, np.ndarray]], device: torch.device | str | None = None) -> Batch:
    """Stack a list of per-decision obs dicts (``EncodedEvent.obs``) into one batch (one upload, see
    :func:`stack_arrays`)."""
    if not observations:
        raise ValueError("empty batch")
    if any(("selection_history" in o) != ("selection_history" in observations[0]) for o in observations):
        raise ValueError("cannot mix selection-history encoding schemas in one batch")
    if any(("attack_history" in o) != ("attack_history" in observations[0]) for o in observations):
        raise ValueError("cannot mix attack-history encoding schemas in one batch")
    keys = [k for k in OBS_KEYS if k in observations[0]]
    host, dev = stack_arrays(observations, keys, device)
    out = Batch(lengths=prefix_lengths(host))
    for key, t in dev.items():
        out[key] = t != 0 if key in MASK_KEYS else t.long()
    return out


def policy_logits(module: nn.Module, batch: Batch) -> Tensor:
    """The actor's action logits of any policy module: ``module.policy_logits(batch)`` when it has one (an actor-critic
    skips its critic; snapshot opponents, KL references, batched evaluation), else ``module(batch).logits``."""
    fn = getattr(module, "policy_logits", None)
    return fn(batch) if fn is not None else module(batch).logits
