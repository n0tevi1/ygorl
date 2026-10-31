"""Export the actor as a serializable traced graph for ygorl._infer (C++ snapshot inference).

See tools/export_snapshot_actor.py for the command line; the trainer calls :func:`export_actor` directly
when ``--opp-cpp`` is on. Inference always takes the embedding_bag path (identical forward values;
``encoders.CategoricalEmbedding`` guards on ``is_grad_enabled``), the graph accepts any batch size and
prefix-sliced widths.
"""

import dataclasses
import hashlib
import json
from pathlib import Path

import torch
from torch import nn


def _example_batch(net_config, event_length: int) -> dict:
    """Full-width zeros of the fixed observation layout (docs/encoding.md; shapes/dtypes are all a trace needs)."""
    obs = {
        "cards": torch.zeros(2, 160, 23, dtype=torch.long),
        "globals": torch.zeros(2, 22, dtype=torch.long),
        "actions": torch.zeros(2, 128, 10, dtype=torch.long),
        "action_mask": torch.zeros(2, 128, dtype=torch.bool),
        "events": torch.zeros(2, event_length, 20, dtype=torch.long),
        "event_mask": torch.ones(2, event_length, dtype=torch.bool),
    }
    # schema markers (encoding.md "Accepted attack choice" / selection history): the network checks presence
    marker = lambda: torch.ones(2, 1, dtype=torch.long)  # noqa: E731
    if getattr(net_config, "selection_history", False):
        obs["selection_history"] = marker()
    if getattr(net_config, "attack_history", False):
        obs["attack_history"] = marker()
    return obs


class ActPath(nn.Module):
    """The traceable policy head: observation dict -> masked action logits."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, obs):
        return self.net.logits(self.net._features(obs))


def export_actor(loaded, out_path: Path) -> dict:
    """Trace the actor of a ``load_policy`` result and save it; returns the artifact metadata."""
    net = loaded.net.eval()
    actpath = ActPath(net).eval()
    example = _example_batch(net.cfg, loaded.event_length)
    with torch.no_grad():
        traced = torch.jit.trace(actpath, example, check_trace=False)
        torch.jit.save(traced, str(out_path))
    cfg = dataclasses.asdict(net.cfg)
    fingerprint = hashlib.sha256(
        json.dumps({"net": cfg, "event_length": loaded.event_length}, sort_keys=True).encode()
    ).hexdigest()[:16]
    return {"format": "ygorl-actpath-trace-v1", "fingerprint": fingerprint, "path": str(out_path)}
