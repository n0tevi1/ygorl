"""PPO checkpoints (T4b.4): one self-contained ``.pt`` file per checkpoint.

Contents (``torch.save`` of plain containers and tensors, loadable with ``weights_only=True``):

| key | content |
|-----|---------|
| ``format`` | :data:`FORMAT` |
| ``config`` | ``TrainConfig.to_dict()`` (includes the PPO hyper-parameters) |
| ``net_config`` | ``NetConfig.to_dict()`` of the actor |
| ``vocab`` | the ``CardVocab`` JSON written by ``CardVocab.save`` (vocab indices are part of the model) |
| ``environment`` | ``Environment.stamp()`` of the training environment, or None |
| ``learner`` | model / EMA reference / optimizer state, update count |
| ``pool`` / ``schedule`` | snapshot pool (keep-best included) and deal counter + RNG, for resuming |
| ``counters`` / ``rng`` | progress counters, torch RNG states |

:func:`load_policy` rebuilds only the actor (a :class:`ygorl.nets.PolicyNet`) for playing and evaluation.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch

from ygorl.cards.cdb import CardVocab
from ygorl.nets.config import NetConfig
from ygorl.nets.policy import PolicyNet
from ygorl.nets.text import TextFeatures

FORMAT = "ygorl-ppo-1"
ACTOR_PREFIX = "actor."


def vocab_to_text(vocab: CardVocab, path: Path) -> str:
    """Write ``vocab`` with ``CardVocab.save`` and return the file's text (embedded in checkpoints)."""
    vocab.save(path)
    return path.read_text()


def vocab_from_text(text: str) -> CardVocab:
    data = json.loads(text)
    if data.get("first_index") != CardVocab.FIRST_INDEX:
        raise ValueError("checkpoint vocab layout mismatch")
    return CardVocab(data["passwords"])


def save_checkpoint(state: dict, path: str | Path) -> Path:
    """Atomically write ``state`` (write to a temporary file, then rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"format": FORMAT, **state}, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path: str | Path) -> dict:
    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or state.get("format") != FORMAT:
        raise ValueError(f"{path}: not a ygorl PPO checkpoint (format {FORMAT})")
    return state


@dataclass
class LoadedPolicy:
    net: PolicyNet  # eval mode, on CPU
    vocab: CardVocab
    event_length: int
    net_config: NetConfig
    environment: dict | None
    update: int
    path: Path


def load_policy(path: str | Path, text_dir: str | Path | None = None) -> LoadedPolicy:
    """The actor of a checkpoint. Frozen text tables (if the net uses them) are reloaded from ``text_dir``
    or the training run's ``config["text_dir"]`` (they are not stored in checkpoints, docs/nets.md)."""
    state = load_checkpoint(path)
    cfg = NetConfig.from_dict(state["net_config"])
    vocab = vocab_from_text(state["vocab"])
    text = None
    if cfg.card_text_dim or cfg.effect_text_dim:
        where = text_dir or state["config"].get("text_dir")
        if where is None:
            raise ValueError(f"{path}: the network uses frozen text tables; pass their directory")
        text = TextFeatures.load(where, vocab)
    net = PolicyNet(cfg, text)
    model = state["learner"]["model"]
    net.load_state_dict({k[len(ACTOR_PREFIX) :]: v for k, v in model.items() if k.startswith(ACTOR_PREFIX)})
    net.eval()
    net.requires_grad_(False)
    return LoadedPolicy(net, vocab, int(state["config"]["event_length"]), cfg, state.get("environment"),
                        int(state["learner"]["updates"]), Path(path))  # fmt: skip


__all__ = ["FORMAT", "LoadedPolicy", "load_checkpoint", "load_policy", "save_checkpoint", "vocab_from_text",
           "vocab_to_text"]  # fmt: skip
