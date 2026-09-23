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

:func:`load_policy` rebuilds only the actor (a :class:`ygorl.nets.PolicyNet`) for playing and evaluation;
:func:`load_actor` does the same for either a PPO checkpoint or a policy checkpoint (``ygorl.nets.agent``, e.g.
from BC), told apart by :func:`checkpoint_format`.
"""

from __future__ import annotations

import functools
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


def checkpoint_format(path: str | Path) -> str | None:
    """The ``format`` field of a ``.pt`` checkpoint (read with ``mmap``: the tensors are not loaded)."""
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"no policy checkpoint at {path}")
    return _format_of(str(p.resolve()), p.stat().st_mtime_ns)


@functools.lru_cache(maxsize=16)
def _format_of(path: str, mtime_ns: int) -> str | None:
    data = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    return data.get("format") if isinstance(data, dict) else None


def load_actor(path: str | Path, text_dir: str | Path | None = None) -> LoadedPolicy:
    """:func:`load_policy` for a PPO checkpoint or a policy checkpoint (``ygorl.nets.agent``, e.g. BC)."""
    from ygorl.nets import agent

    fmt = checkpoint_format(path)
    if fmt == FORMAT:
        return load_policy(path, text_dir)
    if fmt != agent.CHECKPOINT_FORMAT:
        raise ValueError(f"{path}: not a ygorl policy checkpoint (format {fmt!r})")
    text = None
    if text_dir is not None:
        data = torch.load(Path(path), map_location="cpu", weights_only=True, mmap=True)
        text = TextFeatures.load(text_dir, CardVocab(data["vocab"]))
    ckpt = agent.load_checkpoint(path, text)
    ckpt.net.requires_grad_(False)
    return LoadedPolicy(ckpt.net, ckpt.vocab, ckpt.event_length, ckpt.net.cfg, ckpt.environment, 0, Path(path))


def vocab_passwords(vocab: CardVocab) -> list[int]:
    return [vocab.password(i) for i in range(CardVocab.FIRST_INDEX, len(vocab))]


__all__ = ["FORMAT", "LoadedPolicy", "checkpoint_format", "load_actor", "load_checkpoint", "load_policy",
           "save_checkpoint", "vocab_from_text", "vocab_passwords", "vocab_to_text"]  # fmt: skip
