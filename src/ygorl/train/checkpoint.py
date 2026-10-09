"""Checkpoints (T4b.4, #122): writing PPO checkpoints, and rebuilding every network from a PPO or policy checkpoint.

A PPO checkpoint is one self-contained ``.pt`` file (``torch.save`` of plain containers and tensors, loadable with
``weights_only=True``):

| key | content |
|-----|---------|
| ``format`` | :data:`FORMAT` |
| ``config`` | ``TrainConfig.to_dict()`` (includes the PPO hyper-parameters and the critic options) |
| ``net_config`` | ``NetConfig.to_dict()`` of the actor |
| ``vocab`` | the ``CardVocab`` JSON written by ``CardVocab.save`` (vocab indices are part of the model) |
| ``environment`` | ``Environment.stamp()`` of the training environment, or None |
| ``learner`` | model / EMA reference / optimizer state, update count |
| ``pool`` / ``schedule`` | snapshot pool (keep-best included) and deal counter + RNG, for resuming |
| ``counters`` / ``rng`` | progress counters, torch RNG states |

A policy checkpoint (``ygorl.nets.agent``, e.g. from BC) holds only an actor; :func:`checkpoint_format` tells the
two apart. This module is the one place that rebuilds networks from either:

- :func:`load_actor`: the actor (:class:`ygorl.nets.PolicyNet`) of any checkpoint, for playing and evaluation;
  :func:`load_policy` is the same for PPO checkpoints only;
- :func:`load_actor_critic`: the whole :class:`ActorCritic` of a PPO checkpoint, with its critic options
  (:class:`CriticConfig`); :func:`build_actor_critic` builds a fresh one (the trainer, the snapshot pool);
- :func:`warm_start`: copy a loaded actor into a new network, which may add card views;
- :class:`Signature`: what a network reads (card vocab, event window length); ``a.mismatches(b)`` says in words
  why two checkpoints / runs cannot share observations;
- :func:`torch_device`: the device networks run on (with the ROCm settings they need).

Frozen text tables and card facts are not stored in checkpoints (docs/nets.md): they are reloaded from ``text_dir``
or, for a PPO checkpoint, from its run's ``config["text_dir"]``; ``ygorl.nets.text.require_tables`` is the check.
"""

from __future__ import annotations

import functools
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from ygorl.cards.cdb import CardVocab
from ygorl.nets.actor_critic import ActorCritic
from ygorl.nets.agent import vocab_passwords
from ygorl.nets.config import NetConfig
from ygorl.nets.policy import PolicyNet
from ygorl.nets.text import TextFeatures, frozen_views, require_tables

FORMAT = "ygorl-ppo-1"
ACTOR_PREFIX = "actor."
# A warm start may add card views (text tables, card facts, ID dropout) to the checkpoint's network: those modules
# start fresh, every other weight must fit (docs/nets.md「卡片事实」)
CARD_VIEW_FIELDS = frozenset({"card_text", "effect_text", "card_text_dim", "effect_text_dim", "card_facts",
                              "n_archetypes", "n_archetype_slots", "n_reference_slots", "n_categories", "n_queries",
                              "id_dropout"})  # fmt: skip
CARD_VIEW_MODULES = ("text_proj", "archetype", "reference_proj", "category_proj", "query_proj", "effect.proj")


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
    """The raw state of a PPO checkpoint (for resuming); networks come from the ``load_*`` functions below."""
    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or state.get("format") != FORMAT:
        raise ValueError(f"{path}: not a ygorl PPO checkpoint (format {FORMAT})")
    return state


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


def torch_device(device: str | torch.device) -> torch.device:
    """``device`` as a ``torch.device``. On a ROCm GPU this also turns on the fused attention kernels with a mask
    (``TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`` unless set; docs/benchmarks.md), before any attention runs."""
    device = torch.device(device)
    if device.type == "cuda" and torch.version.hip:
        os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    return device


# -- what a network reads -----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Signature:
    """What a network reads besides its weights: the card vocab (the same index must be the same card) and the event
    window length of its observations. Two networks can play on one ``EncodedVecEnv`` iff their signatures are equal;
    hashable, so it also groups checkpoints."""

    passwords: tuple[int, ...]  # vocab passwords in index order (from CardVocab.FIRST_INDEX)
    event_length: int
    selection_history: bool = False

    @classmethod
    def of(cls, vocab: CardVocab, event_length: int, selection_history: bool = False) -> Signature:
        return cls(tuple(vocab_passwords(vocab)), int(event_length), selection_history)

    def mismatches(self, other: Signature, *, event_length: bool = True) -> list[str]:
        """Why ``other`` cannot read this signature's observations, one readable reason per difference (empty =
        compatible); ``event_length=False`` still checks vocabulary and selection schema."""
        out = []
        a, b = self.passwords, other.passwords
        if a != b:
            n = min(len(a), len(b))
            first = next((i for i in range(n) if a[i] != b[i]), n)
            what = (f"index {CardVocab.FIRST_INDEX + first} is card {a[first]} vs {b[first]}" if first < n
                    else f"{len(a)} vs {len(b)} cards")  # fmt: skip
            out.append(f"the card vocab differs ({what})")
        if event_length and self.event_length != other.event_length:
            out.append(f"{self.event_length} vs {other.event_length} event tokens per observation")
        if self.selection_history != other.selection_history:
            out.append("selection-history encoding differs")
        return out


# -- the critic ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CriticConfig:
    """The critic options of an :class:`ActorCritic` (``TrainConfig`` fields ``privileged_critic``,
    ``privileged_dim``, ``critic_hidden``, ``shared_backbone``, ``critic_deck_order``)."""

    privileged: bool = True
    privileged_dim: int = 64
    hidden: int = 128
    shared_backbone: bool = True
    deck_order: bool = False

    @classmethod
    def from_train_config(cls, c: Mapping) -> CriticConfig:
        """From ``TrainConfig.to_dict()``, as stored in a PPO checkpoint's ``config``."""
        return cls(bool(c["privileged_critic"]), int(c["privileged_dim"]), int(c["critic_hidden"]),
                   bool(c["shared_backbone"]), bool(c.get("critic_deck_order", False)))  # fmt: skip


def build_actor_critic(net_config: NetConfig, text: TextFeatures | None, critic: CriticConfig) -> ActorCritic:
    """A freshly initialized actor-critic (CPU, train mode); every ActorCritic is built here."""
    return ActorCritic(net_config, text, privileged=critic.privileged, privileged_dim=critic.privileged_dim,
                       critic_hidden=critic.hidden, shared_backbone=critic.shared_backbone,
                       deck_order=critic.deck_order)  # fmt: skip


# -- loading ------------------------------------------------------------------------------------------------------


@dataclass
class LoadedPolicy:
    net: PolicyNet  # eval mode, no gradients, on CPU
    vocab: CardVocab
    event_length: int
    net_config: NetConfig
    environment: dict | None
    update: int  # PPO updates (0 for a policy checkpoint)
    path: Path

    @property
    def signature(self) -> Signature:
        return Signature.of(self.vocab, self.event_length, self.net_config.selection_history)


@dataclass
class LoadedActorCritic:
    model: ActorCritic  # eval mode, no gradients, on CPU
    critic: CriticConfig
    vocab: CardVocab
    event_length: int
    net_config: NetConfig
    environment: dict | None
    update: int
    path: Path

    @property
    def signature(self) -> Signature:
        return Signature.of(self.vocab, self.event_length, self.net_config.selection_history)


def _tables(cfg: NetConfig, vocab: CardVocab, where: str | Path | None, path) -> TextFeatures | None:
    """The frozen tables ``cfg`` needs, read from ``where`` (None when it needs none)."""
    if not frozen_views(cfg):
        return None
    text = TextFeatures.load(where, vocab) if where is not None else None
    require_tables(cfg, text, path)
    return text


def _frozen(module):
    module.eval()
    return module.requires_grad_(False)


def _ppo(path: str | Path, text_dir: str | Path | None):
    state = load_checkpoint(path)
    cfg = NetConfig.from_dict(state["net_config"])
    vocab = vocab_from_text(state["vocab"])
    return state, cfg, vocab, _tables(cfg, vocab, text_dir or state["config"].get("text_dir"), path)


def load_policy(path: str | Path, text_dir: str | Path | None = None) -> LoadedPolicy:
    """The actor of a PPO checkpoint."""
    state, cfg, vocab, text = _ppo(path, text_dir)
    net = PolicyNet(cfg, text)
    model = state["learner"]["model"]
    net.load_state_dict({k[len(ACTOR_PREFIX) :]: v for k, v in model.items() if k.startswith(ACTOR_PREFIX)})
    return LoadedPolicy(_frozen(net), vocab, int(state["config"]["event_length"]), cfg, state.get("environment"),
                        int(state["learner"]["updates"]), Path(path))  # fmt: skip


def load_actor(path: str | Path, text_dir: str | Path | None = None) -> LoadedPolicy:
    """The actor of a PPO checkpoint or of a policy checkpoint (``ygorl.nets.agent``, e.g. BC)."""
    from ygorl.nets import agent

    fmt = checkpoint_format(path)
    if fmt == FORMAT:
        return load_policy(path, text_dir)
    if fmt != agent.CHECKPOINT_FORMAT:
        raise ValueError(f"{path}: not a ygorl policy checkpoint (format {fmt!r})")
    text = None
    if text_dir is not None:
        data = torch.load(Path(path), map_location="cpu", weights_only=True, mmap=True)
        text = _tables(NetConfig.from_dict(data["config"]), CardVocab(data["vocab"]), text_dir, path)
    ckpt = agent.load_checkpoint(path, text)  # which checks the tables too (require_tables)
    return LoadedPolicy(_frozen(ckpt.net), ckpt.vocab, ckpt.event_length, ckpt.net.cfg, ckpt.environment, 0,
                        Path(path))  # fmt: skip


def load_actor_critic(path: str | Path, text_dir: str | Path | None = None) -> LoadedActorCritic:
    """The whole actor-critic of a PPO checkpoint (the learner's model), built with the run's critic options. A policy
    checkpoint has no critic: :func:`warm_start` puts its actor into a new actor-critic."""
    if checkpoint_format(path) != FORMAT:
        raise ValueError(f"{path}: not a PPO checkpoint (a policy checkpoint has no critic)")
    state, cfg, vocab, text = _ppo(path, text_dir)
    critic = CriticConfig.from_train_config(state["config"])
    model = build_actor_critic(cfg, text, critic)
    model.load_state_dict(state["learner"]["model"])
    return LoadedActorCritic(_frozen(model), critic, vocab, int(state["config"]["event_length"]), cfg,
                             state.get("environment"), int(state["learner"]["updates"]), Path(path))  # fmt: skip


def warm_start(actor: PolicyNet, source: LoadedPolicy) -> list[str]:
    """Copy ``source``'s weights into ``actor``, a new network over the same vocab. The two networks must be the same
    except for added card views (:data:`CARD_VIEW_FIELDS`) or enabling selection history. New card views
    keep their zero-output initialization; selection history inserts a zero event-type row. Returns the config fields that differ (empty = an exact copy)."""
    old, new = source.net_config.to_dict(), actor.cfg.to_dict()
    differ = sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k))
    if set(differ) - CARD_VIEW_FIELDS - {"selection_history"} or (
        old["selection_history"] and not new["selection_history"]
    ):
        raise ValueError(f"{source.path}: its network {old} differs from the configured {new} "
                         "(set the same net options)")  # fmt: skip
    state = source.net.state_dict()
    if "selection_history" in differ:
        from ygorl.env.events import EVENT_TYPES

        # CategoricalEmbedding concatenates columns: insert a type row and shift
        # every later column, rather than silently reinterpreting its old weights.
        boundary = len(EVENT_TYPES) + 1
        target = actor.state_dict()
        for name, value in list(state.items()):
            if name.endswith("embed.categorical.table.weight"):
                expanded = target[name].clone()
                if expanded.shape[0] != value.shape[0] + 1:
                    raise ValueError("selection-history embedding layout mismatch")
                expanded[:boundary] = value[:boundary]
                expanded[boundary].zero_()
                expanded[boundary + 1 :] = value[boundary:]
                state[name] = expanded
    missing, unexpected = actor.load_state_dict(state, strict=not differ)
    if unexpected or any(not any(m in k for m in CARD_VIEW_MODULES) for k in missing):
        raise ValueError(f"{source.path}: weights do not fit (missing {missing}, unexpected {unexpected})")
    return differ


__all__ = ["FORMAT", "CriticConfig", "LoadedActorCritic", "LoadedPolicy", "Signature", "build_actor_critic",
           "checkpoint_format", "load_actor", "load_actor_critic", "load_checkpoint", "load_policy", "save_checkpoint",
           "torch_device", "vocab_from_text", "vocab_to_text", "warm_start"]  # fmt: skip
