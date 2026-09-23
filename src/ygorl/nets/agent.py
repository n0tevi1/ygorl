"""Policy checkpoints and the network policy for :class:`~ygorl.agents.PolicyAgent` (T4a.2).

A checkpoint (``torch.save`` of plain data) holds everything needed to rebuild the network and feed it:
the :class:`NetConfig`, the ``state_dict``, the card vocabulary (passwords in index order), the event
window length and the environment stamp of the data it was trained on. Frozen text tables are not
stored (non-persistent buffers, docs/nets.md); a checkpoint built with them needs the same
``TextFeatures`` at load time.

:class:`NetPolicy` turns a network into a ``PolicyAgent`` policy on the Python ``DecisionPoint`` path
(``Duel.run``, the arena): it encodes each point with :class:`~ygorl.env.observer.PointObserver` from
``observe(point, core)`` calls and returns one logit per legal action::

    agent = PolicyAgent(NetPolicy.from_checkpoint("out/bc/policy.pt"), seed=0)
    Duel(seed, None, deck_a, deck_b).run(agent, RandomAgent(1))

or, by name, ``make_agent("policy:out/bc/policy.pt")`` (``ygorl arena --agent-a policy:PATH``).
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ygorl.cards.cdb import CardVocab
from ygorl.env.encoding import MAX_OPTIONS
from ygorl.env.observer import PointObserver
from ygorl.nets.batch import collate
from ygorl.nets.config import NetConfig
from ygorl.nets.heads import MASKED_LOGIT
from ygorl.nets.policy import PolicyNet
from ygorl.nets.text import TextFeatures

CHECKPOINT_FORMAT = "ygorl-policy"
CHECKPOINT_VERSION = 1


@dataclass
class PolicyCheckpoint:
    net: PolicyNet
    vocab: CardVocab
    event_length: int
    environment: dict | None = None  # {"version", "fingerprint"} of the training data; None without an environment
    meta: dict = field(default_factory=dict)  # trainer, data, hyper-parameters, metrics


def vocab_passwords(vocab: CardVocab) -> list[int]:
    return [vocab.password(i) for i in range(CardVocab.FIRST_INDEX, len(vocab))]


def save_checkpoint(path: str | Path, net: PolicyNet, vocab: CardVocab, *, event_length: int,
                    environment: dict | None = None, meta: dict | None = None) -> Path:  # fmt: skip
    """Write a checkpoint that :func:`load_checkpoint` (and ``PolicyAgent`` via :class:`NetPolicy`) can read."""
    if len(vocab) != net.cfg.vocab_size:
        raise ValueError(f"vocab has {len(vocab)} entries, the network {net.cfg.vocab_size}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    data = {"format": CHECKPOINT_FORMAT, "format_version": CHECKPOINT_VERSION, "config": net.cfg.to_dict(),
            "state_dict": state, "vocab": vocab_passwords(vocab), "event_length": int(event_length),
            "environment": environment, "meta": meta or {}}  # fmt: skip
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path: str | Path, text: TextFeatures | None = None) -> PolicyCheckpoint:
    """Rebuild the network (eval mode, CPU) of a checkpoint written by :func:`save_checkpoint`."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict) or data.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path}: not a ygorl policy checkpoint")
    if data.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError(f"{path}: unsupported checkpoint format_version {data.get('format_version')}")
    cfg = NetConfig.from_dict(data["config"])
    if (cfg.card_text_dim or cfg.effect_text_dim) and text is None:
        raise ValueError(f"{path}: the network was trained with frozen text tables; pass the same TextFeatures")
    vocab = CardVocab(data["vocab"])
    if len(vocab) != cfg.vocab_size:
        raise ValueError(f"{path}: vocab has {len(vocab)} entries, the config {cfg.vocab_size}")
    net = PolicyNet(cfg, text)
    net.load_state_dict(data["state_dict"])
    net.eval()
    return PolicyCheckpoint(net, vocab, data["event_length"], data.get("environment"), data.get("meta", {}))


@functools.lru_cache(maxsize=4)
def _cached_checkpoint(path: str, mtime_ns: int) -> PolicyCheckpoint:
    return load_checkpoint(path)


def cached_checkpoint(path: str | Path) -> PolicyCheckpoint:
    """:func:`load_checkpoint` once per process and file version (arena games build one agent per game)."""
    p = Path(path).resolve()
    return _cached_checkpoint(str(p), p.stat().st_mtime_ns)


class NetPolicy:
    """``point -> logits`` of a :class:`PolicyNet` over ``point.actions``, for :class:`~ygorl.agents.PolicyAgent`.

    Needs ``observe(point, core)`` at every decision point of the duel (``Duel.run`` does it through
    ``PolicyAgent.observe``). Actions beyond the 128 encoded rows get :data:`MASKED_LOGIT` (probability 0).
    One instance per duel; a point index that goes backwards starts a new duel.
    """

    def __init__(self, net: PolicyNet | None, vocab: CardVocab | None, *, event_length: int = 128, cards=None,
                 checkpoint: str | Path | None = None) -> None:  # fmt: skip
        if net is None and checkpoint is None:
            raise ValueError("give a network or a checkpoint path")
        self._net, self._vocab, self._event_length = net, vocab, event_length
        self._checkpoint = checkpoint
        self._cards = cards
        self._observer: PointObserver | None = None
        self._core = None
        self._last = -1

    @classmethod
    def from_checkpoint(cls, path: str | Path, cards=None) -> NetPolicy:
        """Lazy: the file is read (once per process) at the first decision, not here."""
        if not Path(path).is_file():
            raise ValueError(f"no policy checkpoint at {path}")
        return cls(None, None, checkpoint=path, cards=cards)

    @property
    def net(self) -> PolicyNet:
        if self._net is None:
            ckpt = cached_checkpoint(self._checkpoint)
            self._net, self._vocab, self._event_length = ckpt.net, ckpt.vocab, ckpt.event_length
        return self._net

    def observe(self, point, core) -> None:
        if self._observer is None or point.index < self._last:
            _ = self.net  # loads the checkpoint: vocab, event length
            if self._cards is None:
                from ygorl.engine.duel import default_cards

                self._cards = default_cards()
            self._observer = PointObserver(self._cards, self._vocab, self._event_length)
        self._observer.observe(point)
        self._core = core
        self._last = point.index

    def __call__(self, point) -> list[float]:
        if self._observer is None or self._core is None:
            raise RuntimeError("NetPolicy needs observe(point, core) at every decision point (use Duel.run)")
        obs = self._observer.encode(point, self._core)
        with torch.no_grad():
            logits = self.net(collate([obs])).logits[0]
        n = len(point.actions)
        k = min(n, MAX_OPTIONS)
        return logits[:k].tolist() + [MASKED_LOGIT] * (n - k)


def policy_agent_factory(arg: str | None, seed: int):
    """Registry factory for ``policy:PATH[@greedy][@t=T]`` (sampling at temperature 1 by default)."""
    from ygorl.agents.policy import PolicyAgent
    from ygorl.agents.registry import parse_policy_arg

    path, greedy, temperature = parse_policy_arg(arg)
    return PolicyAgent(NetPolicy.from_checkpoint(path), seed=seed, greedy=greedy, temperature=temperature)


__all__ = ["CHECKPOINT_FORMAT", "NetPolicy", "PolicyCheckpoint", "cached_checkpoint", "load_checkpoint",
           "policy_agent_factory", "save_checkpoint", "vocab_passwords"]  # fmt: skip
