"""``policy:<checkpoint.pt>``: a trained PPO or BC checkpoint as an :class:`~ygorl.agents.base.Agent` (T4b.4).

The network reads the C++ encoder's observations (docs/encoding.md), which need the engine state and the
whole event stream, while the Agent protocol only hands over a :class:`DecisionPoint`. The adapter keeps a
C++ ``HostDuel`` in **lockstep** with the Python duel:

- ``on_duel_start(duel)``: :meth:`Duel.run` calls it once; the agent starts a ``HostDuel`` with the duel's
  core seed, rules and loaded decks (the same game).
- ``on_decision(point, index)``: :meth:`Duel.run` reports every answered decision of *both* seats; the agent
  replays it in its host, so the host's event history matches the game.
- ``act(point)``: ``host.observe()`` is exactly the observation ``EncodedVecEnv`` gives the same decision;
  the network samples (``policy``) or takes the argmax (``policy-greedy``) over the legal rows.

The host must ask the same seat with the same number of candidates as the Python tracker; a mismatch raises
(it would mean the two hosts disagree). Only candidates in the first 128 rows can be chosen (the encoding's
truncation). Curriculum modes and augmented starts are not supported (the C++ step loop has no curriculum
filtering yet). Checkpoints are loaded once per process and path; in worker processes (parallel arena)
PyTorch is limited to one thread per process.
"""

from __future__ import annotations

import multiprocessing as mp
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ygorl.engine.duel import DecisionPoint, Duel


@lru_cache(maxsize=8)
def _load(path: str, mtime: float):
    import torch

    from ygorl.train.checkpoint import load_actor

    if mp.current_process().name != "MainProcess":
        torch.set_num_threads(1)
    return load_actor(path)


def load_cached(path: str | Path):
    p = Path(path).resolve()
    if not p.is_file():
        raise ValueError(f"checkpoint not found: {path}")
    return _load(str(p), p.stat().st_mtime)


class CheckpointAgent:
    name = "policy"

    def __init__(self, path: str | Path, seed: int | None = None, *, greedy: bool = False,
                 temperature: float = 1.0) -> None:  # fmt: skip
        import torch

        if not temperature > 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.policy = load_cached(path)
        self.greedy = greedy
        self.temperature = temperature
        self.generator = torch.Generator().manual_seed(int(seed or 0) & ((1 << 63) - 1))
        self.host = None
        self.last_probs: list[float] | None = None

    # -- Duel.run hooks ------------------------------------------------------------------------------
    def on_duel_start(self, duel: Duel) -> None:
        from ygorl import _core

        cfg = duel.config
        if cfg.curriculum != "full" or cfg.augmented_start:
            raise NotImplementedError("policy agents support only curriculum='full' without augmented_start")
        vocab = self.policy.vocab
        passwords = [vocab.password(i) for i in range(vocab.FIRST_INDEX, len(vocab))]
        host = _core.HostDuel(duel.cards.to_core(), duel.scripts, passwords, event_length=self.policy.event_length)
        p = cfg.player
        player = (p.starting_lp, p.starting_hand, p.draw_per_turn)
        decks = [(list(m), list(e)) for m, e in duel.loaded_decks()]
        host.start(list(duel.core_seed), cfg.rule_flags, player, player, decks, cfg.max_turns, cfg.max_decisions)
        self.host = host

    def on_decision(self, point: DecisionPoint, index: int) -> None:
        if self.host is not None:
            self.host.act(index)

    # -- Agent -----------------------------------------------------------------------------------------
    def act(self, point: DecisionPoint) -> int:
        import torch

        from ygorl.nets import collate

        if self.host is None:
            raise RuntimeError("policy agents play through Duel.run (they need on_duel_start to follow the game)")
        n = len(point.actions)
        if self.host.player() != point.player or len(self.host.actions()) != n:
            raise RuntimeError(f"policy agent lost lockstep at decision {point.index}: host asks seat "
                               f"{self.host.player()} with {len(self.host.actions())} actions, the duel seat "
                               f"{point.player} with {n}")  # fmt: skip
        obs = self.host.observe()
        with torch.no_grad():
            probs = torch.softmax(self.policy.net(collate([obs])).logits[0].float() / self.temperature, -1)
        if self.greedy:
            index = int(probs.argmax())
        else:
            index = int(torch.multinomial(probs, 1, generator=self.generator))
        full = np.zeros(n)
        k = min(n, probs.shape[0])
        full[:k] = probs[:k].numpy()
        self.last_probs = full.tolist()
        return index


def make_policy_agent(arg: str | None, seed: int, *, greedy: bool = False, temperature: float = 1.0) -> CheckpointAgent:
    if not arg:
        raise ValueError("policy needs a checkpoint path: policy:<path/to/checkpoint.pt>")
    return CheckpointAgent(arg, seed, greedy=greedy, temperature=temperature)


__all__ = ["CheckpointAgent", "load_cached", "make_policy_agent"]
