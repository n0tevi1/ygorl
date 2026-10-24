"""Shared pieces of the phase-0 offline tools (docs/spikes/first-seat-interruption-plan-2026-10-24.md §6, task T3).

- :class:`Brain`: a PPO checkpoint's actor-critic on CPU: candidate probabilities, critic V, a lockstep ``HostDuel``.
- :func:`run_jobs`: plays a list of :class:`Job` s on an :class:`~ygorl.env.encoded.EncodedVecEnv` in lockstep rounds
  (every running game has exactly one pending event; events are handled in env order, network decisions are batched
  per brain and sampled with each job's own generator), so the results depend only on the jobs and their seeds, not
  on worker timing or batch composition.
- :class:`LockstepGame`: a Python ``DuelSession`` (indices of solver lines, board summaries) with a ``HostDuel`` kept in
  lockstep for the network's observation and the critic's privileged input, as ``CheckpointAgent`` does.
- small helpers: the passive row of an encoded observation, seeded generators, a mean with its standard error.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from ygorl.engine import constants as C
from ygorl.env.encoding import ACTION_KINDS, MAX_OPTIONS
from ygorl.eval.arena import derive_seed
from ygorl.solver.demo import PASSIVE_KINDS, read_jsonl

PASSIVE_ROWS = [ACTION_KINDS.index(k) + 1 for k in PASSIVE_KINDS]  # actions[:, 0] is the kind + 1 (docs/encoding.md)
PASS_KIND = ACTION_KINDS.index("pass") + 1
CHAIN = C.MSG_SELECT_CHAIN
_MASK63 = (1 << 63) - 1


def generator(*keys: int) -> torch.Generator:
    """A CPU generator seeded from the path ``keys`` (stable across runs and processes)."""
    return torch.Generator().manual_seed(derive_seed(*keys) & _MASK63 if len(keys) > 1 else int(keys[0]) & _MASK63)


def turn(obs) -> int:
    return int(obs["globals"][3])


def own_turn(obs) -> bool:
    return bool(int(obs["globals"][2]))


def decision(obs) -> int:
    return int(obs["globals"][18])


def legal_rows(obs) -> np.ndarray:
    n = min(int(obs["globals"][20]), MAX_OPTIONS)
    return np.flatnonzero(obs["action_mask"][:n])


def passive_row(obs) -> int:
    """The first choosable row of a kind in :data:`~ygorl.solver.demo.PASSIVE_KINDS` order (end phase, pass, no,
    cancel, finish), else the first choosable row: the solver's passive answer on an encoded observation."""
    rows = legal_rows(obs)
    kinds = obs["actions"][:, 0]
    for k in PASSIVE_ROWS:
        for r in rows:
            if int(kinds[r]) == k:
                return int(r)
    return int(rows[0])


def is_pass(obs, row: int) -> bool:
    return int(obs["actions"][row][0]) in PASSIVE_ROWS


def mean_se(xs: Sequence[float]) -> dict:
    n = len(xs)
    if n == 0:
        return {"n": 0, "mean": None, "se": None}
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    return {"n": n, "mean": m, "se": math.sqrt(var / n)}


def select_hands(demos: list[str], n: int, min_hand: int, max_hand: int) -> list:
    """``n`` solved plain records with hand index in [min_hand, max_hand), spread evenly over (deck, hand)."""
    recs = {}
    for path in demos:
        for d in read_jsonl(path):
            if d.variant == "plain" and d.status == "solved" and d.lines and min_hand <= d.hand_index < max_hand:
                recs.setdefault((d.deck["name"], d.hand_index), d)
    keys = sorted(recs)
    if n >= len(keys):
        return [recs[k] for k in keys]
    step = len(keys) / n
    return [recs[keys[int(i * step)]] for i in range(n)]


# -- networks -------------------------------------------------------------------------------------------------------


class Brain:
    """A PPO checkpoint's actor-critic on CPU (``map_location="cpu"`` through the checkpoint loader)."""

    def __init__(self, path: str) -> None:
        from ygorl.train.checkpoint import load_actor_critic

        loaded = load_actor_critic(path)
        self.path = str(path)
        self.model = loaded.model
        self.vocab = loaded.vocab
        self.event_length = loaded.event_length
        self.net_config = loaded.net_config
        self.signature = loaded.signature
        self.update = loaded.update

    @torch.no_grad()
    def probs(self, observations: Sequence[dict]) -> torch.Tensor:
        """``[B, A]`` candidate probabilities of the actor (illegal rows 0)."""
        batch = self.model.collate(list(observations), "cpu")
        return torch.softmax(self.model.policy_logits(batch).float(), -1)

    @torch.no_grad()
    def value(self, observations: Sequence[dict], privileged: Sequence[dict | None]) -> torch.Tensor:
        """``[B]`` critic V from the deciding seat (needs the privileged input for a privileged critic)."""
        batch = self.model.collate(list(observations), "cpu")
        priv = self.model.collate_privileged(list(privileged), "cpu")
        return self.model(batch, priv).v.float()

    def host(self, duel):
        """A ``HostDuel`` started on ``duel``'s game (core seed, rules, loaded decks), to be kept in lockstep."""
        from ygorl import _core

        cfg = duel.config
        host = _core.HostDuel(duel.cards.to_core(), duel.scripts, self.vocab.passwords(), event_length=self.event_length,
                              selection_history=self.net_config.selection_history,
                              attack_history=self.net_config.attack_history)  # fmt: skip
        p = cfg.player
        player = (p.starting_lp, p.starting_hand, p.draw_per_turn)
        decks = [(list(m), list(e)) for m, e in duel.loaded_decks()]
        host.start(list(duel.core_seed), cfg.rule_flags, player, player, decks, cfg.max_turns, cfg.max_decisions)
        return host


def sample(probs: torch.Tensor, gen: torch.Generator | None) -> int:
    """One draw from a probability row (``gen`` None: the argmax)."""
    if gen is None:
        return int(probs.argmax())
    return int(torch.multinomial(probs, 1, generator=gen))


# -- the batched lockstep runner --------------------------------------------------------------------------------------


@dataclass
class Net:
    """A decision handed to a brain: sample from ``brain`` with ``gen`` (None: argmax)."""

    brain: str
    gen: torch.Generator | None


STOP = "stop"  # Job.decide: end the game here (the horizon), the env slot is reused


class Job:
    """One game for :func:`run_jobs`. ``decide(ev)`` returns a row, a :class:`Net` request or :data:`STOP`;
    ``chosen(ev, row)`` is told every row stepped; ``finish(result)`` returns the job's record (``result`` None at
    a :data:`STOP`)."""

    spec = None

    def decide(self, ev):  # pragma: no cover - interface
        raise NotImplementedError

    def chosen(self, ev, row: int) -> None:
        pass

    def finish(self, result: dict | None) -> dict:  # pragma: no cover - interface
        raise NotImplementedError


def run_jobs(venv, jobs: Sequence[Job], brains: dict[str, Brain], *, timeout: float = 900.0,
             progress: Callable[[int, int], None] | None = None) -> list[dict]:  # fmt: skip
    """Play every job on ``venv`` and return their records in job order (see the module docstring)."""
    results: list[dict | None] = [None] * len(jobs)
    order = iter(range(len(jobs)))
    active: dict[int, int] = {}
    done = 0

    def start(env: int) -> None:
        j = next(order, None)
        if j is None:
            active.pop(env, None)
            return
        active[env] = j
        venv.reset(env, jobs[j].spec)

    for e in range(min(venv.num_envs, len(jobs))):
        start(e)
    while active:
        events = venv.recv(len(active), timeout=timeout)
        if len(events) != len(active):
            raise RuntimeError(f"lockstep broken: {len(events)} events for {len(active)} running games")
        events.sort(key=lambda ev: ev.env_id)
        requests: dict[str, list] = {}
        for ev in events:
            j = active[ev.env_id]
            job = jobs[j]
            if ev.result is not None:
                results[j] = job.finish(dict(ev.result))
                done += 1
                start(ev.env_id)
                continue
            d = job.decide(ev)
            if isinstance(d, str) and d == STOP:
                results[j] = job.finish(None)
                done += 1
                start(ev.env_id)
            elif isinstance(d, Net):
                requests.setdefault(d.brain, []).append((ev, d))
            else:
                job.chosen(ev, int(d))
                venv.step(ev.env_id, int(d))
        groups: dict[int, tuple[Brain, list]] = {}  # one forward per brain object (names may share one)
        for name in sorted(requests):
            groups.setdefault(id(brains[name]), (brains[name], []))[1].extend(requests[name])
        for brain, reqs in groups.values():  # in the order of the sorted names: deterministic
            probs = brain.probs([ev.obs for ev, _ in reqs])
            for (ev, d), p in zip(reqs, probs):
                row = sample(p, d.gen)
                jobs[active[ev.env_id]].chosen(ev, row)
                venv.step(ev.env_id, row)
        if progress is not None:
            progress(done, len(jobs))
    return results  # type: ignore[return-value]


# -- Python lockstep game ---------------------------------------------------------------------------------------------


class LockstepGame:
    """A ``DuelSession`` on ``duel`` plus ``brain``'s ``HostDuel`` on the same game: ``point`` / ``core`` from Python,
    ``obs()`` / ``privileged()`` (the deciding seat's network input) from the host; ``act(i)`` answers both."""

    def __init__(self, duel, brain: Brain) -> None:
        from ygorl.engine.duel import DuelSession

        self.session = DuelSession(duel)
        self.host = brain.host(duel)
        self.history: list[int] = []

    @property
    def point(self):
        return self.session.point

    @property
    def tracker(self):
        return self.session.tracker

    @property
    def done(self) -> bool:
        return self.session.done

    def obs(self) -> dict:
        point = self.session.point
        if self.host.player() != point.player or len(self.host.actions()) != len(point.actions):
            raise RuntimeError(f"lockstep lost at decision {point.index}")
        return self.host.observe()

    def privileged(self) -> dict:
        return self.host.observe_privileged()

    def act(self, index: int) -> None:
        self.session.act(index)
        self.host.act(index)
        self.history.append(int(index))

    def close(self) -> None:
        self.session.close()


def chosen_row(brain: Brain, game: LockstepGame, gen: torch.Generator | None) -> int:
    """``brain``'s choice at ``game``'s decision (an index into ``point.actions``)."""
    obs = game.obs()
    return sample(brain.probs([obs])[0], gen)


__all__ = ["Brain", "CHAIN", "Job", "LockstepGame", "Net", "PASS_KIND", "STOP", "chosen_row", "decision", "generator",
           "is_pass", "legal_rows", "mean_se", "own_turn", "passive_row", "run_jobs", "sample", "select_hands", "turn"]  # fmt: skip
