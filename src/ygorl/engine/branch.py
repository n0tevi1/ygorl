"""Branching: fork a recorded game at any decision point and roll candidates out.

``fork(replay, t)`` rebuilds the recorded duel, replays it up to agent step
``t`` and stops at that decision point. From there :meth:`Branch.rollout`
plays one candidate action at ``t`` and lets the given policies finish the
game; :meth:`Branch.try_all` does so for every legal action (see
docs/branching.md).

Decision point ``t`` counts agent steps (``DecisionPoint.index``,
``DuelResult.actions``), not engine responses: a multi-select answered with
one ``set_response`` spans several steps (see :mod:`ygorl.engine.actions`), and
``t`` may point into the middle of it.

A replay stores engine responses, not action indices, so the steps before
``t`` are recovered by inverting the action model on each recorded response
(:func:`actions_for_response`); per-step records (``record_steps``) are used
instead when present, and checked against the responses.

This first version replays from the start for every rollout. The arena
snapshot (T2.8) will replace that behind the same interface. Branches share
the recorded hidden state (god's-eye branching): the opponent's hand and both
decks are what they were in the recorded game.
"""

from __future__ import annotations

import copy
import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from ygorl.data.environment import Environment
from ygorl.engine.actions import (
    Action,
    AnnounceAttribState,
    AnnounceRaceState,
    CounterState,
    DecisionState,
    PlaceState,
    SelectCardState,
    SelectSumState,
    SortState,
    TributeState,
)
from ygorl import _core
from ygorl.engine.duel import DecisionPoint, DuelResult, DuelSession, SessionSnapshot
from ygorl.engine.replay import Replay

MAX_SEARCH_NODES = 200_000


class BranchError(ValueError):
    """A replay cannot be forked as asked (t out of range, inconsistent records)."""


# ------------------------------------------------------------------ response -> action indices


def _clone(state: DecisionState) -> DecisionState:
    c = copy.copy(state)
    if hasattr(state, "picked"):
        c.picked = list(state.picked)
    return c


def _card_indices(response: bytes) -> tuple[int, ...] | None:
    if len(response) < 8:
        return None
    mode, count = struct.unpack_from("<iI", response)
    if mode != 0 or len(response) != 8 + 4 * count:
        return None
    return struct.unpack_from(f"<{count}I", response, 8)


def _may_match(state: DecisionState, action: Action, target: bytes) -> bool:
    """Cheap test that ``action`` can still lead to ``target`` (prunes multi-step searches)."""
    picked = getattr(state, "picked", None)
    if isinstance(state, (SelectCardState, TributeState, SelectSumState)):  # response lists picks in order
        if action.kind != "select":
            return True
        picks = _card_indices(target)
        return picks is not None and len(picked) < len(picks) and picks[len(picked)] == action.index
    if isinstance(state, SortState):  # response: position of every card
        if action.kind != "sort":
            return True
        return len(target) == len(state.decision.cards) and struct.unpack_from("<b", target, action.index)[0] == len(picked)
    if isinstance(state, PlaceState):  # response: 3 bytes per zone, in pick order
        k, z = 3 * len(picked), action.value
        return target[k : k + 3] == bytes((z >> 16, (z >> 8) & 0xFF, z & 0xFF))
    if isinstance(state, CounterState):  # response: i16 count per card
        n = len(state.decision.cards)
        if len(target) != 2 * n:
            return False
        return picked.count(action.index) < struct.unpack(f"<{n}h", target)[action.index]
    if isinstance(state, (AnnounceRaceState, AnnounceAttribState)):  # response: bit mask
        return bool(int.from_bytes(target, "little") & action.value)
    return True


def actions_for_response(state: DecisionState, response: bytes, max_nodes: int = MAX_SEARCH_NODES) -> list[int] | None:
    """Action indices that complete ``state`` with exactly ``response`` (None if none do).

    Inverts :mod:`ygorl.engine.actions` by a depth-first search over copies of
    ``state`` (which is left untouched), pruned by what ``response`` says
    about each pick. Where several orders give the same response (counters,
    announced race/attribute bits) the first one found is returned.
    """
    nodes = 0

    def search(st: DecisionState) -> list[int] | None:
        nonlocal nodes
        for i, action in enumerate(st.actions()):
            if not _may_match(st, action, response):
                continue
            nodes += 1
            if nodes > max_nodes:
                raise BranchError(f"no action sequence found for response {response.hex()} within {max_nodes} steps")
            child = _clone(st)
            out = child.step(i)
            if out is None:
                rest = search(child)
                if rest is not None:
                    return [i, *rest]
            elif out == response:
                return [i]
        return None

    return search(state)


def _follow(state: DecisionState, hints: Sequence[int]) -> tuple[list[int], bytes | None]:
    """Apply recorded choices to a copy of ``state`` until the decision completes or they run out."""
    st, used = _clone(state), []
    for idx in hints:
        if not 0 <= idx < len(st.actions()):
            return used + [idx], b""  # an impossible choice never matches a response
        used.append(idx)
        out = st.step(idx)
        if out is not None:
            return used, out
    return used, None


# ------------------------------------------------------------------ replaying recorded steps


class _Reached(Exception):
    def __init__(self, point: DecisionPoint, recorded: int | None) -> None:
        self.point, self.recorded = point, recorded


class _Exhausted(Exception):
    def __init__(self, index: int) -> None:
        self.index = index


class _RecordedSteps:
    """Agent for both seats that answers with the replay's recorded choices, step by step.

    At the first step of every decision it works out the action indices that
    produce the next recorded response (from step records if present, else by
    :func:`actions_for_response`). Raises :class:`_Reached` at ``stop_at`` and
    :class:`_Exhausted` when the record has nothing more to say.
    """

    def __init__(self, replay: Replay, stop_at: int | None) -> None:
        self.responses = replay.responses
        self.hints = _step_hints(replay.steps)
        self.stop_at = stop_at
        self.actions: list[int] = []
        self._sent = 0  # responses the duel had sent when the record last spoke (host answers included)
        self._state: DecisionState | None = None
        self._plan: list[int] = []

    def act(self, point: DecisionPoint) -> int:
        if point.state is not self._state:
            if self._plan:
                raise BranchError(f"step {point.index}: the engine asked a new decision before the recorded one was complete")
            self._state = point.state
            self._plan = self._decide(point)
        if point.index == self.stop_at:
            raise _Reached(point, self._plan[0] if self._plan else None)
        if not self._plan:
            raise _Exhausted(point.index)
        idx = self._plan.pop(0)
        self.actions.append(idx)
        return idx

    def _decide(self, point: DecisionPoint) -> list[int]:
        k = point.response_index  # host answers (curriculum modes) fill the gaps between agent decisions
        target = self.responses[k] if k < len(self.responses) else None
        hints = self.hints[point.index :] if self.hints is not None else []
        self._sent = k
        if target is None:  # after the last response: only step records can continue a started decision
            return _follow(point.state, hints)[0] if hints else []
        self._sent = k + 1
        if hints:
            seq, out = _follow(point.state, hints)
            if out != target:
                raise BranchError(f"step {point.index}: the recorded step choices do not reproduce recorded response {k}")
            return seq
        seq = actions_for_response(point.state, target)
        if seq is None:
            raise BranchError(f"step {point.index}: recorded response {k} ({target.hex()}) matches no legal action "
                              f"sequence for {point.decision.name}")  # fmt: skip
        return seq

    def check_consumed(self, result: DuelResult | None = None) -> None:
        sent = len(result.responses) if result is not None else self._sent
        if sent != len(self.responses) or (result is not None and result.responses != self.responses):
            raise BranchError(f"the duel ended after {sent} of {len(self.responses)} recorded responses; "
                              "the replay does not reproduce")  # fmt: skip


def _step_hints(steps: list[dict]) -> list[int] | None:
    if not steps or any(s.get("index") != i or "chosen" not in s for i, s in enumerate(steps)):
        return None
    return [s["chosen"] for s in steps]


def recorded_actions(replay: Replay, env: Environment | None = None, **duel_kwargs) -> list[int]:
    """Every agent step of the recorded game as action indices (``DuelResult.actions``).

    Replays the whole game once. ``duel_kwargs`` (``cards``, ``scripts``) go
    to :class:`~ygorl.engine.duel.Duel`.
    """
    agent = _RecordedSteps(replay, stop_at=None)
    result = None
    try:
        result = replay.duel(env, **duel_kwargs).run(agent, agent)
    except _Exhausted:
        pass
    agent.check_consumed(result)
    return agent.actions


class RecordedAgent:
    """Plays ``actions[point.index]``: continues a recorded game from any step (both seats)."""

    def __init__(self, actions: Sequence[int]) -> None:
        self.actions = list(actions)

    def act(self, point: DecisionPoint) -> int:
        if point.index >= len(self.actions):
            raise IndexError(f"no recorded action for step {point.index} (the record has {len(self.actions)})")
        return self.actions[point.index]


# ------------------------------------------------------------------ fork / rollout


@dataclass
class CandidateOutcome:
    """Rollout results of one candidate action at the fork point."""

    index: int  # action index at t
    action: Action
    recorded: bool  # the action taken in the recorded game
    side: int  # side to act at t: 0 = deck a, 1 = deck b
    results: list[DuelResult] = field(default_factory=list)

    @property
    def wins(self) -> int:
        return sum(r.winner == self.side for r in self.results)

    @property
    def draws(self) -> int:
        return sum(r.winner is None for r in self.results)

    @property
    def losses(self) -> int:
        return sum(r.winner == 1 - self.side for r in self.results)


class Branch:
    """A recorded game stopped at decision point ``t``; create with :func:`fork`."""

    def __init__(self, replay: Replay, t: int, env: Environment | None, point: DecisionPoint, prefix: list[int],
                 recorded_action: int | None, duel_kwargs: dict) -> None:  # fmt: skip
        self.replay = replay
        self.t = t
        self.env = env
        self.point = point  # the decision point at t: point.actions are the candidates
        self.prefix = prefix  # action indices of steps 0..t-1
        self.recorded_action = recorded_action  # action taken at t in the recorded game (None if not recorded)
        self._duel_kwargs = duel_kwargs
        self._session: DuelSession | None = None  # built on the first rollout: the duel at t, with core snapshots
        self._snapshot: SessionSnapshot | None = None

    @property
    def side(self) -> int:
        """Side to act at ``t`` in (a, b) order: 0 = deck a, 1 = deck b."""
        return (self.replay.first + self.point.player) % 2

    def rollout(self, action: int, policy_a, policy_b, *, record_steps: bool = False,
                record_messages: bool = False) -> DuelResult:  # fmt: skip
        """Replay to ``t``, play ``action`` there, then let the policies (in (a, b) order) finish the game.

        The policies are asked only for steps after ``t``. The result covers the
        whole game from the start (its ``responses`` / ``actions`` include the
        replayed prefix).
        """
        n = len(self.point.actions)
        if not isinstance(action, int) or not 0 <= action < n:
            raise ValueError(f"action {action!r} out of range 0..{n - 1} at t={self.t}")
        if not (record_steps or record_messages) and _core.ARENA_AVAILABLE:
            return self._rollout_from_snapshot(action, (policy_a, policy_b))
        duel = self.replay.duel(self.env, record_steps=record_steps, record_messages=record_messages, **self._duel_kwargs)
        return duel.run(_BranchSeat(self, action, policy_a), _BranchSeat(self, action, policy_b))

    def _at_t(self) -> DuelSession:
        """The session paused at t (replayed once, then restored for every rollout)."""
        if self._session is None:
            session = DuelSession(self.replay.duel(self.env, snapshots=True, **self._duel_kwargs))
            for i, idx in enumerate(self.prefix):
                point = session.point
                if point is None or point.index != i:
                    raise BranchError(f"replay diverged before t={self.t} (step {i})")
                session.act(idx)
            self._session, self._snapshot = session, session.snapshot()
        else:
            self._session.restore(self._snapshot)
        point = self._session.point
        if point is None or point.decision != self.point.decision or point.actions != self.point.actions:
            raise BranchError(f"replay diverged: the decision at t={self.t} differs from the forked one")
        return self._session

    def _rollout_from_snapshot(self, action: int, policies) -> DuelResult:
        session = self._at_t()
        session.act(action)
        duel = session.duel
        while not session.done:
            point = session.point
            if point is None:
                break
            policy = policies[duel.deck_of(point.player)]
            session.act(policy.act(point), getattr(policy, "last_probs", None))
        return session.result()

    def try_all(self, policy_factory: Callable[[int], object], *, candidates: Sequence[int] | None = None,
                rollouts: int = 1, seed: int = 0) -> list[CandidateOutcome]:  # fmt: skip
        """Roll out every candidate (default: all legal actions at ``t``) ``rollouts`` times.

        ``policy_factory(seed)`` builds a fresh policy per side and rollout. Rollout
        ``r`` uses seeds ``seed + 2r`` (a) and ``seed + 2r + 1`` (b) for every
        candidate, so candidates are compared under the same policy randomness.
        """
        n = len(self.point.actions)
        cands = list(range(n)) if candidates is None else list(candidates)
        for c in cands:
            if not isinstance(c, int) or not 0 <= c < n:
                raise ValueError(f"candidate {c!r} out of range 0..{n - 1} at t={self.t}")
        if rollouts < 1:
            raise ValueError(f"rollouts must be >= 1, got {rollouts}")
        out = []
        for c in cands:
            outcome = CandidateOutcome(c, self.point.actions[c], c == self.recorded_action, self.side)
            for r in range(rollouts):
                policy_a, policy_b = policy_factory(seed + 2 * r), policy_factory(seed + 2 * r + 1)
                outcome.results.append(self.rollout(c, policy_a, policy_b))
            out.append(outcome)
        return out


class _BranchSeat:
    """One seat of a rollout: recorded prefix, the candidate at t, then the policy."""

    def __init__(self, branch: Branch, action: int, policy) -> None:
        self.branch, self.action, self.policy = branch, action, policy
        self.last_probs = None

    def act(self, point: DecisionPoint) -> int:
        b, i = self.branch, point.index
        self.last_probs = None
        if i < b.t:
            return b.prefix[i]
        if i == b.t:
            if point.decision != b.point.decision or point.actions != b.point.actions:
                raise BranchError(f"replay diverged: the decision at t={b.t} differs from the forked one")
            return self.action
        idx = self.policy.act(point)
        self.last_probs = getattr(self.policy, "last_probs", None)
        return idx


def fork(replay: Replay, t: int, env: Environment | None = None, **duel_kwargs) -> Branch:
    """Replay ``replay`` up to agent step ``t`` and return the :class:`Branch` there.

    ``t`` indexes agent steps (``DecisionPoint.index``), from 0. ``env`` must be
    the replay's environment (as for :meth:`Replay.play`); ``duel_kwargs``
    (``cards``, ``scripts``) go to :class:`~ygorl.engine.duel.Duel`.
    """
    if not isinstance(t, int) or t < 0:
        raise BranchError(f"t={t} is not a decision point: t counts agent steps from 0")
    agent = _RecordedSteps(replay, stop_at=t)
    try:
        result = replay.duel(env, **duel_kwargs).run(agent, agent)
    except _Reached as reached:
        return Branch(replay, t, env, reached.point, agent.actions[:t], reached.recorded, duel_kwargs)
    except _Exhausted as exhausted:
        last = exhausted.index  # a decision nobody answered: it can be forked, later ones cannot be reached
    else:
        agent.check_consumed(result)
        last = len(agent.actions) - 1
    raise BranchError(f"t={t} is out of range: this replay reaches decision points 0..{last}")


__all__ = ["Branch", "BranchError", "CandidateOutcome", "RecordedAgent", "actions_for_response", "fork", "recorded_actions"]
