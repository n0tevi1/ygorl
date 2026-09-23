"""Lethal search on top of any agent (T4e.1 stage A, #62; prototype for play and evaluation, not for training).

:class:`LethalAgent` wraps an agent. It keeps a *shadow* copy of the duel (same seed, decks and configuration, with
engine snapshots) that follows every decision of both seats (``on_duel_start`` / ``on_decision``, like the
``policy:<checkpoint>`` agent). At its own main / battle phase decisions it snapshots the shadow and plays short
rollouts of the rest of the turn: its own seat with Greedy or random choices, the opponent always passive
(pass / no / cancel / end phase, as the solver's demonstrations assume). A rollout that wins the duel this turn is a
lethal line; the agent then follows that line's actions, and gives control back to the wrapped agent as soon as the
real game departs from the line (e.g. the opponent chains a hand trap).

Strict mode (default) keeps the search to what a player could know: a rollout only counts if it did not draw from
the agent's own deck (the shadow knows the real deck order), toss a coin or die (it knows the RNG), or see the
opponent chain anything (the passive-opponent assumption broke, and the result may depend on hidden cards).
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field

from ygorl.agents.greedy import GreedyAgent
from ygorl.engine import constants as C
from ygorl.engine import messages as M
from ygorl.engine.duel import Duel, DecisionPoint, DuelSession

PASSIVE_KINDS = ("end_phase", "pass", "no", "cancel", "finish")
TRIGGERS = (C.MSG_SELECT_IDLECMD, C.MSG_SELECT_BATTLECMD)


def passive_action(point: DecisionPoint) -> int:
    kinds = [a.kind for a in point.actions]
    return next((kinds.index(k) for k in PASSIVE_KINDS if k in kinds), 0)


def _signature(point: DecisionPoint) -> tuple:
    return point.player, point.decision.TYPE, tuple(a.kind for a in point.actions)


@dataclass
class LethalStats:
    searches: int = 0
    rollouts: int = 0
    found: int = 0  # searches that found a strict lethal line
    rejected: int = 0  # winning rollouts rejected by strict mode
    overrides: int = 0  # decisions taken from a line
    abandoned: int = 0  # lines dropped because the real game departed from them
    desync: int = 0  # the shadow lost lockstep (search disabled for the rest of the game)
    by_turn: dict = field(default_factory=dict)


class LethalAgent:
    """``agent`` plus a lethal search; ``rollouts`` per search (``greedy_rollouts`` of them Greedy, the rest random),
    at most ``max_searches`` searches per turn, rollouts cut after ``max_steps`` decisions."""

    def __init__(self, agent, seed: int | None = None, *, rollouts: int = 12, greedy_rollouts: int = 4,
                 max_searches: int = 4, max_steps: int = 200, strict: bool = True) -> None:  # fmt: skip
        self.agent = agent
        self.name = f"lethal:{getattr(agent, 'name', type(agent).__name__)}"
        self.rng = random.Random(seed)
        self.rollouts, self.greedy_rollouts = rollouts, greedy_rollouts
        self.max_searches, self.max_steps, self.strict = max_searches, max_steps, strict
        self.stats = LethalStats()
        self.shadow: DuelSession | None = None
        self.line: list[tuple] = []  # expected (signature, index) of every decision after the search point
        self._searched: dict[int, int] = {}  # turn -> searches
        self.last_probs = None

    # -- delegation --------------------------------------------------------------------------------
    def observe(self, point, core) -> None:
        hook = getattr(self.agent, "observe", None)
        if hook is not None:
            hook(point, core)

    def on_duel_start(self, duel: Duel) -> None:
        hook = getattr(self.agent, "on_duel_start", None)
        if hook is not None:
            hook(duel)
        shadow = Duel(duel.seed, duel.env, *duel.decks, cards=duel.cards, scripts=duel.scripts, config=duel.config,
                      first=duel.first, snapshots=True, core_seed=duel.core_seed)  # fmt: skip
        self.shadow = DuelSession(shadow)
        self.line, self._searched, self.stats = [], {}, LethalStats()

    def on_decision(self, point: DecisionPoint, index: int) -> None:
        hook = getattr(self.agent, "on_decision", None)
        if hook is not None:
            hook(point, index)
        if self.shadow is None:
            return
        mine = self.shadow.point
        if mine is None or _signature(mine) != _signature(point):
            self.stats.desync += 1
            self.shadow.close()
            self.shadow, self.line = None, []
            return
        if self.line:
            expected_sig, expected_idx = self.line[0]
            if expected_sig == _signature(point) and expected_idx == index:
                self.line.pop(0)
            else:
                self.stats.abandoned += 1
                self.line = []
        self.shadow.act(index)
        if self.shadow.done:
            self._log()

    # -- acting -------------------------------------------------------------------------------------
    def act(self, point: DecisionPoint) -> int:
        idx = self._from_line(point)
        if idx is None and self._should_search(point):
            self._search(point)
            idx = self._from_line(point)
        if idx is not None:
            self.stats.overrides += 1
            self.last_probs = None
            return idx
        idx = self.agent.act(point)
        self.last_probs = getattr(self.agent, "last_probs", None)
        return idx

    def _from_line(self, point: DecisionPoint) -> int | None:
        if self.line and self.line[0][0] == _signature(point):
            return self.line[0][1]
        return None

    def _should_search(self, point: DecisionPoint) -> bool:
        if self.shadow is None or point.decision.TYPE not in TRIGGERS or point.turn_player != point.player:
            return False
        return self._searched.get(point.turn, 0) < self.max_searches

    def _search(self, point: DecisionPoint) -> None:
        shadow, me = self.shadow, point.player
        self._searched[point.turn] = self._searched.get(point.turn, 0) + 1
        self.stats.searches += 1
        snap = shadow.snapshot()
        try:
            for r in range(self.rollouts):
                shadow.restore(snap)
                chooser = GreedyAgent(self.rng.randrange(1 << 30), cards=shadow.duel.cards) \
                    if r < self.greedy_rollouts else random.Random(self.rng.randrange(1 << 30))  # fmt: skip
                line, clean = self._rollout(me, point.turn, chooser)
                self.stats.rollouts += 1
                if line is None:
                    continue
                if self.strict and not clean:
                    self.stats.rejected += 1
                    continue
                self.stats.found += 1
                self.stats.by_turn[point.turn] = self.stats.by_turn.get(point.turn, 0) + 1
                self.line = line
                return
        finally:
            shadow.restore(snap)

    def _rollout(self, me: int, turn: int, chooser) -> tuple[list | None, bool]:
        shadow = self.shadow
        line, clean = [], True
        for _ in range(self.max_steps):
            p = shadow.point
            if p is None or shadow.done or p.turn != turn:
                break
            clean = clean and self._clean(p.events, me)
            if p.player == me:
                idx = chooser.act(p) if isinstance(chooser, GreedyAgent) else chooser.randrange(len(p.actions))
            else:
                idx = passive_action(p)
            line.append((_signature(p), idx))
            shadow.act(idx)
        clean = clean and self._clean(shadow.tracker.events, me)
        won = shadow.done and shadow.tracker.result.reason == "win" and shadow.tracker._engine_winner == me
        return (line if won else None), clean

    @staticmethod
    def _clean(events, me: int) -> bool:
        for e in events:
            if isinstance(e, M.Draw) and e.player == me:
                return False  # the shadow knows the real deck order
            if isinstance(e, (M.TossCoin, M.TossDice)):
                return False  # ... and the RNG
            if isinstance(e, M.Chaining) and e.triggering_controller != me:
                return False  # the opponent answered: the passive assumption broke
        return True

    def _log(self) -> None:
        path = os.environ.get("YGORL_LETHAL_LOG")
        if path:
            with open(path, "a") as f:
                f.write(json.dumps(self.stats.__dict__) + "\n")
        self.shadow.close()
        self.shadow = None


def make_lethal_agent(arg: str | None, seed: int) -> LethalAgent:
    """Registry factory: ``lethal:<inner agent spec>``, e.g. ``lethal:policy:out/ppo/K/policy.pt``."""
    from ygorl.agents.registry import make_agent

    if not arg:
        raise ValueError("lethal needs an inner agent: lethal:<agent spec>")
    return LethalAgent(make_agent(arg, seed), seed)


__all__ = ["LethalAgent", "LethalStats", "make_lethal_agent", "passive_action"]
