"""Real first turns of an opening hand: the observations funnel stage 1 is validated against (T5.6).

The solver works under ``DUEL_PSEUDO_SHUFFLE`` (the core never shuffles the deck after the start)
against a passive template opponent. A *real* first turn here is a standard duel of our host on the
same opening: the same deck order (hand on top), the same core seed words, the same passive opponent,
but the rule flags without ``DUEL_PSEUDO_SHUFFLE`` -- every "shuffle your deck" really shuffles.
Turn 1 is played to the first decision of turn 2 and the board is checked for the target cards.

Three players of that first turn:

* :func:`replay_line` -- the solver's verified line, followed move by move (actions matched by
  what they do -- kind, card, zone -- not by index, since a shuffled deck lists cards in another order);
* :class:`~ygorl.agents.greedy.GreedyAgent` via :func:`play_first_turn`;
* :func:`explore` -- ``n`` randomized rollouts (:class:`ExplorerAgent`) from a snapshot of the opening.

The opponent answers every decision of turn 1 passively (end phase, pass, no, cancel, finish).
"""

from __future__ import annotations

import random
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from ygorl import _core
from ygorl.cards.ydk import Deck
from ygorl.data.environment import Environment
from ygorl.engine import constants as C
from ygorl.engine.actions import Action
from ygorl.engine.duel import DecisionPoint, Duel, DuelConfig, DuelSession, default_cards
from ygorl.solver.batch import PASSIVE_OPPONENT, sample_hand
from ygorl.solver.demo import PASSIVE_KINDS, DemoError, Demonstration, iter_steps
from ygorl.solver.targets import TargetCard, board_summary, board_summary_missing, parse_targets

MAX_TURN_STEPS = 400  # agent steps in turn 1 before the turn is closed passively
_PHASE_KINDS = ("battle_phase", "end_phase", "main2", "shuffle")


@dataclass
class FirstTurn:
    """How one real first turn ended."""

    reached: bool  # the board at the start of turn 2 holds every target card
    missing: list[str] = field(default_factory=list)  # target cards not on that board
    steps: int = 0  # agent steps of player 0 in turn 1
    error: str = ""  # the line could not be followed (replay_line), or the duel stopped
    board: dict | None = None


def real_config(config: DuelConfig | None = None) -> DuelConfig:
    """``config`` as a real duel of a fixed opening: no host shuffle (the order is given), no pseudo-shuffle."""
    config = config or DuelConfig()
    return DuelConfig(rule_flags=config.rule_flags & ~C.DUEL_PSEUDO_SHUFFLE, player=config.player, max_turns=config.max_turns,
                      max_decisions=config.max_decisions, shuffle_decks=False)  # fmt: skip


def first_turn_duel(deck: Deck, hand_seed: int, *, env: Environment | None = None, config: DuelConfig | None = None,
                    cards=None, scripts: _core.ScriptDirectory | None = None, snapshots: bool = False,
                    core_seed: Sequence[int] | None = None) -> Duel:  # fmt: skip
    """A standard duel whose first player opens with the funnel's hand ``hand_seed`` of ``deck``.

    The main deck is loaded in the order :func:`~ygorl.solver.batch.sample_hand` gives (hand on top),
    exactly as the solver loads it, and the core seed words are ``expand_seed(hand_seed)`` like the
    solver's template, unless ``core_seed`` is given.
    """
    config = config or (DuelConfig.from_environment(env) if env is not None else DuelConfig())
    _, order = sample_hand(deck, hand_seed, config.player.starting_hand)
    ordered = Deck(tuple(order), deck.extra, (), deck.name)
    kwargs = {k: v for k, v in (("cards", cards), ("scripts", scripts)) if v is not None}
    return Duel(hand_seed, env, ordered, PASSIVE_OPPONENT, config=real_config(config), snapshots=snapshots,
                core_seed=list(core_seed) if core_seed is not None else None, **kwargs)  # fmt: skip


def passive_index(point: DecisionPoint) -> int:
    kinds = [a.kind for a in point.actions]
    return next((kinds.index(k) for k in PASSIVE_KINDS if k in kinds), 0)


def _finish(session: DuelSession, targets: list[TargetCard], cards, steps: int, error: str = "") -> FirstTurn:
    tracker = session.tracker
    while not session.done and tracker.turn < 2:  # close the turn passively
        session.act(passive_index(session.point))
    crashed = tracker.result.reason == "error"
    if crashed:
        error = error or f"the engine stopped with an error: {tracker.result.error}"
    board = board_summary(session.core, tracker.turn, (tracker.lp[0], tracker.lp[1]))
    missing = board_summary_missing(board, targets, cards)
    return FirstTurn(not missing and not crashed, [t.to_arg() for t in missing], steps, error, board)


def play_first_turn(session: DuelSession, agent, targets: Sequence[str | TargetCard], *, cards=None,
                    max_steps: int = MAX_TURN_STEPS) -> FirstTurn:  # fmt: skip
    """Let ``agent`` (``act(point) -> index``) play player 0's turn 1; the opponent passes."""
    targets = parse_targets(targets)
    cards = cards if cards is not None else default_cards()
    steps = 0
    while not session.done and session.tracker.turn < 2 and steps < max_steps:
        point = session.point
        if point.player == 0:
            session.act(agent.act(point))
            steps += 1
        else:
            session.act(passive_index(point))
    return _finish(session, targets, cards, steps)


class ExplorerAgent:
    """A random first-turn player that keeps going: in an idle phase it takes a proactive action
    (summon, activate, set...) with probability ``p_continue``, otherwise it ends the phase;
    every other decision is uniform."""

    name = "explorer"

    def __init__(self, seed: int | None = None, p_continue: float = 0.9) -> None:
        self.rng = random.Random(seed)
        self.p_continue = p_continue

    def act(self, point: DecisionPoint) -> int:
        acts = point.actions
        if any(a.kind in ("end_phase", "battle_phase") for a in acts):
            proactive = [i for i, a in enumerate(acts) if a.kind not in _PHASE_KINDS]
            if proactive and self.rng.random() < self.p_continue:
                return self.rng.choice(proactive)
            ends = [i for i, a in enumerate(acts) if a.kind == "end_phase"]
            if ends:
                return ends[0]
        return self.rng.randrange(len(acts))


def explore(deck: Deck, hand_seed: int, targets: Sequence[str | TargetCard], n: int, *, seed: int = 0,
            p_continue: float = 0.9, env: Environment | None = None, cards=None,
            scripts: _core.ScriptDirectory | None = None) -> dict:  # fmt: skip
    """``n`` randomized first turns of one opening from a snapshot; how many reached the target board."""
    cards = cards if cards is not None else default_cards()
    targets = parse_targets(targets)
    duel = first_turn_duel(deck, hand_seed, env=env, cards=cards, scripts=scripts, snapshots=True)
    session = DuelSession(duel)
    start = time.monotonic()
    reached, first, steps = 0, None, 0
    try:
        snap = session.snapshot()
        for k in range(n):
            if k:
                session.restore(snap)
            out = play_first_turn(session, ExplorerAgent(seed * 1_000_003 + k, p_continue), targets, cards=cards)
            steps += out.steps
            if out.reached:
                reached += 1
                first = k if first is None else first
    finally:
        session.close()
    return {"n": n, "reached": reached, "first": first, "steps": steps, "wall_s": round(time.monotonic() - start, 2)}


# ------------------------------------------------------------------ the solver's line in a real duel


def _signature(a: Action, loose: bool) -> tuple:
    c = a.card
    if c is None:
        card = None
    elif loose and c.loc.location in (C.LOCATION_DECK, C.LOCATION_HAND):
        card = (c.code, c.loc.controller, c.loc.location)  # order within deck / hand is not comparable
    elif loose and c.loc.location == C.LOCATION_EXTRA:
        # Returning a card can reorder/shuffle the Extra Deck too. Keep its position:
        # face-up Pendulum cards and face-down cards do not have the same summon rules.
        card = (c.code, c.loc.controller, c.loc.location, c.loc.position)
    else:
        card = (c.code, c.loc.controller, c.loc.location, c.loc.sequence, c.loc.position)
    return (a.kind, card, a.description, a.value if a.kind not in ("select", "unselect", "chain") else None)


def match_action(action: Action, candidates: Sequence[Action]) -> int | None:
    """Match kind/card/zone; ignore order within deck, hand and Extra Deck (preserve Extra Deck position)."""
    for loose in (False, True):
        want = _signature(action, loose)
        for i, cand in enumerate(candidates):
            if _signature(cand, loose) == want:
                return i
    return None


def replay_line(demo: Demonstration, line: int = 0, *, env: Environment | None = None, cards=None,
                scripts: _core.ScriptDirectory | None = None) -> FirstTurn:  # fmt: skip
    """Follow the verified solver line ``demo.lines[line]`` in a real duel of the same opening.

    The real duel starts from the record's own start position (deck order, core seed words, player
    rules) with ``DUEL_PSEUDO_SHUFFLE`` cleared. Each step of the line is replayed in the pseudo-shuffle
    duel (:func:`~ygorl.solver.demo.iter_steps`) and matched to the real duel's candidates with
    :func:`match_action`; a step without a match, or played by the other player, ends the attempt
    (``error``), the turn is then closed passively and the board checked like any first turn.
    """
    cards = cards if cards is not None else default_cards()
    targets = parse_targets(demo.targets)
    rep = demo.replay(line)
    rep.rule_flags &= ~C.DUEL_PSEUDO_SHUFFLE
    rep.responses = []
    kwargs = {k: v for k, v in (("cards", cards), ("scripts", scripts)) if v is not None}
    session = DuelSession(rep.duel(env, **kwargs))
    steps, error = 0, ""
    try:
        try:
            for ref, idx in iter_steps(demo, line, env=env, cards=cards, scripts=scripts):
                point = session.point
                if point is None or session.tracker.turn >= 2:
                    break
                if point.player != ref.player:
                    error = f"step {steps}: player {point.player} to act, the line has player {ref.player}"
                    break
                real = match_action(ref.actions[idx], point.actions)
                if real is None:
                    a = ref.actions[idx]
                    error = f"step {steps}: no {a.kind} of {a.card.code if a.card else '-'} in the real duel"
                    break
                session.act(real)
                steps += point.player == 0
        except DemoError as exc:
            error = f"the recorded line does not replay: {exc}"
        return _finish(session, targets, cards, steps, error)
    finally:
        session.close()


__all__ = ["ExplorerAgent", "FirstTurn", "explore", "first_turn_duel", "match_action", "passive_index", "play_first_turn",
           "real_config", "replay_line"]  # fmt: skip
