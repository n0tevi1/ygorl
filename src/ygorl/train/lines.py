"""Scripted-line pool opponents: a solver demonstration played by the opponent seat in a training game.

In a share of the pool games (docs/training.md §8.3 「脚本线对手」) the opponent seat opens with a pre-solved turn-1
line of the demonstration set (docs/solver.md 「示范集格式」): the game starts from the record's exact start (its
core seed words, rule flags and the deck in load order, host shuffle off), the demonstration deck moves first, and
at every decision of that seat the candidate matching the line's next step is played. Once the line ends (turn 2
begins) or breaks (the learner's interruption or a diverging decision leaves no matching candidate), the seat is
played by the pool snapshot of the game, like any other pool game. The learner seat plays as usual and only its
rows are collected; they carry the demonstration's id (``Rollout.line_id``) so that solver labels can be joined
to them later.

The collector only sees encoded observations, so a line is stored as :func:`encoded_action_key` s: what each
demonstrated action is in the encoded candidate row (decision type, kind, card identity and zone, effect
description, value), independent of where the candidate sits in the list and of which equivalent copy it is
(docs/encoding.md 「等价动作去重」). The keys are computed once per record by replaying the line with the Python
reference encoder (:class:`ygorl.env.encoding.ObservationEncoder`, element for element the C++ encoder of
``EncodedVecEnv``), and matched against the environment's rows at acting time.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ygorl.cards.cdb import CardVocab
from ygorl.cards.ydk import Deck
from ygorl.data.environment import Environment, PlayerRules
from ygorl.engine import constants as C
from ygorl.engine.duel import DuelConfig, DuelSession, default_cards, shuffle_deck
from ygorl.env.encoding import (
    ACTION_KINDS,
    CONTROLLER,
    LOCATION,
    MAX_OPTIONS,
    POSITION,
    SEQUENCE,
    VISIBLE,
    ObservationEncoder,
    canonical_action,
)
from ygorl.env.events import CARD as EVENT_CARD, EV, TYPE as EVENT_TYPE
from ygorl.env.pool import GameSpec
from ygorl.solver.demo import DemoError, Demonstration, read_jsonl

LINE_PLAYER = 0  # engine player of the deck under study in every demonstration (it moves first; bc.DEMO_PLAYER)
_UNORDERED_ZONES = (1, 2, 7)  # LOCATION_ENUM of deck, hand, Extra Deck: a card's sequence there says nothing
_LIST_INDEX_KINDS = (ACTION_KINDS.index("select") + 1, ACTION_KINDS.index("unselect") + 1)
_CHAINING, _CHAIN_END = EV["chaining"], EV["chain_end"]
_PILES = (5, 6)  # LOCATION_ENUM of graveyard and banished: a first-seat line matches their cards without the sequence
_PASS_KIND = ACTION_KINDS.index("pass") + 1


def chain_effect(obs) -> int:
    """The effect a MSG_SELECT_CHAIN decision answers, as a card vocab index: the card of the newest ``chaining``
    event in the observation's window (``events`` / ``event_mask``, oldest first) that no ``chain_end`` follows;
    0 at an open window (no chain being built: a trigger window after a summon, an end-phase prompt) or without
    events. Both the learner's rows (``Rollout.line_effect``) and the solver-window labels (train.line_labels)
    identify a chain window by this effect and by how many times the seat has answered it in the game."""
    events, mask = obs.get("events"), obs.get("event_mask")
    if events is None or mask is None:
        return 0
    for i in range(len(events) - 1, -1, -1):
        if not mask[i]:
            continue
        kind = int(events[i][EVENT_TYPE])
        if kind == _CHAIN_END:
            return 0
        if kind == _CHAINING:
            return int(events[i][EVENT_CARD])
    return 0


def encoded_action_key(obs, index: int) -> tuple:
    """What candidate ``index`` of an encoded observation is, as :func:`ygorl.train.bc.action_key` tells choices
    apart: decision type, kind, card (vocab index; controller, zone, position, and the sequence only where it is a
    place -- on the field, for materials and for cards hidden from the decider), effect description, position /
    zone / value parameters. The list index (row 9) and the list-index value of SELECT_UNSELECT_CARD are left out,
    so equivalent copies and reorderings share a key."""
    a = obs["actions"][index]
    decision = int(obs["globals"][18])
    kind, ref, code = int(a[0]), int(a[1]), int(a[2])
    value = int(a[8])
    if decision == C.MSG_SELECT_UNSELECT_CARD and kind in _LIST_INDEX_KINDS:
        value = 0
    where = None
    if ref:
        c = obs["cards"][ref - 1]
        loc = int(c[LOCATION])
        ordered = loc not in _UNORDERED_ZONES or not int(c[VISIBLE])
        where = (int(c[CONTROLLER]), loc, int(c[SEQUENCE]) if ordered else -1, int(c[POSITION]))
    return (decision, kind, code, where, int(a[3]), int(a[4]), int(a[5]), int(a[6]), int(a[7]), value)


def _pile_free(key: tuple) -> tuple:
    """``key`` with the sequence of a graveyard / banished card left out (the order of those piles)."""
    where = key[3]
    if where is not None and where[1] in _PILES:
        return (*key[:3], (where[0], where[1], -1, where[3]), *key[4:])
    return key


def match_loose(obs, key: tuple) -> int | None:
    """:func:`match_action` with graveyard / banished cards matched without their sequence."""
    if len(key) < 4 or key[3] is None or key[3][1] not in _PILES:
        return None
    want = _pile_free(key)
    mask = obs["action_mask"]
    n = min(int(obs["globals"][20]), MAX_OPTIONS)
    for i in range(n):
        if mask[i] and _pile_free(encoded_action_key(obs, i)) == want:
            return i
    return None


def match_action(obs, key: tuple) -> int | None:
    """The first choosable row of ``obs`` with :func:`encoded_action_key` ``key``, or None."""
    mask = obs["action_mask"]
    n = min(int(obs["globals"][20]), MAX_OPTIONS)
    for i in range(n):
        if mask[i] and encoded_action_key(obs, i) == key:
            return i
    return None


@dataclass(frozen=True)
class LineStep:
    key: tuple  # encoded_action_key of the demonstrated action
    forced: bool  # the only choosable row: an environment with skip_forced plays it itself (never asks)


@dataclass(frozen=True)
class LineScript:
    """One demonstration line, ready for the scripted seat: the start position and the seat's steps in order."""

    id: int  # index in the library (the row of ``Rollout.line_id``); stable for a given list of files
    deck: Deck  # the demonstration deck in load order (main) -- the scripted seat's deck
    hand_index: int
    core_seed: tuple[int, int, int, int]
    rule_flags: int  # the record's (includes DUEL_PSEUDO_SHUFFLE: decks are never shuffled during the game)
    player: dict  # PlayerRules of the record
    steps: tuple[LineStep, ...]  # the scripted seat's decisions of the line (other seat's steps left out)
    source: str = ""  # file the record came from

    @property
    def name(self) -> str:
        return f"{self.deck.name}#{self.hand_index}"

    def spec(self, learner_deck: Deck, seed: int, config: DuelConfig | None = None, learner_seat: int = 1) -> GameSpec:
        """The game from the record's exact start (core seed words, rule flags, player rules, the demonstration deck in
        load order). ``learner_seat`` 1 (second seat): the demonstration deck is deck a and moves first, the line is
        played on turn 1; ``learner_deck`` is deck b. ``learner_seat`` 0 (first seat): ``learner_deck`` is deck a and
        moves first, the demonstration deck is deck b and plays the line on its turn 2, with its recorded opening hand
        and one more card drawn. Either way ``learner_deck`` is shuffled with ``seed`` as a normal game would shuffle
        it on that engine seat (``shuffle_deck(main, seed, seat)``)."""
        if learner_seat not in (0, 1):
            raise ValueError("learner_seat must be 0 (first seat) or 1 (second seat)")
        base = config or DuelConfig()
        learner = replace(learner_deck, main=tuple(shuffle_deck(learner_deck.main, seed, learner_seat)))
        cfg = replace(base, rule_flags=self.rule_flags, player=PlayerRules(**self.player), shuffle_decks=False)
        a, b = (self.deck, learner) if learner_seat == 1 else (learner, self.deck)
        return GameSpec(seed=seed, deck_a=a, deck_b=b, first=0, config=cfg, core_seed=self.core_seed)


def line_script(demo: Demonstration, line: int, vocab: CardVocab, *, cards=None, env: Environment | None = None,
                scripts=None, id: int = 0, source: str = "") -> LineScript:  # fmt: skip
    """Replay ``demo.lines[line]`` with the reference encoder and key every step of :data:`LINE_PLAYER`."""
    if demo.start is None:
        raise DemoError("the record has no start position")
    cards = cards if cards is not None else default_cards()
    kwargs = {k: v for k, v in (("cards", cards), ("scripts", scripts)) if v is not None}
    ln = demo.lines[line]
    session = DuelSession(demo.replay(line).duel(env, **kwargs))
    encoder = ObservationEncoder(cards, vocab)
    steps: list[LineStep] = []
    try:
        for step, idx in enumerate(ln.actions):
            point = session.point
            if point is None:
                raise DemoError(f"step {step}: the duel stopped before the line ended")
            if point.player == LINE_PLAYER:
                if idx >= MAX_OPTIONS:
                    raise DemoError(f"step {step}: action {idx} is beyond the {MAX_OPTIONS} encoded rows")
                obs = encoder.encode(point, session.core)
                rep = canonical_action(obs, idx)
                if not obs["action_mask"][rep]:
                    raise DemoError(f"step {step}: the demonstrated action is hidden by the mask (an undo)")
                forced = int(np.count_nonzero(obs["action_mask"])) == 1
                steps.append(LineStep(encoded_action_key(obs, rep), forced))
            session.act(idx)
            if session.tracker.result.retries:
                raise DemoError(f"step {step}: the engine answered MSG_RETRY")
    finally:
        session.close()
    s = demo.start
    deck = Deck(tuple(s["decks"]["a"]["main"]), tuple(s["decks"]["a"]["extra"]), (), demo.deck["name"])
    return LineScript(id, deck, demo.hand_index, tuple(int(w) for w in s["core_seed"]), int(s["rule_flags"]),
                      dict(s["player"]), tuple(steps), source)  # fmt: skip


class LineLibrary:
    """The scripted lines of one or more demonstration files (solved ``plain`` records, their first line), keyed
    and ready to deal. Records that do not replay are left out with a note in ``problems``; the ids of the rest
    are their order over the files."""

    def __init__(self, scripts: Sequence[LineScript], problems: Sequence[str] = ()) -> None:
        self.scripts = list(scripts)
        self.problems = list(problems)
        self.seconds = 0.0
        self.dealable: list[int] | None = (
            None  # ids `sample` draws from; None = every script (TrainConfig.line_labelled_only)
        )

    @classmethod
    def load(cls, paths: Iterable[str | Path], vocab: CardVocab, *, cards=None, env: Environment | None = None,
             variant: str = "plain") -> LineLibrary:  # fmt: skip
        t0 = time.perf_counter()
        scripts, problems = [], []
        for path in paths:
            for demo in read_jsonl(path):
                if demo.variant != variant or demo.status != "solved" or not demo.lines:
                    continue
                if demo.environment is not None and (env is None or env.version != demo.environment["version"]):
                    problems.append(f"{path}: {demo.deck['name']} hand {demo.hand_index}: recorded in environment "
                                    f"{demo.environment['version']}, the run uses {env.version if env else None}")  # fmt: skip
                    continue
                try:
                    scripts.append(line_script(demo, 0, vocab, cards=cards, env=env, id=len(scripts), source=str(path)))
                except (DemoError, ValueError) as exc:
                    problems.append(f"{path}: {demo.deck['name']} hand {demo.hand_index}: {exc}")
        lib = cls(scripts, problems)
        lib.seconds = time.perf_counter() - t0
        return lib

    def __len__(self) -> int:
        return len(self.scripts)

    def __getitem__(self, i: int) -> LineScript:
        return self.scripts[i]

    def sample(self, rng: np.random.Generator) -> LineScript:
        if not self.scripts:
            raise ValueError("the line library is empty")
        if self.dealable is not None:
            if not self.dealable:
                raise ValueError("no dealable line (line_labelled_only with no labelled line)")
            return self.scripts[self.dealable[int(rng.integers(len(self.dealable)))]]
        return self.scripts[int(rng.integers(len(self.scripts)))]


class LinePlayer:
    """The scripted seat's state in one game: which step comes next, and whether the line is still being played.

    ``act(obs)`` answers one decision of the seat: the row matching the next step, or None once the line is over
    (``status`` ``"completed"``: every step played; ``"broken"``: no candidate matched, ``broken_at`` the step), after
    which the caller plays the seat itself. With ``skip_forced`` the environment never asks forced decisions, so
    forced steps are passed over.

    ``first_seat`` (the learner moves first, the line is played on the scripted seat's turn 2): the line waits for the
    scripted seat's own turn (its decisions during the learner's turn 1 return None and the caller plays them, the line
    stays ``"playing"``) and is matched with the tolerances turn 2 needs against a record of turn 1, in order: a
    recorded forced step that turn 2 does ask (the drawn card adds a choice) is played instead of passed over; a
    graveyard / banished card is matched without its sequence (the piles' order can differ); a forced decision (one
    choosable row) or a chain window with a pass row that matches no step is answered with that row without using a
    step (``extra``: a prompt turn 1 did not have, e.g. a hand trap's or a quick-play spell's window); at the line's
    first decision up to ``lead_skip`` leading steps may be passed over to find a match (``skipped``). Completion is
    noticed at any later decision of the seat. Second-seat lines behave as they always did.
    """

    def __init__(self, script: LineScript, skip_forced: bool, *, first_seat: bool = False, lead_skip: int = 3) -> None:
        self.script = script
        self.skip_forced = skip_forced
        self.first_seat = first_seat
        self.lead_skip = lead_skip if first_seat else 0
        self.pos = 0
        self.status = "playing"
        self.broken_at: int | None = None
        self.played = 0
        self.turn: int | None = None  # turn of the line's first played step
        self.skipped = 0  # leading steps passed over (first seat)
        self.extra = 0  # decisions answered outside the line (first seat)

    def _skip_forced_steps(self) -> None:
        steps = self.script.steps
        while self.skip_forced and self.pos < len(steps) and steps[self.pos].forced:
            self.pos += 1

    def act(self, obs) -> int | None:
        if self.status != "playing":
            return None
        if self.first_seat:
            return self._act_first_seat(obs)
        steps = self.script.steps
        self._skip_forced_steps()
        if self.pos >= len(steps):
            self.status = "completed"
            return None
        row = match_action(obs, steps[self.pos].key)
        if row is None:
            self.status, self.broken_at = "broken", self.pos
            return None
        if self.turn is None:
            self.turn = int(obs["globals"][3])
        self.pos += 1
        self.played += 1
        return row

    def _candidates(self, start: int) -> list[int]:
        """The steps the decision may answer from ``start``: the recorded forced steps up to the next unforced one."""
        steps, out, j = self.script.steps, [], start
        while j < len(steps):
            out.append(j)
            if not (self.skip_forced and steps[j].forced):
                break
            j += 1
        return out

    def _act_first_seat(self, obs) -> int | None:
        steps = self.script.steps
        candidates = self._candidates(self.pos)
        if not candidates or (self.skip_forced and steps[candidates[-1]].forced):
            self.status = "completed"  # every step played (the rest forced, never asked)
            return None
        if not int(obs["globals"][2]):  # not the scripted seat's turn: the line has not started yet
            return None
        found = self._find(obs, candidates)
        if found is None:
            n = min(int(obs["globals"][20]), MAX_OPTIONS)
            legal = np.flatnonzero(obs["action_mask"][:n])
            if len(legal) == 1:
                self.extra += 1
                return int(legal[0])
            if int(obs["globals"][18]) == C.MSG_SELECT_CHAIN:
                passes = [int(i) for i in legal if int(obs["actions"][i][0]) == _PASS_KIND]
                if passes:
                    self.extra += 1
                    return passes[0]
            if self.played == 0:
                start = candidates[-1]
                for _ in range(self.lead_skip):
                    start += 1
                    if start >= len(steps):
                        break
                    found = self._find(obs, self._candidates(start))
                    if found is not None:
                        self.skipped = found[0] - self.pos
                        break
        if found is None:
            self.pos = candidates[-1]
            self.status, self.broken_at = "broken", self.pos
            return None
        j, row = found
        if self.turn is None:
            self.turn = int(obs["globals"][3])
        self.pos = j + 1
        self.played += 1
        return row

    def _find(self, obs, candidates: list[int]) -> tuple[int, int] | None:
        """The candidate step a choosable row matches -- the next unforced step first, then the recorded forced steps
        before it in order; exactly, else without the pile sequence: (step, row)."""
        steps = self.script.steps
        order = candidates[-1:] + candidates[:-1]
        for match in (match_action, match_loose):
            for j in order:
                row = match(obs, steps[j].key)
                if row is not None:
                    return j, row
        return None

    def finish(self) -> dict:
        """The line's outcome for the game's record (``FinishedGame.line``): ``seat`` is the learner's seat (1: the
        line moved first; 0: the learner moved first and the line was played on turn 2), ``turn`` the turn of its
        first played step."""
        return {"id": self.script.id, "name": self.script.name, "status": self.status, "played": self.played,
                "steps": len(self.script.steps), "broken_at": self.broken_at, "seat": 0 if self.first_seat else 1,
                "turn": self.turn, "skipped": self.skipped, "extra": self.extra}  # fmt: skip


__all__ = ["LINE_PLAYER", "LineLibrary", "LinePlayer", "LineScript", "LineStep", "chain_effect", "encoded_action_key",
           "line_script", "match_action", "match_loose"]  # fmt: skip
