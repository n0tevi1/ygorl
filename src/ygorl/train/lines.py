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

    def spec(self, learner_deck: Deck, seed: int, config: DuelConfig | None = None) -> GameSpec:
        """The game: the demonstration deck as deck a, moving first, from the record's exact start; ``learner_deck``
        as deck b, shuffled with ``seed`` as a normal game would shuffle it (the engine's seat 1)."""
        base = config or DuelConfig()
        learner = replace(learner_deck, main=tuple(shuffle_deck(learner_deck.main, seed, 1)))
        cfg = replace(base, rule_flags=self.rule_flags, player=PlayerRules(**self.player), shuffle_decks=False)
        return GameSpec(seed=seed, deck_a=self.deck, deck_b=learner, first=0, config=cfg, core_seed=self.core_seed)


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
        return self.scripts[int(rng.integers(len(self.scripts)))]


class LinePlayer:
    """The scripted seat's state in one game: which step comes next, and whether the line is still being played.

    ``act(obs)`` answers one decision of the seat: the row matching the next step, or None once the line is over
    (``status`` ``"completed"``: every step played; ``"broken"``: no candidate matched, ``broken_at`` the step), after
    which the caller plays the seat itself. With ``skip_forced`` the environment never asks forced decisions, so
    forced steps are passed over.
    """

    def __init__(self, script: LineScript, skip_forced: bool) -> None:
        self.script = script
        self.skip_forced = skip_forced
        self.pos = 0
        self.status = "playing"
        self.broken_at: int | None = None
        self.played = 0

    def act(self, obs) -> int | None:
        if self.status != "playing":
            return None
        steps = self.script.steps
        while self.skip_forced and self.pos < len(steps) and steps[self.pos].forced:
            self.pos += 1
        if self.pos >= len(steps):
            self.status = "completed"
            return None
        row = match_action(obs, steps[self.pos].key)
        if row is None:
            self.status, self.broken_at = "broken", self.pos
            return None
        self.pos += 1
        self.played += 1
        return row

    def finish(self) -> dict:
        """The line's outcome for the game's record (``FinishedGame.line``)."""
        return {"id": self.script.id, "name": self.script.name, "status": self.status, "played": self.played,
                "steps": len(self.script.steps), "broken_at": self.broken_at}  # fmt: skip


__all__ = ["LINE_PLAYER", "LineLibrary", "LinePlayer", "LineScript", "LineStep", "chain_effect", "encoded_action_key",
           "line_script", "match_action"]  # fmt: skip
