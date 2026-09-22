"""Demonstrations: solver lines re-run in our engine and turned into our action indices (T4a.1).

A solver line is a list of engine responses (a ``yrp1``). :func:`convert_line` re-runs it in
our core from the file's own seed words, rule flags and decks (host shuffle off) and inverts
every response into the step-wise action indices of :mod:`ygorl.engine.actions` with
:func:`~ygorl.engine.branch.actions_for_response` (a multi-select is one response and several
steps). The conversion is the verification: every response must be consumed, the core must
never answer ``MSG_RETRY``, and the final board must hold the target cards. By default the turn
is then closed by passive answers (end phase, pass, no) up to the first decision of turn 2, so a
demonstration also shows when to stop.

:class:`Demonstration` is one record of the demonstration set (one deck, one opening hand, one
variant); the set is a JSONL file, one record per line (format in docs/solver.md).
:func:`iter_steps` replays a record and yields ``(DecisionPoint, action index)`` pairs, from which
any observation encoder can build training samples.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ygorl import _core
from ygorl.cards.ydk import Deck
from ygorl.data.environment import Environment
from ygorl.engine import messages as M
from ygorl.engine.branch import BranchError, actions_for_response
from ygorl.engine.duel import DecisionPoint, Duel, DuelSession
from ygorl.engine.replay import Replay, YrpFile
from ygorl.solver.targets import TargetCard, board_summary, board_summary_missing, parse_targets

DEMO_FORMAT = "ygorl-demo"
DEMO_FORMAT_VERSION = 1
PASSIVE_KINDS = ("end_phase", "pass", "no", "cancel", "finish")
MAX_CLOSING_STEPS = 400
_CARD_LISTS = (M.SelectCard, M.SelectTribute, M.SelectSum)


class DemoError(ValueError):
    """A solver line does not survive the fresh replay in our engine."""


def canonical_response(decision: M.Decision, response: bytes) -> bytes:
    """``response`` in the encoding our action model produces.

    The core accepts card selections in four layouts (``playerop.cpp``, ``parse_response_cards``):
    u32, u16 or u8 index lists (modes 0/1/2) and a bit field (mode 3). The solver sends mode 2; our
    action model always sends mode 0, which the core reads the same way. Other responses are
    returned unchanged.
    """
    if not isinstance(decision, _CARD_LISTS) or len(response) < 4:
        return response
    (mode,) = struct.unpack_from("<i", response)
    try:
        if mode == 1 or mode == 2:
            (n,) = struct.unpack_from("<I", response, 4)
            fmt = "H" if mode == 1 else "B"
            indices = struct.unpack_from(f"<{n}{fmt}", response, 8)
        elif mode == 3:
            bits = int.from_bytes(response[4:], "little")
            indices = tuple(i for i in range(len(decision.cards)) if bits >> i & 1)
        else:
            return response
    except struct.error:
        return response
    return struct.pack(f"<iI{len(indices)}I", 0, len(indices), *indices)


@dataclass
class DemoLine:
    """One verified line: responses, the action indices that produce them, and where it ends."""

    responses: list[bytes]  # every response, in our encoding (solver part + passive closing)
    actions: list[int]  # action index of every step (a multi-select spans several steps)
    players: list[int]  # engine player of every step (0 = the deck under study, moves first)
    solver_responses: int  # responses[:solver_responses] come from the solver, the rest close the turn
    solver_steps: int  # actions[:solver_steps] produce the solver's responses
    board: dict  # board_summary() where the line ends
    score: dict = field(default_factory=dict)  # solver cost: burned, actions (from the file name), decisions
    source: str = ""  # solver output file

    def to_json(self) -> dict:
        d = asdict(self)
        d["responses"] = [r.hex() for r in self.responses]
        return d

    @classmethod
    def from_json(cls, d: dict) -> DemoLine:
        return cls(**{**d, "responses": [bytes.fromhex(r) for r in d["responses"]]})


def _duel_from(replay: Replay, cards, scripts, env: Environment | None = None) -> Duel:
    kwargs = {k: v for k, v in (("cards", cards), ("scripts", scripts)) if v is not None}
    return replay.duel(env, **kwargs)


def convert_line(yrp: YrpFile, targets: Sequence[TargetCard | str], *, responses: Sequence[bytes] | None = None,
                 cards=None, scripts: _core.ScriptDirectory | None = None, close_turn: bool = True) -> DemoLine:  # fmt: skip
    """Re-run the line of ``yrp`` (or ``responses`` from its start position) and convert it; raise :class:`DemoError`."""
    replay = Replay.from_yrp(yrp)
    wanted = list(replay.responses if responses is None else responses)
    targets = parse_targets(targets)
    session = DuelSession(_duel_from(replay, cards, scripts))
    tracker = session.tracker
    actions: list[int] = []
    players: list[int] = []
    try:
        for k, raw in enumerate(wanted):
            point = session.point
            if point is None:
                raise DemoError(f"the duel stopped ({tracker.result.reason} {tracker.result.error}) before response {k} "
                                f"of {len(wanted)}")  # fmt: skip
            response = canonical_response(point.decision, raw)
            try:
                plan = actions_for_response(point.state, response)
            except BranchError as exc:
                raise DemoError(f"response {k}: {exc}") from None
            if plan is None:
                raise DemoError(f"response {k} ({raw.hex()}) to {point.decision.name} of player {point.player} is not an "
                                f"answer our action model can give")  # fmt: skip
            for idx in plan:
                players.append(session.point.player)
                actions.append(idx)
                session.act(idx)
            if tracker.result.retries:
                raise DemoError(f"response {k} ({raw.hex()}) to {point.decision.name}: the engine answered MSG_RETRY")
        solver_part, solver_steps = len(tracker.result.responses), len(actions)
        if close_turn:
            steps = 0
            while not session.done and tracker.turn < 2:
                point = session.point
                if point is None:
                    break
                kinds = [a.kind for a in point.actions]
                idx = next((kinds.index(k) for k in PASSIVE_KINDS if k in kinds), 0)
                players.append(point.player)
                actions.append(idx)
                session.act(idx)
                steps += 1
                if steps > MAX_CLOSING_STEPS:
                    raise DemoError(f"closing the turn took more than {MAX_CLOSING_STEPS} steps")
            if tracker.result.retries:
                raise DemoError("closing the turn: the engine answered MSG_RETRY")
        res = tracker.result
        if res.reason == "error":
            raise DemoError(f"the engine stopped with an error: {res.error}")
        board = board_summary(session.core, tracker.turn, (tracker.lp[0], tracker.lp[1]))
    finally:
        session.close()
    missing = board_summary_missing(board, targets, cards)
    if missing:
        raise DemoError(f"final board lacks {', '.join(t.to_arg() for t in missing)}")
    return DemoLine(list(res.responses), actions, players, solver_part, solver_steps, board)


@dataclass
class Demonstration:
    """One record of the demonstration set: a deck, an opening hand, a variant and its verified lines."""

    deck: dict  # {"name", "main", "extra"}: the deck list as given (before the hand was drawn)
    hand_index: int
    hand_seed: int
    hand: list[int]  # opening hand (passwords)
    variant: str  # "plain" (solo opening) or "fire" (the opponent plays `fire` at every legal window)
    targets: list[str]  # target cards, password[@zone[:fd]]
    environment: dict | None = None  # {"version", "fingerprint"}, None without an environment
    fire: int | None = None
    status: str = "pending"  # solved / unsolved / unverified / no_window / error
    error: str = ""
    start: dict | None = None  # the duel the lines start from: core_seed, rule_flags, player, decks in load order
    lines: list[DemoLine] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)  # solver lines that failed the fresh replay: {"source", "error"}
    solver: dict = field(default_factory=dict)  # commit, budget, seed, candidates, elapsed_s, events of note
    engine: dict = field(default_factory=lambda: {"ocgcore": list(_core.ocg_version())})

    @property
    def key(self) -> tuple:
        return (self.deck["name"], self.hand_index, self.variant, self.fire)

    @classmethod
    def new(cls, deck: Deck, hand: Sequence[int], *, hand_index: int, hand_seed: int, variant: str,
            targets: Sequence[TargetCard | str], environment: Environment | Mapping | None = None,
            fire: int | None = None) -> Demonstration:  # fmt: skip
        if isinstance(environment, Environment):
            environment = {"version": environment.version, "fingerprint": environment.fingerprint}
        return cls(deck={"name": deck.name, "main": list(deck.main), "extra": list(deck.extra)}, hand_index=hand_index,
                   hand_seed=hand_seed, hand=list(hand), variant=variant,
                   targets=[t.to_arg() for t in parse_targets(targets)], environment=environment, fire=fire)  # fmt: skip

    @classmethod
    def start_from(cls, yrp: YrpFile, **kwargs) -> Demonstration:
        demo = cls.new(**kwargs)
        demo.set_start(yrp)
        return demo

    def set_start(self, yrp: YrpFile) -> None:
        rep = Replay.from_yrp(yrp)
        self.start = {"core_seed": rep.core_seed, "rule_flags": rep.rule_flags, "player": rep.player,
                      "decks": {s: {"main": rep.decks[s]["main"], "extra": rep.decks[s]["extra"]} for s in "ab"}}  # fmt: skip

    def replay(self, line: int = 0) -> Replay:
        """The line as a ygorl :class:`Replay` (load, play, fork, export to ``.yrpX``)."""
        if self.start is None:
            raise ValueError("this record has no start position (no line was solved)")
        s = self.start
        decks = {side: {"name": self.deck["name"] if side == "a" else "opponent", **s["decks"][side], "side": []} for side in "ab"}
        return Replay(seed=0, first=0, rule_flags=s["rule_flags"], player=dict(s["player"]), shuffle_decks=False,
                      decks=decks, responses=list(self.lines[line].responses), environment=self.environment,
                      engine=dict(self.engine), seed_words=list(s["core_seed"]))  # fmt: skip

    # -- storage ----------------------------------------------------------
    def to_json(self) -> dict:
        d = asdict(self)
        d["lines"] = [ln.to_json() for ln in self.lines]
        return {"format": DEMO_FORMAT, "format_version": DEMO_FORMAT_VERSION, **d}

    @classmethod
    def from_json(cls, data: dict) -> Demonstration:
        if data.get("format") != DEMO_FORMAT:
            raise ValueError(f"not a ygorl demonstration (format={data.get('format')!r})")
        if data.get("format_version") != DEMO_FORMAT_VERSION:
            raise ValueError(f"unsupported demonstration format_version {data.get('format_version')} "
                             f"(expected {DEMO_FORMAT_VERSION})")  # fmt: skip
        fields = {k: v for k, v in data.items() if k not in ("format", "format_version")}
        fields["lines"] = [DemoLine.from_json(ln) for ln in fields.get("lines", [])]
        return cls(**fields)

    def append_to(self, path: str | Path) -> None:
        """Append this record as one JSON line (``O_APPEND``: whole-line writes from one process at a time)."""
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(self.to_json(), separators=(",", ":")) + "\n")


def read_jsonl(path: str | Path) -> Iterator[Demonstration]:
    """The records of a demonstration file; a truncated last line (interrupted run) is skipped."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield Demonstration.from_json(data)


def iter_steps(demo: Demonstration, line: int = 0, *, env: Environment | None = None, cards=None,
               scripts: _core.ScriptDirectory | None = None) -> Iterator[tuple[DecisionPoint, int]]:  # fmt: skip
    """Replay ``demo.lines[line]`` and yield every ``(DecisionPoint, chosen action index)``.

    The decision point carries the decoded decision, its legal actions and the events since the
    previous point; observations are encoded from it (``ygorl.env.encoding``) by the consumer.
    A record bound to an environment must be replayed under that environment (``env``).
    """
    ln = demo.lines[line]
    session = DuelSession(_duel_from(demo.replay(line), cards, scripts, env))
    try:
        for i, idx in enumerate(ln.actions):
            point = session.point
            if point is None:
                raise DemoError(f"step {i}: the duel stopped ({session.tracker.result.reason}) before the line ended")
            yield point, idx
            session.act(idx)
        if session.tracker.result.responses != ln.responses:
            raise DemoError("the action indices do not reproduce the recorded responses")
    finally:
        session.close()


__all__ = ["DEMO_FORMAT", "DemoError", "DemoLine", "Demonstration", "canonical_response", "convert_line", "iter_steps",
           "read_jsonl"]  # fmt: skip
