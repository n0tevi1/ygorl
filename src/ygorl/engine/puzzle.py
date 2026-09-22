"""Hand-built duel positions ("puzzles"): cards placed directly in chosen zones.

A :class:`Puzzle` creates a core duel whose cards are added with explicit
location / sequence / position (``OCG_DuelNewCard``) instead of shuffled decks,
so a scenario (a Tribute Summon with a double-tribute monster, a Synchro with a
two-level material, ...) is reached in one or two decisions. Players start with
no hand draw and draw nothing per turn; the duel starts in player 0's first
Main Phase 1.

Puzzles are deterministic like normal duels: the same puzzle and the same
responses give the same message stream (docs/engine.md, determinism). That is
what the T2.3 feasibility checks rely on: they replay a puzzle to a decision
once per candidate response (tests/test_feasible_sets.py).
"""

from __future__ import annotations

from dataclasses import dataclass

from ygorl import _core
from ygorl.engine import constants as C
from ygorl.engine import messages as M
from ygorl.engine.duel import default_cards, default_scripts, expand_seed

@dataclass(frozen=True)
class Placement:
    """One card of a puzzle: ``password`` put in ``location`` / ``sequence`` of ``team``.

    ``position`` defaults to face-up attack on the field and face-down elsewhere.
    """

    password: int
    location: int
    sequence: int = 0
    team: int = 0
    position: int | None = None

    def pos(self) -> int:
        if self.position is not None:
            return self.position
        return C.POS_FACEUP_ATTACK if self.location & C.LOCATION_ONFIELD else C.POS_FACEDOWN_DEFENSE


@dataclass(frozen=True)
class Puzzle:
    cards: tuple[Placement, ...]
    seed: int = 1
    lp: int = 8000
    rule_flags: int = C.DUEL_MODE_MR5

    def start(self, cards=None, scripts=None) -> _core.Duel:
        """A started core duel with every placement added (not yet processed)."""
        cards = cards if cards is not None else default_cards().to_core()
        scripts = scripts if scripts is not None else default_scripts()
        player = (self.lp, 0, 0)  # (starting LP, starting hand, draw per turn)
        core = _core.Duel(expand_seed(self.seed), self.rule_flags, player, player, cards, scripts)
        for base in ("constant.lua", "utility.lua"):
            if not core.load_script(base):
                raise RuntimeError(f"failed to load base script {base}")
        for p in self.cards:
            core.new_card(p.team, 0, p.password, p.team, p.location, p.sequence, p.pos())
        core.start()
        return core


def run_until_stop(core: _core.Duel) -> tuple[int, list[bytes]]:
    """Process until the core awaits a response or the duel ends; return (status, message records)."""
    records: list[bytes] = []
    while True:
        status = core.process()
        records.extend(M.split_messages(core.get_message()))
        if status != _core.DUEL_STATUS_CONTINUE:
            return status, records


def last_decision(records: list[bytes]) -> bytes | None:
    """The last record in ``records`` that is a decision message, or None."""
    for record in reversed(records):
        if isinstance(M.decode_message(record), M.Decision):
            return record
    return None
