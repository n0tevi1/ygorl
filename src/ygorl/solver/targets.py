"""Target boards for the combo solver, and the board summary our verification checks them against.

A target card is ``password[@zone[:fd]]``, the solver's ``--target`` grammar restricted to card
passwords (cards are keyed by password in this repository, never by name). Zones follow the
solver's judge (``search.cpp``, ``BoardKey``): ``atk`` / ``def`` / ``mzone`` all mean "in a
monster zone" -- the battle position is ignored, only the face (up, or down with ``:fd``)
counts; ``szone`` is a spell/trap zone (pendulum zones included); ``hand`` / ``grave`` /
``banished`` are counted off the field. ``field`` and ``extra`` are refused, as the solver does.
Repeated entries require that many copies.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from ygorl.engine import constants as C
from ygorl.engine.query import parse_query_location

FIELD_ZONES = ("atk", "def", "mzone", "szone")
OFF_FIELD_ZONES = ("hand", "grave", "banished")
ZONES = FIELD_ZONES + OFF_FIELD_ZONES
_BOARD_FLAGS = C.QUERY_CODE | C.QUERY_POSITION | C.QUERY_OVERLAY_CARD


@dataclass(frozen=True)
class TargetCard:
    code: int  # card password
    zone: str = "atk"
    facedown: bool = False

    def __post_init__(self) -> None:
        if self.zone not in ZONES:
            raise ValueError(f"unknown target zone {self.zone!r} (expected one of {', '.join(ZONES)})")
        if self.facedown and self.zone not in FIELD_ZONES:
            raise ValueError(f"face-down (:fd) only applies to field zones, not {self.zone!r}")

    @classmethod
    def parse(cls, spec: str) -> TargetCard:
        text = spec.strip()
        card, _, rest = text.partition("@")
        if not card.isdigit():
            raise ValueError(f"target {spec!r}: expected a card password (8-digit number), not a name")
        zone, facedown = "atk", False
        if rest:
            zone, _, flag = rest.partition(":")
            zone = zone.lower()
            if flag not in ("", "fd"):
                raise ValueError(f"target {spec!r}: unknown suffix {flag!r} (only ':fd')")
            facedown = flag == "fd"
            if zone in ("field", "extra"):
                raise ValueError(f"target {spec!r}: zone {zone!r} is refused (ambiguous or not a target zone)")
        return cls(int(card), zone, facedown)

    def to_arg(self) -> str:
        return f"{self.code}@{self.zone}{':fd' if self.facedown else ''}"


def parse_targets(specs: Iterable[str | TargetCard]) -> list[TargetCard]:
    return [s if isinstance(s, TargetCard) else TargetCard.parse(s) for s in specs]


def board_summary(core, turn: int, lp: tuple[int, int]) -> dict:
    """Both players' public and private zones from a live ``_core.Duel`` (codes are passwords)."""
    players = []
    for p in (0, 1):

        def cards(loc: int) -> list[dict]:
            out = []
            for i, c in enumerate(parse_query_location(core.query_location(_BOARD_FLAGS, p, loc))):
                if c is not None and c.get("code"):
                    out.append(
                        {"code": c["code"], "position": c.get("position", 0), "sequence": i, "overlay": c["overlay"]}
                    )
            return out

        extra = cards(C.LOCATION_EXTRA)
        players.append({
            "lp": lp[p],
            "mzone": cards(C.LOCATION_MZONE),
            "szone": cards(C.LOCATION_SZONE),
            "hand": [c["code"] for c in cards(C.LOCATION_HAND)],
            "grave": [c["code"] for c in cards(C.LOCATION_GRAVE)],
            "banished": [c["code"] for c in cards(C.LOCATION_REMOVED)],
            "extra_faceup": [c["code"] for c in extra if c["position"] & C.POS_FACEUP],
            "deck_count": core.query_count(p, C.LOCATION_DECK),
            "extra_count": len(extra),
        })  # fmt: skip
    return {"turn": turn, "players": players}


def board_summary_missing(board: dict, targets: Iterable[TargetCard], cards: Mapping | None = None,
                          player: int = 0) -> list[TargetCard]:  # fmt: skip
    """The target cards ``board`` (from :func:`board_summary`) lacks for ``player``, one entry per missing copy.

    With ``cards`` (a :class:`~ygorl.cards.cdb.CardDB`), alternate artworks count as their original.
    """

    def canon(code: int) -> int:
        return cards.canonical(code) if cards is not None else code

    side = board["players"][player]
    have: Counter = Counter()
    for zone in ("mzone", "szone"):
        for c in side.get(zone, []):
            have[(zone, canon(c["code"]), not c["position"] & C.POS_FACEUP)] += 1
    for zone in OFF_FIELD_ZONES:
        for code in side.get(zone, []):
            have[(zone, canon(code), False)] += 1
    missing = []
    for t in targets:
        zone = "szone" if t.zone == "szone" else ("mzone" if t.zone in FIELD_ZONES else t.zone)
        key = (zone, canon(t.code), t.facedown)
        if have[key] > 0:
            have[key] -= 1
        else:
            missing.append(t)
    return missing


__all__ = ["TargetCard", "board_summary", "board_summary_missing", "parse_targets"]
