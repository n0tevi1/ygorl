"""Board snapshots: what the advisor can see of a game, independent of where it was read (design 10).

A :class:`BoardSnapshot` is the only interface between a state source (a hand-written JSON file, the Master Duel
memory reader, a self-play duel for fidelity tests) and the decision side (:mod:`ygorl.advisor.reconstruct`). It
holds public information plus the advisor's own hand, cards by ``password``; an opponent card the advisor cannot
see (a hand card, a Set card, a face-down banished card) has password ``0``. Sides are ordered (advisor,
opponent), whoever went first in the real game.

JSON form::

    {"turn": 3, "phase": "main1", "turn_player": 0, "went_first": true,
     "normal_summoned": false, "normal_summons": 0, "extra_tribute_summons": 0,
     "activated": [[78665705, "hand"]], "player_effects": [[78665705, 0]],
     "sides": [{"lp": 8000, "deck_count": 33, "extra_count": 15,
                "cards": [{"location": "hand", "password": 14558127},
                          {"location": "mzone", "sequence": 2, "password": 89631139, "position": "faceup_attack"},
                          {"location": "mzone", "sequence": 5, "password": 84013237, "materials": [89631139]}]},
               {"lp": 7200, "deck_count": 30, "extra_count": 13,
                "cards": [{"location": "hand"}, {"location": "szone", "sequence": 1, "position": "facedown"}]}]}

Locations: ``hand``, ``mzone`` (sequence 0-4, Extra Monster Zones 5-6), ``szone`` (0-4, Field Zone 5), ``grave``,
``removed``, ``extra`` (face-up Pendulum cards only; face-down Extra Deck cards are counted in ``extra_count``).
Hand, GY and banished cards take their list order; ``sequence`` matters on the field only.

The turn so far (design 10「重建能还原什么」): per card ``summoned`` (normal / special / flip / set this turn),
``position_locked``, ``attacked``, ``set_this_turn``, ``effect_used`` (an effect of the card was activated this
turn), ``card_activated`` (a face-up Spell / Trap activated this turn); per snapshot ``normal_summons`` (Normal and
Tribute Summons this turn), ``extra_tribute_summons`` (additional Tribute Summons granted this turn) and
``activated`` (cards activated this turn that left the field: their once-per-turn activation is spent) and
``player_effects`` (password, value) of the effects in force on the advisor as a player, e.g. Ancient Chant's
additional Tribute Summon (MD keeps them in the player's effect list, coded by the card they come from).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ygorl.cards.ydk import valid_password
from ygorl.engine import constants as C

LOCATIONS: Mapping[str, int] = {
    "hand": C.LOCATION_HAND,
    "mzone": C.LOCATION_MZONE,
    "szone": C.LOCATION_SZONE,
    "grave": C.LOCATION_GRAVE,
    "removed": C.LOCATION_REMOVED,
    "extra": C.LOCATION_EXTRA,
}
POSITIONS: Mapping[str, int] = {
    "faceup_attack": C.POS_FACEUP_ATTACK,
    "facedown_attack": C.POS_FACEDOWN_ATTACK,
    "faceup_defense": C.POS_FACEUP_DEFENSE,
    "facedown_defense": C.POS_FACEDOWN_DEFENSE,
    "faceup": C.POS_FACEUP,
    "facedown": C.POS_FACEDOWN,
}
_ZONES = {"mzone": 7, "szone": 6}  # sequences per side
PHASES = ("main1", "battle", "main2")  # the advisor's own idle points: Main Phases and the Battle Phase command
UNKNOWN = 0  # password of a card the advisor cannot see
SUMMONED = ("", "normal", "special", "flip", "set")  # how a monster came to the field this turn


class SnapshotError(ValueError):
    """A board snapshot that is malformed or describes an impossible board."""


def default_position(location: str, password: int) -> str:
    if location == "mzone":
        return "faceup_attack" if password else "facedown_defense"
    if location == "szone":
        return "faceup" if password else "facedown"
    if location == "removed" and not password:
        return "facedown"
    return "faceup" if location in ("grave", "removed", "extra") else "facedown"


@dataclass(frozen=True)
class SnapshotCard:
    location: str
    password: int = UNKNOWN
    sequence: int = 0
    position: str = ""  # a POSITIONS name; "" = the location's default (see default_position)
    owner: int | None = None  # 0 advisor / 1 opponent; None = the controller
    materials: tuple[int, ...] = ()  # Xyz materials (mzone only), bottom first
    counters: tuple[tuple[int, int], ...] = ()  # (counter type, count)
    position_locked: bool = False  # summoned, Set or repositioned this turn: cannot change position now
    attacked: bool = False  # declared an attack this turn (Battle Phase snapshots): cannot attack again
    set_this_turn: bool = False  # Set this turn (Spell & Trap Zone): a Trap or Quick-Play Spell cannot activate yet
    effect_used: bool = False  # an effect of this card was activated this turn (its once-per-turn use is spent)
    card_activated: bool = False  # this face-up Spell / Trap was activated this turn
    summoned: str = ""  # SUMMONED: how this monster came to the field this turn ("" = an earlier turn)

    @property
    def pos(self) -> str:
        return self.position or default_position(self.location, self.password)

    @property
    def core_location(self) -> int:
        return LOCATIONS[self.location]

    @property
    def core_position(self) -> int:
        return POSITIONS[self.pos]

    @property
    def known(self) -> bool:
        return self.password != UNKNOWN

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"location": self.location}
        if self.password:
            out["password"] = self.password
        if self.location in _ZONES:
            out["sequence"] = self.sequence
        if self.position:
            out["position"] = self.position
        if self.owner is not None:
            out["owner"] = self.owner
        if self.materials:
            out["materials"] = list(self.materials)
        if self.counters:
            out["counters"] = [list(c) for c in self.counters]
        if self.position_locked:
            out["position_locked"] = True
        if self.attacked:
            out["attacked"] = True
        if self.set_this_turn:
            out["set_this_turn"] = True
        if self.effect_used:
            out["effect_used"] = True
        if self.card_activated:
            out["card_activated"] = True
        if self.summoned:
            out["summoned"] = self.summoned
        return out

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> SnapshotCard:
        if not isinstance(data, Mapping):
            raise SnapshotError(f"a card must be an object, not {data!r}")
        unknown = set(data) - {"location", "password", "sequence", "position", "owner", "materials", "counters",
                               "position_locked", "attacked", "set_this_turn", "effect_used", "card_activated",
                               "summoned"}  # fmt: skip
        if unknown:
            raise SnapshotError(f"unknown card keys {sorted(unknown)}")
        try:
            return cls(location=data["location"], password=int(data.get("password", UNKNOWN)),
                       sequence=int(data.get("sequence", 0)), position=data.get("position", ""),
                       owner=data.get("owner"), materials=tuple(int(p) for p in data.get("materials", ())),
                       counters=tuple((int(t), int(n)) for t, n in data.get("counters", ())),
                       position_locked=bool(data.get("position_locked", False)),
                       attacked=bool(data.get("attacked", False)),
                       set_this_turn=bool(data.get("set_this_turn", False)),
                       effect_used=bool(data.get("effect_used", False)),
                       card_activated=bool(data.get("card_activated", False)),
                       summoned=str(data.get("summoned", "")))  # fmt: skip
        except KeyError:
            raise SnapshotError(f"card without a location: {dict(data)}") from None
        except (TypeError, ValueError) as exc:
            raise SnapshotError(f"malformed card {dict(data)}: {exc}") from None


@dataclass(frozen=True)
class SnapshotSide:
    lp: int
    cards: tuple[SnapshotCard, ...] = ()
    deck_count: int = 0
    extra_count: int = 0  # face-down Extra Deck cards

    def at(self, location: str) -> list[SnapshotCard]:
        return [c for c in self.cards if c.location == location]

    @property
    def hand_count(self) -> int:
        return len(self.at("hand"))

    def seen(self) -> list[int]:
        """Passwords of every known card of this side, Xyz materials included."""
        return [p for c in self.cards for p in (c.password, *c.materials) if p]

    def to_json(self) -> dict[str, Any]:
        return {"lp": self.lp, "deck_count": self.deck_count, "extra_count": self.extra_count,
                "cards": [c.to_json() for c in self.cards]}  # fmt: skip

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> SnapshotSide:
        if not isinstance(data, Mapping):
            raise SnapshotError(f"a side must be an object, not {data!r}")
        unknown = set(data) - {"lp", "cards", "deck_count", "extra_count"}
        if unknown:
            raise SnapshotError(f"unknown side keys {sorted(unknown)}")
        try:
            return cls(lp=int(data["lp"]), cards=tuple(SnapshotCard.from_json(c) for c in data.get("cards", ())),
                       deck_count=int(data.get("deck_count", 0)), extra_count=int(data.get("extra_count", 0)))  # fmt: skip
        except KeyError:
            raise SnapshotError("a side needs its lp") from None
        except (TypeError, ValueError) as exc:
            raise SnapshotError(f"malformed side: {exc}") from None


@dataclass(frozen=True)
class BoardSnapshot:
    turn: int
    phase: str
    sides: tuple[SnapshotSide, SnapshotSide]  # (advisor, opponent)
    turn_player: int = 0  # 0 = the advisor's turn
    went_first: bool = True  # the advisor played first in the real game
    normal_summoned: bool = False  # the advisor has used its Normal Summon / Set this turn
    source: str = ""  # where the snapshot came from ("manual", "selfplay", "mdmem:<build>")
    normal_summons: int = 0  # Normal / Tribute Summons and Sets this turn (0 with normal_summoned: one)
    extra_tribute_summons: int = 0  # additional Tribute Summons granted this turn (Ancient Chant, Card Advance ...)
    activated: tuple[tuple[int, str], ...] = ()  # (password, location) activated this turn, no longer on the field
    player_effects: tuple[tuple[int, int], ...] = ()  # (password, value) of card effects in force on the advisor

    @property
    def summons_used(self) -> int:
        """Normal Summons / Sets used this turn (``normal_summoned`` alone counts as one)."""
        return max(self.normal_summons, int(self.normal_summoned))

    @property
    def normal_summon_left(self) -> bool:
        """The advisor can still Normal Summon / Set (counting the additional Tribute Summons granted)."""
        return self.summons_used < 1 + self.extra_tribute_summons

    def check(self) -> BoardSnapshot:
        """Raise :class:`SnapshotError` unless the snapshot describes a possible board; return self."""
        if self.turn < 1:
            raise SnapshotError(f"turn must be at least 1, not {self.turn}")
        if self.phase not in PHASES:
            raise SnapshotError(f"phase must be one of {PHASES}, not {self.phase!r}")
        if self.turn_player not in (0, 1):
            raise SnapshotError(f"turn_player must be 0 (advisor) or 1 (opponent), not {self.turn_player}")
        if self.turn == 1 and self.went_first != (self.turn_player == 0):
            raise SnapshotError("turn 1 is the first player's turn")
        if len(self.sides) != 2:
            raise SnapshotError("a snapshot has two sides: (advisor, opponent)")
        if self.normal_summons < 0 or self.extra_tribute_summons < 0:
            raise SnapshotError("summon counts must be non-negative")
        for pw, where in self.activated:
            if not valid_password(pw) or where not in LOCATIONS:
                raise SnapshotError(f"activated: ({pw}, {where!r}) is not (password, location)")
        for pw, _ in self.player_effects:
            if not valid_password(pw):
                raise SnapshotError(f"player_effects: invalid card password {pw}")
        for s, side in enumerate(self.sides):
            who = ("advisor", "opponent")[s]
            if side.lp < 1:
                raise SnapshotError(f"{who}: lp must be at least 1 (the game is over otherwise)")
            if side.deck_count < 0 or side.extra_count < 0:
                raise SnapshotError(f"{who}: card counts must be non-negative")
            used: set[tuple[str, int]] = set()
            for card in side.cards:
                _check_card(card, s, who, used)
        return self

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"turn": self.turn, "phase": self.phase, "turn_player": self.turn_player,
                               "went_first": self.went_first, "normal_summoned": self.normal_summoned,
                               "sides": [s.to_json() for s in self.sides]}  # fmt: skip
        if self.normal_summons:
            out["normal_summons"] = self.normal_summons
        if self.extra_tribute_summons:
            out["extra_tribute_summons"] = self.extra_tribute_summons
        if self.activated:
            out["activated"] = [list(a) for a in self.activated]
        if self.player_effects:
            out["player_effects"] = [list(e) for e in self.player_effects]
        if self.source:
            out["source"] = self.source
        return out

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> BoardSnapshot:
        if not isinstance(data, Mapping):
            raise SnapshotError("a snapshot must be a JSON object")
        unknown = set(data) - {"turn", "phase", "turn_player", "went_first", "normal_summoned", "sides", "source",
                               "normal_summons", "extra_tribute_summons", "activated", "player_effects"}  # fmt: skip
        if unknown:
            raise SnapshotError(f"unknown snapshot keys {sorted(unknown)}")
        try:
            sides = tuple(SnapshotSide.from_json(s) for s in data["sides"])
            snap = cls(turn=int(data["turn"]), phase=str(data.get("phase", "main1")), sides=sides,  # type: ignore[arg-type]
                       turn_player=int(data.get("turn_player", 0)), went_first=bool(data.get("went_first", True)),
                       normal_summoned=bool(data.get("normal_summoned", False)),
                       normal_summons=int(data.get("normal_summons", 0)),
                       extra_tribute_summons=int(data.get("extra_tribute_summons", 0)),
                       activated=tuple((int(pw), str(loc)) for pw, loc in data.get("activated", ())),
                       player_effects=tuple((int(pw), int(v)) for pw, v in data.get("player_effects", ())),
                       source=str(data.get("source", "")))  # fmt: skip
        except KeyError as exc:
            raise SnapshotError(f"snapshot is missing {exc.args[0]!r}") from None
        except (TypeError, ValueError) as exc:
            raise SnapshotError(f"malformed snapshot: {exc}") from None
        return snap.check()

    def dumps(self) -> str:
        return json.dumps(self.to_json(), ensure_ascii=False, indent=1) + "\n"


def _check_card(card: SnapshotCard, side: int, who: str, used: set[tuple[str, int]]) -> None:
    where = f"{who}: {card.location} card {card.password or '(unknown)'}"
    if card.location not in LOCATIONS:
        raise SnapshotError(f"{where}: unknown location (expected one of {sorted(LOCATIONS)})")
    if card.position and card.position not in POSITIONS:
        raise SnapshotError(f"{where}: unknown position {card.position!r} (expected one of {sorted(POSITIONS)})")
    for p in (card.password, *card.materials):
        if p and not valid_password(p):
            raise SnapshotError(f"{where}: invalid card password {p}")
    if card.owner not in (None, 0, 1):
        raise SnapshotError(f"{where}: owner must be 0 or 1")
    if side == 0 and not card.known:
        raise SnapshotError(f"{where}: every card of the advisor's own side must be known")
    if not card.known and card.location in ("grave", "extra"):
        raise SnapshotError(f"{where}: {card.location} cards are public, give the password")
    if card.location == "removed" and not card.known and card.pos != "facedown":
        raise SnapshotError(f"{where}: an unknown banished card must be face-down")
    if card.location in _ZONES:
        if not 0 <= card.sequence < _ZONES[card.location]:
            raise SnapshotError(f"{where}: sequence {card.sequence} out of range 0..{_ZONES[card.location] - 1}")
        if (card.location, card.sequence) in used:
            raise SnapshotError(f"{where}: zone {card.location} {card.sequence} holds two cards")
        used.add((card.location, card.sequence))
        if not card.known and card.pos.startswith("faceup"):
            raise SnapshotError(f"{where}: a face-up card must be known")
    if card.attacked and card.location != "mzone":
        raise SnapshotError(f"{where}: only monsters in a Monster Zone attack")
    if card.summoned not in SUMMONED or (card.summoned and card.location != "mzone"):
        raise SnapshotError(f"{where}: summoned must be one of {SUMMONED[1:]} on a monster in a Monster Zone")
    if (card.effect_used or card.card_activated) and card.location not in _ZONES:
        raise SnapshotError(f"{where}: effect_used / card_activated describe a card on the field")
    if card.materials and card.location != "mzone":
        raise SnapshotError(f"{where}: only monsters in a Monster Zone have materials")


# summoned, flip summoned, Set or repositioned this turn: the monster cannot change position
_LOCKED = (C.STATUS_SUMMON_TURN | C.STATUS_FLIP_SUMMON_TURN | C.STATUS_SPSUMMON_TURN | C.STATUS_FORM_CHANGED
           | C.STATUS_SET_TURN)  # fmt: skip


def _position_name(location: str, position: int) -> str:
    if location == "mzone":
        return next((n for n, v in POSITIONS.items() if v == position), "faceup_attack")
    return "faceup" if position & C.POS_FACEUP else "facedown"


def snapshot_from_core(core, point, *, source: str = "selfplay") -> BoardSnapshot:
    """The snapshot the deciding player of ``point`` would read from the live duel ``core`` (fidelity tests).

    Opponent cards the player cannot see get password 0; the deciding player is the advisor (side 0).
    ``normal_summoned`` is not in the core's queries and stays False.
    """
    from ygorl.engine.query import CARD_QUERY_FLAGS, parse_query_location

    viewer = point.player
    sides = []
    for s, con in enumerate((viewer, 1 - viewer)):
        cards: list[SnapshotCard] = []
        extra_count = 0
        for name, loc in LOCATIONS.items():
            for seq, card in enumerate(parse_query_location(core.query_location(CARD_QUERY_FLAGS, con, loc))):
                if card is None:
                    continue
                position = card.get("position", 0)
                if name == "extra" and not position & C.POS_FACEUP:
                    extra_count += 1
                    continue
                known = s == 0 or bool(card["public"])
                owner = card.get("owner", con)
                cards.append(SnapshotCard(
                    location=name, password=card["code"] if known else UNKNOWN,
                    sequence=seq if name in _ZONES else 0, position=_position_name(name, position),
                    owner=None if owner == con else int(owner != viewer), materials=tuple(card["overlay"]),
                    counters=tuple((c & 0xFFFF, c >> 16) for c in card["counters"]) if known else (),
                    position_locked=name == "mzone" and bool(card.get("status", 0) & _LOCKED)))  # fmt: skip
        lp = point.lp[con]
        sides.append(SnapshotSide(lp=lp, cards=tuple(cards), deck_count=core.query_count(con, C.LOCATION_DECK),
                                  extra_count=extra_count))  # fmt: skip
    phase = "main2" if point.phase == C.PHASE_MAIN2 else "main1"
    return BoardSnapshot(turn=point.turn, phase=phase, sides=(sides[0], sides[1]),
                         turn_player=int(point.turn_player != viewer), went_first=viewer == 0, source=source)  # fmt: skip


def load_snapshot(path: str | Path) -> BoardSnapshot:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SnapshotError(f"{path}: {exc}") from None
    try:
        return BoardSnapshot.from_json(data)
    except SnapshotError as exc:
        raise SnapshotError(f"{path}: {exc}") from None


__all__ = ["BoardSnapshot", "LOCATIONS", "POSITIONS", "SUMMONED", "SnapshotCard", "SnapshotError", "SnapshotSide",
           "UNKNOWN", "load_snapshot", "snapshot_from_core"]  # fmt: skip
