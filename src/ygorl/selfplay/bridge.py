"""Bridge between what Master Duel asks and the shadow duel's decision point (design 10 「自博弈的架构」).

:class:`Request` is what a reader of MD's input state produces (``ygorl.md.request`` will; it is defined here so
the bridge does not depend on the reader): a kind, the candidates MD lists and how many to pick. :func:`align`
pairs it with a ygopro :class:`~ygorl.engine.tracker.DecisionPoint`: the kinds must correspond (:data:`DECISIONS`)
and the candidates are paired by ``(password, controller, location[, sequence])`` with multiplicity. What is left
on either side is the data design 10 asks to record (sampled hidden cards, effects of this turn, MD merging or
skipping a step, rule differences), not a failure.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ygorl.engine import constants as C
from ygorl.engine import messages as M
from ygorl.engine.actions import Action
from ygorl.engine.tracker import DecisionPoint

# MD request kind -> the ygopro decisions that ask the same question. MD asks one card at a time where ygopro
# builds a set (SelectCard sub-steps, SelectUnselectCard), so every card list is "select_card".
DECISIONS: dict[str, tuple[type[M.Decision], ...]] = {
    "idle": (M.SelectIdleCmd,),  # Main Phase command point: a card's command icons, the phase button
    "battle": (M.SelectBattleCmd,),  # Battle Phase command point: attack / activate / Main Phase 2 / End Phase
    "place": (M.SelectPlace, M.SelectDisfield),  # a zone on the field
    "position": (M.SelectPosition,),  # Attack / Defense (face-up / face-down) dialog
    "chain": (M.SelectChain,),  # "activate in chain?" window
    "select_card": (M.SelectCard, M.SelectUnselectCard, M.SelectTribute, M.SelectSum),  # card list / target pick
    "yesno": (M.SelectYesNo, M.SelectEffectYn),  # yes / no dialog
    "option": (M.SelectOption,),  # effect choice dialog
    "counter": (M.SelectCounter,),
    "sort": (M.SortCard, M.SortChain),
    "announce": (M.AnnounceCard, M.AnnounceRace, M.AnnounceAttrib, M.AnnounceNumber),
}

# core location -> the snapshot's location names (ygorl.advisor.snapshot.LOCATIONS, plus the deck and Xyz materials)
LOCATION_NAMES: dict[int, str] = {
    C.LOCATION_DECK: "deck",
    C.LOCATION_HAND: "hand",
    C.LOCATION_MZONE: "mzone",
    C.LOCATION_SZONE: "szone",
    C.LOCATION_GRAVE: "grave",
    C.LOCATION_REMOVED: "removed",
    C.LOCATION_EXTRA: "extra",
    C.LOCATION_OVERLAY: "overlay",
}
# actions whose identity is a value, not a card or a zone: Option.value carries it
VALUE_KINDS = ("position", "yes", "no", "option", "number", "race", "attribute", "declare", "rps")


@dataclass(frozen=True)
class Option:
    """One candidate MD offers: a card (password + where it is), a zone (password 0), or a value."""

    password: int
    location: str  # a LOCATION_NAMES value; "" for value options
    sequence: int | None = None  # None: any copy in that location (hand order differs on screen, design 10)
    controller: int = 0  # 0 advisor, 1 opponent
    value: int | None = None  # value options: POS_* for "position", 1 / 0 for "yesno", the description for "option"


@dataclass
class Request:
    """What MD is waiting for (one question; a multi-select is one Request per pick)."""

    kind: str  # a DECISIONS key
    options: list[Option] = field(default_factory=list)
    min: int = 1
    max: int = 1


def location_name(location: int) -> str:
    if location & C.LOCATION_OVERLAY:
        return "overlay"
    return LOCATION_NAMES.get(location, f"0x{location:x}")


Key = tuple  # ("card", password, controller, location name, sequence) | ("value", value)


def action_key(action: Action) -> Key | None:
    """What an action chooses, comparable with an :class:`Option`; None for controls (pass, finish, phases, ...)."""
    if action.kind == "place":
        v = action.value
        return "card", 0, v >> 16, location_name(v >> 8 & 0xFF), v & 0xFF
    if action.kind in VALUE_KINDS:
        return "value", action.description if action.kind == "option" else action.value
    if action.card is None or action.kind == "unselect":
        return None
    loc = action.card.loc
    return "card", action.card.code, loc.controller, location_name(loc.location), loc.sequence


def _matches(option: Option, key: Key, exact: bool) -> bool:
    if option.value is not None or key[0] == "value":
        return key == ("value", option.value)
    _, password, controller, location, sequence = key
    if (password, controller, location) != (option.password, option.controller, option.location):
        return False
    return sequence == option.sequence if exact else option.sequence is None


def choices(point: DecisionPoint) -> tuple[dict[Key, list[int]], list[int]]:
    """The point's choosable actions grouped by what they choose (a card's commands share one key), and the rest."""
    items: dict[Key, list[int]] = {}
    controls: list[int] = []
    for i in point.choosable:
        key = action_key(point.actions[i])
        if key is None:
            controls.append(i)
        else:
            items.setdefault(key, []).append(i)
    return items, controls


@dataclass
class Alignment:
    """An MD request against a ygopro decision point. Indices refer to ``request.options`` / ``point.actions``."""

    kind_match: bool  # the ygopro decision is one DECISIONS lists for the request kind
    pairs: list[tuple[int, tuple[int, ...]]]  # (MD option, ygopro actions on that card / zone / value)
    md_only: list[int]  # MD options with no ygopro action
    ygo_only: list[tuple[int, ...]]  # ygopro choices MD does not offer
    controls: list[int]  # ygopro actions that are not a candidate (pass, finish, cancel, phase changes)

    @property
    def consistent(self) -> bool:
        return self.kind_match and not self.md_only and not self.ygo_only


def _pair(options: Sequence[Option], items: dict[Key, list[int]]) -> tuple[dict[int, Key], list[Key]]:
    free = list(items)
    paired: dict[int, Key] = {}
    for exact in (True, False):  # options naming a sequence first, so wildcards do not take their card
        for o, option in enumerate(options):
            if o in paired:
                continue
            key = next((k for k in free if _matches(option, k, exact)), None)
            if key is not None:
                paired[o] = key
                free.remove(key)
    return paired, free


def align(request: Request, point: DecisionPoint) -> Alignment:
    """Pair MD's candidates with the point's choosable actions (see the module docstring)."""
    if request.kind not in DECISIONS:
        raise ValueError(f"unknown MD request kind {request.kind!r}; known: {sorted(DECISIONS)}")
    items, controls = choices(point)
    paired, free = _pair(request.options, items)
    return Alignment(
        kind_match=isinstance(point.decision, DECISIONS[request.kind]),
        pairs=[(o, tuple(items[k])) for o, k in sorted(paired.items())],
        md_only=[o for o in range(len(request.options)) if o not in paired],
        ygo_only=[tuple(items[k]) for k in free],
        controls=controls,
    )


def find(point: DecisionPoint, option: Option, kind: str | None = None) -> int | None:
    """The first choosable action on ``option``'s card / zone / value (an exact sequence wins), of action ``kind``
    if given (a command point offers several per card: summon, mset, activate, ...), or None."""
    items, _ = choices(point)
    if kind is not None:
        items = {k: v for k, v in ((k, [i for i in v if point.actions[i].kind == kind]) for k, v in items.items()) if v}
    paired, _ = _pair([option], items)
    return items[paired[0]][0] if paired else None


__all__ = ["DECISIONS", "LOCATION_NAMES", "Alignment", "Option", "Request", "action_key", "align", "choices", "find",
           "location_name"]  # fmt: skip
