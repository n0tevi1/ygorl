"""Card-filter IR: compiled from script filter functions, evaluated against the card database.

A script's target filter (``function s.thfilter(c) return c:IsSetCard(X) and ... end``)
is compiled into a small boolean tree over *predicates* on card-database fields:

==============================  ==========================================================
predicate ``Pred(kind, args)``  meaning
==============================  ==========================================================
``setcard (q, ...)``            any card setcode matches any ``q`` (sub-archetype aware)
``code (pw, ...)``              password (or ``alias``) is one of ``pw``
``type_any / type_all (m,)``    ``type & m != 0`` / ``type & m == m``
``type_eq (t,)``                ``type == t`` (Normal Spell / Normal Trap)
``race / attribute (m,)``       ``race & m != 0`` / ``attribute & m != 0``
``level / rank / link``         ``("eq", v, ...)``, ``("le", v)`` or ``("ge", v)``
``attack / defense``            same comparison forms
``has_level ()``                monster that has a level (not Xyz / Link)
``lists_code (pw, ...)``        the card's script lists ``pw`` in ``s.listed_names``
``lists_archetype (q, ...)``    ... lists ``q`` in ``s.listed_series`` (exact)
``lists_code_as_material``      Fusion material codes (``s.material`` / ``Fusion.AddProcMix``)
``lists_archetype_as_material`` material setcodes (``s.material_setcode``)
``hint (action,)``              ``IsAbleToHand`` / ``IsCanBeSpecialSummoned`` / ...: no
                                constraint, but tells what the effect does with the card
``unknown ()``                  anything we cannot interpret (location checks, other cards)
``any ()`` / ``none ()``        literal ``true`` / ``false`` / ``nil`` filter
==============================  ==========================================================

Evaluation returns a bitset over :class:`CardIndex` positions plus an *exact* flag.
Unknown predicates are over-approximated: dropped from conjunctions, and make a
disjunction (or a negation) unconstrained (``None``), so a filter never loses a
card it could really match because of something we failed to parse.
"""

from __future__ import annotations

from collections import ChainMap
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Union

from ygorl.build import lua
from ygorl.build.lua import BinOp, Call, Const, FuncExpr, Index, Method, Name, Table, UnOp
from ygorl.cards.cdb import EXTRA_DECK_TYPES, CardDB
from ygorl.engine import constants as C

# ---------------------------------------------------------------- IR


@dataclass(frozen=True, slots=True)
class Pred:
    kind: str
    args: tuple = ()


@dataclass(frozen=True, slots=True)
class And:
    items: tuple


@dataclass(frozen=True, slots=True)
class Or:
    items: tuple


@dataclass(frozen=True, slots=True)
class Not:
    item: Filter


Filter = Union[Pred, And, Or, Not]

ANY = Pred("any")
NONE = Pred("none")
UNKNOWN = Pred("unknown")

ACTIONS = ("to_hand", "special_summon", "to_grave", "set", "place", "equip", "banish", "to_deck", "discard", "release")


def make_and(items: Iterable[Filter]) -> Filter:
    out: list[Filter] = []
    for it in items:
        parts = it.items if isinstance(it, And) else (it,)
        for p in parts:
            if p == ANY or p in out:
                continue
            out.append(p)
    if NONE in out:
        return NONE
    if not out:
        return ANY
    return out[0] if len(out) == 1 else And(tuple(out))


def make_or(items: Iterable[Filter]) -> Filter:
    out: list[Filter] = []
    for it in items:
        parts = it.items if isinstance(it, Or) else (it,)
        for p in parts:
            if p == NONE or p in out:
                continue
            out.append(p)
    if ANY in out:
        return ANY
    if not out:
        return NONE
    return out[0] if len(out) == 1 else Or(tuple(out))


def make_not(item: Filter) -> Filter:
    if isinstance(item, Not):
        return item.item
    if item.__class__ is Pred and item.kind in ("unknown", "hint"):
        return UNKNOWN
    return Not(item)


def hints(flt: Filter) -> set[str]:
    """Actions announced by ``hint`` predicates outside negations."""
    out: set[str] = set()
    stack = [flt]
    while stack:
        f = stack.pop()
        if isinstance(f, Pred):
            if f.kind == "hint":
                out.add(f.args[0])
        elif isinstance(f, (And, Or)):
            stack.extend(f.items)
    return out


def is_constrained(flt: Filter) -> bool:
    """False if the filter contains no database predicate (only hints / unknowns / ``any``)."""
    if isinstance(flt, Pred):
        return flt.kind not in ("unknown", "hint", "any")
    if isinstance(flt, Not):
        return is_constrained(flt.item)
    return any(is_constrained(i) for i in flt.items)


def setcode_matches(card_setcode: int, query: int) -> bool:
    """Whether a card setcode belongs to the queried archetype (``Card.IsSetCard``).

    The low 12 bits are the archetype, the high 4 bits flag sub-archetypes: a
    query for "Snake-Eye" (0x205) matches "Snake-Eyes" cards (0x1205), a query
    for a sub-archetype only matches cards that carry its sub bits.
    """
    return (card_setcode & 0xFFF) == (query & 0xFFF) and (card_setcode & query & 0xF000) == (query & 0xF000)


def _material_setcode_matches(query: int, listed: int) -> bool:
    # utility.lua: MatchSetcode(set_code, to_match)
    return (query & 0xFFF) == (listed & 0xFFF) and (query & listed) == query


# ---------------------------------------------------------------- evaluation


def _bits(indices: Iterable[int], n: int) -> int:
    buf = bytearray((n + 8) // 8)
    for i in indices:
        buf[i >> 3] |= 1 << (i & 7)
    return int.from_bytes(buf, "little")


class CardIndex:
    """Bitset index over the canonical, non-token cards of a :class:`CardDB`.

    Alternate artworks are folded into their original password; bit ``i`` stands
    for ``passwords[i]``.
    """

    def __init__(self, db: CardDB) -> None:
        self.db = db
        self.passwords: list[int] = sorted(p for p in db if db.canonical(p) == p and not db[p].is_token)
        self.pos: dict[int, int] = {p: i for i, p in enumerate(self.passwords)}
        n = self.n = len(self.passwords)
        self.universe = (1 << n) - 1
        cards = [db[p] for p in self.passwords]
        # codes: an alternate art's password resolves to its original; rule variants
        # (alias to a different name) also answer to the original's code.
        self._code: dict[int, list[int]] = {}
        for i, card in enumerate(cards):
            self._code.setdefault(card.password, []).append(i)
            if card.alias:
                self._code.setdefault(card.alias, []).append(i)
        for p in db:
            canon = db.canonical(p)
            if canon != p and canon in self.pos:
                self._code.setdefault(p, []).append(self.pos[canon])
        self._by_setbase: dict[int, list[tuple[int, int]]] = {}
        for i, card in enumerate(cards):
            for sc in card.setcodes:
                self._by_setbase.setdefault(sc & 0xFFF, []).append((i, sc))

        def flag_bits(attr: str) -> dict[int, int]:
            groups: dict[int, list[int]] = {}
            for i, card in enumerate(cards):
                v = getattr(card, attr)
                while v:
                    low = v & -v
                    groups.setdefault(low, []).append(i)
                    v ^= low
            return {bit: _bits(ix, n) for bit, ix in groups.items()}

        self._type_bits = flag_bits("type")
        self._race_bits = flag_bits("race")
        self._attr_bits = flag_bits("attribute")
        self._types = [c.type for c in cards]
        self._levels = [c.level for c in cards]
        self._attack = [c.attack for c in cards]
        self._defense = [c.defense for c in cards]
        mon = self.type_mask(C.TYPE_MONSTER)
        self.monsters = mon
        self.extra_deck = mon & self.type_mask(EXTRA_DECK_TYPES)
        xyz = self.type_mask(C.TYPE_XYZ)
        link = self.type_mask(C.TYPE_LINK)
        self.main_deck = self.universe & ~self.extra_deck
        self._level_scope = {"level": mon & ~xyz & ~link, "rank": xyz, "link": link}
        self._listings: dict[str, dict[int, list[int]]] = {}
        self._cache: dict[Filter, tuple[int | None, bool]] = {}

    # -- building blocks ---------------------------------------------------
    def type_mask(self, mask: int) -> int:
        bits = 0
        for bit, b in self._type_bits.items():
            if bit & mask:
                bits |= b
        return bits

    def _flags(self, table: dict[int, int], mask: int) -> int:
        bits = 0
        for bit, b in table.items():
            if bit & mask:
                bits |= b
        return bits

    def _where(self, scope: int, values: list[int], test) -> int:
        idx = []
        for i, v in enumerate(values):
            if (scope >> i) & 1 and test(v):
                idx.append(i)
        return _bits(idx, self.n)

    def set_listings(
        self,
        listed_names: Mapping[int, Iterable[int]] = {},
        listed_series: Mapping[int, Iterable[int]] = {},
        material_codes: Mapping[int, Iterable[int]] = {},
        material_setcodes: Mapping[int, Iterable[int]] = {},
    ) -> None:
        """Per-card script metadata used by ``Lists*`` predicates (password -> values)."""
        for kind, table in (
            ("lists_code", listed_names), ("lists_archetype", listed_series),
            ("lists_code_as_material", material_codes), ("lists_archetype_as_material", material_setcodes),
        ):  # fmt: skip
            inv: dict[int, list[int]] = {}
            for pw, values in table.items():
                i = self.pos.get(self.db.canonical(pw)) if pw in self.db else None
                if i is None:
                    continue
                for v in values:
                    inv.setdefault(v, []).append(i)
            self._listings[kind] = inv
        self._cache.clear()

    def members(self, bits: int) -> Iterator[int]:
        pw = self.passwords
        while bits:
            low = bits & -bits
            yield pw[low.bit_length() - 1]
            bits ^= low

    def bits_of(self, passwords: Iterable[int]) -> int:
        return _bits((self.pos[p] for p in passwords if p in self.pos), self.n)

    # -- predicates ----------------------------------------------------------
    def _pred(self, p: Pred) -> int | None:
        k, a = p.kind, p.args
        n = self.n
        if k == "setcard":
            idx = []
            for q in a:
                for i, sc in self._by_setbase.get(q & 0xFFF, ()):
                    if setcode_matches(sc, q):
                        idx.append(i)
            return _bits(idx, n)
        if k == "code":
            return _bits((i for code in a for i in self._code.get(code, ())), n)
        if k == "type_any":
            return self.type_mask(a[0])
        if k == "type_all":
            return self._where(self.type_mask(a[0]), self._types, lambda t: t & a[0] == a[0])
        if k == "type_eq":
            return self._where(self.universe, self._types, lambda t: t == a[0])
        if k == "race":
            return self._flags(self._race_bits, a[0])
        if k == "attribute":
            return self._flags(self._attr_bits, a[0])
        if k in ("level", "rank", "link"):
            return self._where(self._level_scope[k], self._levels, _compare(a))
        if k == "has_level":
            return self._level_scope["level"]
        if k == "attack":
            return self._where(self.monsters, self._attack, _compare(a))
        if k == "defense":
            return self._where(self.monsters & ~self._level_scope["link"], self._defense, _compare(a))
        if k in ("lists_code", "lists_archetype", "lists_code_as_material"):
            inv = self._listings.get(k, {})
            return _bits((i for v in a for i in inv.get(v, ())), n)
        if k == "lists_archetype_as_material":
            inv = self._listings.get(k, {})
            return _bits((i for q in a for listed, ix in inv.items() if _material_setcode_matches(q, listed) for i in ix), n)
        if k == "any":
            return self.universe
        if k == "none":
            return 0
        return None  # unknown, hint

    def evaluate(self, flt: Filter) -> tuple[int | None, bool]:
        """``(bits, exact)``; ``bits is None`` means unconstrained (could be any card)."""
        hit = self._cache.get(flt)
        if hit is not None:
            return hit
        if isinstance(flt, Pred):
            bits = self._pred(flt)
            res = (bits, bits is not None)
        elif isinstance(flt, And):
            bits, exact = None, True
            for item in flt.items:
                b, e = self.evaluate(item)
                exact = exact and e
                if b is not None:
                    bits = b if bits is None else bits & b
            res = (bits, exact and bits is not None)
        elif isinstance(flt, Or):
            bits, exact = 0, True
            for item in flt.items:
                b, e = self.evaluate(item)
                if b is None:
                    bits = None
                    break
                bits |= b
                exact = exact and e
            res = (bits, exact and bits is not None)
        else:
            b, e = self.evaluate(flt.item)
            res = (self.universe & ~b, True) if (b is not None and e) else (None, False)
        self._cache[flt] = res
        return res


def _compare(args: tuple):
    op, *vals = args
    if op == "eq":
        s = set(vals)
        return lambda v: v in s
    if op == "le":
        return lambda v: v <= vals[0]
    if op == "ge":
        return lambda v: v >= vals[0]
    raise ValueError(op)


# ---------------------------------------------------------------- compilation from Lua

_T = C.TYPE_MONSTER, C.TYPE_SPELL, C.TYPE_TRAP
_TYPE_ANY = {
    "IsMonster": C.TYPE_MONSTER, "IsSpell": C.TYPE_SPELL, "IsTrap": C.TYPE_TRAP,
    "IsSpellTrap": C.TYPE_SPELL | C.TYPE_TRAP, "IsMonsterCard": C.TYPE_MONSTER, "IsSpellCard": C.TYPE_SPELL,
    "IsTrapCard": C.TYPE_TRAP, "IsSpellTrapCard": C.TYPE_SPELL | C.TYPE_TRAP, "IsEquipCard": C.TYPE_EQUIP,
}  # fmt: skip
_TYPE_ALL = {
    "IsQuickPlaySpell": C.TYPE_SPELL | C.TYPE_QUICKPLAY, "IsContinuousSpell": C.TYPE_SPELL | C.TYPE_CONTINUOUS,
    "IsEquipSpell": C.TYPE_SPELL | C.TYPE_EQUIP, "IsFieldSpell": C.TYPE_SPELL | C.TYPE_FIELD,
    "IsRitualSpell": C.TYPE_SPELL | C.TYPE_RITUAL, "IsLinkSpell": C.TYPE_SPELL | C.TYPE_LINK,
    "IsContinuousTrap": C.TYPE_TRAP | C.TYPE_CONTINUOUS, "IsCounterTrap": C.TYPE_TRAP | C.TYPE_COUNTER,
    "IsFusionMonster": C.TYPE_MONSTER | C.TYPE_FUSION, "IsRitualMonster": C.TYPE_MONSTER | C.TYPE_RITUAL,
    "IsSynchroMonster": C.TYPE_MONSTER | C.TYPE_SYNCHRO, "IsXyzMonster": C.TYPE_MONSTER | C.TYPE_XYZ,
    "IsPendulumMonster": C.TYPE_MONSTER | C.TYPE_PENDULUM, "IsLinkMonster": C.TYPE_MONSTER | C.TYPE_LINK,
    "IsEffectMonster": C.TYPE_MONSTER | C.TYPE_EFFECT,
}  # fmt: skip
_HINTS = {
    "IsAbleToHand": "to_hand", "IsAbleToHandAsCost": "to_hand",
    "IsCanBeSpecialSummoned": "special_summon",
    "IsAbleToGrave": "to_grave", "IsAbleToGraveAsCost": "to_grave",
    "IsSSetable": "set", "IsAbleToRemove": "banish", "IsAbleToRemoveAsCost": "banish",
    "IsAbleToDeck": "to_deck", "IsAbleToExtra": "to_deck", "IsAbleToDeckAsCost": "to_deck",
    "IsAbleToExtraAsCost": "to_deck", "IsAbleToDeckOrExtraAsCost": "to_deck",
    "IsDiscardable": "discard", "IsReleasable": "release", "IsReleasableByEffect": "release",
}  # fmt: skip
_SETCARD = {"IsSetCard", "IsOriginalSetCard", "IsLinkSetCard", "IsFusionSetCard", "IsSynchroSetCard", "IsXyzSetCard"}
_CODE = {"IsCode", "IsOriginalCode", "IsOriginalCodeRule", "IsCodeRule", "IsFusionCode", "IsLinkCode"}
_TYPE_METHODS = {"IsType", "IsOriginalType", "IsLinkType", "IsFusionType", "IsXyzType", "IsSynchroType"}
_RACE = {"IsRace", "IsOriginalRace", "IsLinkRace", "IsFusionRace", "IsXyzRace", "IsSynchroRace"}
_ATTRIBUTE = {"IsAttribute", "IsOriginalAttribute", "IsLinkAttribute", "IsFusionAttribute", "IsXyzAttribute", "IsSynchroAttribute"}
_STAT = {  # method -> (field, op); op None = equality with varargs
    "IsLevel": ("level", None), "IsOriginalLevel": ("level", None),
    "IsLevelBelow": ("level", "le"), "IsLevelAbove": ("level", "ge"),
    "IsRank": ("rank", None), "IsOriginalRank": ("rank", None),
    "IsRankBelow": ("rank", "le"), "IsRankAbove": ("rank", "ge"),
    "IsLink": ("link", None), "IsLinkBelow": ("link", "le"), "IsLinkAbove": ("link", "ge"),
    "IsAttack": ("attack", None), "IsBaseAttack": ("attack", None), "IsTextAttack": ("attack", None),
    "IsAttackBelow": ("attack", "le"), "IsAttackAbove": ("attack", "ge"),
    "IsDefense": ("defense", None), "IsBaseDefense": ("defense", None), "IsTextDefense": ("defense", None),
    "IsDefenseBelow": ("defense", "le"), "IsDefenseAbove": ("defense", "ge"),
}  # fmt: skip
_GETTERS = {
    "GetLevel": "level", "GetOriginalLevel": "level", "GetRank": "rank", "GetOriginalRank": "rank",
    "GetLink": "link", "GetAttack": "attack", "GetBaseAttack": "attack", "GetTextAttack": "attack",
    "GetDefense": "defense", "GetBaseDefense": "defense", "GetTextDefense": "defense",
}  # fmt: skip
_LISTS = {
    "ListsCode": "lists_code", "ListsArchetype": "lists_archetype",
    "ListsCodeAsMaterial": "lists_code_as_material", "ListsArchetypeAsMaterial": "lists_archetype_as_material",
}  # fmt: skip
_FLIP = {"<": ">", ">": "<", "<=": ">=", ">=": "<=", "==": "==", "~=": "~="}
_MAX_DEPTH = 6


class FilterCompiler:
    """Compiles filter expressions of one script into :data:`Filter` trees.

    ``consts`` maps Lua constant names (``SET_*``, ``TYPE_*``, ``CARD_*``, ...) to
    values; ``card_id`` is the script's own password (the ``id`` of ``GetID()``).
    """

    def __init__(self, script: lua.Script, consts: Mapping[str, int], card_id: int | None) -> None:
        self.script = script
        self.env = dict(consts)
        if card_id is not None:
            self.env["id"] = card_id
        self.scope: Mapping[str, int | None] = self.env  # names visible while compiling
        self._fn_cache: dict[tuple, Filter] = {}
        self._stack: list[str] = []

    # -- helpers -----------------------------------------------------------------
    def ints(self, nodes: Iterable) -> tuple[int, ...] | None:
        out = []
        for node in nodes:
            if isinstance(node, Table):
                vals = self.ints(node.positional())
                if vals is None:
                    return None
                out.extend(vals)
                continue
            v = lua.const_eval(node, self.scope)
            if v is None:
                return None
            out.append(v)
        return tuple(out)

    def method_pred(self, name: str, args: tuple) -> Filter:
        if name in _SETCARD:
            vals = self.ints(args[:1])
            return Pred("setcard", vals) if vals else UNKNOWN
        if name in _CODE:
            vals = self.ints(args)
            return Pred("code", vals) if vals else UNKNOWN
        if name == "IsSummonCode":  # (sc, sumtype, tp, codes...)
            vals = self.ints(args[3:])
            return Pred("code", vals) if vals else UNKNOWN
        if name in _TYPE_METHODS or name in ("IsExactType", "IsCompositeType"):
            vals = self.ints(args[:1])
            if not vals:
                return UNKNOWN
            return Pred("type_any" if name in _TYPE_METHODS else "type_all", vals)
        if name in _RACE or name in _ATTRIBUTE:
            vals = self.ints(args[:1])
            return Pred("race" if name in _RACE else "attribute", vals) if vals else UNKNOWN
        if name in _STAT:
            field, op = _STAT[name]
            vals = self.ints(args if op is None else args[:1])
            if not vals:
                return UNKNOWN
            return Pred(field, ("eq", *vals) if op is None else (op, vals[0]))
        if name == "IsLevelBetween":
            vals = self.ints(args[:2])
            if not vals or len(vals) < 2:
                return UNKNOWN
            return make_and((Pred("level", ("ge", min(vals))), Pred("level", ("le", max(vals)))))
        if name in _TYPE_ANY:
            return Pred("type_any", (_TYPE_ANY[name],))
        if name in _TYPE_ALL:
            return Pred("type_all", (_TYPE_ALL[name],))
        if name == "IsNormalSpell":
            return Pred("type_eq", (C.TYPE_SPELL,))
        if name == "IsNormalTrap":
            return Pred("type_eq", (C.TYPE_TRAP,))
        if name == "IsNormalSpellTrap":
            return make_or((Pred("type_eq", (C.TYPE_SPELL,)), Pred("type_eq", (C.TYPE_TRAP,))))
        if name == "IsContinuousSpellTrap":
            return make_or((Pred("type_all", (_TYPE_ALL["IsContinuousSpell"],)), Pred("type_all", (_TYPE_ALL["IsContinuousTrap"],))))
        if name == "IsNonEffectMonster":
            return make_and((Pred("type_any", (C.TYPE_MONSTER,)), Not(Pred("type_any", (C.TYPE_EFFECT,)))))
        if name in ("HasLevel",):
            return Pred("has_level")
        if name in _LISTS:
            vals = self.ints(args)
            return Pred(_LISTS[name], vals) if vals else UNKNOWN
        if name in _HINTS:
            return Pred("hint", (_HINTS[name],))
        return UNKNOWN

    # -- filter values ----------------------------------------------------------
    def compile_value(self, node, extra: tuple = ()) -> Filter:
        """Compile an expression used *as a filter function* (``s.filter``, ``aux.FilterBoolFunction(...)``).

        ``extra`` are the arguments the caller passes after the card (the trailing
        arguments of ``Duel.IsExistingMatchingCard`` and friends); constant ones are
        bound to the filter function's parameters.
        """
        if isinstance(node, Const):
            return ANY if node.value is None or node.value is True else NONE
        if isinstance(node, FuncExpr):
            if not node.params:
                return UNKNOWN
            return self._with_params(node.params, extra, lambda: self._returns(lua.returns(self.script.tokens, node.body), node.params[0]))
        name = lua.dotted(node) if isinstance(node, (Name, Index)) else None
        if name is not None:
            if name in self.script.functions:
                return self.function(name, extra)
            if name.startswith("Card."):
                return self.method_pred(name[5:], ())
            return UNKNOWN
        if isinstance(node, Call):
            fname = lua.dotted(node.func)
            args = node.args
            if fname in ("aux.FilterBoolFunction", "aux.FilterBoolFunctionEx", "aux.FilterBoolFunctionEx2") and args:
                inner = lua.dotted(args[0])
                if inner and inner.startswith("Card."):
                    return self.method_pred(inner[5:], args[1:])
                return self.compile_value(args[0])
            if fname in ("aux.FaceupFilter", "Synchro.NonTuner", "Synchro.NonTunerEx") and args:
                # (f, args...): f(c, args...)
                inner = lua.dotted(args[0])
                if inner and inner.startswith("Card."):
                    return self.method_pred(inner[5:], args[1:])
                return self.compile_value(args[0], args[1:])
            if fname == "aux.NecroValleyFilter" and args:
                return self.compile_value(args[0], extra)
            if fname == "aux.NOT" and args:
                return make_not(self.compile_value(args[0]))
            if fname == "aux.AND":
                return make_and(self.compile_value(a) for a in args)
            if fname == "aux.OR":
                return make_or(self.compile_value(a) for a in args)
            if fname == "Fusion.IsMonsterFilter":
                return make_and((Pred("type_any", (C.TYPE_MONSTER,)), *(self.compile_value(a) for a in args[:1])))
        return UNKNOWN

    def _bind(self, params: tuple, extra: tuple) -> dict[str, int | None]:
        """Parameter scope of a call: constant arguments bound, the other parameters masked."""
        local: dict[str, int | None] = dict.fromkeys(params[1:])
        for param, arg in zip(params[1:], extra):
            if param != "...":
                local[param] = lua.const_eval(arg, self.scope)
        return local

    def _with_params(self, params: tuple, extra: tuple, body):
        saved = self.scope
        self.scope = ChainMap(self._bind(params, extra), self.env)
        try:
            return body()
        finally:
            self.scope = saved

    def function(self, name: str, extra: tuple = ()) -> Filter:
        fn = self.script.functions.get(name)
        if fn is None or not fn.params or name in self._stack or len(self._stack) >= _MAX_DEPTH:
            return UNKNOWN
        local = self._bind(fn.params, extra)
        key = (name, tuple(sorted((k, v) for k, v in local.items() if v is not None)))
        hit = self._fn_cache.get(key)
        if hit is not None:
            return hit
        self._stack.append(name)
        saved = self.scope
        self.scope = ChainMap(local, self.env)
        try:
            flt = self._returns(self.script.returns(fn), fn.params[0])
        finally:
            self._stack.pop()
            self.scope = saved
        if not self._stack:
            self._fn_cache[key] = flt
        return flt

    def _returns(self, exprs: list, var: str) -> Filter:
        if not exprs:
            return UNKNOWN
        return make_or(self.compile_bool(e, var) for e in exprs)

    # -- boolean expressions about the card variable --------------------------------
    def compile_bool(self, node, var: str) -> Filter:
        if isinstance(node, BinOp):
            if node.op == "and":
                return make_and((self.compile_bool(node.left, var), self.compile_bool(node.right, var)))
            if node.op == "or":
                return make_or((self.compile_bool(node.left, var), self.compile_bool(node.right, var)))
            if node.op in _FLIP:
                return self._comparison(node, var)
            return UNKNOWN
        if isinstance(node, UnOp):
            return make_not(self.compile_bool(node.operand, var)) if node.op == "not" else UNKNOWN
        if isinstance(node, Const):
            if node.value is True:
                return ANY
            if node.value is False or node.value is None:
                return NONE
            return UNKNOWN
        if isinstance(node, Method):
            if node.obj == Name(var):
                return self.method_pred(node.name, node.args)
            return UNKNOWN
        if isinstance(node, Call):
            fname = lua.dotted(node.func)
            if not fname or not node.args or node.args[0] != Name(var):
                return UNKNOWN
            if fname.startswith("Card."):
                return self.method_pred(fname[5:], node.args[1:])
            if fname in self.script.functions:
                return self.function(fname, node.args[1:])
        return UNKNOWN

    def _comparison(self, node: BinOp, var: str) -> Filter:
        op, left, right = node.op, node.left, node.right
        if not _is_getter(left, var):
            left, right, op = right, left, _FLIP[op]
        if not _is_getter(left, var):
            return UNKNOWN
        value = lua.const_eval(right, self.scope)
        if value is None:
            return UNKNOWN
        getter = left.name
        if getter in ("GetType", "GetOriginalType"):
            return Pred("type_eq", (value,)) if op == "==" else UNKNOWN
        if getter in ("GetRace", "GetOriginalRace", "GetAttribute", "GetOriginalAttribute"):
            kind = "race" if "Race" in getter else "attribute"
            return Pred(kind, (value,)) if op == "==" else UNKNOWN
        field = _GETTERS[getter]
        if op == "==":
            return Pred(field, ("eq", value))
        if op == "~=":
            return make_not(Pred(field, ("eq", value)))
        if op == "<=":
            return Pred(field, ("le", value))
        if op == "<":
            return Pred(field, ("le", value - 1))
        if op == ">=":
            return Pred(field, ("ge", value))
        return Pred(field, ("ge", value + 1))


def _is_getter(node, var: str) -> bool:
    return (
        isinstance(node, Method) and node.obj == Name(var) and not node.args
        and (node.name in _GETTERS or node.name in ("GetType", "GetOriginalType", "GetRace", "GetOriginalRace",
                                                     "GetAttribute", "GetOriginalAttribute"))
    )  # fmt: skip
