"""Static analysis of one card script: effects, categories, target queries, procedures.

A *query* is "this card's effect does <action> with a card matching <filter>
taken from <locations> on its controller's side", e.g. Snake-Eye Ash:
``to_hand`` / ``LOCATION_DECK`` / ``level 1 and FIRE``. Queries come from:

* matching-card calls: ``Duel.IsExistingMatchingCard``, ``SelectMatchingCard``,
  ``GetMatchingGroup(Count)``, ``GetFirstMatchingCard``, ``IsExistingTarget``,
  ``SelectTarget`` (filter argument + own-side location argument);
* summon procedures: ``Fusion.CreateSummonEff`` / ``SummonEffTG`` (Fusion
  monsters from the Extra Deck), ``Ritual.AddProc*`` (Ritual monsters, hand);
* material procedures: ``Link/Xyz/Synchro.AddProcedure``, ``Fusion.AddProcMix*``.

The action of a matching call is inferred, in order of confidence
(``Query.evidence``): ``filter`` — an ``IsAbleToHand`` / ``IsCanBeSpecialSummoned`` /
``IsAbleToGrave`` / ``IsSSetable`` ... check inside the filter; ``hintmsg`` — the
``Duel.Hint(HINT_SELECTMSG, tp, HINTMSG_*)`` right before a ``Select*`` call;
``name`` — the filter's function name (``thfilter``, ``spfilter``, ``tgfilter``)
when the effect has the matching category; ``category`` — the effect's
``SetCategory`` names a single relevant action.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

from ygorl import paths
from ygorl.build import lua
from ygorl.build.filters import ANY, Filter, FilterCompiler, Pred, hints, is_constrained, make_and
from ygorl.build.lua import BinOp, Const, FuncExpr, Index, Name, Table

CONSTANT_FILES = ("constant.lua", "archetype_setcode_constants.lua", "card_counter_constants.lua")


@cache
def _load_constants(root: str) -> dict[str, int]:
    env: dict[str, int] = {}
    for name in CONSTANT_FILES:
        path = Path(root) / name
        env.update(lua.read_constants(path.read_text(encoding="utf-8", errors="replace"), env))
    return env


def load_constants(root: str | Path | None = None) -> dict[str, int]:
    """Integer constants of the CardScripts library (``SET_*``, ``CARD_*``, ``TYPE_*``, ...)."""
    return dict(_load_constants(str(root or paths.card_scripts())))


# ---------------------------------------------------------------- data


@dataclass(frozen=True, slots=True)
class Query:
    action: str  # filters.ACTIONS or "material"
    locations: int  # own-side LOCATION_* mask the cards are taken from (0 for materials)
    filter: Filter
    origin: str  # "call" | "fusion_summon" | "ritual_summon" | "material"
    categories: int  # CATEGORY_* of the effect(s) the query belongs to
    evidence: str  # "filter" | "hintmsg" | "name" | "category" | "procedure"


@dataclass(frozen=True, slots=True)
class EffectFact:
    """One effect of a script: what it does and when / from where it is activated.

    ``type`` / ``code`` / ``range`` are the constant values of its ``SetType`` / ``SetCode`` / ``SetRange``
    (0 when absent or not a constant); ``opponent`` is True when a matching or field-group call reached from its
    target / operation / cost can take cards from the opponent's side (``Duel.SelectTarget(tp, f, tp, 0,
    LOCATION_ONFIELD, ...)``, ``Duel.GetFieldGroup(tp, 0, LOCATION_MZONE)``).

    ``count_code`` is the code of a hard once-per-turn limit (``SetCountLimit(1, id)``: one use per card *name*),
    0 without one. ``acts`` is False when the operation never touches a card or a chain -- no negation, destroy,
    banish, send, return, control or position change, no ``EFFECT_DISABLE`` -- e.g. a player lock (Droll & Lock
    Bird) or a draw engine; True when there is no operation to read. ``requires`` lists filters of monsters the
    controller must control for the effect to be usable: own-monster-zone matching calls in its condition or target
    (``Duel.IsExistingTarget(f, tp, LOCATION_MZONE, 0, ...)``), and the ``IsExists`` filter of a condition about
    the chain's targets (``CHAININFO_TARGET_CARDS``: "targets a ... you control").
    """

    categories: int = 0
    type: int = 0
    code: int = 0
    range: int = 0
    opponent: bool = False
    count_code: int = 0
    acts: bool = True
    requires: tuple = ()


EVIDENCE_RANK = {"procedure": 0, "filter": 1, "hintmsg": 2, "name": 3, "category": 4}


@dataclass(frozen=True)
class ScriptFacts:
    password: int
    categories: int = 0
    effects: int = 0
    queries: tuple[Query, ...] = ()
    listed_names: tuple[int, ...] = ()
    listed_series: tuple[int, ...] = ()
    material_codes: tuple[int, ...] = ()
    material_setcodes: tuple[int, ...] = ()
    stats: dict[str, int] = field(default_factory=dict)
    effect_facts: tuple[EffectFact, ...] = ()

    def category_names(self) -> list[str]:
        consts = _category_table()
        return [consts[bit] for bit in sorted(consts) if self.categories & bit]


@cache
def _category_table() -> dict[int, str]:
    out = {}
    for name, value in _load_constants(str(paths.card_scripts())).items():
        if name.startswith("CATEGORY_") and value and value & (value - 1) == 0:
            out[value] = name[len("CATEGORY_") :].lower()
    return out


# ---------------------------------------------------------------- tables

# call -> argument positions of (filter, player, own locations, opponent locations,
# first extra argument passed on to the filter)
MATCH_CALLS = {
    "Duel.IsExistingMatchingCard": (0, 1, 2, 3, 6),
    "Duel.SelectMatchingCard": (1, 2, 3, 4, 8),
    "Duel.GetMatchingGroup": (0, 1, 2, 3, 5),
    "Duel.GetMatchingGroupCount": (0, 1, 2, 3, 5),
    "Duel.GetFirstMatchingCard": (0, 1, 2, 3, 5),
    "Duel.IsExistingTarget": (0, 1, 2, 3, 6),
    "Duel.SelectTarget": (1, 2, 3, 4, 8),
}
SELECT_CALLS = frozenset(("Duel.SelectMatchingCard", "Duel.SelectTarget"))

HINTMSG_ACTIONS = {
    "HINTMSG_ATOHAND": "to_hand", "HINTMSG_SPSUMMON": "special_summon", "HINTMSG_TOGRAVE": "to_grave",
    "HINTMSG_SET": "set", "HINTMSG_REMOVE": "banish", "HINTMSG_TODECK": "to_deck", "HINTMSG_DISCARD": "discard",
    "HINTMSG_RELEASE": "release", "HINTMSG_TOFIELD": "place", "HINTMSG_EQUIP": "equip",
}  # fmt: skip
# hints that say nothing about what happens to the selected card
GENERIC_HINTMSG = frozenset(
    (
        "HINTMSG_TARGET",
        "HINTMSG_SELECT",
        "HINTMSG_FACEUP",
        "HINTMSG_FACEDOWN",
        "HINTMSG_CONFIRM",
        "HINTMSG_OPPO",
        "HINTMSG_SELF",
    )
)

# action -> CATEGORY_* names that announce it
ACTION_CATEGORIES = {
    "to_hand": ("CATEGORY_TOHAND", "CATEGORY_SEARCH"),
    "special_summon": ("CATEGORY_SPECIAL_SUMMON",),
    "to_grave": ("CATEGORY_TOGRAVE", "CATEGORY_DECKDES"),
    "set": ("CATEGORY_SET",),
    "equip": ("CATEGORY_EQUIP",),
}
_NAME_ACTIONS = [
    (re.compile(r"^(th|search|srch|add)"), "to_hand"),
    (re.compile(r"^(sp|ss)"), "special_summon"),
    (re.compile(r"^(tg|gy|tog)"), "to_grave"),
    (re.compile(r"^set"), "set"),
]

FUSION_PROCS = {
    "Fusion.CreateSummonEff": ("handler", "fusfilter", "matfilter", "extrafil", "extraop", "gc", "stage2", "exactcount", "value", "location"),
    "Fusion.RegisterSummonEff": ("handler", "fusfilter", "matfilter", "extrafil", "extraop", "gc", "stage2", "exactcount", "value", "location"),
    "Fusion.SummonEffTG": ("fusfilter", "matfilter", "extrafil", "extraop", "gc", "stage2", "exactcount", "value", "location"),
    "Fusion.SummonEffOP": ("fusfilter", "matfilter", "extrafil", "extraop", "gc", "stage2", "exactcount", "value", "location"),
}  # fmt: skip
RITUAL_PROCS = {
    "Ritual.AddProcGreater": ("handler", "filter", "lv", "desc", "extrafil", "extraop", "matfilter", "stage2", "location"),
    "Ritual.AddProcEqual": ("handler", "filter", "lv", "desc", "extrafil", "extraop", "matfilter", "stage2", "location"),
    "Ritual.CreateProc": ("handler", "lvtype", "filter", "lv", "desc", "extrafil", "extraop", "matfilter", "stage2", "location"),
    "Ritual.AddProc": ("handler", "lvtype", "filter", "lv", "desc", "extrafil", "extraop", "matfilter", "stage2", "location"),
    "Ritual.Target": ("filter", "lvtype", "lv", "extrafil", "extraop", "matfilter", "stage2", "location"),
    "Ritual.Operation": ("filter", "lvtype", "lv", "extrafil", "extraop", "matfilter", "stage2", "location"),
}  # fmt: skip
RITUAL_CODE_PROCS = {"Ritual.AddProcGreaterCode": 3, "Ritual.AddProcEqualCode": 3, "Ritual.AddProcCode": 4}
MATERIAL_PROCS = frozenset(
    (
        "Link.AddProcedure",
        "Xyz.AddProcedure",
        "Synchro.AddProcedure",
        "Fusion.AddProcMix",
        "Fusion.AddProcMixN",
        "Fusion.AddProcMixRep",
    )
)
# Effect:SetType / SetCode / SetRange -> the effect key they set (when to activate it, and from where)
TIMING_SETTERS = {"SetType": "type", "SetCode": "code", "SetRange": "range"}
# setters whose function argument is recorded per role (condition functions do not feed ``refs``)
ROLE_SETTERS = {"SetTarget": "target", "SetOperation": "operation", "SetCost": "cost", "SetCondition": "condition"}
EFFECT_SETTERS = frozenset(("SetCategory", "SetTarget", "SetOperation", "SetCost", "SetCondition", "SetCountLimit",
                            *TIMING_SETTERS))  # fmt: skip
# what an operation must do to interrupt anything: names (methods or constants) that touch cards or the chain
ACTING_NAMES = frozenset((
    "NegateActivation", "NegateEffect", "NegateSummon", "NegateRelatedChain", "ChangeChainOperation", "Destroy",
    "Remove", "SendtoHand", "SendtoDeck", "SendtoGrave", "SendtoExtraP", "GetControl", "ChangePosition", "Release",
    "Overlay", "SwapControl", "RemoveUntil", "EFFECT_DISABLE", "EFFECT_DISABLE_EFFECT", "EFFECT_CANNOT_TRIGGER",
))  # fmt: skip
_PROC_EFFECTS = frozenset(FUSION_PROCS) | frozenset(RITUAL_PROCS) | frozenset(RITUAL_CODE_PROCS)


def _new_effect() -> dict:
    return {"categories": 0, "refs": set(), "type": 0, "code": 0, "range": 0, "count_code": 0,
            "roles": {}, "inline": {}}  # fmt: skip


def _copy_effect(base: dict) -> dict:
    return {**base, "refs": set(base["refs"]), "roles": {k: set(v) for k, v in base["roles"].items()},
            "inline": {k: list(v) for k, v in base["inline"].items()}}  # fmt: skip


def _named(args: tuple, names: tuple) -> dict:
    """Arguments of an ``aux.FunctionWithNamedArgs`` call: a single table or positional."""
    if len(args) == 1 and isinstance(args[0], Table) and any(isinstance(k, str) for k, _ in args[0].items):
        return {k: v for k, v in args[0].items if isinstance(k, str)}
    return dict(zip(names, args))


def _walk(node):
    """Yield every AST node below ``node`` (inclusive)."""
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        if isinstance(n, Table):
            for k, v in n.items:
                stack.append(v)
                if not isinstance(k, (str, type(None))):
                    stack.append(k)
        elif isinstance(n, tuple):
            stack.extend(n)
        elif hasattr(n, "__slots__") and not isinstance(n, (Name, Const)):
            for slot in n.__slots__:
                v = getattr(n, slot)
                if isinstance(v, (tuple, Table)) or hasattr(v, "__slots__"):
                    stack.append(v)


# ---------------------------------------------------------------- analysis


class _Analyzer:
    def __init__(self, src: str, password: int, consts: dict[str, int]) -> None:
        self.password = password
        self.K = consts
        self.script = lua.Script.parse(src)
        self.comp = FilterCompiler(self.script, consts, password)
        self.env = {**consts, "id": password}
        self.stats = {"match_calls": 0, "queries": 0, "opponent_side": 0, "unconstrained": 0,
                      "unknown_action": 0, "unknown_location": 0, "parse_error": 0}  # fmt: skip
        self.queries: dict[tuple, Query] = {}
        self.categories = 0

    def cat(self, *names: str) -> int:
        out = 0
        for n in names:
            out |= self.K.get(n, 0)
        return out

    # -- functions & effects -----------------------------------------------------
    def function_refs(self, start: int, end: int) -> set[str]:
        toks = self.script.tokens
        funcs = self.script.functions
        out = set()
        for j in range(start, min(end, len(toks))):
            kind, value = toks[j]
            if kind != "name" or (j and toks[j - 1][1] in (".", ":")):
                continue
            parts = [value]
            k = j
            while k + 2 < len(toks) and toks[k + 1] == ("op", ".") and toks[k + 2][0] == "name":
                parts.append(toks[k + 2][1])
                k += 2
                name = ".".join(parts)
                if name in funcs:
                    out.add(name)
            if value in funcs:
                out.add(value)
        return out

    def expr_refs(self, node) -> set[str]:
        out = set()
        for n in _walk(node):
            if isinstance(n, (Name, Index)):
                d = lua.dotted(n)
                if d in self.script.functions:
                    out.add(d)
            elif isinstance(n, FuncExpr):
                out |= self.function_refs(*n.body)
        return out

    def effects(self) -> None:
        """Effects (``Effect.CreateEffect`` / ``Clone`` / procedure effects) and what they reference."""
        toks = self.script.tokens
        n = len(toks)
        effects: list[dict] = []
        bound: dict[str, int] = {}
        for j in range(n - 2):
            kind, value = toks[j]
            if kind != "name":
                continue
            nxt = toks[j + 1]
            if nxt == ("op", "=") and toks[j + 2] != ("op", "=") and (j == 0 or toks[j - 1][1] not in (".", ":")):
                rhs = toks[j + 2 : j + 6]
                if rhs[:4] == [("name", "Effect"), ("op", "."), ("name", "CreateEffect"), ("op", "(")]:
                    bound[value] = len(effects)
                    effects.append(_new_effect())
                elif len(rhs) >= 3 and rhs[0][0] == "name" and rhs[1] == ("op", ":") and rhs[2] == ("name", "Clone"):
                    src = bound.get(rhs[0][1])
                    base = effects[src] if src is not None else _new_effect()
                    bound[value] = len(effects)
                    effects.append(_copy_effect(base))
                elif len(rhs) >= 3 and rhs[0][1] in ("Fusion", "Ritual") and rhs[1] == ("op", "."):
                    name = f"{rhs[0][1]}.{rhs[2][1]}"
                    if name in _PROC_EFFECTS:
                        try:
                            node, _ = lua.parse_expr(toks, j + 2)
                        except lua.LuaSyntaxError:
                            continue
                        cats = self.cat("CATEGORY_SPECIAL_SUMMON") | (
                            self.cat("CATEGORY_FUSION_SUMMON") if rhs[0][1] == "Fusion" else 0
                        )
                        bound[value] = len(effects)
                        effects.append({**_new_effect(), "categories": cats, "refs": self.expr_refs(node)})
            elif nxt == ("op", ":") and toks[j + 2][0] == "name" and toks[j + 2][1] in EFFECT_SETTERS:
                idx = bound.get(value)
                if idx is None:
                    continue
                try:
                    args, _ = lua.parse_args(toks, j + 3)
                except lua.LuaSyntaxError:
                    continue
                if not args:
                    continue
                setter = toks[j + 2][1]
                if setter == "SetCategory":
                    v = lua.const_eval(args[0], self.env)
                    if v is not None:
                        effects[idx]["categories"] |= v
                elif setter == "SetCountLimit":  # (count[, code[, flags]]); code may be {id, k}
                    code = args[1] if len(args) > 1 else None
                    if isinstance(code, Table):
                        pos = code.positional()
                        code = pos[0] if pos else None
                    v = lua.const_eval(code, self.env) if code is not None else None
                    effects[idx]["count_code"] = v or 0
                elif setter == "SetCondition":
                    self._role(effects[idx], "condition", args[0])
                elif setter in TIMING_SETTERS:  # the last call wins, as in Lua (a clone may override)
                    v = lua.const_eval(args[0], self.env)
                    if v is not None:
                        effects[idx][TIMING_SETTERS[setter]] = v
                else:
                    effects[idx]["refs"] |= self.expr_refs(args[0])
                    self._role(effects[idx], ROLE_SETTERS[setter], args[0])
        # function -> effects reaching it through the call graph
        graph = {name: self.function_refs(*fn.body) for name, fn in self.script.functions.items()}
        self.graph = graph
        self.fn_effects: dict[str, set[int]] = {}
        for i, eff in enumerate(effects):
            stack = list(eff["refs"])
            seen = set()
            while stack:
                f = stack.pop()
                if f in seen:
                    continue
                seen.add(f)
                self.fn_effects.setdefault(f, set()).add(i)
                stack.extend(graph.get(f, ()))
        self.effect_list = effects
        self.opponent_effects: set[int] = set()
        for eff in effects:
            self.categories |= eff["categories"]
        # SetOperationInfo(0, CATEGORY_X, g, count, player, loc): locations per effect & category
        self.opinfo: list[dict[int, int]] = [{} for _ in effects]
        for pos, _name, args in self.script.find_calls({"Duel.SetOperationInfo", "Duel.SetPossibleOperationInfo"}):
            if len(args) < 6:
                continue
            cat = lua.const_eval(args[1], self.env)
            loc = lua.const_eval(args[5], self.env)
            if not cat or not loc:
                continue
            for i in self.fn_effects.get(self.script.function_at(pos), ()):
                self.opinfo[i][cat] = self.opinfo[i].get(cat, 0) | loc

    def _role(self, eff: dict, role: str, node) -> None:
        """Record the function(s) of a ``SetTarget`` / ``SetOperation`` / ``SetCost`` / ``SetCondition`` call."""
        eff["roles"][role] = self.expr_refs(node)
        eff["inline"][role] = [n.body for n in _walk(node) if isinstance(n, FuncExpr)]

    def _spans(self, eff: dict, *roles: str) -> list[tuple[int, int]]:
        """Token ranges of the functions an effect's ``roles`` reach (named functions closed over calls)."""
        stack = [f for r in roles for f in eff["roles"].get(r, ())]
        spans = [b for r in roles for b in eff["inline"].get(r, ())]
        for b in list(spans):
            stack.extend(self.function_refs(*b))
        seen: set[str] = set()
        while stack:
            f = stack.pop()
            if f in seen or f not in self.script.functions:
                continue
            seen.add(f)
            spans.append(self.script.functions[f].body)
            stack.extend(self.graph.get(f, ()))
        return spans

    def _acts(self, eff: dict) -> bool:
        if not eff["roles"].get("operation") and not eff["inline"].get("operation"):
            return True
        toks = self.script.tokens
        for a, b in self._spans(eff, "operation"):
            for j in range(a, min(b, len(toks))):
                if toks[j][0] == "name" and (toks[j][1] in ACTING_NAMES or toks[j][1].startswith("Negate")):
                    return True
        return False

    def _requires(self, eff: dict, calls: list) -> tuple:
        """Filters of own monsters the effect's condition / target need (see :class:`EffectFact`)."""
        spans = self._spans(eff, "condition", "target")
        if not spans:
            return ()
        inside = lambda pos: any(a <= pos < b for a, b in spans)  # noqa: E731
        toks = self.script.tokens
        mzone = self.K["LOCATION_MZONE"]
        out: list = []
        for pos, name, args in calls:
            fi, pi, si, oi, xi = MATCH_CALLS[name]
            if len(args) <= oi or not inside(pos) or (pos and toks[pos - 1] == ("name", "not")):
                continue
            own, opp = args[si], args[oi]
            if _is_opponent(args[pi]):
                own, opp = opp, own
            if lua.const_eval(own, self.env) != mzone or lua.const_eval(opp, self.env) != 0:
                continue
            flt = self.comp.compile_value(args[fi], args[xi:])
            out.append(flt if is_constrained(flt) else ANY)
        for a, b in self._spans(eff, "condition"):
            body = toks[a : min(b, len(toks))]
            if ("name", "CHAININFO_TARGET_CARDS") not in body:
                continue
            for j in range(a, min(b, len(toks)) - 1):
                if toks[j] == ("name", "IsExists") and toks[j - 1] == ("op", ":"):
                    try:
                        fargs, _ = lua.parse_args(toks, j + 1)
                    except lua.LuaSyntaxError:
                        continue
                    if fargs:
                        flt = self.comp.compile_value(fargs[0], fargs[3:])
                        if is_constrained(flt):
                            out.append(flt)
        return tuple(dict.fromkeys(out))

    def field_groups(self) -> None:
        """Effects reaching ``Duel.GetFieldGroup(Count)(player, own, opponent)`` with an opponent location."""
        for pos, _name, args in self.script.find_calls({"Duel.GetFieldGroup", "Duel.GetFieldGroupCount"}):
            if len(args) < 3:
                continue
            opp = args[1] if _is_opponent(args[0]) else args[2]
            if lua.const_eval(opp, self.env):
                self.opponent_effects |= self.fn_effects.get(self.script.function_at(pos), set())

    def effect_facts(self) -> tuple[EffectFact, ...]:
        calls = self.script.find_calls(set(MATCH_CALLS))
        return tuple(
            EffectFact(
                e["categories"],
                e["type"],
                e["code"],
                e["range"],
                i in self.opponent_effects,
                e["count_code"],
                self._acts(e),
                self._requires(e, calls),
            )  # fmt: skip
            for i, e in enumerate(self.effect_list)
        )

    def call_categories(self, pos: int) -> tuple[int, set[int]]:
        effs = self.fn_effects.get(self.script.function_at(pos), set())
        if not effs:
            return self.categories, effs
        cats = 0
        for i in effs:
            cats |= self.effect_list[i]["categories"]
        return cats, effs

    # -- queries --------------------------------------------------------------------
    def add(self, q: Query) -> None:
        key = (q.action, q.locations, q.filter, q.origin)
        old = self.queries.get(key)
        if old is None:
            self.queries[key] = q
            return
        evidence = min(old.evidence, q.evidence, key=EVIDENCE_RANK.__getitem__)
        self.queries[key] = Query(q.action, q.locations, q.filter, q.origin, old.categories | q.categories, evidence)

    def infer_actions(self, flt: Filter, filter_node, cats: int, hint: str | None) -> tuple[set[str], str]:
        found = hints(flt)
        if found:
            return found, "filter"
        if hint is not None:
            if hint in HINTMSG_ACTIONS:
                return {HINTMSG_ACTIONS[hint]}, "hintmsg"
            if hint not in GENERIC_HINTMSG:
                return set(), "hintmsg"  # e.g. HINTMSG_DESTROY: not a card-moving action we track
        relevant = {a for a, names in ACTION_CATEGORIES.items() if cats & self.cat(*names)}
        fname = lua.dotted(filter_node) if isinstance(filter_node, (Name, Index)) else None
        if fname:
            short = fname.rsplit(".", 1)[-1]
            for rx, action in _NAME_ACTIONS:
                if rx.match(short) and action in relevant:
                    return {action}, "name"
        if len(relevant) == 1:
            return relevant, "category"
        return set(), ""

    def match_calls(self) -> None:
        script = self.script
        hint_calls = []
        for pos, _n, args in script.find_calls({"Duel.Hint"}):
            if len(args) >= 3 and args[0] == Name("HINT_SELECTMSG") and isinstance(args[2], Name):
                hint_calls.append((pos, script.function_at(pos), args[2].id))
        for pos, name, args in script.find_calls(set(MATCH_CALLS)):
            fi, pi, si, oi, xi = MATCH_CALLS[name]
            if len(args) <= oi:
                continue
            self.stats["match_calls"] += 1
            own, opp = args[si], args[oi]
            if _is_opponent(args[pi]):
                own, opp = opp, own
            if lua.const_eval(opp, self.env):
                self.opponent_effects |= self.fn_effects.get(script.function_at(pos), set())
            loc = lua.const_eval(own, self.env)
            if loc == 0:
                self.stats["opponent_side"] += 1
                continue
            flt = self.comp.compile_value(args[fi], args[xi:])
            cats, effs = self.call_categories(pos)
            hint = None
            if name in SELECT_CALLS:
                fn = script.function_at(pos)
                prev = [h for h in hint_calls if h[1] == fn and h[0] < pos]
                if prev:
                    hint = prev[-1][2]
            actions, evidence = self.infer_actions(flt, args[fi], cats, hint)
            if not actions:
                self.stats["unknown_action"] += 1
                continue
            for action in actions:
                where = loc
                if where is None:
                    where = 0
                    names = ACTION_CATEGORIES.get(action, ())
                    mask = self.cat(*names)
                    for i in effs:
                        for cat, info_loc in self.opinfo[i].items():
                            if cat & mask:
                                where |= info_loc
                    if not where:
                        self.stats["unknown_location"] += 1
                        continue
                self.add(Query(action, where, flt, "call", cats, evidence))

    def procedures(self) -> None:
        K = self.K
        comp = self.comp
        mon = K["TYPE_MONSTER"]
        calls = self.script.find_calls(set(FUSION_PROCS) | set(RITUAL_PROCS) | set(RITUAL_CODE_PROCS) | MATERIAL_PROCS)
        material_codes: list[int] = []
        for pos, name, args in calls:
            if name in FUSION_PROCS:
                a = _named(args, FUSION_PROCS[name])
                flt = make_and(
                    (Pred("type_all", (mon | K["TYPE_FUSION"],)), comp.compile_value(a.get("fusfilter", Const(None))))
                )
                loc = lua.const_eval(a["location"], self.env) if "location" in a else None
                cats = self.cat("CATEGORY_SPECIAL_SUMMON", "CATEGORY_FUSION_SUMMON")
                self.categories |= cats
                self.add(Query("special_summon", loc or K["LOCATION_EXTRA"], flt, "fusion_summon", cats, "procedure"))
            elif name in RITUAL_PROCS or name in RITUAL_CODE_PROCS:
                if name in RITUAL_CODE_PROCS:
                    codes = [lua.const_eval(x, self.env) for x in args[RITUAL_CODE_PROCS[name] :]]
                    target = (
                        Pred("code", tuple(c for c in codes if c is not None)) if codes and None not in codes else ANY
                    )
                    loc = None
                else:
                    a = _named(args, RITUAL_PROCS[name])
                    target = comp.compile_value(a.get("filter", Const(None)))
                    loc = lua.const_eval(a["location"], self.env) if "location" in a else None
                flt = make_and((Pred("type_all", (mon | K["TYPE_RITUAL"],)), target))
                cats = self.cat("CATEGORY_SPECIAL_SUMMON")
                self.categories |= cats
                self.add(Query("special_summon", loc or K["LOCATION_HAND"], flt, "ritual_summon", cats, "procedure"))
            else:
                for flt in self.material_filters(name, args, material_codes):
                    if flt != ANY:
                        self.add(Query("material", 0, flt, "material", 0, "procedure"))
        self.material_codes = material_codes

    def material_filters(self, name: str, args: tuple, codes: list[int]) -> list[Filter]:
        comp, K = self.comp, self.K
        get = lambda i: args[i] if len(args) > i else Const(None)  # noqa: E731
        if name == "Link.AddProcedure":
            return [comp.compile_value(get(1))]
        if name == "Xyz.AddProcedure":
            lv = lua.const_eval(get(2), self.env)
            f = comp.compile_value(get(1))
            return [make_and((f, Pred("level", ("eq", lv)))) if lv else f]
        if name == "Synchro.AddProcedure":
            tuner = make_and((comp.compile_value(get(1)), Pred("type_any", (K["TYPE_TUNER"],))))
            return [tuner, comp.compile_value(get(4))]
        if name == "Fusion.AddProcMix":
            mats = args[3:]
        elif name == "Fusion.AddProcMixN":
            mats = args[3::2]
        else:  # Fusion.AddProcMixRep(c,sub,insf,fun1,minc,maxc,...)
            mats = args[3:4] + args[6:]
        out = []
        for m in mats:
            if isinstance(m, Table):
                vals = [lua.const_eval(v, self.env) for v in m.positional()]
                if vals and None not in vals:
                    codes.extend(vals)
                    out.append(Pred("code", tuple(vals)))
                continue
            v = lua.const_eval(m, self.env)
            if v is not None:
                codes.append(v)
                out.append(Pred("code", (v,)))
            else:
                out.append(comp.compile_value(m))
        return out

    def listings(self) -> dict[str, tuple[int, ...]]:
        out: dict[str, tuple[int, ...]] = {}
        for name, node in self.script.top_level_assignments().items():
            key = name.rsplit(".", 1)[-1]
            if key not in ("listed_names", "listed_series", "material", "material_setcode"):
                continue
            items = node.positional() if isinstance(node, Table) else [node]
            vals = [lua.const_eval(v, self.env) for v in items]
            out[key] = tuple(v for v in vals if v is not None)
        return out


def _is_opponent(player) -> bool:
    return isinstance(player, BinOp) and player.op == "-" and player.left == Const(1)


def analyze_script(src: str, password: int, consts: dict[str, int]) -> ScriptFacts:
    """Extract :class:`ScriptFacts` from one script's source."""
    try:
        an = _Analyzer(src, password, consts)
        an.effects()
    except lua.LuaSyntaxError:
        return ScriptFacts(password, stats={"parse_error": 1})
    an.match_calls()
    an.field_groups()
    an.procedures()
    listed = an.listings()
    queries = tuple(an.queries.values())
    an.stats["queries"] = len(queries)
    an.stats["unconstrained"] = sum(1 for q in queries if not is_constrained(q.filter))
    return ScriptFacts(
        password=password,
        categories=an.categories,
        effects=len(an.effect_list),
        queries=queries,
        listed_names=listed.get("listed_names", ()),
        listed_series=listed.get("listed_series", ()),
        material_codes=tuple(dict.fromkeys(an.material_codes + list(listed.get("material", ())))),
        material_setcodes=listed.get("material_setcode", ()),
        stats=an.stats,
        effect_facts=an.effect_facts(),
    )
