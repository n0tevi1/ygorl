"""Scoped search/recovery predicate probe; never an engine-legality replacement.

Only unmodified constructed fixtures with one target and Imperial Iron Wall are
supported. Unknown filter syntax fails closed. Native IsAbleToHand, count limits,
timing and unrelated effects remain explicit assumptions of that fixture scope.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

from ygorl.build import lua
from ygorl.build.scripts import load_constants


class UnsupportedCondition(ValueError):
    pass


@dataclass
class Rule:
    password: int
    filter_name: str
    target_zone: int
    setcode: int | None = None
    race: int | None = None
    type_mask: int | None = None
    level_max: int | None = None
    level_min: int | None = None
    exclude_code: int | None = None
    banish_kind: str = "none"
    banish_count: int = 0
    source_sha256: str = ""
    predicate_source: str = ""
    unresolved_native: tuple[str, ...] = ("IsAbleToHand",)

    def record(self):
        return asdict(self)


def _conjunction(node):
    if isinstance(node, lua.BinOp) and node.op == "and":
        return _conjunction(node.left) + _conjunction(node.right)
    return [node]


def filter_fields(expression, constants, password):
    """Recognize every conjunct; unsupported atoms cannot silently disappear."""
    out = {}
    ability = 0

    def number(node):
        if isinstance(node, lua.Const) and type(node.value) is int:
            return node.value
        if isinstance(node, lua.Name) and node.id in constants:
            return constants[node.id]
        if node == lua.Name("id"):
            return password
        raise UnsupportedCondition(lua.unparse(node))

    def put(key, value):
        if key in out:
            raise UnsupportedCondition(f"duplicate {key}")
        out[key] = value

    for atom in _conjunction(expression):
        negated = isinstance(atom, lua.UnOp) and atom.op == "not"
        node = atom.operand if negated else atom
        if isinstance(node, lua.Method) and node.obj == lua.Name("c"):
            if negated:
                if node.name != "IsCode" or len(node.args) != 1:
                    raise UnsupportedCondition(lua.unparse(atom))
                put("exclude_code", number(node.args[0]))
            elif node.name in ("IsSetCard", "IsRace", "IsType", "IsLevelBelow") and len(node.args) == 1:
                key = {"IsSetCard": "setcode", "IsRace": "race", "IsType": "type_mask", "IsLevelBelow": "level_max"}[
                    node.name
                ]
                put(key, number(node.args[0]))
            elif node.name in ("IsMonster", "IsSpellTrap") and not node.args:
                put(
                    "type_mask",
                    constants["TYPE_MONSTER"]
                    if node.name == "IsMonster"
                    else constants["TYPE_SPELL"] | constants["TYPE_TRAP"],
                )
            elif node.name == "IsAbleToHand" and not node.args:
                ability += 1
            else:
                raise UnsupportedCondition(lua.unparse(atom))
        elif isinstance(node, lua.BinOp) and node.op == ">=" and not negated:
            if node.left != lua.Method(lua.Name("c"), "GetLevel", ()):
                raise UnsupportedCondition(lua.unparse(atom))
            put("level_min", number(node.right))
        else:
            raise UnsupportedCondition(lua.unparse(atom))
    if ability != 1:
        raise UnsupportedCondition("expected one explicit native IsAbleToHand assumption")
    return out


def compile_rule(password: int, scripts: Path) -> Rule:
    if password not in {31562086, 17751597, 32807846, 47325505, 213326, 79816536}:
        raise UnsupportedCondition("card/effect outside validated probe scope")
    path = scripts / "official" / f"c{password}.lua"
    source = path.read_text()
    parsed = lua.Script.parse(source)
    constants = load_constants()
    calls = list(parsed.find_calls({"Duel.IsExistingMatchingCard", "Duel.IsExistingTarget"}))
    if len(calls) != 1:
        raise UnsupportedCondition("expected one scoped target-existence query")
    position, _, args = calls[0]
    filter_name = lua.unparse(args[0])
    function = parsed.functions.get(filter_name)
    if function is None or function.params != ("c",):
        raise UnsupportedCondition("unrecognized target callback")
    values = list(lua.returns(parsed.tokens, function.body))
    body = parsed.tokens[function.body[0] : function.body[1]]
    if len(values) != 1 or not body or body[0][1] != "return":
        raise UnsupportedCondition("filter is not a direct conjunction")
    # Reparse the entire function body so extra statements cannot be ignored.
    expression, end = lua.parse_expr(body, 1)
    if end != len(body):
        raise UnsupportedCondition("trailing filter statements")
    zone_name = lua.unparse(args[2])
    if zone_name not in ("LOCATION_DECK", "LOCATION_GRAVE") or lua.unparse(args[3]) != "0":
        raise UnsupportedCondition("non-local or unsupported target zone")
    targets = [name for name, fn in parsed.functions.items() if fn.body[0] <= position < fn.body[1]]
    if len(targets) != 1:
        raise UnsupportedCondition("ambiguous target scope")
    target = targets[0]
    rule = Rule(password, filter_name, constants[zone_name], **filter_fields(expression, constants, password))
    setters = list(parsed.find_method_calls({"SetTarget", "SetCost"}))
    receivers = [
        receiver for _, receiver, method, a in setters if method == "SetTarget" and lua.unparse(a[0]) == target
    ]
    for _, receiver, method, a in setters:
        if method == "SetCost" and receiver in receivers:
            if lua.unparse(a[0]) != "Cost.SelfBanish":
                raise UnsupportedCondition("unknown selected cost")
            helper = lua.Script.parse((scripts / "utility.lua").read_text())
            fn = helper.functions["Cost.SelfBanish"]
            text = " ".join(v for _, v in helper.tokens[fn.body[0] : fn.body[1]])
            if "IsAbleToRemoveAsCost" not in text:
                raise UnsupportedCondition("self-banish helper contract changed")
            rule.banish_kind, rule.banish_count = "self_cost", 1
    factories = list(parsed.find_calls({"Effect.CreateMysteruneQPEffect"}))
    if factories:
        if len(factories) != 1 or receivers:
            raise UnsupportedCondition("ambiguous effect factory")
        a = factories[0][2]
        if (
            lua.unparse(a[3]) != target
            or not isinstance(a[5], lua.Const)
            or type(a[5].value) is not int
            or a[5].value <= 0
        ):
            raise UnsupportedCondition("unsupported Runick binding")
        helper = (scripts / "cards_specific_functions.lua").read_text()
        if "Duel.IsPlayerCanRemove(tp) and Duel.GetFieldGroupCount(tp,0,LOCATION_DECK)>=ct" not in helper:
            raise UnsupportedCondition("Runick helper contract changed")
        rule.banish_kind, rule.banish_count = "opponent_deck", a[5].value
    elif len(receivers) != 1:
        raise UnsupportedCondition("missing selected effect registration")
    rule.source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    rule.predicate_source = lua.unparse(expression)
    return rule


def set_member(setcodes, query):
    return any((s & 0xFFF) == (query & 0xFFF) and (s & query & 0xF000) == (query & 0xF000) for s in setcodes)


def fixture_predicates(rule: Rule, spec, db, *, assume_unmodified_fixture=False):
    """Eight separate conditions, not an aggregate legal-action label.

    The explicit scope flag is required: dynamic names/types, other continuous
    restrictions, used count limits, extra-deck modes, timing, and IsAbleToHand are
    not modeled. The caller must validate this constructed-fixture scope natively.
    """
    if not assume_unmodified_fixture:
        raise UnsupportedCondition("native fixture assumptions must be explicit")
    if spec["source"] != rule.password:
        raise UnsupportedCondition("wrong source binding")
    c = db[spec["candidate"]]
    banish = rule.banish_kind == "none" or (
        not spec["wall_active"] and (rule.banish_kind == "self_cost" or rule.banish_count <= 4)
    )
    return [
        float(rule.setcode is None or set_member(c.setcodes, rule.setcode)),
        float(rule.race is None or bool(c.race & rule.race)),
        float(rule.type_mask is None or bool(c.type & rule.type_mask)),
        float(rule.level_max is None or c.level <= rule.level_max),
        float(rule.level_min is None or c.level >= rule.level_min),
        float(rule.exclude_code is None or db.canonical(c.password) != db.canonical(rule.exclude_code)),
        float(spec["zone"] == rule.target_zone),
        float(banish),
    ]
