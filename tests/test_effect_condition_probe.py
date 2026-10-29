"""Guard against silent predicate loss and overbroad archetype/state matching."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from ygorl.build import lua

_spec = importlib.util.spec_from_file_location(
    "effect_condition_probe", Path(__file__).parents[1] / "tools" / "effect_condition_probe.py"
)
probe = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = probe
_spec.loader.exec_module(probe)


def test_archetype_subtype_is_not_interchangeable_with_parent():
    assert probe.set_member([0x3008], 0x8)
    assert probe.set_member([0x3008], 0x3008)
    assert not probe.set_member([0xC008, 0x5008], 0x3008)
    assert not probe.set_member([0x160], 0x180)


@pytest.mark.parametrize(
    "expression",
    [
        "c:IsAbleToHand() and c:IsFaceup()",
        "c:IsRace(RACE_WARRIOR) or c:IsAbleToHand()",
        "not c:IsSetCard(SET_HERO) and c:IsAbleToHand()",
        "c:IsAbleToHand() and mystery(c)",
    ],
)
def test_unknown_conditions_fail_closed(expression):
    node, _ = lua.parse_expr(lua.tokenize(expression), 0)
    with pytest.raises(probe.UnsupportedCondition):
        probe.filter_fields(node, {"RACE_WARRIOR": 1, "SET_HERO": 8}, 123)


def test_named_exclusion_and_native_uncertainty_are_preserved():
    node, _ = lua.parse_expr(lua.tokenize("c:IsSetCard(SET_RUNICK) and not c:IsCode(id) and c:IsAbleToHand()"), 0)
    fields = probe.filter_fields(node, {"SET_RUNICK": 0x180}, 31562086)
    assert fields == {"setcode": 0x180, "exclude_code": 31562086}
    rule = probe.Rule(31562086, "s.thfilter", 1, **fields)
    assert rule.unresolved_native == ("IsAbleToHand",)
    with pytest.raises(probe.UnsupportedCondition):
        probe.fixture_predicates(rule, {}, {})


def test_wall_only_affects_effects_that_require_banishment():
    class DB(dict):
        def canonical(self, code):
            return code

    db = DB({20: SimpleNamespace(password=20, setcodes=(0x180,), race=1, type=17, level=4)})
    spec = {"source": 10, "candidate": 20, "zone": 1, "wall_active": True}
    ordinary = probe.Rule(10, "s.filter", 1, race=1, level_max=4)
    banish = probe.Rule(10, "s.filter", 1, setcode=0x180, banish_kind="opponent_deck", banish_count=1)
    assert all(probe.fixture_predicates(ordinary, spec, db, assume_unmodified_fixture=True))
    assert not all(probe.fixture_predicates(banish, spec, db, assume_unmodified_fixture=True))
    spec["wall_active"] = False
    assert all(probe.fixture_predicates(banish, spec, db, assume_unmodified_fixture=True))


def test_unreviewed_card_is_rejected(tmp_path):
    with pytest.raises(probe.UnsupportedCondition, match="outside validated"):
        probe.compile_rule(99999999, tmp_path)
