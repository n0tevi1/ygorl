import pytest

from ygorl.eval.battle_outcomes import decompose_battle_outcome


def battle(atk=2500, target_atk=3100, *, target_position=1, destroyed=(1, 0)):
    return {
        "attacker_atk": atk,
        "attacker_def": 0,
        "attacker": {"position": 1},
        "target_atk": target_atk,
        "target_def": 2000,
        "target": {"position": target_position, "location": 4},
        "attacker_destroyed": destroyed[0],
        "target_destroyed": destroyed[1],
    }


def test_numeric_boost_reverses_damage_direction():
    out = decompose_battle_outcome({"damage": [0, 400], "destroy": [0, 1]}, battle(), [600, 0])
    assert out["numeric_damage_delta"] == [600, -400]
    assert out["non_arithmetic_damage_delta"] == [0, 0]
    assert out["destruction_delta"] == [0, 0]


def test_equal_attack_then_burn_is_distinct_from_destruction_override():
    out = decompose_battle_outcome(
        {"damage": [0, 0], "destroy": [1, 1]}, battle(2400, 2400, destroyed=(1, 1)), [1200, 0]
    )
    assert out["numeric_damage_delta"] == [0, 0]
    assert out["non_arithmetic_damage_delta"] == [1200, 0]
    assert out["destruction_delta"] == [0, 0]
    protected = decompose_battle_outcome(
        {"damage": [0, 500], "destroy": [0, 1]}, battle(3000, 2500, destroyed=(0, 0)), [0, 500]
    )
    assert protected["destruction_delta"] == [0, -1]


def test_components_may_cancel_without_being_zero():
    out = decompose_battle_outcome({"damage": [0, 400], "destroy": [0, 1]}, battle(), [0, 400])
    assert out["numeric_damage_delta"] == [600, -400]
    assert out["non_arithmetic_damage_delta"] == [-600, 400]


@pytest.mark.parametrize("native", [battle(target_position=2), battle(atk=65535)])
def test_unsupported_battle_stats_are_unknown(native):
    assert decompose_battle_outcome({"damage": [0, 0], "destroy": [0, 0]}, native, [0, 0]) is None
