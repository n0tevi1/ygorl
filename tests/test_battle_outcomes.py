import pytest

from ygorl.eval.battle_outcomes import ordinary_battle


def card(atk, defense=0, *, opponent=False, position=1, visible=1):
    row = [0] * 23
    row[4], row[6], row[7], row[17], row[18] = int(opponent), position, visible, atk, defense
    return row


@pytest.mark.parametrize(
    "attack,target,damage,destroy",
    [
        (2000, 1500, [0, 500], [0, 1]),
        (1000, 1500, [500, 0], [1, 0]),
        (1500, 1500, [0, 0], [1, 1]),
        (0, 0, [0, 0], [0, 0]),
    ],
)
def test_attack_position_comparison_and_zero_tie(attack, target, damage, destroy):
    assert ordinary_battle(card(attack), card(target, opponent=True)) == {"damage": damage, "destroy": destroy}


@pytest.mark.parametrize(
    "attack,damage,destroy", [(2200, [800, 0], [0, 0]), (3000, [0, 0], [0, 0]), (3500, [0, 0], [0, 1])]
)
def test_defense_comparison_does_not_destroy_weaker_attacker(attack, damage, destroy):
    assert ordinary_battle(card(attack), card(0, 3000, opponent=True, position=3)) == {
        "damage": damage,
        "destroy": destroy,
    }


def test_direct_attack_and_hidden_information_abstention():
    assert ordinary_battle(card(1700)) == {"damage": [0, 1700], "destroy": [0, 0]}
    assert ordinary_battle(card(1700), card(0, opponent=True, position=4, visible=0)) is None
    assert ordinary_battle(card(1700, position=3), card(1500, opponent=True)) is None
    assert ordinary_battle(card(65535)) is None
    with pytest.raises(ValueError, match="width"):
        ordinary_battle([1, 2])
