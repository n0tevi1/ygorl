"""Boundary and missing-evidence contracts for observational outcome screening."""

import pytest

from ygorl.eval.action_outcomes import action_intervals


def row(index, kind="pass", events=(), player=0):
    return {"index": index, "player": player, "chosen": {"kind": kind, "card": None}, "events": list(events)}


def test_events_precede_action_and_cost_is_not_damage():
    damage = {"type": "Damage", "player": 0, "amount": 200}
    intervals, coverage = action_intervals(
        [
            row(0, "activate", [damage]),
            row(1, events=[{"type": "PayLpCost", "player": 0, "amount": 800}, damage, {"type": "ChainEnd"}]),
        ]
    )
    assert coverage["damage_outside_tracked_intervals"] == [200, 0]
    assert intervals[0]["damage"] == [200, 0]
    assert intervals[0]["lp_cost"] == [800, 0]
    assert intervals[0]["complete"]


def test_terminal_tail_and_phase_change_are_incomplete_not_safe():
    intervals, _ = action_intervals(
        [
            row(0, "activate"),
            row(1, "attack", [{"type": "NewPhase", "phase": 8}]),
        ]
    )
    assert [x["complete"] for x in intervals] == [False, False]
    assert [x["stop"] for x in intervals] == ["NewPhase", "trace_end_unobserved_terminal"]


def test_response_belongs_to_initiator_and_battle_chain_overlap_is_explicit():
    intervals, _ = action_intervals(
        [
            row(0, "attack"),
            row(1, "chain", player=1),
            row(2, "chain"),
            row(3, events=[{"type": "Damage", "player": 0, "amount": 300}, {"type": "ChainEnd"}]),
            row(4, events=[{"type": "Damage", "player": 1, "amount": 500}, {"type": "DamageStepEnd"}]),
        ]
    )
    attack, chain = intervals
    assert attack["damage"] == [300, 500] and chain["damage"] == [300, 0]
    assert chain["initiator"] == 1 and len(chain["responses"]) == 2
    assert attack["complete"] and chain["complete"]


def test_move_within_hand_does_not_count_as_gain_and_draw_is_separate():
    same = {"controller": 0, "location": 2}
    intervals, _ = action_intervals(
        [
            row(0, "activate"),
            row(
                1,
                events=[
                    {"type": "Move", "previous": same, "current": same},
                    {"type": "Draw", "player": 0, "cards": [[123, 1]]},
                    {"type": "ChainEnd"},
                ],
            ),
        ]
    )
    assert intervals[0]["hand_entries"] == [0, 0] and intervals[0]["draws"] == [1, 0]


def test_reordered_trace_is_rejected():
    with pytest.raises(ValueError, match="increase"):
        action_intervals([row(1), row(0)])
