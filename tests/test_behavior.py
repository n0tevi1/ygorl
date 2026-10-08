import copy

from ygorl.eval.behavior import length_stats, summarize_trace


def row(index, events=(), player=0):
    action = {"kind": "cancel", "card": None}
    return dict(
        index=index,
        turn=1,
        phase=4,
        player=player,
        lp=[8000, 8000],
        board=[[], []],
        options=[action],
        chosen=action,
        choice=0,
        mask=[True],
        probs=[1.0],
        undo=[],
        events=list(events),
    )


def test_repeated_choices_require_no_intervening_game_change():
    rows = [row(i) for i in range(10)]
    assert summarize_trace(rows, 0)["counts"]["max_same_choice_without_event"] == 10
    rows[5]["events"] = [{"type": "Damage", "player": 1, "amount": 100}]
    result = summarize_trace(rows, 0)
    assert result["counts"]["max_same_choice_without_event"] == 5 and not result["findings"]


def test_changes_in_player_or_public_board_do_not_count_as_same_state():
    rows = [row(i, player=i % 2) for i in range(8)]
    for i, r in enumerate(rows):
        r["board"] = [[], [{"code": i}]]
    assert summarize_trace(rows, 0)["counts"]["max_same_choice_without_event"] == 1


def test_ash_ownership_uses_current_chain_not_turn_or_previous_chain():
    r = row(0, [{"type": "Chaining", "chain_count": 1, "triggering_controller": 0, "code": 123}])
    r["options"] = [{"kind": "chain", "card": {"code": 14558127}}, {"kind": "pass", "card": None}]
    r.update(chosen=r["options"][0], mask=[True, True], probs=[0.8, 0.2])
    s = copy.deepcopy(r)
    s.update(
        index=1,
        events=[{"type": "ChainEnd"}, {"type": "Chaining", "chain_count": 1, "triggering_controller": 1, "code": 456}],
    )
    result = summarize_trace([r, s], 0)
    assert result["counts"]["ash_self_chosen"] == result["counts"]["ash_opponent_chosen"] == 1
    assert result["findings"][0]["probability"] == 0.8


def test_length_summary_counts_long_tail_and_empty_sample():
    assert length_stats([]) == {"n": 0}
    result = length_stats([2, 3, 4, 9, 30])
    assert result["median"] == 4 and result["gt20"] == 1 and result["le4"] == 3
