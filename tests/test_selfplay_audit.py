"""Self-audit of Solo player traces (design 10 「自审」): one synthetic trace per check, no game client and no out/."""

import json
from types import SimpleNamespace

from ygorl.md.ui import my_monster_xy, my_spell_xy
from ygorl.selfplay.selfaudit import (
    ANCIENT_CHANT,
    CARD_ADVANCE,
    MILLENNIUM_REVELATION,
    TYPE_FIELD,
    audit_records,
    audit_trace,
    main,
)

# MD card id -> password, and a card database stand-in (name, ATK, type)
PW = {101: 1001, 102: 1002, 103: 1003, 104: 1004, 105: 1005, 150: MILLENNIUM_REVELATION, 151: ANCIENT_CHANT,
      152: 2000, 153: CARD_ADVANCE}  # fmt: skip
MONSTER, SPELL = 0x1, 0x2
CARDS = {
    1001: SimpleNamespace(name="Low", attack=1000, type=MONSTER),
    1002: SimpleNamespace(name="Mid", attack=2000, type=MONSTER),
    1003: SimpleNamespace(name="High", attack=2500, type=MONSTER),
    1004: SimpleNamespace(name="Searched", attack=1800, type=MONSTER),
    1005: SimpleNamespace(name="Other", attack=1500, type=MONSTER),
    MILLENNIUM_REVELATION: SimpleNamespace(name="Millennium Revelation", attack=0, type=SPELL),
    ANCIENT_CHANT: SimpleNamespace(name="Ancient Chant", attack=0, type=SPELL),
    2000: SimpleNamespace(name="Field", attack=0, type=SPELL | TYPE_FIELD),
    CARD_ADVANCE: SimpleNamespace(name="Card Advance", attack=0, type=SPELL),
}


def side(field=None, hand=(), grave=(), deck=30):
    slots = [None] * 13
    for s, cid in (field or {}).items():
        slots[s] = [cid, 1, 0]
    return {"lp": 8000, "field": slots, "hand": list(hand), "grave": list(grave),
            "counts": {"hand": len(hand), "deck": deck, "grave": len(grave)}}  # fmt: skip


def step(seq, me=None, opp=None, decision=None, observed=None, clicks=(), turn=3, **extra):
    return {"kind": "step", "seq": seq, "step": seq, "turn": turn, "phase": "main1", "turn_player": "me",
            "dialog": None, "memory": {"me": me or side(), "opp": opp or side()}, "decision": decision,
            "observed": observed or {}, "clicks": [{"xy": list(xy), "why": why} for xy, why in clicks],
            "check": None, "refusal": None, "stuck": None, **extra}  # fmt: skip


def game(n=1, exit=0):
    return {"kind": "game", "game": n, "exit": exit, "finished": exit == 0, "result": 2, "lp": [0, 8000], "turn": 5}


def audit(*records):
    return audit_records(list(records), PW, CARDS)


def kinds(rep, severity=None):
    return [f.kind for f in rep.findings if severity is None or f.severity == severity]


def policy(action, card, p=0.9, **extra):
    return {"by": "policy", "action": action, "card": card, "p": p, "phase": "main1", "top": [], **extra}


def test_a_clean_game_is_correct_and_counts_rules():
    rep = audit(step(1, decision={"by": "rule", "rule": "draw-continue"}),
                step(2, decision=policy("summon", 1001), observed={"selected": {"want": 101, "got": 101}}),
                game())  # fmt: skip
    (g,) = rep.games
    assert g.verdict == "correct" and g.rules["draw-continue"] == 1 and g.policy == 1 and not rep.findings


def test_games_split_at_their_game_records():
    rep = audit(step(1), game(1), step(1), step(2), game(2), step(1))
    assert [g.game for g in rep.games] == [1, 2, 3] and [g.steps for g in rep.games] == [1, 2, 1]
    assert rep.games[2].exit is None  # the trace ends inside game 3


def test_hand_action_on_another_card_is_major():
    rep = audit(step(1, decision=policy("summon", 1001), observed={"selected": {"want": 101, "got": 102}}), game())
    assert kinds(rep, "major") == ["selected-mismatch"] and rep.games[0].verdict == "major"


def test_prompt_card_meant_but_passed_is_major():
    rec = step(1, decision={"by": "rule", "rule": "prompt-pass"}, clicks=[((1026, 934), "tick"), ((870, 1117), "pass")],
               observed={"prompt_card": {"want": 1001, "got": None}, "prompt_list": [101]})  # fmt: skip
    assert "prompt-card" in kinds(audit(rec, game()), "major")


def test_prompt_card_unconfirmed_but_only_offered_is_info():
    rec = step(1, decision={"by": "rule", "rule": "prompt-respond"}, clicks=[((1180, 1117), "activate")],
               observed={"prompt_card": {"want": MILLENNIUM_REVELATION, "got": None}, "prompt_list": [150]})  # fmt: skip
    rep = audit(rec, game())
    assert rep.games[0].verdict == "correct" and "prompt-card" in kinds(rep, "info")


def test_prompt_card_another_card_offered_is_minor():
    rec = step(1, decision={"by": "rebuild", "question": "respond", "pick": 0}, clicks=[((870, 1117), "pass")],
               observed={"prompt_card": {"want": 1001, "got": 1002}, "prompt_list": [102]})  # fmt: skip
    assert kinds(audit(rec, game()), "minor") == ["prompt-card"]


def test_list_pick_of_another_card_is_major():
    cands = [[104, "deck", 3], [105, "deck", 7]]
    lst = step(1, decision={"by": "shadow", "dialog": "list", "pick": 0},
               observed={"list_check": {"want": 104, "by": "memory"}, "list_candidates": cands})  # fmt: skip
    after = step(2, me=side(grave=[105], deck=29))
    rep = audit(lst, after, game())
    (f,) = rep.findings
    assert (f.severity, f.kind, f.evidence["moved"]) == ("major", "list-pick", [105])


def test_list_pick_of_the_wanted_card_is_correct():
    cands = [[104, "hand", 0], [105, "hand", 1]]
    lst = step(1, me=side(hand=[104, 105]), observed={"list_check": {"want": 104}, "list_candidates": cands})
    rep = audit(lst, step(2, me=side(hand=[105], grave=[104])), game())
    assert not rep.findings


def test_list_pick_unverified_is_minor():
    lst = step(1, observed={"list_check": {"want": 104, "unverified": True}, "list_candidates": []})
    assert kinds(audit(lst, game()), "minor") == ["list-pick"]


def test_attack_target_other_than_the_rebuild_pick_is_major():
    rec = step(1, opp=side(field={1: 102, 3: 103}), decision=policy("attack", 1003, target={"by": "rebuild", "pick": 3}),
               observed={"target_slot": 1})  # fmt: skip
    assert kinds(audit(rec, game()), "major") == ["attack-target"]


def test_attack_on_an_empty_zone_is_major():
    rec = step(1, opp=side(field={1: 102}), decision=policy("attack", 1003, target={"by": "rule"}),
               observed={"target_slot": 2})  # fmt: skip
    assert kinds(audit(rec, game()), "major") == ["attack-target"]


def test_failed_checks_of_one_action_in_a_turn_group_into_one_major():
    fail = {"action": "attack", "card": 1003, "ok": False, "problem": "attack: nothing changed"}
    rep = audit(step(1, decision=policy("attack", 1003)), step(2, check=fail), step(3, check=fail), game())
    (f,) = [f for f in rep.findings if f.kind == "check-failed"]
    assert f.severity == "major" and f.evidence["seqs"] == [2, 3] and "2 times" in f.message


def test_tribute_rule_taking_a_stronger_monster_is_major():
    me = side(field={0: 101, 1: 102, 2: 103})
    bad = step(1, me=me, decision=policy("summon", 1004, then=[{"by": "rule", "rule": "tribute-lowest-atk"}]),
               clicks=[(my_monster_xy(2), "tribute High")])  # fmt: skip
    good = step(1, me=me, decision=policy("summon", 1004, then=[{"by": "rule", "rule": "tribute-lowest-atk"}]),
                clicks=[(my_monster_xy(0), "tribute Low")])  # fmt: skip
    assert kinds(audit(bad, game()), "major") == ["tribute-not-lowest"]
    assert not audit(good, game()).findings


def test_tribute_candidates_and_the_policys_pick_bound_the_rule():
    # the policy took slot 0 (the lowest) itself; the rule's pick among the recorded candidates was the lowest left
    rec = step(1, me=side(field={0: 101, 1: 102, 2: 103}),
               decision=policy("summon", 1004, then=[{"by": "rule", "rule": "tribute-lowest-atk"}]),
               observed={"tribute_candidates": [0, 2]},
               clicks=[(my_monster_xy(0), "tribute (policy)"), (my_monster_xy(2), "tribute High")])  # fmt: skip
    assert not audit(rec, game()).findings


def test_zone_click_on_an_occupied_zone_is_minor():
    rec = step(1, me=side(field={7: 150}), decision=policy("sset", 1005), clicks=[(my_spell_xy(0), "zone for sset")])
    assert kinds(audit(rec, game()), "minor") == ["zone-occupied"]


def test_field_spell_zone_click():
    act = policy("activate", 2000)
    harmless = [
        step(1, decision=act, clicks=[(my_spell_xy(0), "zone for activate")]),
        step(2, me=side(field={12: 152})),
    ]
    lost = [step(1, decision=act, clicks=[(my_spell_xy(0), "zone for activate")]), step(2)]
    assert [(f.severity, f.kind) for f in audit(*harmless, game()).findings] == [("minor", "field-spell-zone")]
    assert [(f.severity, f.kind) for f in audit(*lost, game()).findings] == [("major", "field-spell-zone")]


def test_refusals():
    chant = {
        "action": "activate",
        "card": ANCIENT_CHANT,
        "reason": "MD refused: no Activate for Ancient Chant in the Graveyard",
    }
    other = {"action": "activate", "card": 1005, "reason": "MD refused: no Activate for Other on the field"}
    rep = audit(step(1, refusal=chant), step(2, refusal=chant), step(3, refusal=other), step(4, refusal=other),
                step(5, turn=5, refusal=other), game())  # fmt: skip
    got = sorted((f.severity, f.kind) for f in rep.findings)
    assert got == [("info", "refusal"), ("info", "refusal"), ("minor", "refusal-repeated")]


def test_recoveries_are_minor_and_counted():
    rec = {"kind": "step", "step": -1, "observed": {"recovery": {"why": "nothing to do for 120 s", "level": 1}}}
    rep = audit(step(1), rec, rec, game())
    assert rep.games[0].recoveries == {"nothing to do for 120 s": 2} and kinds(rep, "minor") == ["recovery"]


def test_failures_by_cause_are_minor_and_counted():
    fl = {"failure": {"cause": "no-effect", "message": "activate: nothing changed", "count": 1}}
    rep = audit(step(1, observed=fl), step(2, observed=fl), game())
    assert rep.games[0].recoveries == {"failure: no-effect": 2} and kinds(rep, "minor") == ["recovery"]


def test_a_duel_given_up_is_major():
    rep = audit(step(1), game(exit=3))
    assert kinds(rep, "major") == ["duel-given-up"] and rep.games[0].verdict == "major"


def test_rule_picks_accepted_mechanical_and_others():
    rep = audit(step(1, decision=policy("activate", CARD_ADVANCE)),
                step(2, decision={"by": "rule", "rule": "list-set-first-accepted"}),
                step(3, decision={"by": "rule", "rule": "zone-first-free"}),
                step(4, decision={"by": "rule", "rule": "summon-pick-first-offered"}),
                game())  # fmt: skip
    assert [(f.severity, f.kind) for f in rep.findings] == [("info", "rule-accepted"), ("info", "rule-decision")]
    assert rep.games[0].rules == {"list-set-first-accepted": 1, "zone-first-free": 1, "summon-pick-first-offered": 1}


def test_policy_picks_after_masks_are_info():
    rep = audit(step(1, decision=policy("end_phase", 0, p=0.0)), game())
    assert [(f.severity, f.kind) for f in rep.findings] == [("info", "low-p")]


def test_cli_prints_a_table_or_json_and_exits_0(tmp_path, capsys):
    path = tmp_path / "run.jsonl"
    recs = [{"kind": "start", "env": "md-2026-09"}, step(1), game(1), step(1), game(2, exit=3)]
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    assert audit_trace(path, PW, CARDS).to_json()["verdicts"] == {"correct": 1, "major": 1}
    assert main([str(path)]) == 0
    assert "duel-given-up" in capsys.readouterr().out
    assert main([str(path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["verdicts"] == {"correct": 1, "major": 1}
    assert main([str(tmp_path / "missing.jsonl")]) == 0
