"""Fusion constraints apply to complete groups, including nonmonotone real-card requirements."""

import hashlib
import json
from pathlib import Path

import pytest

from ygorl import _core, paths
from ygorl.engine import constants as C
from ygorl.engine.duel import default_cards, default_scripts, expand_seed
from ygorl.engine.script_patches import FUSION_SHA256, fusion_override


def test_fusion_patch_is_content_pinned():
    original = _core.ScriptDirectory([str(p) for p in paths.script_directories()]).read("proc_fusion.lua")
    assert hashlib.sha256(original).hexdigest() == FUSION_SHA256
    assert default_scripts().read("proc_fusion.lua") == fusion_override(original) != original
    assert fusion_override(original + b"\n-- unknown version") is None
    assert fusion_override(None) is None


@pytest.mark.parametrize("case,expected", [
    ("no_constraint", True), ("at_most_one_deck", True),
    ("required_two_deck", False), ("selected_two_deck", False),
    ("forbidden_card", False), ("substitute_allowed", True), ("substitute_denied", False),
    ("exact_count_mismatch", False), ("no_zone_needed", True),
    ("requires_future_material", True), ("constraint_never_sees_partial_group", True),
])  # fmt: skip
def test_fusion_constraints_and_group_restoration(case, expected):
    for patched in (False, True):
        ids = (46986414, 89631139, 67441435, 14558127, 52687916)
        overrides = {f"c{code}.lua": "local s,id=GetID() function s.initial_effect(c) end" for code in ids}
        if patched:
            overrides["proc_fusion.lua"] = default_scripts().read("proc_fusion.lua")
        constraint = "function(tp,g) return g:FilterCount(Card.IsLocation,nil,LOCATION_DECK)<=1 end"
        if case == "no_constraint":
            constraint = "nil"
        if case == "forbidden_card":
            constraint = "function(tp,g) return not g:IsExists(Card.IsCode,1,nil,46986414) end"
        if case == "requires_future_material":
            constraint = "function(tp,g) return g:IsExists(Card.IsCode,1,nil,67441435) end"
        if case == "constraint_never_sees_partial_group":
            constraint = "function(tp,g) assert(#g==3,'partial group checked') return true end"
        must = "Group.FromCards(d1,d2)" if case == "required_two_deck" else "Group.CreateGroup()"
        selected = "Group.FromCards(d1)" if case == "selected_two_deck" else "Group.CreateGroup()"
        candidate = "d2" if case == "selected_two_deck" else "h1"
        functions = "aux.TRUE,aux.TRUE,aux.TRUE"
        if case.startswith("substitute_"):
            functions = "function(c,fc,sub) return sub and c==h1 end,aux.TRUE,aux.TRUE"
        sub = "true" if case == "substitute_allowed" else "false"
        chkf = "PLAYER_NONE" if case == "no_zone_needed" else "0"
        overrides["probe.lua"] = f"""
            local h1=Duel.GetMatchingGroup(Card.IsCode,0,LOCATION_HAND,0,nil,46986414):GetFirst()
            local h2=Duel.GetMatchingGroup(Card.IsCode,0,LOCATION_HAND,0,nil,89631139):GetFirst()
            local d1=Duel.GetMatchingGroup(Card.IsCode,0,LOCATION_DECK,0,nil,67441435):GetFirst()
            local d2=Duel.GetMatchingGroup(Card.IsCode,0,LOCATION_DECK,0,nil,14558127):GetFirst()
            local fc=Duel.GetMatchingGroup(Card.IsCode,0,LOCATION_EXTRA,0,nil,52687916):GetFirst()
            local mg=Group.FromCards(h1,h2,d1,d2) local before=mg:Clone()
            local sg={selected} local sg_before=sg:Clone() local must={must} local must_before=must:Clone()
            Fusion.CheckAdditional={constraint}
            Fusion.CheckExact={2 if case == "exact_count_mismatch" else "nil"}
            local result=Fusion.SelectMix({candidate},0,mg,sg,must,fc,{sub},false,false,SUMMON_TYPE_FUSION|MATERIAL_FUSION,{chkf},{functions})
            assert(result=={str(expected).lower()},"wrong fusion legality: "..tostring(result))
            assert(#mg==#before and mg:Includes(before),"candidate group changed")
            assert(#sg==#sg_before and sg:Includes(sg_before),"selected group not restored")
            assert(#must==#must_before and must:Includes(must_before),"required group changed")
            Debug.Message("fusion legality and group restoration passed")
        """
        scripts = _core.ScriptDirectory([str(p) for p in paths.script_directories()], overrides)
        core = _core.Duel(
            expand_seed(20261004), C.DUEL_MODE_MR5, (8000, 0, 0), (8000, 0, 0), default_cards().to_core(), scripts
        )
        try:
            assert core.load_script("constant.lua") and core.load_script("utility.lua")
            for code, loc in zip(
                ids, [C.LOCATION_HAND, C.LOCATION_HAND, C.LOCATION_DECK, C.LOCATION_DECK, C.LOCATION_EXTRA], strict=True
            ):
                core.new_card(0, 0, code, 0, loc, 0, C.POS_FACEDOWN_DEFENSE)
            assert core.load_script("probe.lua"), core.pop_logs()
            assert core.pop_logs() == [(1, b"fusion legality and group restoration passed")]
        finally:
            core.close()


def test_real_fusion_failure_continues_with_identical_prefix():
    case = json.loads((Path(__file__).parent / "data/material-budget-replays.json").read_text())["cases"][0]
    original = _core.ScriptDirectory([str(p) for p in paths.script_directories()])
    old = _core.set_max_script_steps(100000)
    try:
        points = []
        for patched in (False, True):
            host = _core.HostDuel(default_cards().to_core(), default_scripts() if patched else original, [])
            host.start(
                case["core_seed"], case["rule_flags"], (8000, 5, 1), (8000, 5, 1), case["loaded_decks"], 200, 4000
            )
            for i, action in enumerate(case["actions"]):
                assert not host.done()
                obs = host.observe()
                signature = (host.player(), repr(host.actions()), tuple((k, obs[k].tobytes()) for k in sorted(obs)))
                if patched:
                    assert signature == points[i]
                else:
                    points.append(signature)
                host.act(action)
            if patched:
                assert not host.done() and host.actions()
            else:
                assert host.done() and host.result()["reason"] == "error"
                assert "script budget" in host.result()["error"]
    finally:
        _core.set_max_script_steps(old)


@pytest.mark.parametrize("case", json.loads((Path(__file__).parent / "data/fusion-search-replays.json").read_text())["cases"],
                         ids=lambda c: c["name"])  # fmt: skip
def test_complete_fusion_game_matches_unmodified_high_budget_reference(case):
    from ygorl.cards.ydk import Deck
    from ygorl.engine.duel import Duel, DuelConfig, DuelSession

    decks = [Deck(main=m, extra=e) for m, e in case["loaded_decks"]]
    old = _core.set_max_script_steps(100000)
    session = DuelSession(Duel(
        0, None, *decks, core_seed=case["core_seed"], record_messages=True,
        config=DuelConfig(rule_flags=case["rule_flags"], max_decisions=4000, shuffle_decks=False),
    ))  # fmt: skip
    try:
        for action in case["actions"]:
            assert not session.done
            session.act(action)
        assert session.done
        result = session.result()
        reference = case["reference"]
        assert dict(reason=result.reason, winner=result.winner, lp=list(result.lp),
                    turns=result.turns, decisions=result.decisions) == reference["result"]  # fmt: skip
        assert len(result.message_log) == reference["message_buffers"]
        for buffers, key in ((result.message_log, "messages_sha256"), (result.responses, "responses_sha256")):
            digest = hashlib.sha256()
            for buffer in buffers:
                digest.update(len(buffer).to_bytes(8, "little"))
                digest.update(buffer)
            assert digest.hexdigest() == reference[key]
    finally:
        session.close()
        _core.set_max_script_steps(old)
