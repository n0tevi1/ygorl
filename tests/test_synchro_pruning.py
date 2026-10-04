"""Material legality must survive version-pinned pruning, including callbacks which change levels."""

import hashlib
import json
from pathlib import Path

import pytest

from ygorl import _core, paths
from ygorl.engine import constants as C
from ygorl.engine.duel import default_cards, default_scripts, expand_seed
from ygorl.engine.script_patches import SYNCHRO_SHA256, synchro_override

DM, BE, EXTRA, FUTURE = 46986414, 89631139, 52687916, 67441435


def original_scripts():
    return _core.ScriptDirectory([str(p) for p in paths.script_directories()])


def test_unknown_script_versions_are_not_rewritten():
    source = original_scripts().read("proc_synchro.lua")
    assert hashlib.sha256(source).hexdigest() == SYNCHRO_SHA256
    assert default_scripts().read("proc_synchro.lua") == synchro_override(source) != source
    assert synchro_override(source + b"\n-- upstream update") is None
    assert synchro_override(None) is None


def single_effect(code, *, value=None, operation=None, target=None, label=None):
    body = f"local e=Effect.CreateEffect(c) e:SetType(EFFECT_TYPE_SINGLE) e:SetCode({code}) "
    for name, content in (("Value", value), ("Operation", operation), ("Target", target), ("Label", label)):
        if content is not None:
            body += f"e:Set{name}({content}) "
    return body + "c:RegisterEffect(e) "


@pytest.mark.parametrize("case,expected", [
    ("ordinary_over_level", False), ("ordinary_equal", True),
    ("alternate_level", True), ("dual_level", True), ("packed_numeric_level", True), ("custom_material", True),
    ("future_custom_material", True), ("group_constraint_changes_level", True),
    ("unknown_hand_check_changes_level", True), ("replaced_library_hand_check", True),
])  # fmt: skip
def test_material_checks_keep_legal_combinations_and_restore_groups(case, expected):
    messages = []
    for patched in (False, True):
        overrides = {}
        if patched:
            overrides["proc_synchro.lua"] = default_scripts().read("proc_synchro.lua")
        tuner = single_effect("EFFECT_ADD_TYPE", value="TYPE_TUNER")
        material, future, setup, req2, level = "", "", "", "nil", 9
        has_future = case in (
            "future_custom_material",
            "unknown_hand_check_changes_level",
            "replaced_library_hand_check",
        )
        if case == "ordinary_equal":
            level = 15
        if case in ("alternate_level", "dual_level"):
            value = "2" if case == "alternate_level" else "(8<<16)|2"
            material = single_effect("EFFECT_SYNCHRO_LEVEL", value=value)
        if case in ("custom_material", "future_custom_material"):
            effect = single_effect("EFFECT_SYNCHRO_MATERIAL_CUSTOM", operation="function() return true,true end")
            if has_future:
                future = effect
            else:
                material = effect
        if case == "group_constraint_changes_level":
            change = single_effect("EFFECT_CHANGE_LEVEL", value="2")
            change = change.replace("Effect.CreateEffect(c)", "Effect.CreateEffect(n)").replace(
                "c:RegisterEffect", "n:RegisterEffect"
            )
            req2 = "function() " + change + " return true end"
        if case in ("unknown_hand_check_changes_level", "replaced_library_hand_check"):
            tuner += single_effect("EFFECT_HAND_SYNCHRO", label="123")
            callback = """function(e,c,sg,tg,ntg,tsg,ntsg)
                if c then
                    local n=Duel.GetMatchingGroup(Card.IsCode,0,LOCATION_MZONE,0,nil,89631139):GetFirst()
                    local change=Effect.CreateEffect(n) change:SetType(EFFECT_TYPE_SINGLE)
                    change:SetCode(EFFECT_CHANGE_LEVEL) change:SetValue(1) n:RegisterEffect(change)
                end
                return true,Group.CreateGroup(),Group.CreateGroup()
            end"""
            if case == "unknown_hand_check_changes_level":
                future = single_effect("EFFECT_HAND_SYNCHRO+EFFECT_SYNCHRO_CHECK", target=callback, label="123")
            else:
                # Register through the reviewed helper, then replace its target: identity must be rechecked.
                setup = f"""
                    local hand=Synchro.CreateHandMaterialEffect(t,123,nil,nil,false,t)
                    t:RegisterEffect(hand)
                    local f=Duel.GetMatchingGroup(Card.IsCode,0,LOCATION_HAND,0,nil,{FUTURE}):GetFirst()
                    assert(hand:GetValue()(hand,f,sc))
                    f:GetCardEffect(EFFECT_HAND_SYNCHRO+EFFECT_SYNCHRO_CHECK):SetTarget({callback})
                """
        future = single_effect("EFFECT_REMOVE_TYPE", value="TYPE_TUNER") + future
        for card, body in ((DM, tuner), (BE, material), (FUTURE, future)):
            overrides[f"c{card}.lua"] = f"local s,id=GetID() function s.initial_effect(c) {body} end"
        location = "LOCATION_HAND" if case == "replaced_library_hand_check" else "LOCATION_MZONE"
        overrides["probe.lua"] = f"""
            local t=Duel.GetMatchingGroup(Card.IsCode,0,LOCATION_MZONE,0,nil,{DM}):GetFirst()
            local n=Duel.GetMatchingGroup(Card.IsCode,0,LOCATION_MZONE,0,nil,{BE}):GetFirst()
            local sc=Duel.GetMatchingGroup(Card.IsCode,0,LOCATION_EXTRA,0,nil,{EXTRA}):GetFirst()
            {setup}
            local tsg=Group.FromCards(t) local ntsg=Group.CreateGroup() local sg=Group.FromCards(t)
            local ntg=Group.FromCards(n)
            if {str(has_future).lower()} then
                ntg:Merge(Duel.GetMatchingGroup(Card.IsCode,0,{location},0,nil,{FUTURE}))
            end
            local original_ntg=ntg:Clone()
            local result=Synchro.CheckP42(n,ntg,tsg,ntsg,sg,1,99,{req2},nil,{level},sc,0,Group.CreateGroup(),false,nil,nil)
            assert(result=={str(expected).lower()},"wrong material legality: "..tostring(result))
            assert(#ntg==#original_ntg and ntg:Includes(original_ntg) and #ntsg==0 and #sg==1 and sg:IsContains(t) and #tsg==1,"groups not restored")
            Debug.Message("material legality and group restoration passed")
        """
        scripts = _core.ScriptDirectory([str(p) for p in paths.script_directories()], overrides)
        cards = default_cards().to_core()
        if case == "packed_numeric_level":
            # Sum checks interpret both 16-bit halves as alternatives, even without an explicit effect.
            changed = _core.CardDatabase()
            for code in (DM, BE, EXTRA):
                row = list(cards.get(code))
                if code == BE:
                    row[4] = (1 << 16) | 2
                changed.add(*row)
            cards = changed
        core = _core.Duel(expand_seed(20261004), C.DUEL_MODE_MR5, (8000, 0, 0), (8000, 0, 0), cards, scripts)
        try:
            assert core.load_script("constant.lua") and core.load_script("utility.lua")
            core.new_card(0, 0, DM, 0, C.LOCATION_MZONE, 0, C.POS_FACEUP_ATTACK)
            core.new_card(0, 0, BE, 0, C.LOCATION_MZONE, 1, C.POS_FACEUP_ATTACK)
            core.new_card(0, 0, EXTRA, 0, C.LOCATION_EXTRA, 0, C.POS_FACEDOWN_DEFENSE)
            if has_future:
                loc = C.LOCATION_HAND if case == "replaced_library_hand_check" else C.LOCATION_MZONE
                core.new_card(0, 0, FUTURE, 0, loc, 2, C.POS_FACEUP_ATTACK)
            assert core.load_script("probe.lua"), core.pop_logs()
            logs = core.pop_logs()
            assert logs == [(1, b"material legality and group restoration passed")]
            messages.append(logs)
        finally:
            core.close()
    assert messages[0] == messages[1]


REPLAYS = json.loads((Path(__file__).parent / "data/synchro-search-replays.json").read_text())["cases"]


def buffers_digest(buffers):
    digest = hashlib.sha256()
    for buffer in buffers:
        digest.update(len(buffer).to_bytes(8, "little"))
        digest.update(buffer)
    return digest.hexdigest()


@pytest.mark.parametrize("case", REPLAYS, ids=lambda c: c["name"])
def test_known_synchro_prefix_continues_under_the_original_budget(case):
    old = _core.set_max_script_steps(100000)
    try:
        observations = []
        for patched in (False, True):
            _core.script_steps_peak(True)
            host = _core.HostDuel(default_cards().to_core(), default_scripts() if patched else original_scripts(), [])
            host.start(
                case["core_seed"], case["rule_flags"], (8000, 5, 1), (8000, 5, 1), case["loaded_decks"], 200, 4000
            )
            for i, action in enumerate(case["actions"][: case["failure_decisions"]]):
                assert not host.done()
                obs = host.observe()
                signature = (host.player(), repr(host.actions()), buffers_digest(obs[k].tobytes() for k in sorted(obs)))
                if patched:
                    assert signature == observations[i]
                else:
                    observations.append(signature)
                host.act(action)
            if patched:
                assert not host.done() and host.actions()
                assert _core.script_steps_peak() < 100000
            else:
                assert host.done() and host.result()["reason"] == "error"
                assert "script budget" in host.result()["error"]
    finally:
        _core.set_max_script_steps(old)


@pytest.mark.parametrize("case", [c for c in REPLAYS if "reference" in c], ids=lambda c: c["name"])
def test_complete_game_matches_unmodified_high_budget_reference(case):
    from ygorl.cards.ydk import Deck
    from ygorl.engine.duel import Duel, DuelConfig, DuelSession

    old = _core.set_max_script_steps(100000)
    decks = [Deck(main=m, extra=e) for m, e in case["loaded_decks"]]
    config = DuelConfig(rule_flags=case["rule_flags"], max_decisions=4000, shuffle_decks=False)
    session = DuelSession(Duel(0, None, *decks, core_seed=case["core_seed"], config=config, record_messages=True))
    try:
        for action in case["actions"]:
            assert not session.done
            session.act(action)
        assert session.done
        result = session.result()
        reference = case["reference"]
        actual = dict(
            reason=result.reason,
            winner=result.winner,
            lp=list(result.lp),
            turns=result.turns,
            decisions=result.decisions,
        )
        assert actual == reference["result"]
        assert len(result.message_log) == reference["message_buffers"]
        assert buffers_digest(result.message_log) == reference["messages_sha256"]
        assert buffers_digest(result.responses) == reference["responses_sha256"]
    finally:
        session.close()
        _core.set_max_script_steps(old)
