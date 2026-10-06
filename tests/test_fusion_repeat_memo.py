"""Repeated-material memoization must preserve group-dependent predicate semantics."""

import pytest

from ygorl import _core, paths
from ygorl.engine import constants as C
from ygorl.engine.duel import default_cards, default_scripts, expand_seed


@pytest.mark.parametrize("case,expected", [
    ("compatible", True), ("two_nonrepeat", False), ("group_dependent", True),
    ("group_cannot_start", False), ("different_roles", True), ("one_substitute", True),
    ("two_substitutes", False), ("required_outside", True), ("required_missing", False),
    ("exact_count", False), ("nested", True),
])  # fmt: skip
def test_selected_repeat_states_preserve_legality_and_groups(case, expected):
    repeat = "function(c) return c~=cards[1] end"
    roles = "function(c) return c==cards[1] end"
    if case == "two_nonrepeat":
        repeat = "function(c) return c~=cards[1] and c~=cards[2] end"
        roles = "aux.TRUE"
    elif case == "group_dependent":
        repeat = "function(c,fc,sub,sub2,mg,g) return c~=cards[1] and (c~=cards[2] or g:IsContains(cards[3])) end"
    elif case == "group_cannot_start":
        repeat = "function(c,fc,sub,sub2,mg,g) return c~=cards[1] and #g>=3 end"
    elif case == "different_roles":
        repeat = "function(c) return c~=cards[1] and c~=cards[2] end"
        roles = "function(c) return c==cards[1] or c==cards[2] end,function(c) return c==cards[1] end"
    elif case == "one_substitute":
        roles = "function(c,fc,sub) return c==cards[1] and sub end"
    elif case == "two_substitutes":
        repeat = "function(c) return c~=cards[1] and c~=cards[2] end"
        roles = "function(c,fc,sub) return c==cards[1] and sub end,function(c,fc,sub) return c==cards[2] and sub end"
    elif case == "nested":
        repeat = """function(c)
            if not nested then
                nested=true
                local inner=Group.FromCards(table.unpack(cards,1,5))
                assert(Fusion.SelectMixRep(cards[6],0,mg,inner,Group.CreateGroup(),fc,
                    true,true,false,SUMMON_TYPE_FUSION|MATERIAL_FUSION,PLAYER_NONE,aux.TRUE,6,7,aux.TRUE))
                nested=false
            end
            return c~=cards[1]
        end"""
    for variant in ("baseline", "memo", "cache_full"):
        overrides = {f"c{code}.lua": "local s,id=GetID() function s.initial_effect(c) end"
                     for code in (46986414, 52687916)}  # fmt: skip
        if variant != "baseline":
            source = default_scripts().read("proc_fusion.lua")
            if variant == "cache_full":
                # The fallback executes the same search after the bounded cache fills.
                assert b"entries<4096" in source
                source = source.replace(b"entries<4096", b"entries<0")
            overrides["proc_fusion.lua"] = source
        overrides["probe.lua"] = f"""
            local mg=Duel.GetMatchingGroup(Card.IsCode,0,LOCATION_HAND,0,nil,46986414)
            local cards={{}} for tc in aux.Next(mg) do cards[#cards+1]=tc end
            assert(#cards==8)
            local fc=Duel.GetMatchingGroup(Card.IsCode,0,LOCATION_EXTRA,0,nil,52687916):GetFirst()
            local sg=Group.FromCards(table.unpack(cards,1,5))
            local mustg={"Group.FromCards(cards[8])" if case.startswith("required_") else "Group.CreateGroup()"}
            {"mg:RemoveCard(cards[8])" if case == "required_missing" else ""}
            local before=mg:Clone() local selected=sg:Clone() local required=mustg:Clone()
            local nested=false
            Fusion.CheckExact={5 if case == "exact_count" else "nil"}
            local result=Fusion.SelectMixRep(cards[6],0,mg,sg,mustg,fc,true,true,false,
                SUMMON_TYPE_FUSION|MATERIAL_FUSION,PLAYER_NONE,{repeat},6,7,{roles})
            assert(result=={str(expected).lower()},"unexpected legality: "..tostring(result))
            assert(#mg==#before and mg:Includes(before),"candidate group changed")
            assert(#sg==#selected and sg:Includes(selected),"selected group changed")
            assert(#mustg==#required and mustg:Includes(required),"required group changed")
            Debug.Message("repeat groups and legality passed")
        """
        scripts = _core.ScriptDirectory([str(p) for p in paths.script_directories()], overrides)
        core = _core.Duel(expand_seed(20261006), C.DUEL_MODE_MR5, (8000, 0, 0), (8000, 0, 0),
                          default_cards().to_core(), scripts)  # fmt: skip
        try:
            assert core.load_script("constant.lua") and core.load_script("utility.lua")
            for i in range(8):
                core.new_card(0, 0, 46986414, 0, C.LOCATION_HAND, i, C.POS_FACEDOWN_DEFENSE)
            core.new_card(0, 0, 52687916, 0, C.LOCATION_EXTRA, 0, C.POS_FACEDOWN_DEFENSE)
            assert core.load_script("probe.lua"), (variant, core.pop_logs())
            assert core.pop_logs() == [(1, b"repeat groups and legality passed")]
        finally:
            core.close()
