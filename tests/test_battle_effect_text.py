"""The script channel must retain applicability, inheritance, and uncertainty."""

import importlib.util
from pathlib import Path

from ygorl.build import lua

_spec = importlib.util.spec_from_file_location(
    "battle_effect_text", Path(__file__).parents[1] / "tools" / "battle_effect_text.py"
)
module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(module)


def test_conditions_clones_and_registration_survive():
    source = """
-- public prose must not leak into the script channel
local s,id=GetID()
function s.initial_effect(c)
 local e1=Effect.CreateEffect(c)
 e1:SetCode(EFFECT_SET_BASE_ATTACK)
 e1:SetCondition(s.condition)
 e1:SetValue(200)
 c:RegisterEffect(e1)
 local e2=e1:Clone()
 e2:SetValue(300)
 c:RegisterEffect(e2)
end
function s.condition(e)
 return Duel.GetCurrentPhase()==PHASE_DAMAGE_CAL
  and Duel.GetTurnPlayer()~=e:GetHandlerPlayer()
end
"""
    result = module.script_document("fixture", source)
    assert lua.tokenize(result["canonical_source"]) == lua.tokenize(source)
    assert "public prose" not in result["text"]
    condition = next(r for r in result["setters"] if r["setter"] == "SetCondition")
    assert condition["function"] == "s.initial_effect"
    assert condition["expression"] == "( s . condition )"
    assert "PHASE_DAMAGE_CAL" in result["text"]
    assert "Clone" in result["text"] and "RegisterEffect" in result["text"]
    assert "currently active" in result["text"]


def test_missing_script_does_not_claim_absence_of_effects():
    result = module.script_document("fixture", None)
    assert result["status"] == "missing_script"
    assert "unknown" in result["text"]
