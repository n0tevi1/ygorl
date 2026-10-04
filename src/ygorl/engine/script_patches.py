"""Content-pinned, in-memory script fixes; unreviewed upstream versions stay untouched."""

from hashlib import sha256


SYNCHRO_SHA256 = "cacd92d496ab653e6e5f304cb2b38b831e4f6d41ab1e842ffab909a0efb1543d"
FUSION_SHA256 = "3779bd72c57d95ce7d330c04f9bc2479966337f50ad23ce3a1d5bf9304a6fdc9"

# Only the library-created hand-material checks are known to preserve card levels and selected groups.
# Keep this identity set private to the Lua chunk, rather than trusting card-provided labels or flags.
_HELPERS = """local ygorl_static_hand_checks=setmetatable({}, {__mode="k"})
local ygorl_level_exceptions={EFFECT_SYNCHRO_LEVEL,EFFECT_SYNCHRO_MATERIAL_CUSTOM,
\tEFFECT_SYNCHRO_CHECK,EFFECT_SYNCHRO_MAT_RESTRICTION,
\tEFFECT_UPDATE_LEVEL,EFFECT_CHANGE_LEVEL,EFFECT_CHANGE_LEVEL_FINAL}
local function ygorl_unsafe_synchro_level(tc,sc)
\tfor _,code in ipairs(ygorl_level_exceptions) do
\t\tif tc:IsHasEffect(code) then return true end
\tend
\tfor _,te in ipairs({tc:GetCardEffect(EFFECT_HAND_SYNCHRO+EFFECT_SYNCHRO_CHECK)}) do
\t\tif not ygorl_static_hand_checks[te:GetTarget()] then return true end
\tend
\tlocal level=tc:GetLevel()
\treturn level<=0 or level>0xffff or tc:GetSynchroLevel(sc)~=level
end
"""

_BOUND = """\tlocal safe_levels=not req2 and not reqm and not Synchro.CheckAdditional
\t\tand not ntg:IsExists(ygorl_unsafe_synchro_level,1,nil,sc)
\t\tand not sg:IsExists(ygorl_unsafe_synchro_level,1,nil,sc)
\tif safe_levels and sg:GetSum(Card.GetLevel)>lv then
\t\tres=false
\telseif max and (tsg_count+ntsg_count)>max then"""


def synchro_override(source: bytes | None) -> bytes | None:
    """Return the reviewed optimization, or None to use an unknown/missing upstream source verbatim."""
    if source is None or sha256(source).hexdigest() != SYNCHRO_SHA256:
        return None
    text = source.decode("utf-8")
    text = text.replace("function Synchro.NonTuner(f,a,b,c)", _HELPERS + "function Synchro.NonTuner(f,a,b,c)", 1)
    start = text.index("function Synchro.CheckP42(")
    end = text.index("function Synchro.CheckLabel(", start)
    old = "\tif max and (tsg_count+ntsg_count)>max then"
    assert text[start:end].count(old) == 1
    text = text[:start] + text[start:end].replace(old, _BOUND) + text[end:]
    start = text.index("function Synchro.CreateHandMaterialEffect(")
    old = "\n\tlocal function synval(e,c,sc)"
    new = "\n\tygorl_static_hand_checks[synchktg]=true" + old
    assert text[start:].count(old) == 1
    text = text[:start] + text[start:].replace(old, new)
    return text.encode("utf-8")


def fusion_override(source: bytes | None) -> bytes | None:
    """Check the complete group's constraint before enumerating material-role permutations.

    Do not check partial groups: real card constraints can require an as-yet-unselected material,
    despite the upstream monotonicity comment (Fusion Destiny is one such counterexample).
    """
    if source is None or sha256(source).hexdigest() != FUSION_SHA256:
        return None
    text = source.decode("utf-8")
    start = text.index("function Fusion.CheckMixGoal(")
    end = text.index("function Fusion.SelectMix(", start)
    body = text[start:end]
    check = "(not Fusion.CheckAdditional or Fusion.CheckAdditional(tp,sg,fc,sumtype,tp))"
    assert body.count(check) == 1
    body = body.replace("return sg:IsExists(", f"return {check} and sg:IsExists(")
    body = body.replace("\n\t\tand " + check, "")
    return (text[:start] + body + text[end:]).encode("utf-8")
