"""Content-pinned, in-memory script fixes; unreviewed upstream versions stay untouched."""

from hashlib import sha256


SYNCHRO_SHA256 = "cacd92d496ab653e6e5f304cb2b38b831e4f6d41ab1e842ffab909a0efb1543d"
FUSION_SHA256 = "3779bd72c57d95ce7d330c04f9bc2479966337f50ad23ce3a1d5bf9304a6fdc9"
CHAOS_ANGEL_SHA256 = "69ab42828aaf335d8caa90ce7e411674aefc30afa51f452d0d55001774a58afc"
CLOWN_CREW_SHA256 = "7fbfd6cee3ed1d90c78fd02c42d70bf47436176dc75ca793281ca71e1996b4fd"


def clown_crew_override(source: bytes | None) -> bytes | None:
    """Do not pay a tribute cost whose only remaining targets are its attached equips."""
    if source is None or sha256(source).hexdigest() != CLOWN_CREW_SHA256:
        return None
    text = source.decode("utf-8")
    old = """function s.rthcostfilter(c)
\treturn Duel.IsExistingTarget(Card.IsAbleToHand,0,LOCATION_ONFIELD,LOCATION_ONFIELD,1,c)
end"""
    new = """function s.rthremainingfilter(tc,rc)
\treturn tc:IsAbleToHand() and tc:GetEquipTarget()~=rc
end
function s.rthcostfilter(c)
\treturn Duel.IsExistingTarget(s.rthremainingfilter,0,LOCATION_ONFIELD,LOCATION_ONFIELD,1,c,c)
end"""
    assert text.count(old) == 1
    return text.replace(old, new).encode("utf-8")


def chaos_angel_override(source: bytes | None) -> bytes | None:
    """Evaluate prospective immunity without borrowing an unrelated chain's player/type."""
    if source is None or sha256(source).hexdigest() != CHAOS_ANGEL_SHA256:
        return None
    text = source.decode("utf-8")
    old = "\tlocal trig_p,trig_typ=Duel.GetChainInfo(0,CHAININFO_TRIGGERING_PLAYER,CHAININFO_TRIGGERING_TYPE)"
    new = """\tlocal ce,trig_p,trig_typ=Duel.GetChainInfo(0,CHAININFO_TRIGGERING_EFFECT,CHAININFO_TRIGGERING_PLAYER,CHAININFO_TRIGGERING_TYPE)
\tif ce~=te then
\t\ttrig_p=te:GetHandlerPlayer()
\t\tif Duel.GetReasonEffect()==te then trig_p=Duel.GetReasonPlayer() end
\t\ttrig_typ=te:GetActiveType()
\tend"""
    assert text.count(old) == 1
    return text.replace(old, new).encode("utf-8")


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


# Predicates are read-only, but may depend on the entire assigned group and remaining roles.
_FUSION_SELECTED_MEMO = """-- Failed states are local to one selected-material search, never to a duel or card.
local function ygorl_check_mix_rep_selected(tp,mg,sg,mustg,fc,sub,sub2,contact,sumtype,chkf,fun1,minc,maxc,...)
	local cards={}
	for tc in aux.Next(sg) do cards[#cards+1]=tc end
	local roles={} local role_count=0
	for _,f in ipairs({...}) do
		if not roles[f] then role_count=role_count+1 roles[f]=role_count end
	end
	local failed={} local entries=0
	local selected,cond
	selected=function(c,...) return Fusion.CheckMixRepTemplate(c,cond,...) end
	cond=function(tp,mg,sg,mustg,g,fc,sub,sub2,contact,sumtype,chkf,fun1,minc,maxc,...)
		local key={tostring(sub),tostring(minc),tostring(maxc)}
		for _,tc in ipairs(cards) do
			key[#key+1]=g:IsContains(tc) and "1" or "0"
			key[#key+1]=mustg:IsContains(tc) and "1" or "0"
		end
		for _,f in ipairs({...}) do key[#key+1]=tostring(roles[f]) end
		key=table.concat(key,",")
		if failed[key] then return false end
		local res
		if #g<#sg then
			res=sg:IsExists(selected,1,g,tp,mg,sg,mustg,g,fc,sub,sub2,contact,sumtype,chkf,fun1,minc,maxc,...)
		else
			res=Fusion.CheckSelectMixRep(tp,mg,sg,mustg,g,fc,sub,sub2,contact,sumtype,chkf,fun1,minc,maxc,...)
		end
		if not res and entries<4096 then failed[key]=true entries=entries+1 end
		return res
	end
	local g=Group.CreateGroup()
	return sg:IsExists(selected,1,nil,tp,mg,sg,mustg,g,fc,sub,sub2,contact,sumtype,chkf,fun1,minc,maxc,...)
end
"""


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
    text = text[:start] + body + text[end:]
    start = text.index("function Fusion.SelectMixRep(")
    end = text.index("function Fusion.AddProcMixRepUnfix(", start)
    body = text[start:end]
    old = "res=sg:IsExists(Fusion.CheckMixRepSelected,1,nil,tp,mg2,sg,mustg,g,fc,sub,sub2,contact,sumtype,chkf,fun1,minc,maxc,...)"
    assert body.count(old) == 1
    new = (
        "if #sg<6 then\n\t\t\t" + old + "\n\t\telse\n\t\t\t"
        "res=ygorl_check_mix_rep_selected(tp,mg2,sg,mustg,fc,sub,sub2,contact,sumtype,chkf,fun1,minc,maxc,...)"
        "\n\t\tend"
    )
    return (text[:start] + _FUSION_SELECTED_MEMO + body.replace(old, new) + text[end:]).encode("utf-8")
