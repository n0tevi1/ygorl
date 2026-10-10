"""A tribute must leave a target after its attached equips leave by game rules."""

import hashlib
import json
import random
from pathlib import Path

import pytest

from ygorl import _core, paths
from ygorl.cards.ydk import Deck
from ygorl.engine.duel import Duel, DuelConfig, DuelSession, default_cards, default_scripts
from ygorl.engine.script_patches import CLOWN_CREW_SHA256, clown_crew_override


CASE = json.loads((Path(__file__).parent / "data/clown-release-replay.json").read_text())


def scripts(patched=True, late_removal=None):
    original = _core.ScriptDirectory([str(p) for p in paths.script_directories()])
    overrides = {n: default_scripts().read(n) for n in ("proc_synchro.lua", "proc_fusion.lua", "c22850702.lua")}
    overrides["c83232904.lua"] = default_scripts().read("c83232904.lua") if patched else original.read("c83232904.lua")
    if late_removal is not None:
        overrides["c83232904.lua"] += f"""
local original=s.rthop
function s.rthop(e,tp,...)
    if Duel.GetTurnCount()~=6 then return original(e,tp,...) end
    local tc=Duel.GetFirstTarget()
    assert(tc and tc:IsOnField(),'activation must have a real field target')
    if {str(late_removal).lower()} then Duel.Remove(tc,POS_FACEUP,REASON_EFFECT) end
    original(e,tp,...)
    assert(tc:IsLocation({"LOCATION_REMOVED" if late_removal else "LOCATION_HAND"}),'unexpected target destination')
    Debug.Message('clown resolution passed')
end
""".encode()
    return _core.ScriptDirectory([str(p) for p in paths.script_directories()], overrides)


def host(patched=True):
    h = _core.HostDuel(default_cards().to_core(), scripts(patched), [])
    h.start(CASE["core_seed"], CASE["rule_flags"], (8000, 5, 1), (8000, 5, 1), CASE["loaded_decks"], 200, 4000)
    return h


def session(patched=True, late_removal=None):
    decks = [Deck(main=m, extra=e) for m, e in CASE["loaded_decks"]]
    return DuelSession(
        Duel(
            0,
            None,
            *decks,
            core_seed=CASE["core_seed"],
            scripts=scripts(patched, late_removal),
            config=DuelConfig(rule_flags=CASE["rule_flags"], shuffle_decks=False, max_decisions=4000),
        )
    )


def test_patch_is_content_pinned():
    source = _core.ScriptDirectory([str(p) for p in paths.script_directories()]).read("c83232904.lua")
    assert hashlib.sha256(source).hexdigest() == CLOWN_CREW_SHA256
    assert default_scripts().read("c83232904.lua") == clown_crew_override(source) != source
    assert clown_crew_override(source + b"\n-- unknown") is None
    assert clown_crew_override(None) is None


def test_actual_failure_removes_only_unsafe_tribute():
    old, fixed = host(False), host(True)
    for a in CASE["actions"][:392]:
        assert old.actions() == fixed.actions()
        x, y = old.observe(), fixed.observe()
        assert all(x[k].tobytes() == y[k].tobytes() for k in x)
        old.act(a)
        fixed.act(a)
    assert old.result()["responses"] == fixed.result()["responses"]
    unsafe = [a for a in old.actions() if a not in fixed.actions()]
    assert len(unsafe) == 1 and unsafe[0][3] == 70088809
    assert [a[3] for a in fixed.actions()] == [82159583, 82159583, 91800273]
    assert all(a in old.actions() for a in fixed.actions())
    for a in CASE["actions"][392:]:
        old.act(a)
    assert old.done() and old.result()["reason"] == "error"
    assert "c83232904.lua" in old.result()["error"] and "nil value" in old.result()["error"]
    assert not fixed.done() and not fixed.result()["script_errors"]


def test_original_error_matches_python_host():
    native = host(False)
    with_session = session(False)
    try:
        for a in CASE["actions"]:
            native.act(a)
            with_session.act(a)
        assert native.done() and with_session.done
        a, b = native.result(), with_session.result()
        for key in ("reason", "winner", "turns", "decisions", "lp", "error", "script_errors", "retries", "responses"):
            assert a[key] == getattr(b, key), key
    finally:
        with_session.close()


@pytest.mark.parametrize("tribute", range(3))
def test_all_remaining_tributes_finish_healthy(tribute):
    h = host()
    for a in CASE["actions"][:392]:
        h.act(a)
    h.act(tribute)
    rng = random.Random(tribute)
    while not h.done():
        mask = h.observe()["action_mask"]
        h.act(rng.choice([i for i in range(len(h.actions())) if mask[i]]))
    result = h.result()
    assert result["reason"] == "win", result
    assert not any(result[k] for k in ("error", "script_errors", "retries", "unknown_messages"))


@pytest.mark.parametrize("late_removal", [False, True])
def test_legal_target_can_leave_after_activation(late_removal):
    # Deliberately move the real target at the operation boundary, after payment
    # and targeting. This is a resolution check, not a naturally occurring chain.
    s = session(late_removal=late_removal)
    try:
        for a in CASE["actions"][:392]:
            s.act(a)
        s.act(0)  # Legal hand tribute leaves both field cards.
        assert type(s.point.decision).__name__ == "SelectCard"
        assert len(s.point.actions) == 2
        s.act(next(i for i, a in enumerate(s.point.actions) if a.card.code == 70088809))
        events = []
        for _ in range(10):
            assert s.point is not None
            events.extend(s.point.events)
            if type(s.point.decision).__name__ == "SelectIdleCmd":
                break
            s.act(next((i for i, a in enumerate(s.point.actions) if a.kind == "pass"), 0))
        assert type(s.point.decision).__name__ == "SelectIdleCmd"
        assert not s.tracker.result.error and not s.tracker.result.script_errors
        from ygorl.engine import constants as C
        from ygorl.engine import messages as M

        moves = [e for e in events if isinstance(e, M.Move) and e.code == 70088809]
        assert any(e.current.location == (C.LOCATION_REMOVED if late_removal else C.LOCATION_HAND) for e in moves)
    finally:
        s.close()


@pytest.mark.parametrize("controller", [0, 1])
@pytest.mark.parametrize("independent_target", [False, True])
def test_field_tribute_requires_a_target_not_equipped_to_it(controller, independent_target):
    from ygorl.engine import constants as C
    from ygorl.engine.duel import expand_seed

    probe = f"""
-- No active reason effect in this board probe: enumerate matching cards.
-- Real activation/targetability is covered by the complete recorded replay above.
Duel.IsExistingTarget=Duel.IsExistingMatchingCard
local rc=Duel.GetFieldCard({controller},LOCATION_MZONE,0)
local ec=Duel.GetFieldCard({1 - controller},LOCATION_SZONE,0)
assert(Debug.PreEquip(ec,rc))
assert(ec:GetEquipTarget()==rc)
assert(c83232904.rthcostfilter(rc)=={str(independent_target).lower()},'unsafe tribute eligibility')
local hand=Duel.GetFieldGroup({controller},LOCATION_HAND,0):GetFirst()
assert(c83232904.rthcostfilter(hand),'hand tribute must leave field targets')
Debug.Message('cost contexts passed')
"""
    overrides = {"c83232904.lua": default_scripts().read("c83232904.lua"), "probe.lua": probe}
    lua = _core.ScriptDirectory([str(p) for p in paths.script_directories()], overrides)
    core = _core.Duel(
        expand_seed(20261010), C.DUEL_MODE_MR5, (8000, 0, 0), (8000, 0, 0), default_cards().to_core(), lua
    )
    try:
        assert core.load_script("constant.lua") and core.load_script("utility.lua")
        core.new_card(controller, 0, 83232904, controller, C.LOCATION_GRAVE, 0, C.POS_FACEUP_ATTACK)
        core.new_card(controller, 0, 70088809, controller, C.LOCATION_MZONE, 0, C.POS_FACEUP_ATTACK)
        core.new_card(1 - controller, 0, 82119326, 1 - controller, C.LOCATION_SZONE, 0, C.POS_FACEUP_ATTACK)
        core.new_card(controller, 0, 82159583, controller, C.LOCATION_HAND, 0, C.POS_FACEDOWN_DEFENSE)
        if independent_target:
            core.new_card(1 - controller, 0, 82159583, 1 - controller, C.LOCATION_MZONE, 0, C.POS_FACEUP_ATTACK)
        assert core.load_script("probe.lua"), core.pop_logs()
        assert core.pop_logs() == [(1, b"cost contexts passed")]
    finally:
        core.close()
