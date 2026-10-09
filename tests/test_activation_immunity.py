"""Prospective activation immunity and the actual seed-2 cold training failure."""

import hashlib
import json
import random
from pathlib import Path

import pytest

from ygorl import _core, paths
from ygorl.engine import constants as C
from ygorl.engine.duel import default_cards, default_scripts, expand_seed
from ygorl.engine.script_patches import CHAOS_ANGEL_SHA256, chaos_angel_override


def test_chaos_angel_patch_is_content_pinned():
    original = _core.ScriptDirectory([str(p) for p in paths.script_directories()]).read("c22850702.lua")
    assert hashlib.sha256(original).hexdigest() == CHAOS_ANGEL_SHA256
    assert default_scripts().read("c22850702.lua") == chaos_angel_override(original) != original
    assert chaos_angel_override(original + b"\n-- unknown revision") is None
    assert chaos_angel_override(None) is None


def test_immunity_uses_matching_chain_or_prospective_effect():
    # Exercise the installed Lua predicate. Mock chain metadata to cover historical
    # controller/type preservation and unrelated chains, not just the nil-chain case.
    probe = """
    local e={GetHandlerPlayer=function() return 1 end}
    local te={IsActivated=function() return true end,
              GetHandlerPlayer=function() return 0 end,
              GetActiveType=function() return TYPE_MONSTER end}
    Duel.GetChainInfo=function() end
    Duel.GetReasonEffect=function() end
    local immune=c22850702.immval
    assert(immune(e,te),'prospective opponent monster must be immune')
    te.GetHandlerPlayer=function() return 1 end
    assert(not immune(e,te),'own monster must not be immune')
    te.GetHandlerPlayer=function() return 0 end
    te.GetActiveType=function() return TYPE_SPELL|TYPE_PENDULUM end
    assert(not immune(e,te),'pendulum spell activation must not be immune')
    te.GetActiveType=function() return TYPE_MONSTER end
    te.IsActivated=function() return false end
    assert(not immune(e,te),'continuous monster effect must not be immune')
    te.IsActivated=function() return true end
    Duel.GetChainInfo=function() return {},1,TYPE_SPELL end
    assert(immune(e,te),'unrelated own spell chain must not hide opponent monster immunity')
    te.GetActiveType=function() return TYPE_SPELL end
    Duel.GetChainInfo=function() return {},0,TYPE_MONSTER end
    assert(not immune(e,te),'unrelated opponent monster chain must not block spell')
    Duel.GetChainInfo=function() return te,0,TYPE_MONSTER end
    te.GetHandlerPlayer=function() return 1 end
    assert(immune(e,te),'actual chain retains player/type after control or type change')
    Duel.GetChainInfo=function() return te,1,TYPE_SPELL end
    te.GetHandlerPlayer=function() return 0 end
    te.GetActiveType=function() return TYPE_MONSTER end
    assert(not immune(e,te),'actual own spell chain retains player/type')
    Duel.GetChainInfo=function() end
    Duel.GetReasonEffect=function() return te end
    Duel.GetReasonPlayer=function() return 1 end
    assert(not immune(e,te),'prospective activating player can differ from handler controller')
    Debug.Message('immunity contexts passed')
    """
    scripts = _core.ScriptDirectory(
        [str(p) for p in paths.script_directories()],
        {
            "c22850702.lua": default_scripts().read("c22850702.lua"),
            "probe.lua": probe,
        },
    )
    core = _core.Duel(
        expand_seed(20261009), C.DUEL_MODE_MR5, (8000, 0, 0), (8000, 0, 0), default_cards().to_core(), scripts
    )
    try:
        assert core.load_script("constant.lua") and core.load_script("utility.lua")
        core.new_card(1, 0, 22850702, 1, C.LOCATION_MZONE, 0, C.POS_FACEUP_ATTACK)
        assert core.load_script("probe.lua"), core.pop_logs()
        assert core.pop_logs() == [(1, b"immunity contexts passed")]
    finally:
        core.close()


def start_replay(patched):
    case = json.loads((Path(__file__).parent / "data/verte-immunity-replay.json").read_text())
    scripts = default_scripts()
    if not patched:
        scripts = _core.ScriptDirectory(
            [str(p) for p in paths.script_directories()],
            {name: scripts.read(name) for name in ("proc_synchro.lua", "proc_fusion.lua")},
        )
    host = _core.HostDuel(default_cards().to_core(), scripts, [])
    host.start(case["core_seed"], case["rule_flags"], (8000, 5, 1), (8000, 5, 1), case["loaded_decks"], 200, 4000)
    return case, host


def test_actual_failure_removes_only_invalid_activation_before_cost():
    case, old = start_replay(False)
    _, fixed = start_replay(True)
    for action in case["actions"]:
        assert old.actions() == fixed.actions()
        a, b = old.observe(), fixed.observe()
        assert all(a[k].tobytes() == b[k].tobytes() for k in a)
        old.act(action)
        fixed.act(action)
    assert len(case["actions"]) == 410
    assert old.result()["responses"] == fixed.result()["responses"]
    removed = [a for a in old.actions() if a not in fixed.actions()]
    assert len(removed) == 1 and removed[0][3] == 70369116 and removed[0][5] == 73787366178817
    assert all(a in old.actions() for a in fixed.actions())
    assert fixed.result()["lp"] == (6900, 5100)
    old.act(old.actions().index(removed[0]))
    assert old.done() and old.result()["reason"] == "error"
    assert "c70369116.lua" in old.result()["error"] and "nil value" in old.result()["error"]
    assert not fixed.done() and not fixed.result()["script_errors"]


@pytest.mark.parametrize("seed", range(3))
def test_failed_game_can_finish_after_corrected_legality(seed):
    case, host = start_replay(True)
    for action in case["actions"]:
        host.act(action)
    rng = random.Random(seed)
    while not host.done():
        mask = host.observe()["action_mask"]
        host.act(rng.choice([i for i in range(len(host.actions())) if mask[i]]))
    result = host.result()
    assert result["reason"] == "win", result
    assert not result["error"] and not result["script_errors"] and not result["retries"]


@pytest.mark.parametrize("late_immunity", [False, True])
def test_legal_copied_fusion_rechecks_materials_at_resolution(late_immunity):
    """Inject immunity becoming effective after legal activation, before its operation.

    This isolates a legitimate state change at the resolution boundary; the injection
    is not a claim that Chaos Angel naturally changes immunity during this replay.
    """
    from ygorl.cards.ydk import Deck
    from ygorl.engine.duel import Duel, DuelConfig, DuelSession

    case = json.loads((Path(__file__).parent / "data/verte-immunity-replay.json").read_text())
    overrides = {n: default_scripts().read(n) for n in ("proc_synchro.lua", "proc_fusion.lua", "c22850702.lua")}
    overrides["c22850702.lua"] += b"""
local original=s.immval
function s.immval(e,te) return ygorl_enable_immunity and original(e,te) or false end
"""
    overrides["c70369116.lua"] = (
        default_scripts().read("c70369116.lua")
        + f"""
local original=s.operation
function s.operation(e,tp,...)
    ygorl_enable_immunity={str(late_immunity).lower()}
    local before=Duel.GetFieldGroupCount(tp,LOCATION_EXTRA,0)
    original(e,tp,...)
    assert(Duel.GetFieldGroupCount(tp,LOCATION_EXTRA,0)==before-{0 if late_immunity else 1},
           'unexpected fusion resolution')
end
""".encode()
    )
    scripts = _core.ScriptDirectory([str(p) for p in paths.script_directories()], overrides)
    decks = [Deck(main=m, extra=e) for m, e in case["loaded_decks"]]
    session = DuelSession(
        Duel(
            0,
            None,
            *decks,
            core_seed=case["core_seed"],
            scripts=scripts,
            config=DuelConfig(rule_flags=case["rule_flags"], shuffle_decks=False, max_decisions=4000),
        )
    )
    try:
        for index in case["actions"]:
            session.act(index)
        index = next(
            i
            for i, a in enumerate(session.point.actions)
            if a.card and a.card.code == 70369116 and a.description == 73787366178817
        )
        session.act(index)
        fusion_selection = False
        for _ in range(20):
            assert session.point is not None
            if type(session.point.decision).__name__ == "SelectIdleCmd":
                break
            fusion_selection |= any(a.card and a.card.code == 13243124 for a in session.point.actions)
            index = next((i for i, a in enumerate(session.point.actions) if a.kind not in ("cancel", "finish")), 0)
            session.act(index)
        assert type(session.point.decision).__name__ == "SelectIdleCmd"
        assert session.point.index == (416 if late_immunity else 421)
        assert fusion_selection == (not late_immunity)
        assert session.point.lp == (4900, 5100)  # legal activation paid its cost in both cases
        assert not session.tracker.result.script_errors and not session.tracker.result.error
    finally:
        session.close()


def test_python_host_stops_on_same_lua_error_as_native_host():
    from ygorl.cards.ydk import Deck
    from ygorl.engine.duel import Duel, DuelConfig, DuelSession

    case, host = start_replay(False)
    scripts = _core.ScriptDirectory(
        [str(p) for p in paths.script_directories()],
        {n: default_scripts().read(n) for n in ("proc_synchro.lua", "proc_fusion.lua")},
    )
    decks = [Deck(main=m, extra=e) for m, e in case["loaded_decks"]]
    session = DuelSession(
        Duel(
            0,
            None,
            *decks,
            core_seed=case["core_seed"],
            scripts=scripts,
            config=DuelConfig(rule_flags=case["rule_flags"], shuffle_decks=False, max_decisions=4000),
        )
    )
    try:
        for index in case["actions"]:
            host.act(index)
            session.act(index)
        index = next(i for i, a in enumerate(host.actions()) if a[3] == 70369116 and a[5] == 73787366178817)
        host.act(index)
        session.act(index)
        assert host.done() and session.done and session.point is None
        native, python = host.result(), session.result()
        for key in ("reason", "winner", "turns", "decisions", "lp", "error", "script_errors", "retries", "responses"):
            assert native[key] == getattr(python, key), key
    finally:
        session.close()
