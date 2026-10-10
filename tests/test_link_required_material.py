"""Required Link materials are not removable; other selected materials still are."""

import hashlib
import json
from pathlib import Path

from ygorl import _core, paths
from ygorl.cards.ydk import Deck
from ygorl.engine.duel import Duel, DuelConfig, DuelSession, default_cards, default_scripts, expand_seed
from ygorl.engine.script_patches import LINK_SHA256, link_override


CASES = json.loads((Path(__file__).parent / "data/link-required-material-replays.json").read_text())


def scripts(patched):
    raw = _core.ScriptDirectory([str(p) for p in paths.script_directories()])
    names = ("proc_link.lua", "proc_synchro.lua", "proc_fusion.lua", "c22850702.lua", "c83232904.lua")
    overrides = {name: default_scripts().read(name) for name in names}
    if not patched:
        overrides["proc_link.lua"] = raw.read("proc_link.lua")
    return _core.ScriptDirectory([str(p) for p in paths.script_directories()], overrides)


def session(case, patched):
    return DuelSession(
        Duel(
            case["seed"],
            None,
            *[Deck(**case["decks"][x]) for x in ("a", "b")],
            first=case["first"],
            scripts=scripts(patched),
            config=DuelConfig(rule_flags=case["rule_flags"], shuffle_decks=False),
        )
    )


def test_link_patch_is_content_pinned():
    original = _core.ScriptDirectory([str(p) for p in paths.script_directories()]).read("proc_link.lua")
    assert hashlib.sha256(original).hexdigest() == LINK_SHA256
    assert default_scripts().read("proc_link.lua") == link_override(original) != original
    assert link_override(original + b"\n-- upstream drift") is None
    assert link_override(None) is None


def test_actual_required_material_noop_removed_without_changing_completion():
    case = CASES[0]
    old, fixed = session(case, False), session(case, True)
    try:
        for choice in case["actions"][: case["stop"]]:
            assert old.point.actions == fixed.point.actions
            assert old.point.decision == fixed.point.decision
            old.act(choice)
            fixed.act(choice)
        assert old.tracker.result.responses == fixed.tracker.result.responses
        decision = old.point.decision
        assert [a.kind for a in old.point.actions] == ["select", "unselect"]
        assert old.point.actions[1].card.code == 65741786  # required I:P
        assert fixed.point.actions == old.point.actions[:1]
        assert fixed.point.actions[0].card.code == 19899073
        for _ in range(10):
            old.act(1)
            assert old.point.decision == decision  # ignored, not an engine retry
        old.act(0)
        fixed.act(0)
        for choice in case["actions"][case["stop"] + 11 :]:
            assert old.point.actions == fixed.point.actions
            old.act(choice)
            fixed.act(choice)
        a, b = old.result(), fixed.result()
        assert a.reason == b.reason == "win"
        assert (a.winner, a.turns, a.lp) == (b.winner, b.turns, b.lp)
        assert a.decisions == b.decisions + 10
        assert a.responses[case["stop"] + 10 :] == b.responses[case["stop"] :]
        for result in (a, b):
            assert not any(getattr(result, k) for k in ("error", "script_errors", "retries", "unknown_messages"))
    finally:
        old.close()
        fixed.close()


def test_optional_material_deselection_remains_available_and_changes_group():
    case = CASES[1]
    old, fixed = session(case, False), session(case, True)
    try:
        for choice in case["actions"][: case["stop"]]:
            assert old.point.actions == fixed.point.actions
            old.act(choice)
            fixed.act(choice)
        before = fixed.point.decision
        choice = next(i for i, a in enumerate(fixed.point.actions) if a.kind == "unselect")
        assert old.point.actions == fixed.point.actions
        old.act(choice)
        fixed.act(choice)
        assert fixed.point.decision != before
        assert old.point.decision == fixed.point.decision
        assert old.tracker.result.responses == fixed.tracker.result.responses
    finally:
        old.close()
        fixed.close()


def test_native_host_uses_same_required_material_fix():
    case = CASES[0]
    decks = [(case["decks"][x]["main"], case["decks"][x]["extra"]) for x in ("a", "b")]
    host = _core.HostDuel(default_cards().to_core(), scripts(True), [])
    host.start(expand_seed(case["seed"]), case["rule_flags"], (8000, 5, 1), (8000, 5, 1), decks, 200, 6000)
    for choice in case["actions"][: case["stop"]]:
        host.act(choice)
    assert len(host.actions()) == 1 and host.actions()[0][3] == 19899073
    host.act(0)
    for choice in case["actions"][case["stop"] + 11 :]:
        host.act(choice)
    assert host.done()
    result = host.result()
    assert result["reason"] == "win" and result["winner"] == 0
    assert not any(result[k] for k in ("error", "script_errors", "retries", "unknown_messages"))
