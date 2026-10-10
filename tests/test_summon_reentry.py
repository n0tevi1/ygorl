"""Reentry restriction is opt-in and must not infer progress from tensors alone."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from summon_reentry import SummonReentryGuard
from tests.test_selection_history import setup, step
from ygorl.engine import messages as M
from ygorl.engine.actions import Action


def point(index, idle=True, **changes):
    decision = (M.SelectIdleCmd(0, (), (), (), (), (), (), False, True, False) if idle
                else M.SelectUnselectCard(0, False, True, 1, 2, (), ()))  # fmt: skip
    actions = [Action("spsummon", 0), Action("end_phase")] if idle else [Action("select"), Action("cancel")]
    return SimpleNamespace(**(dict(index=index, player=0, turn=3, phase=4, lp=(8000, 8000),
                                  decision=decision, events=(decision,), actions=actions) | changes))  # fmt: skip


def aborted(guard, obs, events=None):
    root = point(0)
    guard.observe(root)
    assert guard.restrict(root, obs)[0] is obs
    guard.on_decision(root, 0)
    selection = point(1, False)
    if events is not None:
        selection.events = events
    guard.observe(selection)
    assert guard.restrict(selection, obs)[0] is obs  # never remove cancel
    guard.on_decision(selection, 1)
    root = point(2)
    guard.observe(root)
    return guard.restrict(root, obs)


def test_only_retry_is_removed_and_inputs_are_unchanged():
    obs = {"action_mask": np.array([True, True, False]), "cards": np.array([123])}
    after, info = aborted(SummonReentryGuard(), obs)
    assert info == {"aborted_attempt": 1, "blocked": [0], "sole_exit_preserved": 0}
    np.testing.assert_array_equal(after["action_mask"], [False, True, False])
    np.testing.assert_array_equal(obs["action_mask"], [True, True, False])
    assert after["cards"] is obs["cards"]


def test_preserve_sole_exit_and_retry_budget():
    obs = {"action_mask": np.array([True, False])}
    after, info = aborted(SummonReentryGuard(), obs)
    assert after is obs and info["sole_exit_preserved"] == 1
    obs = {"action_mask": np.array([True, True])}
    assert aborted(SummonReentryGuard(4), obs)[1]["blocked"] == []
    with pytest.raises(ValueError):
        SummonReentryGuard(0)


@pytest.mark.parametrize("events", [(M.Damage(0, 1),), (M.Retry(),), (M.Waiting(),), (M.NewPhase(4),)])
def test_real_or_uncertain_progress_invalidates_even_with_identical_observation(events):
    obs = {"action_mask": np.array([True, True])}
    assert aborted(SummonReentryGuard(), obs, events)[0] is obs


@pytest.mark.parametrize("change", [dict(player=1), dict(turn=4), dict(phase=8), dict(lp=(7999, 8000))])
def test_context_changes_clear_history(change):
    obs = {"action_mask": np.array([True, True])}
    guard = SummonReentryGuard()
    aborted(guard, obs)
    root = point(3, **change)
    guard.observe(root)
    assert guard.restrict(root, obs)[0] is obs


def test_changed_observation_other_action_and_finished_selection_allow_retry():
    obs = {"action_mask": np.array([True, True]), "cards": np.array([1])}
    guard = SummonReentryGuard()
    aborted(guard, obs)
    root = point(3)
    guard.observe(root)
    changed = {**obs, "cards": np.array([2])}
    assert guard.restrict(root, changed)[0] is changed
    # A different idle action is not assumed to be a no-op.
    guard.on_decision(root, 1)
    root = point(4)
    guard.observe(root)
    assert guard.restrict(root, obs)[0] is obs
    guard.on_decision(root, 0)
    selection = point(5, False, actions=[Action("finish")])
    guard.observe(selection)
    guard.on_decision(selection, 0)
    root = point(6)
    guard.observe(root)
    assert guard.restrict(root, obs)[0] is obs


def test_canceling_a_target_is_not_a_canceled_summon():
    obs = {"action_mask": np.array([True, True])}
    guard = SummonReentryGuard()
    root = point(0)
    guard.observe(root)
    guard.restrict(root, obs)
    guard.on_decision(root, 0)
    target = point(1, False, decision=M.SelectCard(0, True, 1, 1, ()))
    guard.observe(target)
    guard.restrict(target, obs)
    guard.on_decision(target, 1)
    root = point(2)
    guard.observe(root)
    assert guard.restrict(root, obs)[0] is obs


def test_recorded_summon_can_complete_with_the_remaining_material():
    case = json.loads((Path(__file__).parent / "data/summon_reentry.json").read_text())
    session, host, observer, _ = setup(case, enabled=False)
    guard = SummonReentryGuard()
    try:
        for index in case["actions"][:236] + [0]:
            p = session.point
            guard.observe(p)
            if p.player == 0:
                _, info = guard.restrict(p, host.observe())
                assert not info["blocked"]
            if p.index == 236:
                assert p.actions[index].kind == "select" and p.actions[index].card.code == 33854624
            guard.on_decision(p, index)
            step(session, host, observer, index)
        assert isinstance(session.point.decision, M.SelectPlace)
        assert host.result()["responses"] == session.tracker.result.responses
    finally:
        session.close()


@pytest.mark.parametrize("budget", [1, 4])
def test_recorded_chaos_angel_abort_reentry_native_reference_parity(budget):
    case = json.loads((Path(__file__).parent / "data/summon_reentry.json").read_text())
    session, host, observer, _ = setup(case, enabled=False)
    native_guard, reference_guard = SummonReentryGuard(budget), SummonReentryGuard(budget)
    roots = []
    try:
        for index in case["actions"]:
            p = session.point
            observer.observe(p)
            native, reference = host.observe(), observer.encode(p, session.core)
            for key in reference:
                np.testing.assert_array_equal(native[key], reference[key])
            # Hook sees BOTH players, but restriction applies only to candidate.
            for guard in (native_guard, reference_guard):
                guard.observe(p)
            if p.player == 0:
                n, info = native_guard.restrict(p, native)
                r, other = reference_guard.restrict(p, reference)
                assert info == other
                np.testing.assert_array_equal(n["action_mask"], r["action_mask"])
                assert np.all(n["action_mask"] <= native["action_mask"]) and np.any(n["action_mask"])
                if p.index in (233, 240, 247, 254, 261):
                    roots.append((p.index, info["blocked"]))
                    assert session.tracker._selection_cancels == 0
                if p.index == 233 + budget * 7:
                    assert info["blocked"] == [1]
                    assert p.actions[1].card.code == 22850702
                    assert n["action_mask"][next(i for i, a in enumerate(p.actions) if a.kind == "end_phase")]
                    break
                assert n["action_mask"][index]
            for guard in (native_guard, reference_guard):
                guard.on_decision(p, index)
            step(session, host, observer, index)
        else:
            # budget4 restriction is at the next point after the 261-action prefix.
            p = session.point
            observer.observe(p)
            native_guard.observe(p)
            _, info = native_guard.restrict(p, host.observe())
            assert p.index == 261 and info["blocked"] == [1]
        assert roots[0] == (233, [])
        assert host.result()["responses"] == session.tracker.result.responses
        assert not session.tracker.result.script_errors
        # A no-event return WITHOUT cancel does not constitute an aborted attempt.
        guard = SummonReentryGuard()
        original = replace(p, index=0, events=())
        guard.observe(original)
        guard.restrict(original, host.observe())
        guard.on_decision(original, 1)
        back = replace(original, index=1)
        guard.observe(back)
        assert not guard.restrict(back, host.observe())[1]["blocked"]
    finally:
        session.close()
