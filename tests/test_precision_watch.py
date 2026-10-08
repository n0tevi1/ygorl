"""The observer separates useful speed evidence from fewer accepted updates."""

import importlib
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
watch = importlib.import_module("precision_watch")


def row(update=129, seconds=20, minibatches=16, evaluated=16, early=0, games=110, truncated=1):
    return {
        "rows": 16384,
        "minibatches": minibatches,
        "evaluated_minibatches": evaluated,
        "early_stop": early,
        "update_s": seconds,
        "collect_s": 10,
        "step_s": seconds + 10,
        "errors": 0,
        "total": {"updates": update, "games": games, "truncated": truncated},
    }


def test_health_is_cumulative_since_original_start_not_last_recovery():
    result = watch.aggregate([row(games=300, truncated=7)], {"games": 100, "truncated": 2})
    assert result["health"] == {"games": 200, "truncated": 5, "errors": 0, "truncation_fraction": 0.025}
    units = {role: {"ActiveState": "active"} for role in ("controller", "evaluator")}
    assert "unhealthy" in watch.stop_reason({}, {}, units, {"arm": result}, [], 0)


def test_startup_grace_and_normal_completed_service_exit():
    inactive = {role: {"ActiveState": "inactive"} for role in ("controller", "evaluator")}
    assert watch.stop_reason({}, {}, inactive, {}, [], 119) is None
    assert "unexpectedly inactive" in watch.stop_reason({}, {}, inactive, {}, [], 120)
    assert watch.stop_reason({}, {"status": "complete"}, inactive, {}, list(range(6)), 1000) is None
    assert "controller reported failed" == watch.stop_reason({"stage": "failed"}, {}, inactive, {}, [], 0)


def test_less_work_is_not_counted_as_a_matched_precision_speedup():
    arms = {}
    origin = {"games": 100, "truncated": 1}
    for seed in range(3):
        arms[f"seed-{seed}-fp32"] = watch.aggregate([row(seconds=20)], origin)
        # Twice as fast because half the minibatches were accepted: unmatched, not a speed win.
        arms[f"seed-{seed}-bf16"] = watch.aggregate([row(seconds=10, minibatches=8, evaluated=9, early=1)], origin)
    report = watch.efficiency_report(arms, [], 100, 500)
    assert report["matched_work_strata"] == []
    assert report["descriptive_at_least_10_percent"] is False
    assert report["auto_promote"] is False
    for seed in range(3):
        arms[f"seed-{seed}-bf16"] = watch.aggregate([row(seconds=17)], origin)
    report = watch.efficiency_report(arms, [], 100, 500)
    assert report["median_of_stratum_time_reductions"] == pytest.approx(0.15)
    assert report["descriptive_at_least_10_percent"] is True
    assert report["costs"]["checkpoint_transfer_seconds"] is None


def test_report_names_only_measured_intervals_and_rejects_nonfinite():
    origin = {"games": 100, "truncated": 1}
    arms = {f"seed-{seed}-{arm}": watch.aggregate([row()], origin) for seed in range(3) for arm in ("fp32", "bf16")}
    events = [
        {"stage": "allocated", "time": 100, "session": "s"},
        {"stage": "launched", "time": 120, "session": "s"},
        {"stage": "preempted", "time": 150, "session": "s"},
        {"stage": "allocated", "time": 170, "session": "t"},
        {"stage": "launched", "time": 200, "session": "t"},
    ]
    result = watch.efficiency_report(arms, events, 90, 500)
    assert result["costs"]["allocation_complete_to_first_launch_seconds"] == [20, 30]
    assert result["costs"]["preemption_detected_to_relaunch_seconds"] == [50]
    with pytest.raises(ValueError, match="nonfinite"):
        watch.aggregate([row(seconds=float("nan"))], origin)


def test_cached_validation_still_rejects_later_metric_changes(tmp_path, monkeypatch):
    observer = object.__new__(watch.Watch)
    observer.verified = {}
    for name in ("checkpoint.pt", "manifest.json", "precision-manifest.json", "games.jsonl.gz"):
        (tmp_path / name).write_bytes(b"fixture")
    (tmp_path / "metrics.jsonl").write_text(json.dumps(row()) + "\n")
    monkeypatch.setattr(watch, "validate_precision", lambda *args: None)
    digest = watch.sha(tmp_path / "checkpoint.pt")
    assert observer.verified_rows("arm", tmp_path, {}, digest) == [row()]
    (tmp_path / "metrics.jsonl").write_text(json.dumps(row(seconds=1)) + "\n")
    with pytest.raises(ValueError, match="immutable snapshot changed"):
        observer.verified_rows("arm", tmp_path, {}, digest)


def test_monitor_stop_preserves_existing_evaluator_reason(tmp_path):
    original = {"owner": "precision_evaluate", "reason": "bad evaluation"}
    watch.atomic(tmp_path / "STOP.json", original)
    watch.request_stop(tmp_path, "inactive controller")
    assert watch.read(tmp_path / "STOP.json") == original
