"""Execute the actual polling payload locally to verify lease fencing and bounded retries."""

import contextlib
import hashlib
import importlib
import io
import json
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
r = importlib.import_module("precision_recover")


def fixture(tmp_path, failures=(), after=None):
    cls = r.controller_class(Path(__file__).parents[1].resolve())
    c = object.__new__(cls)
    c.root = tmp_path
    c.serial, c.generation = 100, 4
    c.deadline_unix = time.time() + 900
    c.guard = lambda **kw: None
    c.events, c.calls = [], []
    c.event = lambda stage, **kw: c.events.append((stage, kw))
    remote = tmp_path / "remote"
    remote.mkdir()
    (remote / "worker.pid").write_text(str(os.getpid()))
    (remote / "worker.log").write_text("healthy update")
    status = {
        "stage": "running",
        "generation": 4,
        "updated_unix": time.time(),
        "counters": {"updates": 158, "games": 200, "truncated": 0, "errors": 0},
    }
    (remote / "status.json").write_text(json.dumps(status))
    lease = tmp_path / "lease.json"
    lease.write_text(json.dumps({"job_id": "pending", "generation": 4, "expires_unix": time.time() + 240}))
    job = {
        "out": str(remote),
        "lease": str(lease),
        "job_id": "pending",
        "generation": 4,
        "identity": {"origin_counts": {"games": 0, "truncated": 0, "errors": 0}},
    }

    def cli(args, timeout):
        c.calls.append({"args": args, "timeout": timeout})
        number = len(c.calls)
        if after:
            after(number, job)
        if (number, "before") in failures:
            raise RuntimeError("colab exec failed; see retained-log")
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            exec(Path(args[-1]).read_text(), {})
        if (number, "after") in failures:
            raise RuntimeError("colab exec timed out; remote state is unknown")
        return buffer.getvalue()

    c.cli_call = cli
    return c, job


@pytest.mark.parametrize("lost", ["before", "after"])
def test_transient_poll_or_lost_reply_retries_same_fence_once(tmp_path, lost):
    c, job = fixture(tmp_path, [(1, lost)])
    result = c.poll_job("existing-vm", job)
    assert result["status"]["counters"]["updates"] == 158
    assert result["poll_fence"]["generation"] == 4
    assert [call["timeout"] for call in c.calls] == [45, 45, 45]
    assert all(call["args"][2] == "existing-vm" for call in c.calls)
    assert [stage for stage, _ in c.events] == ["poll_transport_error", "poll_retry"]


@pytest.mark.parametrize(
    "change",
    [
        {"generation": 3},
        {"job_id": "another-arm"},
        {"expires_unix": 0},
        {"expires_unix": float("nan")},
        {"expires_unix": float("inf")},
    ],
)
def test_initial_fence_or_expiry_rejection_does_not_renew_or_retry(tmp_path, change):
    c, job = fixture(tmp_path)
    lease = Path(job["lease"])
    value = json.loads(lease.read_text()) | change
    lease.write_text(json.dumps(value))
    before = lease.read_bytes()
    with pytest.raises(ValueError, match="poll fence rejected"):
        c.poll_job("vm", job)
    assert len(c.calls) == 1 and lease.read_bytes() == before


@pytest.mark.parametrize(
    "failure",
    ["failed", "dead", "expired", "wrong-generation", "errors", "truncations", "stale", "future", "nonfinite"],
)
def test_probe_never_retries_failure_dead_worker_or_bad_fence(tmp_path, failure):
    def change(number, job):
        if number != 2:
            return
        remote = Path(job["out"])
        state = json.loads((remote / "status.json").read_text())
        if failure == "failed":
            state.update(stage="failed", error="nonfinite")
        elif failure == "dead":
            (remote / "worker.pid").write_text("99999999")
        elif failure in ("expired", "wrong-generation"):
            path = Path(job["lease"])
            lease = json.loads(path.read_text())
            lease.update({"expires_unix": 0} if failure == "expired" else {"generation": 3})
            path.write_text(json.dumps(lease))
        elif failure == "errors":
            state["counters"]["errors"] = 1
        elif failure == "truncations":
            state["counters"]["truncated"] = 3
        elif failure == "stale":
            state["updated_unix"] = time.time() - 301
        elif failure == "future":
            state["updated_unix"] = time.time() + 300
        else:
            state["counters"]["games"] = float("nan")
        (remote / "status.json").write_text(json.dumps(state))

    c, job = fixture(tmp_path, [(1, "before")], change)
    with pytest.raises(ValueError):
        c.poll_job("vm", job)
    assert len(c.calls) == 2
    assert "poll_retry" not in [stage for stage, _ in c.events]


def test_successful_failed_worker_status_is_returned_without_retry(tmp_path):
    c, job = fixture(tmp_path)
    path = Path(job["out"]) / "status.json"
    path.write_text(json.dumps({"stage": "failed", "generation": 4, "error": "bad rollout"}))
    assert c.poll_job("vm", job)["status"]["stage"] == "failed"
    assert len(c.calls) == 1


def test_exhaustion_preserves_transport_failure_for_existing_cleanup(tmp_path):
    c, job = fixture(tmp_path, [(1, "before"), (3, "before")])
    with pytest.raises(RuntimeError, match="colab exec failed;"):
        c.poll_job("vm", job)
    assert len(c.calls) == 3
    assert len([stage for stage, _ in c.events if stage == "poll_retry"]) == 1


def test_probe_transport_failure_does_not_retry_unknown_worker(tmp_path):
    c, job = fixture(tmp_path, [(1, "before"), (2, "before")])
    with pytest.raises(RuntimeError, match="colab exec failed;"):
        c.poll_job("vm", job)
    assert len(c.calls) == 2


def test_optional_resource_probe_failure_does_not_discard_healthy_status(tmp_path, monkeypatch):
    original = Path.read_text

    def unavailable(path, *args, **kwargs):
        if str(path) == "/proc/loadavg":
            raise OSError("optional proc data unavailable")
        return original(path, *args, **kwargs)

    c, job = fixture(tmp_path, [(1, "before")])
    monkeypatch.setattr(Path, "read_text", unavailable)
    assert c.poll_job("vm", job)["status"]["stage"] == "running"
    probe = json.loads(next(tmp_path.glob("poll-probe-*.json")).read_text())
    assert probe["host_probe"] == {"read_error": "OSError"}
    assert len(c.calls) == 3


def test_original_deadline_and_short_lease_prevent_retry(tmp_path):
    c, job = fixture(tmp_path, [(1, "before")])
    c.deadline_unix = time.time() + 50
    with pytest.raises(ValueError, match="insufficient lease/deadline"):
        c.poll_job("vm", job)
    assert len(c.calls) == 2


@pytest.mark.parametrize("games,truncated", [(200, 2), (99, 2)])
def test_probe_preserves_original_truncation_health_threshold(tmp_path, games, truncated):
    c, job = fixture(tmp_path, [(1, "before")])
    path = Path(job["out"]) / "status.json"
    status = json.loads(path.read_text())
    status["counters"].update(games=games, truncated=truncated)
    path.write_text(json.dumps(status))
    assert c.poll_job("vm", job)["status"]["stage"] == "running"
    assert len(c.calls) == 3


def test_transient_poll_does_not_replace_healthy_worker_in_controller_loop(tmp_path, monkeypatch):
    def complete(number, job):
        if number == 3:
            remote = Path(job["out"])
            state = json.loads((remote / "status.json").read_text())
            state["stage"] = "complete"
            state["counters"]["updates"] = 160
            (remote / "status.json").write_text(json.dumps(state))
            (remote / "worker.pid").write_text("99999999")

    c, job = fixture(tmp_path, [(1, "before")], complete)
    c.registration = {"jobs": [{"job_id": "pending"}]}
    c.audit_path = tmp_path / "audit.json"
    c.audit_path.write_text("{}")
    (tmp_path / "identity.json").write_text("{}")
    (tmp_path / "STOP.json").write_text("{}")
    c.audit = {"stop_sha256": hashlib.sha256((tmp_path / "STOP.json").read_bytes()).hexdigest()}
    c.admission = lambda: {"verified": True}
    c.owned, allocations, releases = [], [], []

    def allocate():
        allocations.append("vm")
        c.owned.append("vm")
        return "vm"

    c.allocate = allocate
    c.setup = lambda session: None
    c.launch_job = lambda session, spec: job
    c.finish = lambda spec: None
    c.release = lambda session: releases.append(session)
    c.main()
    assert allocations == ["vm"] and releases == ["vm"]
    assert "transport_recovery" not in [stage for stage, _ in c.events]
    assert c.events[-1][0] == "training_complete"


def controlled_stop():
    events = [
        {
            "stage": "transport_recovery",
            "time": 1,
            "generation": 4,
            "error": "RuntimeError('colab exec failed; see cli-0788.log')",
            "diagnostic": {"status": {"stage": "running", "generation": 4, "counters": {"errors": 0, "truncated": 0}}},
        },
        {
            "stage": "transport_recovery",
            "time": 2,
            "generation": 5,
            "error": "RuntimeError('colab exec failed; see cli-0834.log')",
            "diagnostic": {
                "diagnostic_error": "RuntimeError('colab exec timed out; remote state is unknown')",
                "unverified_suffix_unknown": True,
            },
        },
    ]
    stop = {
        "source": "precision_recovery_transport_audit",
        "time": 3,
        "reason": "Stop repeated poll transport recovery before exhausting original allocation budget; retain all verified artifacts pending fenced poll retry amendment.",
        "transport_events": json.loads(json.dumps(events)),
    }
    failure = {"stage": "failed", "time": 4, "error": "RuntimeError('STOP.json requested study stop')"}
    audit = {"last_failure": failure, "transport_events": json.loads(json.dumps(events))}
    return stop, [*events, failure], audit


def test_only_exact_audited_controlled_transport_stop_is_admitted():
    stop, events, audit = controlled_stop()
    r.admit_transport_stop(stop, events, audit)
    events[0]["time"] = 0
    with pytest.raises(ValueError, match="transport events differ"):
        r.admit_transport_stop(stop, events, audit)


@pytest.mark.parametrize(
    "change", ["health-stop", "operator-stop", "diagnostic-failed", "health-failure", "event-unbound"]
)
def test_health_and_unrelated_operator_stops_cannot_use_transport_admission(change):
    stop, events, audit = controlled_stop()
    if change == "health-stop":
        stop["source"] = "precision_evaluate"
    elif change == "operator-stop":
        stop["reason"] = "stop for unrelated reasons"
    elif change == "diagnostic-failed":
        for collection in (events, stop["transport_events"], audit["transport_events"]):
            collection[0]["diagnostic"]["status"]["stage"] = "failed"
    elif change == "health-failure":
        events.insert(1, {"stage": "failed", "time": 1.5, "error": "nonfinite"})
    else:
        del audit["transport_events"]
    with pytest.raises(ValueError):
        r.admit_transport_stop(stop, events, audit)


def test_original_observer_transport_stop_still_requires_timeout_failure():
    stop = {"source": "precision_watch", "reason": "required controller service unexpectedly inactive: failed"}
    failure = {"stage": "failed", "error": "RuntimeError('colab exec timed out; remote state is unknown')"}
    r.admit_transport_stop(stop, [failure], {"last_failure": failure})
    failure["error"] = "nonfinite model"
    with pytest.raises(ValueError, match="not the audited transport failure"):
        r.admit_transport_stop(stop, [failure], {"last_failure": failure})
