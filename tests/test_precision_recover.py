"""Controller restart admission and recovery preserve budgets, work and ownership order."""

import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
r = importlib.import_module("precision_recover")


def test_budget_keeps_original_clock_balance_and_allocations():
    identity = {"deadline_unix": 1000, "initial_balance": 251.13}
    registration = {"max_units": 12, "max_allocations": 8}
    value = r.remaining_budget(identity, registration, 248.90, 2, 900)
    assert value["seconds"] == 100 and value["units"] == pytest.approx(9.77) and value["allocations"] == 6
    for balance, count, now in [(239, 2, 900), (248.9, 8, 900), (248.9, 2, 1000)]:
        with pytest.raises(ValueError, match="budget exhausted"):
            r.remaining_budget(identity, registration, balance, count, now)


def test_serials_include_partial_downloads_and_never_reuse_worker_generation(tmp_path):
    for name in [
        "cli-0742.log",
        "rpc-0740.py",
        "upload-801-000.part",
        "job-3.json",
        "remote-2.json",
        "durable/job/download-00820/partial",
    ]:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    assert r.high_water(tmp_path) == (820, 4)


def make_controller(tmp_path, monkeypatch):
    cls = r.controller_class(Path(__file__).parents[1].resolve())
    c = object.__new__(cls)
    c.root = tmp_path
    c.audit_path = tmp_path / "audit.json"
    c.audit_path.write_text("{}")
    (tmp_path / "identity.json").write_text('{"original":true}')
    (tmp_path / "STOP.json").write_text('{"original_failure":true}')
    c.audit = {"stop_sha256": importlib.import_module("colab_checkpoint").sha(tmp_path / "STOP.json")}
    c.registration = {"jobs": [{"job_id": "done"}, {"job_id": "pending"}]}
    (tmp_path / "durable/done").mkdir(parents=True)
    (tmp_path / "durable/done/latest.json").write_text("{}")
    c.generation = 4
    c.owned = []
    c.calls = []
    c.admission = lambda: {"verified": True}
    c.event = lambda stage, **kw: c.calls.append(("event", stage))
    c.snapshot = lambda spec: (None, {"counters": {"updates": 160}})
    c.finish = lambda spec: c.calls.append(("finish", spec["job_id"]))

    def allocate():
        name = "new-" + str(len(c.owned))
        c.owned.append(name)
        c.calls.append(("allocate", name))
        return name

    c.allocate = allocate
    c.setup = lambda session: None
    c.launch_job = lambda session, spec: c.calls.append(("launch", spec["job_id"])) or spec
    c.release = lambda session: c.calls.append(("release", session))
    c.guard = lambda **kw: None
    c.diagnose = lambda *args: {"diagnostic_error": "timeout", "unverified_suffix_unknown": True}
    c.preserve_failure = lambda *args: c.calls.append(("preserve", "failure"))
    c.poll_job = lambda *args: {"status": {"stage": "complete"}, "alive": False, "snapshots": []}
    return c


def test_resume_skips_completed_jobs_archives_stop_and_preserves_identity(tmp_path, monkeypatch):
    c = make_controller(tmp_path, monkeypatch)
    c.main()
    assert ("finish", "done") in c.calls and ("launch", "done") not in c.calls
    assert ("launch", "pending") in c.calls
    assert json.loads((tmp_path / "recoveries/1/previous-STOP.json").read_text()) == {"original_failure": True}
    assert json.loads((tmp_path / "identity.json").read_text()) == {"original": True}
    assert not (tmp_path / "STOP.json").exists()
    assert c.calls[-1] == ("event", "training_complete")


def test_rejected_admission_never_clears_stop_or_allocates(tmp_path, monkeypatch):
    c = make_controller(tmp_path, monkeypatch)

    def reject():
        raise ValueError("partial evaluation")

    c.admission = reject
    with pytest.raises(ValueError, match="partial evaluation"):
        c.main()
    assert not c.calls and (tmp_path / "STOP.json").exists()


def test_new_stop_after_admission_is_never_cleared(tmp_path, monkeypatch):
    c = make_controller(tmp_path, monkeypatch)

    def change_stop():
        (tmp_path / "STOP.json").write_text('{"new_health_failure":true}')
        return {}

    c.admission = change_stop
    with pytest.raises(ValueError, match="STOP changed during admission"):
        c.main()
    assert json.loads((tmp_path / "STOP.json").read_text()) == {"new_health_failure": True}
    assert not c.calls


def test_timeout_confirms_endpoint_release_before_replacement(tmp_path, monkeypatch):
    c = make_controller(tmp_path, monkeypatch)
    calls = iter(
        [
            RuntimeError("colab exec timed out; remote state is unknown"),
            {"status": {"stage": "complete"}, "alive": False, "snapshots": []},
        ]
    )

    def poll(*args):
        value = next(calls)
        if isinstance(value, Exception):
            raise value
        return value

    c.poll_job = poll
    c.main()
    assert c.calls.index(("release", "new-0")) < c.calls.index(("allocate", "new-1"))
    assert c.generation == 6


@pytest.mark.parametrize("failure", ["worker", "setup", "diagnostic"])
def test_reported_failure_is_not_retried_and_survives_cleanup(tmp_path, monkeypatch, failure):
    c = make_controller(tmp_path, monkeypatch)

    def poll(*args):
        if failure == "setup":
            raise RuntimeError("remote build failed")
        if failure == "diagnostic":
            raise RuntimeError("colab exec timed out; remote state is unknown")
        return {"status": {"stage": "failed", "error": "nonfinite"}, "alive": False, "snapshots": []}

    c.poll_job = poll
    if failure == "diagnostic":
        c.diagnose = lambda *args: {"status": {"stage": "failed", "error": "worker lease expired"}}
    with pytest.raises((ValueError, RuntimeError)):
        c.main()
    assert len([x for x in c.calls if x[0] == "allocate"]) == 1
    assert (tmp_path / "STOP.json").exists()
    assert c.calls[-1] == ("event", "failed")
