"""An uncommitted fatal rollout remains an incident until a responder acknowledges it."""

import json

import pytest

from study_watch import T3Delivery, atomic, digest, inspect_study, update_incident


def setup_study(tmp_path):
    study = tmp_path / "study"
    study.mkdir()
    atomic(study / "identity.json", {"seed": 2})
    atomic(study / "pipeline-status.json", {"stage": "running", "updated_unix": 1000})
    config = {"study": str(study), "study_sha256": digest(study / "identity.json"), "completion_report": "report.json"}
    service = {"ActiveState": "active", "InvocationID": "one"}
    return study, config, service


def test_first_heartbeat_grace_is_bound_to_process_start_not_watcher_start(tmp_path):
    study, config, service = setup_study(tmp_path)
    (study / "pipeline-status.json").unlink()
    config["invocation"] = "one"
    service["ExecMainStartTimestampMonotonic"] = "1000000000"
    assert not inspect_study(config, service, 9000, monotonic_now=1004)["alerts"]
    # Recreating a watcher cannot extend a grace window anchored to this process.
    assert (
        "pipeline heartbeat stale (>180 seconds)" in inspect_study(config, service, 9000, monotonic_now=1181)["alerts"]
    )
    atomic(study / "STOP.json", {"reason": "startup failure"})
    assert any("startup failure" in a for a in inspect_study(config, service, 9000, monotonic_now=1004)["alerts"])


@pytest.mark.parametrize(
    "change",
    [
        {"InvocationID": "two"},
        {"ActiveState": "failed"},
        {"ExecMainStartTimestampMonotonic": "0"},
        {"ExecMainStartTimestampMonotonic": "bad"},
        {"ExecMainStartTimestampMonotonic": "2000000000"},
    ],
)
def test_first_heartbeat_grace_requires_valid_bound_live_process(tmp_path, change):
    study, config, service = setup_study(tmp_path)
    (study / "pipeline-status.json").unlink()
    config["invocation"] = "one"
    service.update({"ExecMainStartTimestampMonotonic": "1000000000", **change})
    assert (
        "pipeline heartbeat stale (>180 seconds)" in inspect_study(config, service, 9000, monotonic_now=1004)["alerts"]
    )


def test_startup_grace_never_exempts_an_existing_stale_heartbeat(tmp_path):
    study, config, service = setup_study(tmp_path)
    config["invocation"] = "one"
    service["ExecMainStartTimestampMonotonic"] = "1000000000"
    assert (
        "pipeline heartbeat stale (>180 seconds)" in inspect_study(config, service, 9000, monotonic_now=1004)["alerts"]
    )


def test_failed_uncommitted_rollout_and_stop_do_not_count_as_complete(tmp_path):
    study, config, service = setup_study(tmp_path)
    (study / "errors.jsonl").write_text(json.dumps({"error": "Lua failure"}) + "\n")
    atomic(study / "pipeline-status.json", {"stage": "stopped", "updated_unix": 1000})
    atomic(study / "STOP.json", {"reason": "unhealthy rollout before optimizer"})
    snapshot = inspect_study(config, service, 1010)
    assert snapshot["failed_rollouts"] == [{"path": "errors.jsonl", "count": 1}]
    assert not snapshot["complete"]
    assert any("uncommitted" in s or "not committed" in s for s in snapshot["alerts"])
    monitor = tmp_path / "monitor"
    monitor.mkdir()
    incident = update_incident(monitor, config, snapshot)
    assert incident["kind"] == "attention" and incident["acknowledgment"] is None
    # Restart + later heartbeat staleness must preserve the same delivery IDs.
    later = update_incident(monitor, config, inspect_study(config, {"ActiveState": "inactive"}, 2000))
    assert later == incident
    assert not (monitor / "finished.json").exists()


def test_new_stop_does_not_inherit_acknowledged_startup_warning(tmp_path):
    study, config, service = setup_study(tmp_path)
    monitor = tmp_path / "monitor"
    monitor.mkdir()
    first = update_incident(monitor, config, inspect_study(config, service, 2000))
    first["acknowledgment"] = {"id": first["id"]}
    first["delivery"] = {"verified_unix": 2000}
    atomic(monitor / "incident.json", first)
    atomic(study / "STOP.json", {"reason": "no eligible samples"})
    second = update_incident(monitor, config, inspect_study(config, service, 2010))
    assert second["id"] != first["id"]
    assert second["acknowledgment"] is None and second["delivery"] is None
    assert any("no eligible samples" in a for a in second["alerts"])
    assert json.loads((monitor / f"incident-{first['id']}.json").read_text()) == first
    assert update_incident(monitor, config, inspect_study(config, service, 2020)) == second


def test_same_warning_after_recovery_is_a_new_occurrence(tmp_path):
    study, config, service = setup_study(tmp_path)
    monitor = tmp_path / "monitor"
    monitor.mkdir()
    first = update_incident(monitor, config, inspect_study(config, service, 2000))
    first["acknowledgment"] = {"id": first["id"]}
    atomic(monitor / "incident.json", first)
    atomic(study / "pipeline-status.json", {"stage": "running", "updated_unix": 2001})
    cleared = update_incident(monitor, config, inspect_study(config, service, 2002))
    assert cleared["cleared_unix"] == 2002
    second = update_incident(monitor, config, inspect_study(config, service, 2200))
    assert second["id"] != first["id"] and second["signature"] == first["signature"]
    assert second["acknowledgment"] is None


def test_completion_needs_bound_healthy_report(tmp_path):
    study, config, service = setup_study(tmp_path)
    atomic(study / "pipeline-status.json", {"stage": "complete", "updated_unix": 1000})
    assert not inspect_study(config, service, 1010)["complete"]
    atomic(study / "report.json", {"study_sha256": "wrong", "healthy": True})
    assert not inspect_study(config, service, 1010)["complete"]
    atomic(study / "report.json", {"study_sha256": config["study_sha256"], "healthy": True})
    assert not inspect_study(config, service, 1010)["complete"]  # process still running
    service.update(ActiveState="inactive", SubState="dead", Result="success", ExecMainStatus="0")
    assert inspect_study(config, service, 1010)["complete"]


def test_delivery_preserves_mode_and_reuses_ids_after_ambiguous_failure(tmp_path):
    config = {"thread_id": "existing-thread", "runtime_mode": "auto", "study": str(tmp_path)}
    delivery = T3Delivery(config)
    incident = {
        "id": "incident",
        "kind": "attention",
        "created_unix": 1000,
        "alerts": ["STOP"],
        "command_id": "same-command",
        "message_id": "same-message",
    }
    commands = []

    def request(path, payload=None):
        if payload:
            commands.append(payload)
            return {"sequence": 42}
        return {
            "thread": {
                "runtimeMode": "auto",
                "modelSelection": {
                    "instanceId": "codex",
                    "model": "gpt-6-astra",
                    "options": [{"id": "reasoningEffort", "value": "high"}],
                },
                "interactionMode": "default",
                "messages": [{"id": "same-message"}] if commands else [],
            }
        }

    delivery.request = request
    for _ in range(2):
        assert delivery.deliver(incident, tmp_path)["sequence"] == 42
    assert commands[0] == commands[1]
    assert commands[0]["threadId"] == "existing-thread" and commands[0]["runtimeMode"] == "auto"
    assert commands[0]["modelSelection"]["options"] == [{"id": "reasoningEffort", "value": "high"}]
    assert commands[0]["message"]["text"].startswith("[自动实验监控；不是新的用户指令]")


@pytest.mark.parametrize(
    "invocation,state,substate,result,status,expected",
    [
        ("", "inactive", "dead", "success", "0", True),
        ("one", "active", "exited", "success", "0", True),
        ("two", "inactive", "dead", "success", "0", False),
        ("", "active", "running", "success", "0", False),
        ("one", "active", "running", "success", "0", False),
        ("", "failed", "failed", "exit-code", "1", False),
        ("", "inactive", "dead", "success", "1", False),
    ],
)
def test_terminal_invocation_and_process_exit(tmp_path, invocation, state, substate, result, status, expected):
    study, config, _ = setup_study(tmp_path)
    config["invocation"] = "one"
    atomic(study / "pipeline-status.json", {"stage": "complete", "updated_unix": 1000})
    atomic(study / "report.json", {"study_sha256": config["study_sha256"], "healthy": True})
    service = {
        "InvocationID": invocation,
        "ActiveState": state,
        "SubState": substate,
        "Result": result,
        "ExecMainStatus": status,
    }
    snapshot = inspect_study(config, service, 1010)
    assert snapshot["complete"] == expected
    if invocation == "two":
        assert any("invocation" in a for a in snapshot["alerts"])
    if expected:
        (study / "errors.jsonl").write_text('{"error":"uncommitted Lua failure"}\n')
        assert not inspect_study(config, service, 1010)["complete"]


def test_archived_ack_does_not_acknowledge_new_incident(tmp_path):
    from study_watch import acknowledge

    old_id, new_id = "a" * 64, "b" * 64
    current = {"id": new_id, "kind": "complete", "acknowledgment": None}
    old = {"id": old_id, "kind": "attention"}
    atomic(tmp_path / "incident.json", current)
    atomic(tmp_path / f"incident-{old_id}.json", old)
    acknowledge(tmp_path, old_id, "investigated archived inspection failure")
    assert json.loads((tmp_path / f"ack-{old_id}.json").read_text())["id"] == old_id
    assert not (tmp_path / f"ack-{new_id}.json").exists()
    assert json.loads((tmp_path / "incident.json").read_text()) == current
    assert json.loads((tmp_path / f"incident-{old_id}.json").read_text()) == old


@pytest.mark.parametrize("incident_id,note", [("../escape", "note"), ("a" * 64, " "), ("a" * 64, None)])
def test_ack_rejects_invalid_id_or_empty_note(tmp_path, incident_id, note):
    from study_watch import acknowledge

    with pytest.raises(ValueError):
        acknowledge(tmp_path, incident_id, note)
    assert not list(tmp_path.iterdir())


def test_ack_rejects_mismatched_archive(tmp_path):
    from study_watch import acknowledge

    atomic(tmp_path / "incident.json", {"id": "b" * 64})
    atomic(tmp_path / f"incident-{'a' * 64}.json", {"id": "c" * 64})
    with pytest.raises(ValueError, match="does not match"):
        acknowledge(tmp_path, "a" * 64, "note")
    assert not list(tmp_path.glob("ack-*.json"))


def test_launcher_keeps_success_terminal_and_crashes_restartable(tmp_path, monkeypatch):
    import study_watch

    atomic(tmp_path / "config.json", {"unit": "ygorl-training.service"})
    calls = []
    monkeypatch.setattr(study_watch.subprocess, "run", lambda argv, **kw: calls.append((argv, kw)))
    study_watch.launch_monitor(tmp_path, "ygorl-training-watch.service")
    argv, kwargs = calls[0]
    assert argv[:3] == ["systemd-run", "--user", "--unit=ygorl-training-watch.service"]
    assert "--property=Restart=on-failure" in argv
    assert "--property=Restart=always" not in argv
    assert "--property=Type=notify" in argv
    assert "--property=NotifyAccess=main" in argv
    assert "--property=WatchdogSec=120s" in argv
    assert argv[-4:] == [
        study_watch.sys.executable,
        str(study_watch.Path(study_watch.__file__).resolve()),
        "--monitor",
        str(tmp_path.resolve()),
    ]
    assert kwargs == {"check": True}


@pytest.mark.parametrize("unit", ["ygorl-training.service", "other.service", "ygorl-x.service;bad"])
def test_launcher_rejects_study_unit_and_invalid_names(tmp_path, monkeypatch, unit):
    import study_watch

    atomic(tmp_path / "config.json", {"unit": "ygorl-training.service"})

    def unexpected(*args, **kwargs):
        pytest.fail("invalid launch must not invoke systemd")

    monkeypatch.setattr(study_watch.subprocess, "run", unexpected)
    with pytest.raises(ValueError):
        study_watch.launch_monitor(tmp_path, unit)
