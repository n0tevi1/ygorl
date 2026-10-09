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
