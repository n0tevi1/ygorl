"""Persistent experiment incidents with verified delivery to the existing T3 thread.

Run under a restarting user service. Tokens are short-lived, held in memory, and
never written to reports. Delivery does not acknowledge or resolve an incident.
"""

import argparse
import fcntl
import hashlib
import json
import os
import socket
import subprocess
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path


def atomic(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_study(config, service, now):
    root = Path(config["study"])
    alerts = []
    if digest(root / "identity.json") != config["study_sha256"]:
        alerts.append("study identity changed")
    stop = root / "STOP.json"
    if stop.exists():
        alerts.append("study STOP: " + str(json.loads(stop.read_text()).get("reason", "unknown")))
    errors = []
    for path in sorted(root.glob("**/errors.jsonl")):
        lines = path.read_text().splitlines(keepends=True)
        records = [json.loads(s) for s in lines if s.endswith("\n")]
        if records:
            errors.append({"path": str(path.relative_to(root)), "count": len(records)})
    if errors:
        alerts.append("failed rollouts recorded (including those not committed to metrics)")
    state_path = root / "pipeline-status.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    complete = state.get("stage") == "complete"
    if state.get("stage") == "stopped":
        alerts.append("pipeline stopped")
    if not complete:
        if service.get("ActiveState") not in ("active", "activating"):
            alerts.append("study service is not active")
        if now - state.get("updated_unix", 0) > 180:
            alerts.append("pipeline heartbeat stale (>180 seconds)")
    progress = [
        p.stat().st_mtime
        for pattern in config.get("progress_globs", ["**/metrics.jsonl", "evaluation/**/*.jsonl"])
        for p in root.glob(pattern)
    ]
    if not complete and progress and now - max(progress) > 1800:
        alerts.append("no training/evaluation progress for 30 minutes")
    if complete:
        report = root / config["completion_report"]
        if not report.is_file():
            alerts.append("completion report missing")
        else:
            report_data = json.loads(report.read_text())
            if report_data.get("study_sha256") != config["study_sha256"] or not report_data.get("healthy"):
                alerts.append("completion report identity/health failed")
    clean_exit = (
        (service.get("ActiveState"), service.get("SubState")) in (("inactive", "dead"), ("active", "exited"))
        and service.get("Result") == "success"
        and service.get("ExecMainStatus") == "0"
    )
    if service.get("ActiveState") == "failed" or service.get("Result") not in (None, "success"):
        alerts.append("study service failed: " + str(service.get("Result")))
    if complete and service.get("ExecMainStatus") not in (None, "0"):
        alerts.append("study service returned nonzero status")
    if config.get("invocation") and service.get("InvocationID") != config["invocation"]:
        # systemd may discard identity after normal exit. Never exempt an actual new ID.
        if service.get("InvocationID") or not (complete and clean_exit and not alerts):
            alerts.append("study service invocation changed or unavailable before verified completion")
    return {
        "checked_unix": now,
        "pipeline": state,
        "service": service,
        "alerts": alerts,
        "failed_rollouts": errors,
        "complete": complete and clean_exit and not alerts,
    }


def update_incident(monitor, config, snapshot):
    path = monitor / "incident.json"
    prior = json.loads(path.read_text()) if path.exists() else None
    kind = "attention" if snapshot["alerts"] else "complete" if snapshot["complete"] else None
    if kind is None:
        return prior
    key = hashlib.sha256(json.dumps([config["study_sha256"], kind], sort_keys=True).encode()).hexdigest()
    if prior and prior["id"] == key:
        return prior
    if prior:
        atomic(monitor / f"incident-{prior['id']}.json", prior)
    incident = {
        "id": key,
        "kind": kind,
        "created_unix": snapshot["checked_unix"],
        "alerts": snapshot["alerts"],
        "command_id": str(uuid.uuid4()),
        "message_id": str(uuid.uuid4()),
        "delivery": None,
        "acknowledgment": None,
    }
    atomic(path, incident)
    return incident


class T3Delivery:
    def __init__(self, config):
        self.config = config
        self.token = None
        self.expires = 0

    def request(self, path, payload=None):
        if self.expires < time.time():
            self.token = subprocess.check_output(
                [
                    self.config.get("t3_cli", "/home/ya0guang/.local/bin/t3"),
                    "auth",
                    "session",
                    "issue",
                    "--ttl",
                    "15m",
                    "--label",
                    "ygorl-study-monitor",
                    "--token-only",
                ],
                text=True,
                timeout=30,
            ).strip()
            self.expires = time.time() + 600
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            self.config.get("t3_url", "http://127.0.0.1:3773") + path,
            data=data,
            headers={"Authorization": "Bearer " + self.token, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)

    def deliver(self, incident, monitor):
        thread_id = self.config["thread_id"]
        endpoint = f"/api/orchestration/threads/{thread_id}?turnLimit=1"
        thread = self.request(endpoint)["thread"]
        # Never upgrade permissions, switch models, or dispatch into an archived/deleted thread.
        assert thread["runtimeMode"] == self.config["runtime_mode"] == "auto"
        assert thread["interactionMode"] == "default"
        assert not thread.get("archivedAt") and not thread.get("deletedAt")
        text = (
            f"[自动实验监控；不是新的用户指令] {incident['kind']}\n"
            f"研究：{self.config['study']}\n事件 ID：{incident['id']}\n"
            f"异常：{json.dumps(incident['alerts'], ensure_ascii=False)}\n"
            f"证据：{monitor / 'latest.json'}\n"
            f"请按用户既有授权检查并接手。接手后运行：python3 {Path(__file__).resolve()} "
            f"--monitor {monitor} --ack {incident['id']} --note '接手说明'。"
            "送达不表示已接手或修复；保留原研究证据，不自动清除 STOP。"
        )
        command = {
            "type": "thread.turn.start",
            "commandId": incident["command_id"],
            "threadId": thread_id,
            "message": {"messageId": incident["message_id"], "role": "user", "text": text, "attachments": []},
            "modelSelection": thread["modelSelection"],
            "runtimeMode": thread["runtimeMode"],
            "interactionMode": thread["interactionMode"],
            "createdAt": datetime.fromtimestamp(incident["created_unix"], timezone.utc).isoformat(),
        }
        receipt = self.request("/api/orchestration/dispatch", command)
        # Reuse the command/message IDs after interruption or ambiguous network failure.
        # The server deduplicates the command; verify storage independently of HTTP success.
        thread = self.request(endpoint)["thread"]
        assert any(m["id"] == incident["message_id"] for m in thread["messages"]), "delivery not visible yet"
        return {"verified_unix": time.time(), "sequence": receipt["sequence"], "message_id": incident["message_id"]}


def notify_service():
    address = os.environ.get("NOTIFY_SOCKET")
    if address:
        if address.startswith("@"):
            address = "\0" + address[1:]
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
            sock.connect(address)
            sock.sendall(b"READY=1\nWATCHDOG=1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monitor", type=Path, required=True)
    parser.add_argument("--ack")
    parser.add_argument("--note")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    monitor = args.monitor.resolve()
    lock = (monitor / "watch.lock").open("a")
    if args.ack:
        # A separate file allows acknowledgment while the service holds its process lock.
        incident = json.loads((monitor / "incident.json").read_text())
        assert incident["id"] == args.ack and args.note
        atomic(monitor / f"ack-{args.ack}.json", {"id": args.ack, "acknowledged_unix": time.time(), "note": args.note})
        return
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = json.loads((monitor / "config.json").read_text())
    delivery = T3Delivery(config)
    while True:
        notify_service()
        now = time.time()
        try:
            out = subprocess.check_output(
                [
                    "systemctl",
                    "--user",
                    "show",
                    config["unit"],
                    "--property=ActiveState",
                    "--property=SubState",
                    "--property=InvocationID",
                    "--property=Result",
                    "--property=ExecMainStatus",
                ],
                text=True,
                timeout=15,
            )
            service = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
            snapshot = inspect_study(config, service, now)
        except Exception as exc:
            snapshot = {"checked_unix": now, "alerts": ["monitor inspection failed: " + repr(exc)], "complete": False}
        atomic(monitor / "latest.json", snapshot)
        incident = update_incident(monitor, config, snapshot)
        if incident:
            ack = monitor / f"ack-{incident['id']}.json"
            if ack.exists():
                acknowledgment = json.loads(ack.read_text())
                assert acknowledgment["id"] == incident["id"]
                incident["acknowledgment"] = acknowledgment
            if (
                incident["delivery"]
                and not incident["acknowledgment"]
                and now - incident["delivery"]["verified_unix"] >= 1800
            ):
                incident.setdefault("delivery_history", []).append(incident["delivery"])
                incident["delivery"] = None
                incident["command_id"] = str(uuid.uuid4())
                incident["message_id"] = str(uuid.uuid4())
                atomic(monitor / "incident.json", incident)
            if not incident["delivery"] and not incident["acknowledgment"]:
                try:
                    incident["delivery"] = delivery.deliver(incident, monitor)
                    incident.pop("delivery_error", None)
                except Exception as exc:
                    incident["delivery_error"] = {"at": time.time(), "error": repr(exc)}
            atomic(monitor / "incident.json", incident)
            if snapshot["complete"] and incident["kind"] == "complete" and incident["acknowledgment"]:
                atomic(monitor / "finished.json", snapshot)
                return
        if args.once:
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
