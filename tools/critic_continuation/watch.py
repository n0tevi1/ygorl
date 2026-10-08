"""Independent observer: no writes inside the registered study and no process control."""

import argparse
import fcntl
import json
import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from audit import audit, sha

STUDY = Path(__file__).resolve().parent
ROOT = STUDY / "monitor"
CONFIG = json.loads((ROOT / "config.json").read_text())
IDENTITY = CONFIG["study_sha256"]
UNIT = CONFIG["unit"]
INVOCATION = CONFIG["invocation"]
TAG = "ygorl-monitor:terminal-critic-continuation-restart-2026-10-08"
START = f"<!-- {TAG}:start -->"
END = f"<!-- {TAG}:end -->"
CACHE = {}


def atomic(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def service():
    names = (
        "ActiveState",
        "SubState",
        "MainPID",
        "Result",
        "ExecMainStatus",
        "InvocationID",
        "ExecMainStartTimestamp",
        "ExecMainExitTimestamp",
    )
    command = ["systemctl", "--user", "show", UNIT] + [f"--property={n}" for n in names]
    s = subprocess.check_output(command, text=True, timeout=15)
    return dict(line.split("=", 1) for line in s.splitlines() if "=" in line)


def snapshot():
    assert sha(STUDY / "identity.json") == IDENTITY, "study identity changed"
    identity = json.loads((STUDY / "identity.json").read_text())
    assert all(sha(STUDY / name) == digest for name, digest in identity["drivers"].items()), "study driver changed"
    state = json.loads((STUDY / "pipeline-status.json").read_text())
    svc = service()
    assert svc["InvocationID"] == INVOCATION, "unexpected study service invocation"
    totals = dict(updates=0, rows=0, games=0, truncated=0, errors=0)
    phases = {}
    activity = []
    for p in sorted(STUDY.glob("seed-*/*/run/metrics.jsonl")):
        rows = [json.loads(s) for s in p.read_text().splitlines(keepends=True) if s.endswith("\n")]
        if not rows:
            continue
        source = json.loads((p.parent.parent / "start-audit.json").read_text())["counters"]
        counter = {k: rows[-1]["total"][k] - source[k] for k in totals}
        phases[str(p.parent.parent.relative_to(STUDY))] = counter
        for k in totals:
            totals[k] += counter[k]
        activity.append(p.stat().st_mtime)
    cells = []
    for p in sorted((STUDY / "evaluation").glob("*/*.json")):
        raw = p.with_suffix(".jsonl")
        key = str(p)
        stamp = (p.stat().st_mtime_ns, raw.stat().st_mtime_ns, raw.stat().st_size)
        if key not in CACHE or CACHE[key][0] != stamp:
            cell = json.loads(p.read_text())
            assert cell["identity"]["study_sha256"] == IDENTITY
            assert cell["games"] == 256 and cell["all_healthy"] and sha(raw) == cell["raw_sha256"]
            rows = [json.loads(s) for s in raw.read_text().splitlines()]
            assert len(rows) == 256 and [r["game_id"] for r in rows] == list(range(256))
            assert all(
                r["result"]["reason"] == "win"
                and r["result"]["winner"] in (0, 1)
                and not any(
                    r["result"].get(k)
                    for k in ("error", "retries", "unknown_messages", "undecodable_messages", "script_errors")
                )
                for r in rows
            )
            assert sum(r["result"]["winner"] == 0 for r in rows) == cell["wins"]
            CACHE[key] = (stamp, {"cell": str(p.relative_to(STUDY / "evaluation")), "games": 256, "wins": cell["wins"]})
        cells.append(CACHE[key][1])
    retention_progress = STUDY / "retention/progress.json"
    if retention_progress.exists():
        activity.append(retention_progress.stat().st_mtime)
    for p in (STUDY / "evaluation").glob("*/*.jsonl"):
        activity.append(p.stat().st_mtime)
    now = time.time()
    idle = now - max(activity) if activity else None
    alerts = []
    stop = json.loads((STUDY / "STOP.json").read_text()) if (STUDY / "STOP.json").exists() else None
    if stop:
        alerts.append("study STOP: " + stop.get("reason", "unknown"))
    if totals["errors"]:
        alerts.append("training errors recorded")
    if svc["ActiveState"] == "failed":
        alerts.append("study service failed: " + svc["Result"] + "/" + svc["ExecMainStatus"])
    if any(c["games"] >= 100 and c["truncated"] / c["games"] > 0.01 for c in phases.values()):
        alerts.append("training truncations exceeded the registered 1% limit")
    if state["stage"] not in ("complete", "stopped") and svc["SubState"] != "running":
        alerts.append("service not running before completion")
    if state["stage"] not in ("complete", "stopped") and now - state["updated_unix"] > 180:
        alerts.append("pipeline heartbeat older than 180 seconds")
    if state["stage"] not in ("complete", "stopped") and idle is not None and idle > 1800:
        alerts.append("no training or raw evaluation progress for 30 minutes")
    retention = STUDY / "retention/progress.json"
    retention_status = json.loads(retention.read_text()) if retention.exists() else None
    if state["stage"] == "complete":
        assert retention_status and retention_status["stage"] == "complete"
        assert (STUDY / "retention/analysis.json").exists()
    result = {
        "retention": retention_status,
        "checked_unix": now,
        "checked_local": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(),
        "study_sha256": IDENTITY,
        "pipeline": state,
        "service": svc,
        "training": totals,
        "phases": phases,
        "completed_evaluation_games": sum(c["games"] for c in cells),
        "verified_cells": cells,
        "data_idle_seconds": idle,
        "alerts": alerts,
        "stop": stop,
        "independent_audit": None,
    }
    if state["stage"] == "complete" and not alerts:
        p = ROOT / "completion-audit.json"
        if p.exists():
            checked = json.loads(p.read_text())
            assert checked["analysis_sha256"] == sha(STUDY / "analysis.json")
            assert checked["auditor_sha256"] == sha(STUDY / "audit.py")
        else:
            checked = audit(STUDY)
            assert checked["training"]["updates"] == 2304 and checked["training"]["rows"] == 37748736
            atomic(p, checked)
        result["independent_audit"] = checked
    return result


def section(status):
    t = status.get("training", {})
    lines = [
        START,
        "## 独立监控：终局critic 128→512更新PPO续训",
        "",
        f"核验时间：{status['checked_local']}。训练 **{t.get('updates', '?')}/2304新增更新**，"
        f"正常/已计数终局 **{t.get('games', '?')}**，截断 **{t.get('truncated', '?')}**、错误 **{t.get('errors', '?')}**；"
        f"已SHA/原始胜负核验的完整评估 **{status.get('completed_evaluation_games', '?')}/19,456局**。",
        f"服务状态：`{status.get('service', {}).get('ActiveState', 'unknown')}/{status.get('service', {}).get('SubState', 'unknown')}`；"
        f"实验阶段：`{status.get('pipeline', {}).get('stage', 'unknown')}`。",
    ]
    if status.get("retention"):
        r = status["retention"]
        lines.append(f"固定BC参考任务保留诊断：`{r['stage']}`，已完成节点 {r.get('completed', '进行中')}/18。")
    if status.get("alerts"):
        lines += ["", "**需要处理：** " + "；".join(status["alerts"])]
    checked = status.get("independent_audit")
    if checked:
        lines += [
            "",
            "**已完成且独立审计通过。** 原始胜负、模型/阶段SHA、配对输入、20,000次交叉及条件bootstrap均核对。",
        ]
        for name, label in [("warm_minus_cold", "warm−cold"), ("warm_minus_initial", "warm−initial")]:
            value = checked["primary"][name]
            lo, hi = value["crossed_ci95"]
            lines.append(
                f"固定update512 {label}：**{100 * value['mean']:+.2f}pp，95%交叉CI [{100 * lo:+.2f},{100 * hi:+.2f}]pp**。"
            )
        growth = checked["growth"]["warm_512_minus_128"]
        lo, hi = growth["crossed_ci95"]
        lines.append(
            f"warm512−warm128：**{100 * growth['mean']:+.2f}pp，95%交叉CI [{100 * lo:+.2f},{100 * hi:+.2f}]pp**。"
        )
        vals = checked["primary"]["warm_minus_cold"]["per_training_seed"]
        lines.append("三个seed warm−cold：" + ", ".join(f"{100 * v:+.2f}pp" for v in vals) + "。")
        history = checked["growth"]["warm_historical_512_minus_128"]
        lo, hi = history["crossed_ci95"]
        lines.append(
            f"历史RL对手warm512−warm128：{100 * history['mean']:+.2f}pp，95% CI [{100 * lo:+.2f},{100 * hi:+.2f}]pp。"
        )
        lines.append(
            "预登记加长训练条件：**" + str(checked["criterion_to_test_longer_training"]) + "**。仍不代表顶尖对局验收。"
        )
    else:
        lines += ["", "研究尚未通过完整结果审计；不根据部分seed或节点作强度结论。"]
    lines += [
        "",
        "独立监控证据：`out/research/terminal-critic-continuation-restart-2026-10-08/monitor/`。每30秒检查；"
        "本节每10分钟及完成/异常转换时同步。只读取实验，保留原STOP与失败数据，不修改参数或替换对局。",
        END,
    ]
    return "\n".join(lines) + "\n"


def publish(status):
    block = section(status)
    for number in (61, 83, 92):
        cmd = ["/usr/bin/gh", "issue", "view", str(number), "--repo", "n0tevi1/ygorl", "--json", "body"]
        before = json.loads(subprocess.check_output(cmd, text=True, timeout=30))["body"]
        if START in before:
            assert before.count(START) == before.count(END) == 1
            left, rest = before.split(START, 1)
            _, right = rest.split(END, 1)
            updated = left + block.rstrip("\n") + right
        else:
            updated = block + "\n---\n\n" + before
        assert len(updated) < 65000
        path = ROOT / f"issue-{number}-body.md"
        path.write_text(updated)
        assert json.loads(subprocess.check_output(cmd, text=True, timeout=30))["body"] == before, (
            "concurrent issue edit"
        )
        subprocess.run(
            ["/usr/bin/gh", "issue", "edit", str(number), "--repo", "n0tevi1/ygorl", "--body-file", str(path)],
            check=True,
            timeout=30,
            capture_output=True,
            text=True,
        )
        assert json.loads(subprocess.check_output(cmd, text=True, timeout=30))["body"] == updated


def main():
    lock = (ROOT / "monitor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    last_publish = 0
    last_kind = None
    previous = None
    while True:
        try:
            current = snapshot()
        except Exception as exc:
            current = {
                "checked_unix": time.time(),
                "checked_local": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(),
                "alerts": ["monitor verification failed: " + repr(exc)],
                "independent_audit": None,
                "traceback": traceback.format_exc(),
            }
        atomic(ROOT / "latest.json", current)
        key = (
            current.get("pipeline", {}).get("stage"),
            current.get("training", {}).get("updates"),
            current.get("completed_evaluation_games"),
            tuple(current["alerts"]),
        )
        if key != previous:
            with (ROOT / "events.jsonl").open("a") as f:
                f.write(json.dumps(current) + "\n")
            print(json.dumps({"checked": current["checked_local"], "progress": key}), flush=True)
            previous = key
        terminal = current.get("pipeline", {}).get("stage") in ("complete", "stopped") or current.get(
            "service", {}
        ).get("ActiveState") in ("failed", "inactive")
        kind = "complete" if current.get("independent_audit") else "attention" if current["alerts"] else "running"
        published = not args.publish
        if args.publish and (time.time() - last_publish >= 600 or kind != last_kind or terminal):
            try:
                publish(current)
                last_publish = time.time()
                last_kind = kind
                published = True
                atomic(
                    ROOT / "publication.json", {"published_unix": last_publish, "kind": kind, "issues": [61, 83, 92]}
                )
            except Exception:
                with (ROOT / "publication-errors.log").open("a") as f:
                    f.write(traceback.format_exc() + "\n")
        if args.once:
            return
        if terminal and published:
            atomic(ROOT / "finished.json", current)
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
