"""Independent bounded supervision of registered precision training and evaluation."""

import argparse
import json
import math
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path

from colab_checkpoint import atomic, run_lock, sha
from precision_worker import validate_precision
from ygorl.solver.resume import output_lock


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def request_stop(root, reason):
    with output_lock(root / "STOP.json"):
        if not (root / "STOP.json").exists():
            atomic(root / "STOP.json", {"source": "precision_watch", "reason": reason, "time": time.time()})


def aggregate(rows, origin):
    if not rows:
        return {"updates": 0, "health": {"games": 0, "truncated": 0, "errors": 0}, "strata": {}}
    if any(isinstance(v, float) and not math.isfinite(v) for row in rows for v in row.values()):
        raise ValueError("nonfinite committed metrics")
    strata = defaultdict(list)
    for row in rows:
        key = (row["rows"], row["minibatches"], row["evaluated_minibatches"], row["early_stop"])
        strata[json.dumps(key)].append(row["update_s"])
    health = {k: rows[-1]["total"].get(k, 0) - origin.get(k, 0) for k in ("games", "truncated")}
    health["errors"] = sum(row.get("errors", 0) for row in rows)
    health["truncation_fraction"] = health["truncated"] / health["games"] if health["games"] else 0
    return {
        "updates": len(rows),
        "last_update": rows[-1]["total"]["updates"],
        "rows": sum(r["rows"] for r in rows),
        "accepted_minibatches": sum(r["minibatches"] for r in rows),
        "evaluated_minibatches": sum(r["evaluated_minibatches"] for r in rows),
        "early_stops": sum(r["early_stop"] for r in rows),
        "health": health,
        "timing": {
            key: {"sum": sum(r[key] for r in rows), "median": statistics.median(r[key] for r in rows)}
            for key in ("update_s", "collect_s", "step_s")
        },
        "strata": {
            key: {"updates": len(values), "median_update_s": statistics.median(values)}
            for key, values in strata.items()
        },
    }


def stop_reason(controller, evaluator, units, arms, completed, elapsed):
    if controller.get("stage") in ("failed", "stopped", "cleanup_failed"):
        return "controller reported " + controller["stage"]
    if evaluator.get("status") in ("failed", "stopped", "deadline-incomplete"):
        return "evaluator reported " + evaluator["status"]
    for name, arm in arms.items():
        health = arm["health"]
        if health["errors"] or (health["games"] >= 100 and health["truncated"] / health["games"] > 0.01):
            return "unhealthy committed training prefix: " + name
    if elapsed >= 120:
        for role, done in (("controller", len(completed) == 6), ("evaluator", evaluator.get("status") == "complete")):
            state = units[role]
            if not done and state["ActiveState"] not in ("active", "activating", "reloading"):
                return f"required {role} service unexpectedly inactive: {state}"
    return None


def efficiency_report(arms, events, started, now):
    matched = []
    for seed in range(3):
        fp32, bf16 = (arms[f"seed-{seed}-{p}"] for p in ("fp32", "bf16"))
        for key in sorted(fp32["strata"].keys() & bf16["strata"].keys()):
            a, b = fp32["strata"][key], bf16["strata"][key]
            reduction = 1 - b["median_update_s"] / a["median_update_s"] if a["median_update_s"] else None
            matched.append(
                {
                    "seed": seed,
                    "rows_accepted_evaluated_earlystop": json.loads(key),
                    "fp32": a,
                    "bf16": b,
                    "median_time_reduction": reduction,
                }
            )
    allocated = {}
    setup_and_first_input = []
    recoveries = []
    pending_preemption = None
    for event in events:
        if event["stage"] == "allocated":
            allocated[event["session"]] = event["time"]
        elif event["stage"] == "preempted":
            pending_preemption = event["time"]
        elif event["stage"] == "launched":
            if event["session"] in allocated:
                setup_and_first_input.append(event["time"] - allocated.pop(event["session"]))
            if pending_preemption is not None:
                recoveries.append(event["time"] - pending_preemption)
                pending_preemption = None
    reductions = [s["median_time_reduction"] for s in matched if s["median_time_reduction"] is not None]
    median = statistics.median(reductions) if reductions else None
    steps = sum(a["timing"]["step_s"]["sum"] for a in arms.values())
    elapsed = now - started
    return {
        "arms": arms,
        "matched_work_strata": matched,
        "median_of_stratum_time_reductions": median,
        "descriptive_at_least_10_percent": median is not None and median >= 0.1,
        "study_elapsed_seconds": elapsed,
        "committed_step_seconds": steps,
        "committed_rows_per_study_second": sum(a["rows"] for a in arms.values()) / elapsed if elapsed > 0 else None,
        "costs": {
            "allocation_complete_to_first_launch_seconds": setup_and_first_input,
            "preemption_detected_to_relaunch_seconds": recoveries,
            "checkpoint_transfer_seconds": None,
            "allocation_seconds": None,
            "unattributed_elapsed_minus_committed_steps": elapsed - steps,
        },
        "cost_caveat": "Intervals can overlap; allocation-to-launch combines setup and input transfer. Snapshot transfer and recomputed work are not separately timed. Unattributed time is not pure overhead.",
        "interpretation": "Descriptive matched-work speed evidence only; fewer accepted minibatches is not an infrastructure speedup.",
        "quality_and_behavior_review": "required separately; selected behavior cases are diagnostic, not population frequencies",
        "auto_promote": False,
    }


class Watch:
    def __init__(self, args):
        self.args = args
        self.root = args.root.resolve()
        self.out = self.root / "watch"
        self.out.mkdir(parents=True, exist_ok=True)
        self.started = time.time()
        self.verified = {}
        self.last_publish = 0
        self.last_stage = None

    def verified_rows(self, name, directory, expected, checkpoint_sha):
        key = (name, checkpoint_sha)
        names = ("checkpoint.pt", "manifest.json", "precision-manifest.json", "metrics.jsonl", "games.jsonl.gz")
        hashes = {filename: sha(directory / filename) for filename in names}
        if hashes["checkpoint.pt"] != checkpoint_sha:
            raise ValueError("snapshot pointer checkpoint hash differs")
        if key not in self.verified:
            validate_precision(directory, expected)
            self.verified[key] = hashes
        elif self.verified[key] != hashes:
            raise ValueError("previously verified immutable snapshot changed")
        return [json.loads(s) for s in (directory / "metrics.jsonl").read_text().splitlines()]

    def units(self):
        result = {}
        for role, unit in (("controller", self.args.controller_unit), ("evaluator", self.args.evaluator_unit)):
            response = subprocess.run(
                ["systemctl", "--user", "show", unit, "--property=ActiveState,SubState,Result,ExecMainStatus"],
                capture_output=True,
                text=True,
                timeout=15,
                check=True,
            )
            result[role] = dict(line.split("=", 1) for line in response.stdout.splitlines() if "=" in line)
            if "ActiveState" not in result[role]:
                raise ValueError("service state missing")
        return result

    def snapshot(self):
        now = time.time()
        identity = read(self.root / "identity.json")
        arms = {}
        if identity:
            for name, expected in identity["jobs"].items():
                store = self.root / "durable" / name
                pointer = read(store / "latest.json")
                if not pointer:
                    continue
                directory = (store / pointer["path"]).resolve()
                if not directory.is_relative_to(store.resolve()):
                    raise ValueError("snapshot escapes registered store")
                rows = self.verified_rows(name, directory, expected, pointer["sha256"])
                arms[name] = aggregate(rows, expected["origin_counts"])
        completed = []
        for path in sorted((self.root / "completed").glob("*.json")):
            value = read(path)
            name = value["job_id"]
            if (
                not identity
                or value["identity"] != identity["jobs"].get(name)
                or sha(value["checkpoint"]) != value["checkpoint_sha256"]
            ):
                raise ValueError("completed pointer identity differs")
            if path.stem != name or name in completed:
                raise ValueError("duplicate/misnamed completion")
            directory = Path(value["snapshot"])
            # A completion can appear just after the earlier latest-pointer read.
            rows = self.verified_rows(name, directory, identity["jobs"][name], value["checkpoint_sha256"])
            arms[name] = aggregate(rows, identity["jobs"][name]["origin_counts"])
            if arms[name]["last_update"] != 160:
                raise ValueError("completion lacks verified endpoint")
            completed.append(name)
        remotes = sorted(self.root.glob("remote-*.json"), key=lambda p: int(p.stem.split("-")[-1]))
        controller = read(self.root / "status.json", {})
        evaluator = read(self.root / "evaluator/status.json", {})
        unit_states = self.units()
        reason = stop_reason(controller, evaluator, unit_states, arms, completed, now - self.started)
        if reason:
            request_stop(self.root, reason)
        stage = (
            "stopped"
            if (self.root / "STOP.json").exists()
            else "complete"
            if len(completed) == 6 and evaluator.get("status") == "complete"
            else "monitoring"
        )
        status = {
            "stage": stage,
            "checked_unix": now,
            "units": unit_states,
            "controller": controller,
            "evaluator": evaluator,
            "completed": completed,
            "verified_arms": arms,
            "live_remote_uncommitted": read(remotes[-1]) if remotes else None,
            "behavior": read(self.args.behavior_root / "latest.json"),
            "behavior_interpretation": "selected cases and anomaly signals require review; no automatic strength label",
            "stop": read(self.root / "STOP.json"),
            "auto_promote": False,
        }
        atomic(self.out / "latest.json", status)
        if len(completed) == 6:
            raw_events = (self.root / "events.jsonl").read_text()
            lines = raw_events.splitlines()
            events = [json.loads(s) for s in (lines if raw_events.endswith("\n") else lines[:-1])]
            # Freeze training completion time; evaluation and observation do not inflate training cost.
            endpoint = next(
                (e["time"] for e in reversed(events) if e["stage"] == "training_complete"),
                max(read(self.root / "completed" / (name + ".json"))["completed_unix"] for name in completed),
            )
            atomic(
                self.root / "efficiency-report.json",
                efficiency_report(arms, events, identity["started_unix"], endpoint),
            )
        return status

    def publish(self, status):
        terminal_change = status["stage"] in ("complete", "stopped") and status["stage"] != self.last_stage
        if not self.args.publish or (time.time() - self.last_publish < 600 and not terminal_change):
            return
        self.last_publish = time.time()
        self.last_stage = status["stage"]
        marker = "<!-- ygorl-precision-watch:" + read(self.root / "registration.json")["run_id"] + " -->"
        body = [
            marker,
            "Learner precision pilot monitor",
            "",
            f"State: **{status['stage']}**; completed training arms: {len(status['completed'])}/6.",
            "",
            "| Arm | Updates | Rows | Accepted / evaluated minibatches | KL stops | Games / truncations / errors |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, arm in status["verified_arms"].items():
            h = arm["health"]
            body.append(
                f"| {name} | {arm['updates']} | {arm['rows']} | {arm['accepted_minibatches']} / {arm['evaluated_minibatches']} | {arm['early_stops']} | {h['games']} / {h['truncated']} / {h['errors']} |"
            )
        body.extend(
            [
                "",
                "Metrics are locally verified prefixes; remote progress can be ahead. Behavior signals require review; no automatic promotion.",
                f"Evaluator: `{status['evaluator'].get('status', 'starting')}`. STOP: `{status['stop']}`.",
            ]
        )
        text = "\n".join(body) + "\n"
        (self.out / "issue-body.md").write_text(text)
        atomic(self.out / "issue-payload.json", {"body": text})
        saved = read(self.out / "publication.json", {})
        try:
            if not saved.get("id"):
                response = subprocess.run(
                    [
                        "gh",
                        "api",
                        "--paginate",
                        f"repos/n0tevi1/ygorl/issues/{self.args.issue}/comments",
                        "--jq",
                        f".[] | select(.body | contains({json.dumps(marker)})) | {{id,html_url}}",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=True,
                )
                found = [json.loads(line) for line in response.stdout.splitlines() if line.strip()]
                if len(found) > 1:
                    raise ValueError("multiple monitor comments found")
                saved = found[0] if found else {}
            endpoint = (
                f"repos/n0tevi1/ygorl/issues/comments/{saved['id']}"
                if saved.get("id")
                else f"repos/n0tevi1/ygorl/issues/{self.args.issue}/comments"
            )
            response = subprocess.run(
                [
                    "gh",
                    "api",
                    "--method",
                    "PATCH" if saved.get("id") else "POST",
                    endpoint,
                    "--input",
                    str(self.out / "issue-payload.json"),
                ],
                capture_output=True,
                text=True,
                timeout=60,
                check=True,
            )
            value = json.loads(response.stdout)
            atomic(
                self.out / "publication.json",
                {"id": value["id"], "html_url": value["html_url"], "updated_unix": time.time()},
            )
            self.last_stage = status["stage"]
        except Exception as exc:
            atomic(self.out / "publication-error.json", {"error": repr(exc), "time": time.time()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--controller-unit", required=True)
    parser.add_argument("--evaluator-unit", required=True)
    parser.add_argument("--behavior-root", type=Path, required=True)
    parser.add_argument("--issue", type=int, default=60)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    watch = Watch(args)
    with run_lock(watch.out):
        while time.time() - watch.started < 7 * 3600:
            try:
                status = watch.snapshot()
                watch.publish(status)
            except Exception as exc:
                atomic(watch.out / "error.json", {"error": repr(exc), "time": time.time()})
                # Unreadable/corrupt verified data is unsafe; observation transport failure alone is not.
                if isinstance(exc, (ValueError, KeyError)):
                    request_stop(watch.root, repr(exc))
                if args.once:
                    raise
            if args.once:
                return
            time.sleep(30)
        atomic(watch.out / "expired.json", {"time": time.time(), "reason": "seven-hour supervisor limit"})


if __name__ == "__main__":
    main()
