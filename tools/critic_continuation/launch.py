"""Run the preregistered bounded producer/consumer pipeline; stop safely on any failed contract."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time

from run import ROOT, check_stop, checked_node, phase_root, sha, stop, verify_identity
from ygorl.solver.resume import output_lock
from ygorl.train.registration import atomic_json


def status(stage, **extra):
    value = {"stage": stage, "updated_unix": time.time(), **extra}
    atomic_json(ROOT / "pipeline-status.json", value)
    print(json.dumps(value), flush=True)


def produce():
    verify_identity()
    completed = []
    with output_lock(ROOT / "producer-report.json"):
        for seed_id in range(3):
            for arm in ("cold", "warm"):
                check_stop()
                root = phase_root(seed_id, arm)
                root.mkdir(parents=True, exist_ok=True)
                report = root / "report.json"
                if report.exists():
                    data = json.loads(report.read_text())
                    assert data["study_sha256"] == sha(ROOT / "identity.json")
                    for node in data["nodes"]:
                        checked_node(seed_id, arm, node["update"])
                else:
                    with (root / "training.log").open("x") as log:
                        subprocess.run(
                            [sys.executable, str(ROOT / "run.py"), "train", "--seed-id", str(seed_id), "--arm", arm],
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=True,
                        )
                completed.append({"seed_id": seed_id, "arm": arm, "report_sha256": sha(report)})
                print("COMPLETED PHASE", seed_id, arm, flush=True)
        atomic_json(ROOT / "producer-report.json", {"study_sha256": sha(ROOT / "identity.json"), "phases": completed})


def progress():
    result = {}
    for path in sorted(ROOT.glob("seed-*/*/run/metrics.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines(keepends=True) if line.endswith("\n")]
        result[str(path.parent.parent.relative_to(ROOT))] = len(rows)
    return result


def pipeline():
    verify_identity()
    check_stop()
    processes = []
    with output_lock(ROOT / "pipeline-status.json"):
        if (ROOT / "pipeline-status.json").exists():
            prior = ROOT / f"pipeline-status-before-resume-{time.time_ns()}.json"
            prior.write_bytes((ROOT / "pipeline-status.json").read_bytes())
        try:
            for label, command in [
                ("producer", [sys.executable, str(ROOT / "launch.py"), "produce"]),
                ("consumer", ["nice", "-n", "19", sys.executable, str(ROOT / "run.py"), "consume"]),
                ("retention", ["nice", "-n", "19", sys.executable, str(ROOT / "retention.py")]),
            ]:
                log_path = ROOT / (
                    f"{label}.log" if not (ROOT / f"{label}.log").exists() else f"{label}-resume-{time.time_ns()}.log"
                )
                with log_path.open("x") as log:
                    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                processes.append((label, child))
            started = last = time.monotonic()
            status("running", children={label: child.pid for label, child in processes})
            while any(child.poll() is None for _, child in processes):
                for label, child in processes:
                    assert child.poll() in (None, 0), f"{label} failed with exit {child.returncode}"
                check_stop()
                assert time.monotonic() - started < 48 * 3600, "study timed out"
                if time.monotonic() - last >= 60:
                    status(
                        "running", updates=progress(), child_exits={label: child.poll() for label, child in processes}
                    )
                    last = time.monotonic()
                time.sleep(5)
            assert all(child.returncode == 0 for _, child in processes)
            status("analyzing", updates=progress())
            analysis_log = ROOT / (
                "analysis.log" if not (ROOT / "analysis.log").exists() else f"analysis-resume-{time.time_ns()}.log"
            )
            with analysis_log.open("x") as log:
                subprocess.run(
                    [sys.executable, str(ROOT / "analyze.py")], stdout=log, stderr=subprocess.STDOUT, check=True
                )
            from audit import audit

            atomic_json(ROOT / "completion-audit.json", audit(ROOT))
            atomic_json(
                ROOT / "report.json",
                {
                    "study_sha256": sha(ROOT / "identity.json"),
                    "healthy": True,
                    "completion_audit_sha256": sha(ROOT / "completion-audit.json"),
                },
            )
            result = json.loads((ROOT / "analysis.json").read_text())
            status(
                "complete",
                continue_to_longer=result["criterion_to_test_longer_training"],
                primary=result["contrasts"]["512"],
                growth=result["growth"],
            )
        except BaseException as exc:
            stop(repr(exc), stage="pipeline")
            status("stopped", error=repr(exc), updates=progress())
            # Give owned children time to observe STOP at a safe boundary before terminating their own groups.
            deadline = time.monotonic() + 120
            while any(child.poll() is None for _, child in processes) and time.monotonic() < deadline:
                time.sleep(2)
            for _, child in processes:
                if child.poll() is None:
                    assert os.getpgid(child.pid) == child.pid
                    os.killpg(child.pid, signal.SIGTERM)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("produce", "all"))
    args = parser.parse_args()
    try:
        produce() if args.mode == "produce" else pipeline()
    except BaseException as exc:
        stop(repr(exc), mode=args.mode)
        raise
