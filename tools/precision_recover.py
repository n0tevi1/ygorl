"""Audited local-only controller recovery; worker and evaluation inputs stay frozen."""

import argparse
import importlib
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

from colab_owned_endpoint import endpoint_call, owned_endpoint


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def high_water(root):
    serial, generation = 0, -1
    for p in root.rglob("*"):
        if match := re.match(r"(?:cli|rpc|upload|cleanup|download)-(\d+)", p.name):
            serial = max(serial, int(match[1]))
        if match := re.fullmatch(r"(?:job|remote)-(\d+)\.json", p.name):
            generation = max(generation, int(match[1]))
    return serial, generation + 1


def remaining_budget(identity, registration, balance, allocations, now):
    remaining = {
        "seconds": identity["deadline_unix"] - now,
        "units": registration["max_units"] - (identity["initial_balance"] - balance),
        "allocations": registration["max_allocations"] - allocations,
    }
    require(all(v > 0 for v in remaining.values()), "original budget exhausted")
    return remaining


def stopped_units(units):
    for unit in units:
        result = subprocess.check_output(
            ["systemctl", "--user", "show", unit, "--property=ActiveState", "--value"], text=True, timeout=15
        ).strip()
        require(result in ("inactive", "failed"), f"prior service is not quiescent: {unit}: {result}")


def admit_transport_stop(stop, events, audit):
    """Only the original timeout stop or the specifically documented churn stop may resume."""
    failures = [e for e in events if e["stage"] == "failed"]
    require(failures and failures[-1] == audit["last_failure"], "audited failure differs")
    if stop.get("source") == "precision_watch":
        require(
            failures[-1]["error"] == "RuntimeError('colab exec timed out; remote state is unknown')",
            "not the audited transport failure",
        )
        require(
            stop.get("reason", "").startswith("required controller service unexpectedly inactive:"),
            "health or unrelated STOP requires separate investigation",
        )
        return
    require(
        stop.get("source") == "precision_recovery_transport_audit",
        "health or unrelated STOP requires separate investigation",
    )
    require(
        stop.get("reason")
        == "Stop repeated poll transport recovery before exhausting original allocation budget; retain all verified artifacts pending fenced poll retry amendment.",
        "unrelated operator STOP requires separate investigation",
    )
    require(failures[-1]["error"] == "RuntimeError('STOP.json requested study stop')", "operator STOP failure differs")
    recorded = [e for e in events if e["stage"] == "transport_recovery"][-2:]
    require(
        len(recorded) == 2 and recorded == stop.get("transport_events") == audit.get("transport_events"),
        "audited transport events differ",
    )
    require(
        recorded[0]["time"] < recorded[1]["time"] <= stop["time"] <= failures[-1]["time"],
        "operator stop chronology differs",
    )
    require(
        not [e for e in failures[:-1] if e["time"] >= recorded[0]["time"]],
        "additional failure requires separate investigation",
    )
    for event in recorded:
        require(
            re.fullmatch(r"RuntimeError\('colab exec failed; see cli-\d+\.log'\)", event["error"]) is not None,
            "not an audited polling transport failure",
        )
        diagnostic = event.get("diagnostic", {})
        if diagnostic.get("status"):
            status = diagnostic["status"]
            require(
                status.get("stage") in ("running", "complete") and status.get("generation") == event["generation"],
                "diagnostic worker failed or differs",
            )
            require(
                status.get("counters", {}).get("errors") == 0 and status.get("counters", {}).get("truncated") == 0,
                "diagnostic health requires separate investigation",
            )
        else:
            require(
                diagnostic.get("unverified_suffix_unknown") is True
                and diagnostic.get("diagnostic_error")
                == "RuntimeError('colab exec timed out; remote state is unknown')",
                "unbound diagnostic uncertainty",
            )


def controller_class(frozen_repo):
    """Import the exact registered driver, not a second copy from this amendment's worktree."""
    sys.path.insert(0, str(frozen_repo / "tools"))
    base = importlib.import_module("precision_colab")
    require(Path(base.__file__).resolve() == frozen_repo / "tools/precision_colab.py", "wrong frozen controller import")
    checkpoint = importlib.import_module("colab_checkpoint")
    evaluate = importlib.import_module("precision_evaluate")
    trainer = importlib.import_module("ygorl.train.trainer")
    require(
        Path(trainer.__file__).resolve() == frozen_repo / "src/ygorl/train/trainer.py",
        "actual training imports are not from frozen source; set PYTHONPATH to frozen/src",
    )
    atomic, sha = checkpoint.atomic, checkpoint.sha

    class Recovery(base.PrecisionPilot):
        def __init__(self, root, audit):
            super().__init__(root)
            self.audit_path = Path(audit).resolve()
            self.audit = read(self.audit_path)
            self.identity = read(self.root / "identity.json")
            self.deadline_unix = self.identity["deadline_unix"]
            self.deadline = time.monotonic() + max(0, self.deadline_unix - time.time())
            self.initial_balance = self.identity["initial_balance"]
            self.owned = read(self.root / "owned-sessions.json")
            require(len(set(self.owned)) == len(self.owned), "duplicate owned sessions")
            require(
                self.owned == [f"{self.run_id}-vm{i}" for i in range(len(self.owned))], "allocation history differs"
            )
            self.allocation_count = len(self.owned)
            self.serial, self.generation = high_water(self.root)
            self.history = Path.home() / ".config/colab-cli/history"
            self.endpoints = {}
            self.attempt = None
            require(self.identity["jobs"] == self.identities, "saved job identities differ")
            require(self.identity["registration_sha256"] == self.registration_sha, "registration differs")
            require(self.identity["package_sha256"] == self.package_sha, "package differs")
            require(self.identity["run_id"] == self.run_id, "run differs")

        def ownership(self, session):
            history = self.history / (session + ".jsonl")
            endpoint = self.endpoints.get(session) or owned_endpoint(
                session, history, self.root / "owned-sessions.json"
            )
            self.endpoints[session] = endpoint
            return dict(
                session=session,
                endpoint=endpoint,
                history=str(history),
                owned_sessions=str(self.root / "owned-sessions.json"),
            )

        def balance(self):
            value = self.cli_call(["usage"])
            match = re.search(r"Current balance:\s*([\d.]+)", value)
            require(match is not None, "missing account balance")
            return float(match[1])

        def admission(self):
            require(sha(self.root / "identity.json") == self.audit["study_sha256"], "study identity changed")
            require(sha(self.root / "STOP.json") == self.audit["stop_sha256"], "STOP changed")
            for path, expected in self.audit["recovery_files"].items():
                require(sha(path) == expected, "recovery implementation changed")
            require(str(Path(__file__).resolve()) in self.audit["recovery_files"], "unbound recovery controller")
            require(
                str(Path(__file__).with_name("colab_owned_endpoint.py").resolve()) in self.audit["recovery_files"],
                "unbound endpoint helper",
            )
            events = [json.loads(s) for s in (self.root / "events.jsonl").read_text().splitlines()]
            stop = read(self.root / "STOP.json")
            admit_transport_stop(stop, events, self.audit)
            stopped_units(self.audit["quiescent_units"])
            assignments = endpoint_call("list")
            require(not assignments["endpoints"], "server assignments still present; no replacement permitted")
            budget = remaining_budget(
                self.identity, self.registration, self.balance(), self.allocation_count, time.time()
            )
            require(
                not [p for p in self.root.glob("evaluation/*/*.jsonl") if not p.with_suffix(".json").exists()],
                "partial evaluation cell; refuse reroll",
            )
            evaluate.bind_inputs(self.root)  # includes independent evaluator identity/source/environment hashes
            for p in self.root.glob("evaluation/*/*.json"):
                evaluate.read_cell(p)
            progress = {}
            for spec in self.registration["jobs"]:
                name = spec["job_id"]
                pointer = self.root / "durable" / name / "latest.json"
                progress[name] = 128
                if pointer.exists():
                    directory, manifest = self.snapshot(spec)
                    progress[name] = manifest["counters"]["updates"]
                completed = self.root / "completed" / (name + ".json")
                if completed.exists():
                    evaluate.completed_candidate(self.root, spec, self.identity)
                    require(progress[name] == 160, "completed job missing latest endpoint")
                    require(
                        read(completed)["checkpoint_sha256"] == sha(directory / "checkpoint.pt"),
                        "completed/latest endpoint disagree",
                    )
            return {
                "budget": budget,
                "progress": progress,
                "server": assignments,
                "next_generation": self.generation,
                "next_allocation": self.allocation_count,
                "next_serial_above": self.serial,
            }

        def snapshot(self, spec):
            store = self.root / "durable" / spec["job_id"]
            pointer = read(store / "latest.json")
            directory = (store / pointer["path"]).resolve()
            require(directory.is_relative_to(store.resolve()), "snapshot escapes store")
            base.validate_precision(directory, self.identities[spec["job_id"]])
            manifest = read(directory / "manifest.json")
            require(pointer["sha256"] == sha(directory / "checkpoint.pt"), "latest checkpoint changed")
            require(pointer["update"] == manifest["counters"]["updates"], "latest counter mismatch")
            return directory, manifest

        def finish(self, spec):
            directory, manifest = self.snapshot(spec)
            require(manifest["counters"]["updates"] == 160, "endpoint incomplete")
            target = self.root / "completed" / (spec["job_id"] + ".json")
            if target.exists():
                evaluate.completed_candidate(self.root, spec, self.identity)
                return
            atomic(
                target,
                {
                    **spec,
                    "checkpoint": str(directory / "checkpoint.pt"),
                    "checkpoint_sha256": sha(directory / "checkpoint.pt"),
                    "metrics": str(directory / "metrics.jsonl"),
                    "games": str(directory / "games.jsonl.gz"),
                    "snapshot": str(directory),
                    "identity": self.identities[spec["job_id"]],
                    "completed_unix": time.time(),
                },
            )
            self.event("arm_complete", job_id=spec["job_id"], snapshot=str(directory))

        def allocate(self):
            require(not endpoint_call("list")["endpoints"], "server assignment remains before allocation")
            session = super().allocate()
            ownership = self.ownership(session)
            require(ownership["endpoint"] in endpoint_call("list")["endpoints"], "new endpoint absent from server")
            atomic(self.root / ("endpoint-" + session + ".json"), ownership)
            helper = Path(__file__).with_name("colab_owned_endpoint.py")
            args = [sys.executable, str(helper), "stop"]
            for key, value in ownership.items():
                args.extend(["--" + key.replace("_", "-"), value])
            subprocess.run(
                [
                    "systemd-run",
                    "--user",
                    "--unit=" + session + "-endpoint-expiry",
                    "--on-active=" + str(max(1, int(self.deadline_unix - time.time()))) + "s",
                    *args,
                ],
                check=True,
                capture_output=True,
            )
            self.event("endpoint_bound", **ownership)
            return session

        def release(self, session):
            require(session in self.owned, "unowned session")
            record = endpoint_call("stop", **self.ownership(session))
            require(record["absent"], "endpoint termination unconfirmed")
            self.event("session_released", session=session, endpoint=self.endpoints[session])

        def diagnose(self, session, job):
            self.serial += 1
            path = self.root / f"rpc-{self.serial:04d}.py"
            path.write_text(
                "import json,pathlib\nr=pathlib.Path(" + repr(job["out"]) + ")\n"
                "s=r/'status.json';p=r/'worker.log'\n"
                "print('YGORL_JSON:'+json.dumps({'status':json.loads(s.read_text()) if s.exists() else None,"
                "'log_tail':p.read_text()[-4000:] if p.exists() else ''}))\n"
            )
            try:
                result = self.cli_call(["exec", "-s", session, "-f", str(path)], timeout=30)
                value = json.loads(next(x.split(":", 1)[1] for x in result.splitlines() if x.startswith("YGORL_JSON:")))
            except Exception as exc:
                value = {"diagnostic_error": repr(exc), "unverified_suffix_unknown": True}
            atomic(self.root / f"diagnostic-{self.generation}.json", value)
            return value

        def poll_job(self, session, job):
            """Retry only a fenced status/lease RPC, after a fresh read-only health probe."""

            def rpc(*, probe=False):
                self.guard()
                remaining = self.deadline_unix - time.time()
                require(remaining > 0, "original study deadline reached")
                self.serial += 1
                path = self.root / f"rpc-{self.serial:04d}.py"
                path.write_text(f"""import json,pathlib,time,os,math
root=pathlib.Path({job["out"]!r});lease=pathlib.Path({job["lease"]!r})
now=time.time();old=json.loads(lease.read_text())
rejected=None
if old.get('generation')!={job["generation"]} or old.get('job_id')!={job["job_id"]!r}: rejected='wrong job or generation'
elif not isinstance(old.get('expires_unix'),(int,float)) or not math.isfinite(old['expires_unix']): rejected='invalid lease expiry'
elif now>=min(old['expires_unix'],{self.deadline_unix}): rejected='expired lease or deadline'
if rejected:
 print('YGORL_JSON:'+json.dumps({{'poll_rejected':rejected}}))
else:
 if not {probe!r}:
  old['expires_unix']=min(now+300,{self.deadline_unix})
  tmp=lease.with_suffix('.tmp');tmp.write_text(json.dumps(old));tmp.replace(lease)
 pid=int((root/'worker.pid').read_text())
 try:
  os.kill(pid,0);alive=pathlib.Path(f'/proc/{{pid}}/stat').read_text().split(') ',1)[1].split()[0]!='Z'
 except (ProcessLookupError,FileNotFoundError): alive=False
 def read(name):
  p=root/name;return json.loads(p.read_text()) if p.exists() else None
 host=None
 if {probe!r}:
  try:
   p=pathlib.Path(f'/proc/{{pid}}/status');worker=p.read_text().splitlines() if p.exists() else []
   host={{'loadavg':pathlib.Path('/proc/loadavg').read_text().strip(),'cpus':os.cpu_count(),'memory':[line for line in pathlib.Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')],'worker':[line for line in worker if line.startswith(('VmRSS:','Threads:'))]}}
  except OSError as error: host={{'read_error':type(error).__name__}}
 print('YGORL_JSON:'+json.dumps({{'alive':alive,'status':read('status.json'),'restore':read('restore.json'),'precision':read('precision-evidence.json'),'snapshots':sorted(str(p.parent) for p in (root/'snapshots').glob('*/precision-manifest.json')),'log_tail':(root/'worker.log').read_text()[-4000:],'host_probe':host,'poll_fence':{{'job_id':old['job_id'],'generation':old['generation'],'lease_expires_unix':old['expires_unix'],'server_unix':time.time()}}}}))
""")
                output = self.cli_call(["exec", "-s", session, "-f", str(path)], timeout=min(45, remaining))
                lines = [s for s in output.splitlines() if s.startswith("YGORL_JSON:")]
                require(bool(lines), "poll acknowledgement missing; execution outcome unknown")
                value = json.loads(lines[-1].split(":", 1)[1])
                require(not value.get("poll_rejected"), "poll fence rejected: " + str(value.get("poll_rejected")))
                fence = value["poll_fence"]
                require(
                    all(math.isfinite(fence[k]) for k in ("lease_expires_unix", "server_unix")), "nonfinite poll fence"
                )
                require(
                    fence["generation"] == job["generation"] and fence["job_id"] == job["job_id"],
                    "poll acknowledgement fence differs",
                )
                status = value.get("status") or {}
                require(status.get("generation", job["generation"]) == job["generation"], "worker generation differs")
                return value

            try:
                return rpc()
            except RuntimeError as exc:
                # These are stable errors emitted by our CLI wrapper, not parsed remote traceback text.
                if not str(exc).startswith(("colab exec failed;", "colab exec timed out;")):
                    raise
                self.event(
                    "poll_transport_error",
                    job_id=job["job_id"],
                    generation=job["generation"],
                    error=repr(exc),
                    outcome="unknown; only lease/status RPC may have executed",
                )
                try:
                    info = rpc(probe=True)
                except ValueError:
                    raise  # a confirmed fence/identity rejection must stop the study
                except Exception as probe_error:
                    self.event(
                        "poll_probe_unavailable",
                        job_id=job["job_id"],
                        generation=job["generation"],
                        error=repr(probe_error),
                    )
                    raise exc  # existing endpoint-confirmed cleanup handles unresolved transport failure
                atomic(self.root / f"poll-probe-{job['generation']}-{self.serial}.json", info)
                status = info.get("status") or {}
                require(status.get("stage") != "failed", "poll probe reports worker failure")
                require(info["alive"] or status.get("stage") == "complete", "poll probe reports dead worker")
                require(status.get("stage") in ("running", "complete"), "poll probe cannot establish healthy worker")
                updated = status.get("updated_unix", float("nan"))
                require(
                    math.isfinite(updated) and 0 <= info["poll_fence"]["server_unix"] - updated <= 300,
                    "poll probe status is stale or has invalid time",
                )
                counts = status.get("counters", {})
                require(
                    all(not isinstance(value, (int, float)) or math.isfinite(value) for value in counts.values()),
                    "poll probe reports nonfinite counters",
                )
                origin = job["identity"]["origin_counts"]
                require(counts.get("errors", 0) <= origin.get("errors", 0), "poll probe reports training errors")
                games = counts.get("games", 0) - origin.get("games", 0)
                truncated = counts.get("truncated", 0) - origin.get("truncated", 0)
                require(games < 100 or truncated / games <= 0.01, "poll probe reports unhealthy truncations")
                lease_margin = info["poll_fence"]["lease_expires_unix"] - info["poll_fence"]["server_unix"]
                require(
                    lease_margin >= 60 and self.deadline_unix - time.time() >= 60,
                    "insufficient lease/deadline margin for poll retry",
                )
                self.event(
                    "poll_retry", job_id=job["job_id"], generation=job["generation"], lease_margin=lease_margin, retry=1
                )
                return rpc()  # exactly one retry; its transport failure follows existing cleanup/recovery

        def main(self, plan_only=False):
            with checkpoint.run_lock(self.root / "controller"):
                plan = self.admission()
                if plan_only:
                    print(json.dumps(plan, indent=2))
                    return
                recoveries = self.root / "recoveries"
                recoveries.mkdir(exist_ok=True)
                number = max([int(p.name) for p in recoveries.iterdir() if p.is_dir() and p.name.isdigit()] + [0]) + 1
                self.attempt = recoveries / str(number)
                self.attempt.mkdir()
                atomic(
                    self.attempt / "admission.json",
                    {
                        "audit_sha256": sha(self.audit_path),
                        "plan": plan,
                        "started_unix": time.time(),
                        "original_identity_sha256": sha(self.root / "identity.json"),
                    },
                )
                # Prior supervisors are quiescent; preserve the original STOP rather than overwrite or delete it.
                with evaluate.output_lock(self.root / "STOP.json"):
                    require(sha(self.root / "STOP.json") == self.audit["stop_sha256"], "STOP changed during admission")
                    (self.root / "STOP.json").rename(self.attempt / "previous-STOP.json")
                self.event("recovering", admission=str(self.attempt / "admission.json"))
                session = None
                terminal = None
                try:
                    for spec in self.registration["jobs"]:
                        name = spec["job_id"]
                        pointer = self.root / "durable" / name / "latest.json"
                        if pointer.exists() and self.snapshot(spec)[1]["counters"]["updates"] == 160:
                            self.finish(spec)
                            continue
                        while True:
                            job = None
                            try:
                                if session is None:
                                    session = self.allocate()
                                    self.setup(session)
                                job = self.launch_job(session, spec)
                                seen = set()
                                while True:
                                    info = self.poll_job(session, job)
                                    atomic(self.root / f"remote-{self.generation}.json", info)
                                    status = info["status"]
                                    if status and status["stage"] == "failed":
                                        self.preserve_failure(session, job)
                                        raise ValueError("worker failure: " + status["error"])
                                    if not info["alive"] and (not status or status["stage"] != "complete"):
                                        self.preserve_failure(session, job)
                                        raise ValueError("worker died without completion")
                                    for path in info["snapshots"]:
                                        if path not in seen:
                                            self.sync_job(session, path, job)
                                            seen.add(path)
                                    if status and status["stage"] == "complete" and not info["alive"]:
                                        self.finish(spec)
                                        break
                                    self.event("running", job_id=name, generation=self.generation, status=status)
                                    time.sleep(30)
                                self.generation += 1
                                break
                            except RuntimeError as exc:
                                self.guard(usage=True)
                                if not (
                                    str(exc).startswith("colab ")
                                    and ("timed out" in str(exc) or " failed;" in str(exc))
                                ):
                                    raise
                                if session is None:
                                    raise
                                diagnostic = self.diagnose(session, job) if job else {"setup_failure": repr(exc)}
                                if (diagnostic.get("status") or {}).get("stage") == "failed":
                                    raise ValueError("diagnostic reports worker failure") from exc
                                # Name tracking can disappear without real preemption. Terminate the exact owned endpoint.
                                self.release(session)
                                self.event(
                                    "transport_recovery",
                                    session=session,
                                    job_id=name,
                                    generation=self.generation,
                                    error=repr(exc),
                                    diagnostic=diagnostic,
                                )
                                session = None
                                self.generation += 1
                                if pointer.exists() and self.snapshot(spec)[1]["counters"]["updates"] == 160:
                                    self.finish(spec)
                                    break
                    terminal = {"stage": "training_complete", "completed_arms": 6}
                except BaseException as exc:
                    terminal = {"stage": "failed", "error": repr(exc)}
                    with evaluate.output_lock(self.root / "STOP.json"):
                        if not (self.root / "STOP.json").exists():
                            atomic(
                                self.root / "STOP.json",
                                {"source": "precision_recover", "reason": repr(exc), "time": time.time()},
                            )
                    raise
                finally:
                    for owned in self.owned:
                        try:
                            self.release(owned)
                        except Exception as exc:
                            self.event("cleanup_failed", session=owned, error=repr(exc))
                            terminal = {"stage": "failed", "error": "cleanup unconfirmed: " + repr(exc)}
                    if terminal:
                        self.event(**terminal)

    return Recovery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--frozen-repo", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    controller_class(args.frozen_repo.resolve())(args.root, args.audit).main(args.plan_only)


if __name__ == "__main__":
    main()
