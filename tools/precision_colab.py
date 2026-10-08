"""One owned L4 at a time; bounded paired precision jobs and verified recovery."""

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

from colab_checkpoint import atomic, promote, run_lock, sha
from colab_pilot import Pilot
from precision_worker import digest, expected_config, validate_precision
from ygorl.train.checkpoint import load_checkpoint


def validate_registration(registration):
    fixed = {"origin_update": 128, "target_update": 160, "snapshot_every": 4, "lease_seconds": 300}
    if any(registration.get(k) != v for k, v in fixed.items()):
        raise ValueError("registration differs from approved pilot")
    for key, ceiling in (("max_hours", 6), ("max_units", 12), ("max_allocations", 8)):
        if not 0 < registration[key] <= ceiling:
            raise ValueError("budget exceeds approved pilot")
    if not re.fullmatch(r"[a-z0-9-]{1,48}", registration["run_id"]):
        raise ValueError("unsafe run ID")
    expected = [(0, "fp32"), (0, "bf16"), (1, "bf16"), (1, "fp32"), (2, "fp32"), (2, "bf16")]
    jobs = registration["jobs"]
    if [(j["seed"], j["learner_precision"]) for j in jobs] != expected:
        raise ValueError("registered paired job order differs")
    if any(j["job_id"] != f"seed-{j['seed']}-{j['learner_precision']}" for j in jobs):
        raise ValueError("invalid job ID")
    for seed in range(3):
        pair = [j for j in jobs if j["seed"] == seed]
        if pair[0]["checkpoint_sha256"] != pair[1]["checkpoint_sha256"]:
            raise ValueError("paired initial checkpoints differ")


def active_names(listing):
    names = set(re.findall(r"^\[([^\]]+)\].*\| Hardware:", listing, re.MULTILINE))
    if not names and "No active sessions found on server." not in listing:
        raise RuntimeError("cannot interpret server session listing")
    return names


class PrecisionPilot(Pilot):
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.repo = Path(__file__).resolve().parents[1]
        self.cli = "/home/ya0guang/.local/bin/colab"
        self.registration = json.loads((self.root / "registration.json").read_text())
        validate_registration(self.registration)
        self.registration_sha = sha(self.root / "registration.json")
        self.package = json.loads((self.root / "package.json").read_text())
        self.package_sha = sha(self.root / "package.json")
        self.run_id = self.registration["run_id"]
        self.deadline_unix = time.time() + self.registration["max_hours"] * 3600
        self.deadline = time.monotonic() + self.registration["max_hours"] * 3600
        self.serial = 0
        self.owned = []
        self.events = self.root / "events.jsonl"
        self.identities = {}
        self.decks = {}
        self.allocation_count = 0
        self.generation = 0
        self.last_usage_check = 0
        self.check_inputs()

    def check_inputs(self):
        if (
            sha(self.root / "registration.json") != self.registration_sha
            or sha(self.root / "package.json") != self.package_sha
        ):
            raise ValueError("controller inputs changed after registration")
        for name in ("archive", "manifest", "bootstrap"):
            if sha(self.package[name]) != self.package[name + "_sha256"]:
                raise ValueError(f"package {name} differs")
        manifest = json.loads(Path(self.package["manifest"]).read_text())
        for name in (
            "precision_colab.py",
            "precision_worker.py",
            "colab_pilot.py",
            "colab_checkpoint.py",
            "colab_worker.py",
        ):
            if manifest["files"].get("tools/" + name) != sha(self.repo / "tools" / name):
                raise ValueError("live driver differs from registered source")
        for job in self.registration["jobs"]:
            if sha(job["checkpoint"]) != job["checkpoint_sha256"]:
                raise ValueError("registered initial checkpoint differs")
            source = load_checkpoint(job["checkpoint"])
            if source["counters"]["updates"] != self.registration["origin_update"]:
                raise ValueError("initial update differs")
            self.identities[job["job_id"]] = {
                "run_id": self.run_id + "/" + job["job_id"],
                "job_id": job["job_id"],
                "seed": job["seed"],
                "learner_precision": job["learner_precision"],
                "registration_sha256": self.registration_sha,
                "package_sha256": self.package_sha,
                "source_checkpoint_sha256": job["checkpoint_sha256"],
                "environment": source["environment"],
                "origin_counts": source["counters"],
                "origin_update": self.registration["origin_update"],
                "target_update": self.registration["target_update"],
                "config_sha256": digest(expected_config(source["config"], job["learner_precision"])),
            }
            self.decks[job["job_id"]] = {Path(p).name: sha(p) for p in source["config"]["decks"]}

    def guard(self, *, usage=False):
        if (self.root / "STOP.json").exists():
            raise RuntimeError("STOP.json requested study stop")
        if time.time() >= self.deadline_unix:
            raise RuntimeError("study wall budget expired")
        if usage or time.monotonic() - self.last_usage_check >= 120:
            value = self.cli_call(["usage"])
            match = re.search(r"Current balance:\s*([\d.]+)", value)
            if not match:
                raise RuntimeError("cannot verify compute-unit balance")
            balance = float(match[1])
            if self.initial_balance - balance >= self.registration["max_units"]:
                raise RuntimeError("study compute-unit budget reached")
            self.last_usage_check = time.monotonic()

    def allocate(self):
        self.guard(usage=True)
        if self.allocation_count >= self.registration["max_allocations"]:
            raise RuntimeError("allocation budget exhausted")
        self.check_inputs()
        listing = self.cli_call(["sessions"])
        if active_names(listing).intersection(self.owned):
            raise RuntimeError("old owned VM still exists; refuse duplicate allocation")
        session = f"{self.run_id}-vm{self.allocation_count}"
        if session in active_names(listing):
            raise RuntimeError("session name already exists")
        self.owned.append(session)
        self.allocation_count += 1
        atomic(self.root / "owned-sessions.json", self.owned)
        # A local timer exists before allocation: even an ambiguous allocation timeout remains bounded.
        remaining = max(1, int(self.deadline_unix - time.time()))
        subprocess.run(
            [
                "systemd-run",
                "--user",
                f"--unit={session}-expiry",
                f"--on-active={remaining}s",
                self.cli,
                "stop",
                "-s",
                session,
            ],
            check=True,
            capture_output=True,
        )
        self.cli_call(["new", "-s", session, "--gpu", "L4"], timeout=300)
        self.event("allocated", session=session, allocations=self.allocation_count)
        return session

    def setup(self, session):
        self.upload_large(session, self.package["archive"], "/content/source-v2.tar.gz")
        self.upload(session, self.package["manifest"], "/content/source-manifest.json")
        self.cli_call(["exec", "-s", session, "-f", self.package["bootstrap"]])
        until = min(self.deadline_unix, time.time() + 1200)
        while time.time() < until:
            self.guard()
            status = self.remote(
                session,
                """import json,pathlib
p=pathlib.Path('/content/setup.exit')
print('YGORL_JSON:'+json.dumps({'exit':p.read_text().strip() if p.exists() else None}))
""",
            )
            if status["exit"] is not None:
                if status["exit"] != "0":
                    self.cli_call(
                        ["download", "-s", session, "/content/setup.log", str(self.root / f"setup-{session}.log")],
                        check=False,
                    )
                    raise RuntimeError("remote build failed")
                self.cli_call(
                    ["download", "-s", session, "/content/setup.log", str(self.root / f"setup-{session}.log")]
                )
                return
            self.event("building", session=session)
            time.sleep(30)
        raise RuntimeError("remote setup timeout")

    def launch_job(self, session, spec):
        self.guard()
        identity = self.identities[spec["job_id"]]
        store = self.root / "durable" / spec["job_id"]
        pointer = store / "latest.json"
        directory = store / json.loads(pointer.read_text())["path"] if pointer.exists() else None
        if directory:
            validate_precision(directory, identity)
        source = directory / "checkpoint.pt" if directory else Path(spec["checkpoint"])
        remote_input = f"/content/input-{self.generation}"
        self.remote(
            session, f"import pathlib,json;pathlib.Path({remote_input!r}).mkdir(exist_ok=True);print('YGORL_JSON:{{}}')"
        )
        self.upload_large(session, source, remote_input + "/checkpoint.pt")
        if directory:
            for name in ("metrics.jsonl", "games.jsonl.gz"):
                self.upload_large(session, directory / name, remote_input + "/" + name)
        job = {
            **spec,
            **{k: self.registration[k] for k in ("origin_update", "target_update", "snapshot_every")},
            "run_id": identity["run_id"],
            "identity": identity,
            "environment": identity["environment"],
            "generation": self.generation,
            "migration": directory is None,
            "checkpoint": remote_input + "/checkpoint.pt",
            "checkpoint_sha256": sha(source),
            "metrics": remote_input + "/metrics.jsonl" if directory else None,
            "games": remote_input + "/games.jsonl.gz" if directory else None,
            "deck_sha256": self.decks[spec["job_id"]],
            "deadline_unix": self.deadline_unix,
            "out": f"/content/precision/g{self.generation}",
            "lease": "/content/precision/lease.json",
        }
        local = self.root / f"job-{self.generation}.json"
        atomic(local, job)
        self.upload(session, local, "/content/job.json")
        result = self.remote(
            session,
            f"""import hashlib,json,pathlib,os,subprocess,time
root=pathlib.Path('/content/ygorl')
m=json.loads(pathlib.Path('/content/source-manifest.json').read_text())
assert hashlib.sha256(pathlib.Path('/content/source-manifest.json').read_bytes()).hexdigest()=={self.package["manifest_sha256"]!r}
for name,digest in m['files'].items():
 p=root/name; assert p.resolve().is_relative_to(root.resolve()),name
 assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,name
job=json.loads(pathlib.Path('/content/job.json').read_text())
assert hashlib.sha256(pathlib.Path('/content/job.json').read_bytes()).hexdigest()=={sha(local)!r}
out=pathlib.Path(job['out']);out.mkdir(parents=True,exist_ok=False)
lease=pathlib.Path(job['lease']);lease.parent.mkdir(parents=True,exist_ok=True)
lease.write_text(json.dumps({{'job_id':job['job_id'],'generation':job['generation'],'expires_unix':min(time.time()+300,job['deadline_unix'])}}))
env={{**os.environ,'PYTHONPATH':'src','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}}
with (out/'worker.log').open('w') as log:
 p=subprocess.Popen([str(root/'.venv/bin/python'),'-u','tools/precision_worker.py','--config','/content/job.json'],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(out/'worker.pid').write_text(str(p.pid))
print('YGORL_JSON:'+json.dumps({{'pid':p.pid,'source_files_verified':len(m['files'])}}))
""",
        )
        self.event("launched", session=session, job_id=spec["job_id"], generation=self.generation, **result)
        return job

    def poll_job(self, session, job):
        self.guard()
        return self.remote(
            session,
            f"""import json,pathlib,time,os
root=pathlib.Path({job["out"]!r});lease=pathlib.Path({job["lease"]!r})
old=json.loads(lease.read_text());assert old['generation']=={job["generation"]} and old['job_id']=={job["job_id"]!r} and time.time()<old['expires_unix']
tmp=lease.with_suffix('.tmp');tmp.write_text(json.dumps({{'generation':{job["generation"]},'job_id':{job["job_id"]!r},'expires_unix':min(time.time()+300,{self.deadline_unix})}}));tmp.replace(lease)
pid=int((root/'worker.pid').read_text())
try:
 os.kill(pid,0);alive=pathlib.Path(f'/proc/{{pid}}/stat').read_text().split(') ',1)[1].split()[0]!='Z'
except ProcessLookupError: alive=False
def read(name):
 p=root/name;return json.loads(p.read_text()) if p.exists() else None
print('YGORL_JSON:'+json.dumps({{'alive':alive,'status':read('status.json'),'restore':read('restore.json'),'precision':read('precision-evidence.json'),'snapshots':sorted(str(p.parent) for p in (root/'snapshots').glob('*/precision-manifest.json')),'log_tail':(root/'worker.log').read_text()[-4000:]}}))
""",
        )

    def sync_job(self, session, remote, job):
        self.guard()
        store = self.root / "durable" / job["job_id"]
        self.serial += 1
        staging = store / f"download-{self.serial:05d}"
        staging.mkdir(parents=True, exist_ok=False)
        for name in ("manifest.json", "checkpoint.pt", "metrics.jsonl", "games.jsonl.gz", "precision-manifest.json"):
            self.poll_job(session, job)
            self.cli_call(["download", "-s", session, remote + "/" + name, str(staging / name)], timeout=180)
        self.poll_job(session, job)
        validate_precision(staging, job["identity"])
        with run_lock(store):
            destination = promote(staging, store, job["run_id"], job["environment"])
        self.event("checkpoint_verified", job_id=job["job_id"], generation=job["generation"], snapshot=str(destination))
        return destination

    def release(self, session):
        if session not in self.owned:
            raise ValueError("refuse to release unowned session")
        self.cli_call(["stop", "-s", session], check=False)
        # Cleanup works after the deadline; a failed listing is never evidence of absence.
        listing = subprocess.run([self.cli, "sessions"], capture_output=True, text=True, timeout=60, check=True)
        self.serial += 1
        (self.root / f"cleanup-{self.serial:04d}.log").write_text(listing.stdout + listing.stderr)
        if session in active_names(listing.stdout):
            raise RuntimeError("owned session termination not confirmed")
        self.event("session_released", session=session)

    def preserve_failure(self, session, job):
        for name in (
            "worker.log",
            "status.json",
            "restore.json",
            "precision-evidence.json",
            "run/errors.jsonl",
            "run/truncations.jsonl",
            "run/games.jsonl.gz",
        ):
            destination = self.root / "failures" / str(job["generation"]) / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            self.cli_call(["download", "-s", session, job["out"] + "/" + name, str(destination)], check=False)

    def main(self):
        with run_lock(self.root / "controller"):
            if (self.root / "identity.json").exists():
                raise RuntimeError("controller already started; inspect retained identity before manual recovery")
            usage = self.cli_call(["usage"])
            match = re.search(r"Current balance:\s*([\d.]+)", usage)
            if not match:
                raise RuntimeError("cannot verify starting compute balance")
            self.initial_balance = float(match[1])
            atomic(
                self.root / "identity.json",
                {
                    "run_id": self.run_id,
                    "registration_sha256": self.registration_sha,
                    "package_sha256": self.package_sha,
                    "jobs": self.identities,
                    "started_unix": time.time(),
                    "deadline_unix": self.deadline_unix,
                    "initial_balance": self.initial_balance,
                },
            )
            session = None
            setup_done = False
            try:
                for spec in self.registration["jobs"]:
                    while True:
                        job = None
                        try:
                            if session is None:
                                session = self.allocate()
                                setup_done = False
                            if not setup_done:
                                self.setup(session)
                                setup_done = True
                            job = self.launch_job(session, spec)
                            seen = set()
                            while True:
                                info = self.poll_job(session, job)
                                atomic(self.root / f"remote-{self.generation}.json", info)
                                status = info["status"]
                                if status and status["stage"] == "failed":
                                    self.preserve_failure(session, job)
                                    raise ValueError("remote worker failed: " + status["error"])
                                if not info["alive"] and (not status or status["stage"] != "complete"):
                                    self.preserve_failure(session, job)
                                    raise ValueError("worker died without completion")
                                for path in info["snapshots"]:
                                    if path not in seen:
                                        self.sync_job(session, path, job)
                                        seen.add(path)
                                if status and status["stage"] == "complete":
                                    store = self.root / "durable" / spec["job_id"]
                                    pointer = json.loads((store / "latest.json").read_text())
                                    if pointer["update"] != self.registration["target_update"]:
                                        raise ValueError("completion without final verified checkpoint")
                                    directory = store / pointer["path"]
                                    validate_precision(directory, job["identity"])
                                    if info["alive"]:
                                        time.sleep(5)
                                        continue
                                    atomic(
                                        self.root / "completed" / (spec["job_id"] + ".json"),
                                        {
                                            **spec,
                                            "checkpoint": str(directory / "checkpoint.pt"),
                                            "checkpoint_sha256": pointer["sha256"],
                                            "metrics": str(directory / "metrics.jsonl"),
                                            "games": str(directory / "games.jsonl.gz"),
                                            "snapshot": str(directory),
                                            "identity": job["identity"],
                                            "completed_unix": time.time(),
                                        },
                                    )
                                    self.event("arm_complete", job_id=spec["job_id"], snapshot=str(directory))
                                    break
                                self.event("running", job_id=spec["job_id"], generation=self.generation, status=status)
                                time.sleep(30)
                            self.generation += 1
                            break
                        except (RuntimeError, subprocess.TimeoutExpired) as exc:
                            self.guard()
                            # Transport/setup/download failures are recoverable only after server-confirmed VM loss.
                            if session is None or session in active_names(self.cli_call(["sessions"])):
                                raise
                            self.event(
                                "preempted",
                                session=session,
                                job_id=spec["job_id"],
                                generation=self.generation,
                                error=repr(exc),
                            )
                            session = None
                            self.generation += 1
                self.event("training_complete", completed_arms=len(self.registration["jobs"]))
            except BaseException as exc:
                self.event("stopped" if (self.root / "STOP.json").exists() else "failed", error=repr(exc))
                raise
            finally:
                for owned in self.owned:
                    try:
                        self.release(owned)
                    except BaseException as exc:
                        self.event("cleanup_failed", session=owned, error=repr(exc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    PrecisionPilot(parser.parse_args().root).main()
