"""Bounded two-VM recovery pilot using the installed Colab CLI and verified local snapshots."""

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

from colab_checkpoint import atomic, promote, run_lock, sha, validate
from ygorl.train.checkpoint import load_checkpoint


class Pilot:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.repo = Path(__file__).resolve().parents[1]
        self.cli = "/home/ya0guang/.local/bin/colab"
        self.run_id = "ygorl-colab-recovery-20261008"
        self.deadline = time.monotonic() + 7200
        self.serial = 0
        self.owned = []
        self.store = self.root / "durable"
        self.package = json.loads((self.root / "package.json").read_text())
        self.source = load_checkpoint(self.package["checkpoint"])
        self.environment = self.source["environment"]
        self.events = self.root / "events.jsonl"

    def event(self, stage, **kw):
        record = {"stage": stage, "time": time.time(), **kw}
        with self.events.open("a") as f:
            f.write(json.dumps(record) + "\n")
        atomic(self.root / "status.json", record)
        print(json.dumps(record), flush=True)

    def cli_call(self, args, timeout=180, check=True):
        assert time.monotonic() < self.deadline or not check, "pilot wall budget expired"
        try:
            proc = subprocess.run([self.cli, *args], text=True, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"colab {args[0]} timed out; remote state is unknown") from exc
        self.serial += 1
        (self.root / f"cli-{self.serial:04d}.log").write_text(proc.stdout + proc.stderr)
        if check and proc.returncode:
            raise RuntimeError(f"colab {args[0]} failed; see cli-{self.serial:04d}.log")
        return proc.stdout

    def remote(self, session, code):
        self.serial += 1
        path = self.root / f"rpc-{self.serial:04d}.py"
        path.write_text(code)
        out = self.cli_call(["exec", "-s", session, "-f", str(path)])
        lines = [s for s in out.splitlines() if s.startswith("YGORL_JSON:")]
        if not lines:
            raise RuntimeError(f"no remote acknowledgement; see {path.name} and CLI log")
        return json.loads(lines[-1].split(":", 1)[1])

    def upload(self, session, local, remote):
        self.cli_call(["upload", "-s", session, str(local), remote])

    def upload_large(self, session, local, remote):
        local = Path(local)
        self.serial += 1
        parts = []
        with local.open("rb") as f:
            while content := f.read(40 * 1024 * 1024):
                part = self.root / f"upload-{self.serial}-{len(parts):03d}.part"
                part.write_bytes(content)
                target = remote + f".part-{len(parts):03d}"
                self.upload(session, part, target)
                parts.append((target, sha(part)))
        expected = sha(local)
        self.remote(
            session,
            f"""import hashlib,json,pathlib
parts={parts!r}
target=pathlib.Path({remote!r});target.parent.mkdir(parents=True,exist_ok=True)
with target.open('wb') as f:
 for name,digest in parts:
  data=pathlib.Path(name).read_bytes();assert hashlib.sha256(data).hexdigest()==digest;f.write(data)
assert hashlib.sha256(target.read_bytes()).hexdigest()=={expected!r}
print('YGORL_JSON:'+json.dumps({{'assembled':str(target)}}))
""",
        )

    def allocate(self, generation):
        session = f"ygorl-recovery-20261008-{chr(97 + generation)}"
        self.owned.append(session)
        if generation:
            usage = self.cli_call(["usage"])
            balance = float(re.search(r"Current balance:\s*([\d.]+)", usage)[1])
            assert self.initial_balance - balance < 6, "compute-unit budget reached"
            self.cli_call(["new", "-s", session, "--gpu", "L4"], timeout=300)
            subprocess.run(
                [
                    "systemd-run",
                    "--user",
                    f"--unit=ygorl-colab-pilot-{generation}-expiry-20261008",
                    "--on-active=45m",
                    self.cli,
                    "stop",
                    "-s",
                    session,
                ],
                check=True,
                capture_output=True,
            )
            self.upload(session, self.root / "source-v2.tar.gz", "/content/source-v2.tar.gz")
            self.cli_call(["exec", "-s", session, "-f", str(self.root / "bootstrap-v2.py")])
        setup_deadline = min(self.deadline, time.monotonic() + 1200)
        while True:
            status = self.remote(
                session,
                """import json,pathlib
p=pathlib.Path('/content/setup.exit')
print('YGORL_JSON:'+json.dumps({'exit':p.read_text().strip() if p.exists() else None}))
""",
            )
            if status["exit"] is not None:
                assert status["exit"] == "0", "remote build failed; retain logs and stop"
                break
            assert time.monotonic() < setup_deadline, "remote setup budget expired"
            self.event("building", session=session, generation=generation)
            time.sleep(30)
        self.upload(session, self.root / "source-v2-manifest.json", "/content/source-manifest.json")
        for name in ("colab_checkpoint.py", "colab_worker.py"):
            self.upload(session, self.repo / "tools" / name, f"/content/ygorl/tools/{name}")
        return session

    def launch(self, session, generation):
        latest = self.store / "latest.json"
        directory = self.store / json.loads(latest.read_text())["path"] if latest.exists() else None
        source = directory / "checkpoint.pt" if directory else Path(self.package["checkpoint"])
        metrics = None
        if directory:
            validate(directory, self.run_id, self.environment)
            self.upload_large(session, source, "/content/input.pt")
            self.upload(session, directory / "metrics.jsonl", "/content/input-metrics.jsonl")
            metrics = "/content/input-metrics.jsonl"
        elif generation == 0:
            self.remote(
                session,
                f"""import pathlib,hashlib,json
parts={self.package["parts"]!r}
with open('/content/input.pt','wb') as f:
 for part in parts:
  data=(pathlib.Path('/content')/part['file']).read_bytes();assert hashlib.sha256(data).hexdigest()==part['sha256'];f.write(data)
assert hashlib.sha256(pathlib.Path('/content/input.pt').read_bytes()).hexdigest()=={self.package["sha256"]!r}
print('YGORL_JSON:'+json.dumps({{'assembled':True}}))
""",
            )
        else:
            self.upload_large(session, source, "/content/input.pt")
        job = {
            "run_id": self.run_id,
            "generation": generation,
            "origin_update": 128,
            "target_update": 136,
            "checkpoint": "/content/input.pt",
            "checkpoint_sha256": sha(source),
            "metrics": metrics,
            "migration": directory is None,
            "out": f"/content/recovery/g{generation}",
            "lease": "/content/recovery/lease.json",
            "environment": self.environment,
            "deck_sha256": {Path(p).name: sha(p) for p in self.source["config"]["decks"]},
            "drivers": {name: sha(self.repo / "tools" / name) for name in ("colab_checkpoint.py", "colab_worker.py")},
        }
        path = self.root / f"job-{generation}.json"
        atomic(path, job)
        self.upload(session, path, "/content/job.json")
        result = self.remote(
            session,
            f"""import hashlib,json,pathlib,os,subprocess,time
root=pathlib.Path('/content/ygorl')
m=json.loads(pathlib.Path('/content/source-manifest.json').read_text())
for name,digest in m['files'].items():
 assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
job=json.loads(pathlib.Path('/content/job.json').read_text())
for name,digest in job['drivers'].items(): assert hashlib.sha256((root/'tools'/name).read_bytes()).hexdigest()==digest,name
out=pathlib.Path(job['out']);out.mkdir(parents=True,exist_ok=True)
lease=pathlib.Path(job['lease']);lease.write_text(json.dumps({{'generation':{generation},'expires_unix':time.time()+300}}))
env={{**os.environ,'PYTHONPATH':'src','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}}
with (out/'worker.log').open('w') as log:
 p=subprocess.Popen([str(root/'.venv/bin/python'),'-u','tools/colab_worker.py','--config','/content/job.json'],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(out/'worker.pid').write_text(str(p.pid))
print('YGORL_JSON:'+json.dumps({{'pid':p.pid,'source_files_verified':len(m['files'])}}))
""",
        )
        self.event("launched", session=session, generation=generation, **result)

    def poll(self, session, generation):
        return self.remote(
            session,
            f"""import json,pathlib,time,os
root=pathlib.Path('/content/recovery/g{generation}')
lease=pathlib.Path('/content/recovery/lease.json');tmp=lease.with_suffix('.tmp')
tmp.write_text(json.dumps({{'generation':{generation},'expires_unix':time.time()+300}}));tmp.replace(lease)
p=root/'status.json';r=root/'restore.json';pid=int((root/'worker.pid').read_text())
try:
 os.kill(pid,0)
 alive=pathlib.Path(f'/proc/{{pid}}/stat').read_text().split(') ',1)[1].split()[0]!='Z'
except ProcessLookupError: alive=False
snapshots=sorted(str(p.parent) for p in (root/'snapshots').glob('*/manifest.json'))
print('YGORL_JSON:'+json.dumps({{'alive':alive,'status':json.loads(p.read_text()) if p.exists() else None,'restore':json.loads(r.read_text()) if r.exists() else None,'snapshots':snapshots,'log_tail':(root/'worker.log').read_text()[-2500:]}}))
""",
        )

    def sync(self, session, remote):
        self.serial += 1
        staging = self.store / f"download-{self.serial:04d}"
        staging.mkdir(parents=True, exist_ok=False)
        for name in ("manifest.json", "checkpoint.pt", "metrics.jsonl"):
            self.cli_call(["download", "-s", session, remote + "/" + name, str(staging / name)], timeout=180)
        destination = promote(staging, self.store, self.run_id, self.environment)
        self.event(
            "checkpoint_verified",
            session=session,
            update=json.loads((destination / "manifest.json").read_text())["counters"]["updates"],
            snapshot=str(destination),
        )
        return destination

    def release(self, session):
        self.cli_call(["stop", "-s", session], check=False)
        listing = self.cli_call(["sessions"])
        assert session not in listing, "old session termination not confirmed; refuse duplicate worker"
        self.event("session_released", session=session)

    def main(self):
        self.initial_balance = 251.96
        forced = False
        with run_lock(self.store):
            assert not self.events.exists(), "pilot already started; inspect it instead of reusing its generations"
            atomic(
                self.root / "identity.json",
                {
                    "run_id": self.run_id,
                    "started_unix": time.time(),
                    "source_archive_sha256": sha(self.root / "source-v2.tar.gz"),
                    "source_manifest_sha256": sha(self.root / "source-v2-manifest.json"),
                    "checkpoint_sha256": self.package["sha256"],
                    "drivers": {
                        name: sha(self.repo / "tools" / name)
                        for name in ("colab_checkpoint.py", "colab_worker.py", "colab_pilot.py")
                    },
                },
            )
            try:
                for generation in range(3):
                    session = self.allocate(generation)
                    self.launch(session, generation)
                    seen = set()
                    failures = 0
                    while True:
                        released_for_test = False
                        try:
                            info = self.poll(session, generation)
                            failures = 0
                        except RuntimeError:
                            failures += 1
                            listing = self.cli_call(["sessions"])
                            if session not in listing:
                                self.event("preempted", session=session, generation=generation)
                                break
                            if failures >= 3:
                                raise
                            time.sleep(30)
                            continue
                        atomic(self.root / f"remote-{generation}.json", info)
                        status = info["status"]
                        if status and status["stage"] == "failed":
                            raise RuntimeError(status["error"])
                        if not info["alive"] and (not status or status["stage"] != "complete"):
                            raise RuntimeError("remote worker exited before completion; inspect retained log")
                        for path in info["snapshots"]:
                            if path not in seen:
                                self.sync(session, path)
                                seen.add(path)
                                if not forced:
                                    self.event(
                                        "forced_preemption",
                                        session=session,
                                        generation=generation,
                                        worker_alive=info["alive"],
                                        remote_status=status,
                                    )
                                    self.release(session)
                                    forced = True
                                    released_for_test = True
                                    break
                        if released_for_test:
                            break
                        if status and status["stage"] == "complete":
                            pointer = json.loads((self.store / "latest.json").read_text())
                            assert pointer["update"] == 136 and generation > 0 and forced
                            assert (
                                info["restore"]["all_saved_fields_equal"]
                                and not info["restore"]["migration_cuda_rng_reset"]
                            )
                            self.release(session)
                            self.event(
                                "complete",
                                final_update=136,
                                accepted_updates=8,
                                forced_preemption=True,
                                generations=generation + 1,
                            )
                            return
                        self.event("running", session=session, generation=generation, status=status)
                        time.sleep(30)
                raise RuntimeError("allocation budget exhausted")
            except BaseException as e:
                self.event("failed", error=repr(e))
                raise
            finally:
                for session in self.owned:
                    self.cli_call(["stop", "-s", session], check=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    Pilot(p.parse_args().root).main()
