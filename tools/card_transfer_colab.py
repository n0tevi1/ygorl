"""Single owned Colab worker for the registered card-transfer pilot.

Downloads and verifies a full optimizer/RNG checkpoint after every epoch.
Only proven endpoint preemption permits reallocation; worker errors stop.
"""

import argparse
import json
import re
import subprocess
import threading
import time
from pathlib import Path

import torch

from colab_checkpoint import atomic, sha
from colab_owned_endpoint import endpoint_call, owned_endpoint
from colab_pilot import Pilot


class TransferPilot(Pilot):
    def __init__(self, root):
        self.root = root.resolve()
        self.repo = Path(__file__).resolve().parents[1]
        self.cli = "/home/ya0guang/.local/bin/colab"
        self.reg = json.loads((root / "registration.json").read_text())
        self.package = json.loads((root / "package.json").read_text())
        self.identity_sha = sha(root / "identity.json")
        identity = json.loads((root / "identity.json").read_text())
        assert sha(root / "registration.json") == identity["registration_sha256"]
        assert sha(root / "package.json") == identity["package_sha256"]
        assert sha(Path(__file__)) == identity["controller_sha256"]
        assert 0 < self.reg["max_hours"] <= 2 and 0 < self.reg["max_units"] <= 6
        assert 0 < self.reg["max_allocations"] <= 2
        self.end = min(time.time() + self.reg["max_hours"] * 3600, self.reg.get("deadline_unix", float("inf")))
        self.deadline = time.monotonic() + max(0, self.end - time.time())
        self.prior_allocations = self.reg.get("prior_allocations", 0)
        assert isinstance(self.prior_allocations, int) and 0 <= self.prior_allocations < self.reg["max_allocations"]
        self.serial = int(time.time())
        self.events = root / "events.jsonl"
        self.owned = []
        self.session = self.endpoint = None
        self.last_usage = 0
        self.stage = "starting"
        self.done = threading.Event()
        self.beat = threading.Thread(target=self.heartbeat, daemon=True)
        self.beat.start()

    def heartbeat(self):
        while not self.done.is_set():
            atomic(self.root / "pipeline-status.json", {"stage": self.stage, "updated_unix": time.time()})
            self.done.wait(15)

    def guard(self):
        assert time.time() < self.end, "wall budget expired"
        assert not (self.root / "STOP.json").exists(), "study STOP"
        if time.time() - self.last_usage > 120:
            output = self.cli_call(["usage"])
            match = re.search(r"Current balance:\s*([\d.]+)", output)
            assert match and self.reg["initial_balance"] - float(match[1]) < self.reg["max_units"], "compute budget"
            self.last_usage = time.time()

    def absent(self):
        return self.endpoint not in endpoint_call("list")["endpoints"]

    def read_remote(self, session, code):
        """Retry only known idempotent status/lease reads, never launches or allocations."""
        assert session == self.session
        for attempt in range(3):
            self.guard()
            self.serial += 1
            path = self.root / f"rpc-{self.serial:04d}.py"
            path.write_text(code)
            try:
                output = self.cli_call(["exec", "-s", self.session, "-f", str(path)], timeout=40)
                lines = [line for line in output.splitlines() if line.startswith("YGORL_JSON:")]
                if not lines:
                    raise RuntimeError("status response missing")
                return json.loads(lines[-1].split(":", 1)[1])
            except RuntimeError as exc:
                self.event("status_retry", session=self.session, attempt=attempt + 1, error=str(exc))
                if attempt == 2 or self.absent():
                    raise

    def stop_owned(self):
        if self.session is None:
            return
        history = Path.home() / ".config/colab-cli/history" / (self.session + ".jsonl")
        endpoint = self.endpoint or owned_endpoint(self.session, history, self.root / "owned-sessions.json")
        proof = endpoint_call(
            "stop",
            session=self.session,
            endpoint=endpoint,
            history=history,
            owned_sessions=self.root / "owned-sessions.json",
        )
        atomic(self.root / (self.session + "-termination.json"), proof)
        self.session = self.endpoint = None

    def setup(self):
        self.guard()
        assert len(self.owned) + self.prior_allocations < self.reg["max_allocations"], "allocation budget"
        self.session = f"ygorl-card-transfer-20261010-vm{len(self.owned) + self.prior_allocations}"
        assert not (Path.home() / ".config/colab-cli/history" / (self.session + ".jsonl")).exists(), (
            "reused session name"
        )
        self.owned.append(self.session)
        atomic(self.root / "owned-sessions.json", self.owned)
        subprocess.run(
            [
                "systemd-run",
                "--user",
                "--unit=" + self.session + "-expiry",
                f"--on-active={max(1, int(self.end - time.time()))}s",
                self.cli,
                "stop",
                "-s",
                self.session,
            ],
            check=True,
            capture_output=True,
        )
        self.cli_call(["new", "-s", self.session, "--gpu", "L4"], timeout=300)
        history = Path.home() / ".config/colab-cli/history" / (self.session + ".jsonl")
        self.endpoint = owned_endpoint(self.session, history, self.root / "owned-sessions.json")
        self.stage = "setup"
        for key, target in [("source", "/content/source.tar.gz"), ("data", "/content/data.tar.gz")]:
            assert sha(self.package[key]) == self.package[key + "_sha256"]
            self.upload_large(self.session, self.package[key], target)
        assert sha(self.package["bootstrap"]) == self.package["bootstrap_sha256"]
        self.cli_call(["exec", "-s", self.session, "-f", self.package["bootstrap"]])
        until = min(self.end, time.time() + 1500)
        while time.time() < until:
            self.guard()
            status = self.read_remote(
                self.session,
                "import json,pathlib\np=pathlib.Path('/content/setup.exit')\nprint('YGORL_JSON:'+json.dumps({'exit':p.read_text().strip() if p.exists() else None}))",
            )
            if status["exit"] is not None:
                self.cli_call(
                    [
                        "download",
                        "-s",
                        self.session,
                        "/content/setup.log",
                        str(self.root / (self.session + "-setup.log")),
                    ]
                )
                assert status["exit"] == "0", "remote setup failed"
                return
            time.sleep(20)
        raise RuntimeError("remote setup deadline")

    def epoch(self, seed, arm, epoch):
        name = f"seed-{seed}-{arm}"
        dest = self.root / "jobs" / name
        dest.mkdir(parents=True, exist_ok=True)
        remote_dir = f"/content/study/jobs/{name}"
        self.remote(
            self.session,
            f"import pathlib,json\npathlib.Path({remote_dir!r}).mkdir(parents=True,exist_ok=True)\nprint('YGORL_JSON:{{}}')",
        )
        if (dest / "latest.pt").exists():
            self.upload_large(self.session, dest / "latest.pt", remote_dir + "/latest.pt")
        cmd = [
            "/content/ygorl/.venv/bin/python",
            "/content/ygorl/tools/card_transfer_worker.py",
            "--root",
            "/content/study",
            "--seed",
            str(seed),
            "--arm",
            arm,
            "--stop-after",
            str(epoch),
        ]
        self.remote(
            self.session,
            f"""import json,pathlib,subprocess,os,time
p=pathlib.Path('/content/epoch.exit');p.unlink(missing_ok=True)
pathlib.Path('/content/lease').write_text(str(time.time()+300))
env=dict(os.environ,PYTHONPATH='/content/ygorl/src',YGORL_LEASE='/content/lease',OMP_NUM_THREADS='2')
wrapper={("import pathlib,subprocess; p=subprocess.run(" + repr(cmd) + '); pathlib.Path("/content/epoch.exit").write_text(str(p.returncode))')!r}
f=open('/content/epoch.log','w');p=subprocess.Popen(['/content/ygorl/.venv/bin/python','-c',wrapper],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print('YGORL_JSON:'+json.dumps({{'pid':p.pid}}))
""",
        )
        errors = 0
        while True:
            self.guard()
            try:
                status = self.read_remote(
                    self.session,
                    """import json,pathlib,time
pathlib.Path('/content/lease').write_text(str(time.time()+300))
p=pathlib.Path('/content/epoch.exit')
print('YGORL_JSON:'+json.dumps({'exit':p.read_text().strip() if p.exists() else None}))
""",
                )
                errors = 0
            except RuntimeError:
                errors += 1
                if errors >= 3:
                    if self.absent():
                        self.event("preempted", session=self.session, job=name, epoch=epoch)
                        self.session = self.endpoint = None
                        self.setup()
                        return self.epoch(seed, arm, epoch)
                    raise
                time.sleep(20)
                continue
            if status["exit"] is not None:
                break
            time.sleep(20)
        self.cli_call(["download", "-s", self.session, "/content/epoch.log", str(dest / f"epoch-{epoch}.log")])
        assert status["exit"] == "0", f"worker failed {name} epoch {epoch}"
        expected = self.read_remote(
            self.session,
            f"""import hashlib,pathlib,json
p=pathlib.Path({remote_dir!r})/'latest.pt'
print('YGORL_JSON:'+json.dumps({{'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()}}))
""",
        )["sha256"]
        temp = dest / "download.pt"
        self.cli_call(["download", "-s", self.session, remote_dir + "/latest.pt", str(temp)])
        assert sha(temp) == expected, "download hash mismatch"
        state = torch.load(temp, map_location="cpu", weights_only=True)
        assert state["identity"] == {
            "registration_sha256": sha(self.root / "registration.json"),
            "seed": seed,
            "arm": arm,
        }
        assert state["epoch"] == epoch and len(state["history"]) == epoch
        assert {"optimizer", "torch_rng", "cuda_rng", "model"} <= state.keys()
        temp.replace(dest / "latest.pt")
        atomic(dest / "metrics.json", state["history"])
        atomic(dest / "progress.json", {"epoch": epoch, "updated_unix": time.time(), "checkpoint_sha256": expected})
        if epoch == self.reg["epochs"]:
            # Keep the zero-shot boundary separate from subsequent adaptation.
            import shutil

            shutil.copy2(dest / "latest.pt", dest / "zero-shot.pt")
        if epoch == self.reg["epochs"] + self.reg["adapt_epochs"]:
            self.cli_call(["download", "-s", self.session, remote_dir + "/report.json", str(dest / "report.json")])
        self.event("epoch_verified", job=name, epoch=epoch, checkpoint_sha256=expected)

    def execute(self):
        try:
            self.setup()
            total = self.reg["epochs"] + self.reg["adapt_epochs"]
            for seed in self.reg["seeds"]:
                arms = list(self.reg["arms"])
                if seed % 2:
                    arms.reverse()
                for arm in arms:
                    self.stage = f"training-seed-{seed}-{arm}"
                    for epoch in range(1, total + 1):
                        self.epoch(seed, arm, epoch)
            self.stop_owned()
            atomic(
                self.root / "report.json",
                {
                    "study_sha256": self.identity_sha,
                    "healthy": True,
                    "jobs": len(self.reg["seeds"]) * len(self.reg["arms"]),
                    "scope": "offline transfer; requires paired analysis and full-game follow-up",
                },
            )
            self.stage = "complete"
        except BaseException as exc:
            atomic(self.root / "STOP.json", {"reason": repr(exc), "time": time.time()})
            self.stage = "stopped"
            try:
                self.stop_owned()
            except Exception as cleanup:
                atomic(self.root / "cleanup-failure.json", {"reason": repr(cleanup)})
            raise
        finally:
            self.done.set()
            self.beat.join()
            atomic(self.root / "pipeline-status.json", {"stage": self.stage, "updated_unix": time.time()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    assert not (args.root / "owned-sessions.json").exists(), (
        "fresh controller only; review partial state before restart"
    )
    TransferPilot(args.root).execute()
