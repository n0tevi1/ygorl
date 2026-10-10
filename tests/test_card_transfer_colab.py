import importlib.util
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def controller():
    sys.path.insert(0, str(TOOLS))
    try:
        spec = importlib.util.spec_from_file_location("transfer_test_controller", TOOLS / "card_transfer_colab.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.TransferPilot
    finally:
        sys.path.remove(str(TOOLS))


@pytest.mark.parametrize("preempted", [False, True])
def test_status_timeout_retries_without_allocating(tmp_path, preempted):
    cls = controller()
    pilot = cls.__new__(cls)
    pilot.root, pilot.serial, pilot.session = tmp_path, 0, "owned-session"
    pilot.guard = lambda: None
    pilot.absent = lambda: preempted
    events, calls = [], []
    pilot.event = lambda *a, **k: events.append((a, k))

    def cli(args, timeout):
        calls.append((args, timeout))
        if len(calls) == 1:
            raise RuntimeError("status timeout")
        return 'YGORL_JSON:{"exit":"0"}\n'

    pilot.cli_call = cli
    if preempted:
        with pytest.raises(RuntimeError, match="timeout"):
            pilot.read_remote(pilot.session, "pass")
        assert len(calls) == 1
    else:
        assert pilot.read_remote(pilot.session, "pass") == {"exit": "0"}
        assert len(calls) == 2
    assert events and all(c[0][0] == "exec" and c[1] == 40 for c in calls)


def test_persistent_status_failure_is_bounded(tmp_path):
    cls = controller()
    pilot = cls.__new__(cls)
    pilot.root, pilot.serial, pilot.session = tmp_path, 0, "owned-session"
    pilot.guard = lambda: None
    pilot.absent = lambda: False
    events = []
    pilot.event = lambda *a, **k: events.append(k)

    def failed(*a, **k):
        raise RuntimeError("timeout")

    pilot.cli_call = failed
    with pytest.raises(RuntimeError):
        pilot.read_remote(pilot.session, "pass")
    assert len(events) == 3


@pytest.mark.parametrize("receipt_valid", [False, True])
def test_launch_timeout_reconciles_without_relaunch(tmp_path, receipt_valid):
    cls = controller()
    pilot = cls.__new__(cls)
    pilot.session = "owned-session"
    command, identity = ["python", "worker"], {"epoch": 4}
    launches, reads = [], []
    pilot.event = lambda *a, **k: None

    def remote(*args):
        launches.append(args)
        raise RuntimeError("launch reply lost")

    def read(*args):
        reads.append(args)
        if len(reads) == 1:
            return {}
        return dict(state="launched", command=command, identity=identity, pid=42) if receipt_valid else None

    pilot.remote, pilot.read_remote = remote, read
    if receipt_valid:
        assert pilot.launch_epoch(command, "/epoch-4", identity)["pid"] == 42
    else:
        with pytest.raises(RuntimeError, match="not proven"):
            pilot.launch_epoch(command, "/epoch-4", identity)
    assert len(launches) == 1 and len(reads) == 2


def test_local_epoch_rejects_corrupt_checkpoint(tmp_path):
    import hashlib
    import json
    import torch

    sys.path.insert(0, str(TOOLS))
    try:
        from card_transfer_local import verify_epoch
    finally:
        sys.path.remove(str(TOOLS))
    (tmp_path / "registration.json").write_text("{}")
    directory = tmp_path / "jobs/seed-0-id"
    directory.mkdir(parents=True)
    history = [{"epoch": 1}]
    state = dict(
        identity=dict(registration_sha256=hashlib.sha256(b"{}").hexdigest(), seed=0, arm="id"),
        epoch=1,
        history=history,
        model={},
        optimizer={},
        torch_rng=[],
        cuda_rng=[],
    )
    torch.save(state, directory / "latest.pt")
    digest = hashlib.sha256((directory / "latest.pt").read_bytes()).hexdigest()
    (directory / "progress.json").write_text(json.dumps(dict(epoch=1, checkpoint_sha256=digest)))
    (directory / "metrics.json").write_text(json.dumps(history))
    assert verify_epoch(tmp_path, 0, "id", 1) == digest
    with (directory / "latest.pt").open("ab") as f:
        f.write(b"corrupt")
    with pytest.raises(AssertionError, match="digest mismatch"):
        verify_epoch(tmp_path, 0, "id", 1)
