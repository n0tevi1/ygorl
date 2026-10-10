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
