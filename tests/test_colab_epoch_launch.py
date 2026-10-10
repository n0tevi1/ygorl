import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
spec = importlib.util.spec_from_file_location("epoch_launch_test", TOOLS / "colab_epoch_launch.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_repeated_launch_runs_worker_once(tmp_path):
    counter = tmp_path / "runs"
    command = [sys.executable, "-c", f"open({str(counter)!r},'a').write('run\\n')"]
    identity = {"epoch": 4, "study": "test"}
    code = module.launch_code(command, str(tmp_path), identity, {})
    # Simulate the first reply being lost by discarding it.
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)
    recovered = subprocess.check_output([sys.executable, "-c", module.receipt_code(str(tmp_path))], text=True)
    receipt = module.validate_receipt(json.loads(recovered.split("YGORL_JSON:")[1]), command, identity)
    second = subprocess.check_output([sys.executable, "-c", code], text=True)
    assert json.loads(second.split("YGORL_JSON:")[1])["pid"] == receipt["pid"]
    for _ in range(100):
        if (tmp_path / "exit").exists():
            break
        time.sleep(0.01)
    assert (tmp_path / "exit").read_text() == "0"
    assert counter.read_text() == "run\n"


def test_claim_without_receipt_never_relaunches(tmp_path):
    command = [sys.executable, "-c", "raise AssertionError('must not run')"]
    identity = {"epoch": 4}
    (tmp_path / "launch.json").write_text(json.dumps(dict(state="claimed", command=command, identity=identity)))
    result = subprocess.run(
        [sys.executable, "-c", module.launch_code(command, str(tmp_path), identity, {})], capture_output=True
    )
    assert result.returncode and b"ambiguous launch" in result.stderr
    assert not (tmp_path / "exit").exists()


@pytest.mark.parametrize("receipt", [None, {}, {"state": "launched", "command": [], "identity": {}, "pid": 1}])
def test_wrong_or_missing_receipt_is_not_success(receipt):
    with pytest.raises(RuntimeError, match="not proven"):
        module.validate_receipt(receipt, ["expected"], {"epoch": 4})
