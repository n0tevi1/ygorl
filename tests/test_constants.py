import subprocess
import sys
from pathlib import Path

from ygorl.engine import constants as C

ROOT = Path(__file__).resolve().parents[1]


def test_constants_match_header():
    """The generated module must be in sync with the pinned ygopro-core header."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "gen_constants.py"), "--check"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_known_values():
    assert C.LOCATION_MZONE == 0x04
    assert C.MSG_SELECT_CARD == 15
    assert C.MSG_REMOVE_CARDS == 190
    assert C.LINK_MARKER_TOP == 0o200  # octal in the C header
    assert C.RACE_ALL == 0x40000000FFFFFFFF
    assert C.DUEL_MODE_MR5 == C.DUEL_PZONE | C.DUEL_EMZONE | C.DUEL_FSX_MMZONE | (
        C.DUEL_TRAP_MONSTERS_NOT_USE_ZONE | C.DUEL_TRIGGER_ONLY_IN_LOCATION
    )
