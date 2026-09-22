import ygorl
from ygorl import _core


def test_import():
    assert ygorl.__version__
    assert _core.hello() == "ygorl._core"


def test_ocg_version():
    major, minor = _core.ocg_version()
    assert major == 11
    assert minor >= 0
