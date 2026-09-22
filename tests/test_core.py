import ygorl
from ygorl import _core


def test_import():
    assert ygorl.__version__
    assert _core.hello() == "ygorl._core"
