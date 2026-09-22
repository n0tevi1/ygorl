from pathlib import Path

import pytest

from ygorl.cards.lflist import LflistError, load_lflist, parse_lflist, select
from ygorl.cards.ydk import YdkError, parse_ydk

LFLISTS = Path(__file__).resolve().parents[1] / "third_party" / "LFLists"

SAMPLE = """#[2026.09 TCG]
!2026.09 TCG
#Forbidden
21044178 0 --Abyss Dweller
#Limited
14558127 1 --Ash Blossom
10000 2
!Second
$whitelist
89631139 3
"""


def test_parse_lflist():
    lists = parse_lflist(SAMPLE)
    assert [lst.name for lst in lists] == ["2026.09 TCG", "Second"]
    tcg = select(lists)
    assert tcg.limit(21044178) == 0
    assert tcg.limit(14558127) == 1
    assert tcg.limit(10000) == 2
    assert tcg.limit(12345) == 3
    assert tcg.forbidden() == {21044178}
    wl = select(lists, "Second")
    assert wl.whitelist and wl.limit(89631139) == 3 and wl.limit(12345) == 0
    assert parse_lflist(tcg.to_text())[0] == tcg


@pytest.mark.parametrize(
    "text, msg",
    [
        ("123 1\n", "before any '!name'"),
        ("!x\n123\n", "expected '<password> <limit>'"),
        ("!x\n123 4\n", "out of range"),
        ("!x\nabc 1\n", "non-integer"),
        ("!x\n1 1\n1 2\n", "listed twice"),
    ],
)
def test_lflist_errors(text, msg):
    with pytest.raises(LflistError, match=msg):
        parse_lflist(text)


@pytest.mark.parametrize("path", sorted(LFLISTS.glob("*.lflist.conf")), ids=lambda p: p.name)
def test_upstream_lflists_parse(path):
    lists = load_lflist(path, strict=False)
    assert lists and all(lst.limits for lst in lists)


def test_lenient_mode_mirrors_edopro():
    (lst,) = parse_lflist("!x\n5 3\n5 0\n1 -1\n", strict=False)
    assert lst.limit(5) == 0  # last entry wins
    assert 1 not in lst.limits  # sentinel skipped


def test_parse_ydk_roundtrip():
    deck = parse_ydk("#created by x\n#main\n1\n1\n2\n#extra\n3\n!side\n4\n", name="d")
    assert deck.main == (1, 1, 2) and deck.extra == (3,) and deck.side == (4,)
    assert deck.counts()[1] == 2
    assert parse_ydk(deck.to_ydk(), name="d") == deck


def test_ydk_errors():
    with pytest.raises(YdkError, match="before '#main'"):
        parse_ydk("123\n")
    with pytest.raises(YdkError, match="expected a card password"):
        parse_ydk("#main\nfoo\n")
