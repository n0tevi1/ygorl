"""The 10 Master Duel meta-style test decks under tests/decks/ (T1.5)."""

from pathlib import Path

import pytest

from ygorl.cards.cdb import CardDB
from ygorl.cards.legality import validate_deck
from ygorl.cards.ydk import load_ydk

DECKS = sorted((Path(__file__).parent / "decks").glob("*.ydk"))


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


def test_ten_decks():
    assert len(DECKS) == 10


@pytest.mark.parametrize("path", DECKS, ids=lambda p: p.stem)
def test_deck_is_legal_and_scripted(db, path):
    from ygorl import paths

    deck = load_ydk(path)
    assert validate_deck(deck, cards=db, banlist=None) == []
    assert 40 <= len(deck.main) <= 60 and len(deck.extra) <= 15
    official = paths.card_scripts() / "official"
    missing = [p for p in set(deck.main + deck.extra) if not (official / f"c{p}.lua").exists()]
    assert missing == [], f"cards without a script: {missing}"


def test_generator_is_reproducible(tmp_path):
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, str(root / "tools" / "make_test_decks.py"), "--out", str(tmp_path)],
        check=True,
        capture_output=True,
    )
    for path in DECKS:
        assert (tmp_path / path.name).read_text() == path.read_text()
