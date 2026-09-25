"""Tests for cards.cdb loading and bit-field decoding (T1.2)."""

import json
import re
from pathlib import Path

import pytest

from ygorl import paths
from ygorl.cards.cdb import Card, CardDB, CardVocab, decode_row, flag_names
from ygorl.engine import constants as C

DATA = Path(__file__).parent / "data"
SAMPLES = json.loads((DATA / "card_samples.json").read_text())["cards"]


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


def test_loads_babelcdb(db):
    assert len(db) > 10000
    assert isinstance(db[89631139], Card)


@pytest.mark.parametrize("expected", SAMPLES, ids=lambda s: s["name"])
def test_decoded_fields_match_official_data(db, expected):
    card = db[expected["password"]]
    assert card.name == expected["name"]
    assert card.type_names == expected["types"]
    if card.is_monster:
        assert card.attribute_names == [expected["attribute"]]
        assert card.race_names == [expected["race"]]
        assert card.attack == expected["attack"]
        assert card.level == expected.get("level", expected.get("rank", expected.get("link")))
        assert card.rank == expected.get("rank", 0)
        assert card.link == expected.get("link", 0)
        if card.is_link:
            assert card.defense == 0
            assert card.link_marker_names == expected["link_markers"]
        else:
            assert card.defense == expected["defense"]
            assert card.link_marker == 0
        assert [card.lscale, card.rscale] == expected.get("scales", [0, 0])
        assert card.is_extra_deck == expected.get("extra_deck", False)


def test_decode_row_packed_fields():
    level = 7 | (5 << 16) | (3 << 24)  # level 7, right scale 5, left scale 3
    row = (
        1,
        3,
        0,
        0x10DD | (0x5 << 16) | (0x123 << 48),
        C.TYPE_MONSTER | C.TYPE_PENDULUM,
        2500,
        2000,
        level,
        C.RACE_DRAGON,
        C.ATTRIBUTE_DARK,
        0,
    )
    card = decode_row(row, None)
    assert (card.level, card.lscale, card.rscale) == (7, 3, 5)
    assert card.setcodes == (0x10DD, 0x5, 0x123)
    assert card.name == "" and card.strings == ("",) * 16
    link = decode_row(
        (2, 3, 0, 0, C.TYPE_MONSTER | C.TYPE_LINK, 1000, C.LINK_MARKER_TOP | C.LINK_MARKER_BOTTOM, 2, 1, 1, 0), None
    )
    assert link.defense == 0 and link.link_marker_names == ["bottom", "top"] and link.link == 2


def test_texts_and_effect_strings(db):
    ash = db[14558127]
    assert "negate that effect" in ash.desc
    assert ash.effect_string(0) == "Negate that effect"
    assert len(ash.strings) == 16


def test_flag_names():
    assert flag_names(C.TYPE_MONSTER | C.TYPE_EFFECT | C.TYPE_TUNER, "TYPE") == ["monster", "effect", "tuner"]
    assert flag_names(C.RACE_YOKAI, "RACE") == ["yokai"]


def test_core_tuple_matches_native_db(db):
    native = db.to_core()
    assert len(native) == len(db)
    for pw in (89631139, 1861629, 16178681):
        assert native.get(pw) == db[pw].to_core_tuple()
    assert db.to_core() is native  # cached


def test_snapshot_roundtrip(db, tmp_path):
    small = CardDB(db[p] for p in (89631139, 14558127, 1861629))
    small.save_snapshot(tmp_path / "cards.json")
    again = CardDB.load_snapshot(tmp_path / "cards.json")
    assert dict(again) == dict(small)


def test_vocab_is_stable_when_extended(tmp_path):
    v = CardVocab([30, 10, 20])
    assert v.index(30) == CardVocab.FIRST_INDEX and v.index(999) == CardVocab.UNKNOWN
    v.save(tmp_path / "vocab.json")
    v2 = CardVocab.load(tmp_path / "vocab.json")
    v2.extend([5, 10])
    assert [v2.index(p) for p in (30, 10, 20)] == [v.index(p) for p in (30, 10, 20)]
    assert v2.index(5) == len(v2) - 1 and v2.password(v2.index(5)) == 5
    with pytest.raises(KeyError):
        v2.password(CardVocab.PAD)


def test_canonical_resolves_alternate_art(db):
    # Dark Magician alternate artwork 36996508 has alias 46986414
    if 36996508 in db:
        assert db.canonical(36996508) == 46986414
    assert db.canonical(46986414) == 46986414


STRINGID = re.compile(r"SetDescription\(\s*aux\.Stringid\(\s*id\s*,\s*(\d+)\s*\)\s*\)")


def test_script_descriptions_point_at_existing_strings(db):
    """``e:SetDescription(aux.Stringid(id, n))`` in official scripts resolves to ``str{n+1}``.

    ``aux.Stringid(id, n)`` indexes ``texts.str{n+1}`` of the card itself. About
    12% of references are empty upstream (BabelCDB has no text for them, e.g.
    Starlight Junktion str1), so this guards coverage rather than demanding 100%;
    T5.2 needs a placeholder for missing effect text. The sampled records in
    tests/data/effect_strings_sample.json keep 50 resolved strings for human review.
    """
    missing = []
    checked = 0
    for script in sorted((paths.card_scripts() / "official").glob("c*.lua")):
        password = int(script.stem[1:])
        if password not in db:
            continue
        for n in {int(m) for m in STRINGID.findall(script.read_text(errors="replace"))}:
            checked += 1
            if n >= 16 or not db[password].strings[n]:
                missing.append((password, n))
    assert checked > 5000
    assert len(missing) / checked < 0.15, missing[:20]


def test_recorded_effect_string_sample(db):
    sample = json.loads((DATA / "effect_strings_sample.json").read_text())["samples"]
    assert len(sample) >= 50
    for s in sample:
        assert db[s["password"]].strings[s["index"]] == s["text"]


def test_vocab_from_db_with_a_base_only_appends():
    """A saved vocab stays valid when the card database grows: from_db(db, base) keeps every old index."""
    from ygorl.cards.cdb import CardVocab

    base = CardVocab([30, 10])  # e.g. loaded from a checkpoint; not in password order
    grown = {5: None, 10: None, 20: None, 30: None}  # 5 and 20 are new cards
    v = CardVocab.from_db(grown, base=base)
    assert [v.index(p) for p in (30, 10)] == [base.index(30), base.index(10)] == [2, 3]
    assert (v.index(5), v.index(20)) == (4, 5)  # new cards appended in password order
    assert len(base) == 4  # the base itself is not modified
    assert CardVocab.from_db(grown).index(5) == 2  # without a base: fresh, password order
