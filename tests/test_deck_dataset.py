"""The full deck dataset (ygorl.data.deck_dataset, tools/build_deck_dataset.py) on a tiny raw sample."""

from __future__ import annotations

import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from ygorl.cards.cdb import CardDB
from ygorl.cards.ydk import load_ydk
from ygorl.data import masterduelmeta as mdm
from ygorl.data.cardmap import CardMapper
from ygorl.data.deck_dataset import build_deck_dataset, load_deck_dataset
from ygorl.data.fetch import read_raw

from .test_data_sources import DECKS, FENRIR, write_fixture_raw

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def db() -> CardDB:
    return CardDB.load()


@pytest.fixture(scope="module")
def raw(tmp_path_factory, db) -> Path:
    path = tmp_path_factory.mktemp("raw")
    write_fixture_raw(path, db)
    return path


def records(raw: Path, db: CardDB) -> list[mdm.DeckRecord]:
    ids = mdm.card_ids(read_raw(raw / mdm.CARDS_FILE))
    return mdm.parse_top_decks(read_raw(raw / mdm.TOP_DECKS_FILE), ids, CardMapper(db), since="")


def test_every_mapped_list_once_newest_first_with_legality_and_sources(raw, db, tmp_path):
    def problems(r):  # stand-in rule: Kashtira Fenrir is limited to 1
        return ["over_limit"] if r.counts()[FENRIR] > 1 else []

    ds = build_deck_dataset(records(raw, db), problems, {"environment": {"environment": "test"}})
    # 8 raw lists: the Yubel list has an unmapped card; the Branded list appears four times (twice as "Branded", as
    # "Tenpai" and as "Old"): the newest copy (Branded, 2026-09-20) stays
    s = ds.meta["stats"]
    assert (s["raw_lists"], s["unmapped_lists"], s["duplicates"], s["lists"]) == (8, 1, 3, 4)
    assert ds.types.tolist() == ["Branded", "Kashtira"] and s["types"] == 2
    assert ds.date.tolist() == ["2026-09-20", "2026-09-18", "2026-09-12", "2026-09-11"]
    assert ds.copies.tolist() == [4, 1, 1, 1]
    assert ds.legal.tolist() == [True, True, False, True] and ds.problems[2] == "over_limit"
    assert s["legal"] == 3 and s["legal_share"] == 0.75 and s["lists_per_type"]["median"] == 2.0
    assert set(ds.kind.tolist()) == {"ranked"} and set(ds.source.tolist()) == {"Master I"}

    branded = load_ydk(DECKS / "branded_despia.ydk")
    assert ds.type_name(0) == "Branded" and ds.cards(0) == dict(Counter(branded.main + branded.extra))
    assert (ds.n_main[0], ds.n_extra[0]) == (len(branded.main), len(branded.extra))
    m = ds.matrix()
    assert m.shape == (4, len(ds.passwords)) and m.sum() == sum(ds.n_main) + sum(ds.n_extra)
    assert np.all(np.diff(ds.passwords) > 0)

    back = load_deck_dataset(ds.save(tmp_path / "ds"))
    assert back.meta == ds.meta and back.cards(3) == ds.cards(3) and back.problems.tolist() == ds.problems.tolist()
    (tmp_path / "ds" / "meta.json").write_text('{"format": "something else"}')
    with pytest.raises(ValueError, match="ygorl-deck-dataset"):
        load_deck_dataset(tmp_path / "ds")


def test_build_tool_on_the_raw_sample(raw, tmp_path):
    out = tmp_path / "out"
    res = subprocess.run([sys.executable, str(ROOT / "tools" / "build_deck_dataset.py"), "md-2026-09",
                          "--raw-dir", str(raw), "--out", str(out)], cwd=ROOT, capture_output=True, text=True)  # fmt: skip
    assert res.returncode == 0, res.stderr
    ds = load_deck_dataset(out)
    assert len(ds) == 4 and ds.meta["environment"]["environment"] == "md-2026-09"
    assert ds.meta["stats"]["bytes"] > 0 and ds.meta["sha256"]
