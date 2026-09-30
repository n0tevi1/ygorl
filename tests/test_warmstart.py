"""Warm start of the card-value model from earlier paired data (ygorl.build.warmstart, #145)."""

import json
import math

import pytest

from ygorl.build.selection import SD_PAIR
from ygorl.build.signals import CardValueModel
from ygorl.build.warmstart import BLANK, load, parse_edit, pooled_stderr

M2 = [{"type": "Therion", "file": "decks/therion.ydk", "parent": 0.57,
       "loo": [{"card": 84332527, "name": "Therion \"Bull\" Ain", "copies": 1, "value": 0.048, "ci": 0.0196},
               {"card": 43270827, "name": "Therion Cross", "copies": 3, "value": -0.008, "ci": 0.0098}]}]  # fmt: skip
M1 = {"checkpoint": "c.pt", "pairs": 300, "decks": [{"type": "Megalith", "file": "decks/megalith-3.ydk",
      "children": [{"edit": "main: -63233638 +81439173", "diff": 0.0025, "sd_diff_pair": 0.349},
                   {"edit": "extra: -21225115 +65741786", "diff": -0.0075, "sd_diff_pair": 0.0}]}]}  # fmt: skip
LINEAGE = [
    {"child": "r0001-p0-c2", "parent": {"id": "corpus:therion", "type": "Therion"},
     "edits": [{"out": 48686504, "into": 52038441, "section": "main"},
               {"out": 21887075, "into": 52038441, "section": "main"}],
     "all_pairs": {"pairs": 700, "diff": 0.03, "stderr": 0.01},
     "search": {"pairs": 600, "diff": 0.038},
     "learned": [{"source": "first_batch", "pairs": 25, "diff": 0.04, "stderr": 0.0407},
                 {"source": "validation", "pairs": 100, "diff": -0.07, "stderr": 0.031}]},
    {"child": "r0001-p0-c3", "parent": {"id": "corpus:therion", "type": "Therion"},
     "edits": [{"out": 97526666, "into": 30271097, "section": "main"}], "learned": []},
]  # fmt: skip
COMPARE_OLD = {"args": {}, "parents": [{"type": "Pendulum Magician", "file": "decks/pendulum-magician-3.ydk",
               "mvp": {"accepted": True, "diff": 0.05, "ci": [0.01, 0.09]},
               "evo": {"arms": [{"index": 0, "pairs": 300, "observed": 0.1}],
                       "chosen": ["main: -25311006 +82224646", "extra: -47349116 +26435595"],
                       "decision": "futile", "validation_pairs": 400, "diff": 0.0, "ci": [-0.0377, 0.0184]}}]}  # fmt: skip


def write(tmp_path, name, data):
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return p


def by_id(items):
    return dict(items)


def test_m2_leave_one_out_becomes_blank_in_card_out(tmp_path):
    items = by_id(load(write(tmp_path, "m2.json", M2)))
    o = items["warm:m2:m2:decks/therion.ydk:84332527"]
    # value = parent - child (card -> blank): the child is worse by 0.048
    assert (o.deck_type, o.into, o.out, o.diff) == ("Therion", (BLANK,), (84332527,), -0.048)
    assert o.stderr == pytest.approx(0.01)
    # the model learns that Bull Ain is worth more than the blank, Cross slightly less
    model = CardValueModel()
    for obs in items.values():
        model.add(obs)
    assert model.gain([84332527], [BLANK], "Therion")[0] > 0 > model.gain([43270827], [BLANK], "Therion")[0]


def test_m1_swaps_keep_their_sign_and_pool_the_variance(tmp_path):
    items = list(load(write(tmp_path, "m1.json", M1), inflate=2.0))
    (_, a), (_, b) = items
    assert (a.deck_type, a.into, a.out, a.diff) == ("Megalith", (81439173,), (63233638,), 0.0025)
    assert a.stderr == pytest.approx(2 * pooled_stderr(0.349, 300))
    assert b.out == (21225115,) and b.stderr > 0  # a zero sd still gets the pseudo-pairs' error
    assert pooled_stderr(SD_PAIR, 50) == pytest.approx(SD_PAIR / math.sqrt(50))
    assert parse_edit("extra: -1 +2") == ("extra", 1, 2)
    with pytest.raises(ValueError):
        parse_edit("main: -Ash +Droll")


def test_lineage_gives_only_the_non_adaptive_pairs(tmp_path):
    state = tmp_path / "r111-therion"
    state.mkdir()
    (state / "lineage.jsonl").write_text("\n".join(json.dumps(r) for r in LINEAGE) + "\n{torn")
    items = by_id(load(state))
    assert set(items) == {"warm:lineage:r111-therion:r0001-p0-c2/first_batch",
                          "warm:lineage:r111-therion:r0001-p0-c2/validation"}  # fmt: skip
    o = items["warm:lineage:r111-therion:r0001-p0-c2/validation"]
    assert (o.into, o.out, o.diff, o.stderr) == ((52038441, 52038441), (48686504, 21887075), -0.07, 0.031)
    # never the search's or the all-pairs estimate (selection-biased)
    assert all(x.diff in (0.04, -0.07) for x in items.values())
    assert by_id(load(state / "lineage.jsonl")) == items


def test_the_tuner_comparison_gives_fresh_validations_only(tmp_path):
    items = by_id(load(write(tmp_path, "cmp.json", COMPARE_OLD)))
    (uid,) = items  # the old format logs no MVP edit; the Thompson arms are search pairs
    o = items[uid]
    assert o.into == (82224646, 26435595) and o.out == (25311006, 47349116) and o.diff == 0.0
    assert o.stderr == pytest.approx(0.0184 / 1.645)  # from the futility bound
    new = json.loads(json.dumps(COMPARE_OLD))
    new["parents"][0]["evo"]["stderr"] = 0.012
    new["parents"][0]["mvp"]["finalists"] = [{"edit": "main: -1 +2", "pairs": 200, "diff": 0.05, "stderr": 0.015}]
    items = by_id(load(write(tmp_path, "cmp2.json", new)))
    assert len(items) == 2 and items["warm:compare:cmp2:0:evo"].stderr == 0.012
    fin = items["warm:compare:cmp2:0:mvp0"]
    assert (fin.into, fin.out, fin.diff, fin.stderr) == ((2,), (1,), 0.05, 0.015)
    with pytest.raises(ValueError):
        load(write(tmp_path, "x.json", {"what": 1}))
