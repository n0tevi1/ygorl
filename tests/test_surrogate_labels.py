"""Tests for real-game surrogate labels (T5.7): deck fingerprints, label aggregation, the JSONL cache."""

import json
from pathlib import Path

import pytest

from ygorl.build.labels import LabelCache, LabelConfig, deck_fingerprint, label_decks, labels_from_reports
from ygorl.cards.ydk import Deck, load_ydk
from ygorl.eval.arena import Arena, GameRecord, summarize

DECKS = Path(__file__).parent / "decks"


def test_deck_fingerprint_ignores_order_name_and_side():
    a = Deck(main=(3, 1, 2, 2), extra=(9, 8), side=(5,), name="a")
    b = Deck(main=(2, 2, 1, 3), extra=(8, 9), side=(), name="b")
    assert deck_fingerprint(a) == deck_fingerprint(b)
    assert len(deck_fingerprint(a)) == 64
    assert deck_fingerprint(a) != deck_fingerprint(Deck(main=(3, 1, 2), extra=(9, 8)))
    assert deck_fingerprint(Deck(main=(1,), extra=(2,))) != deck_fingerprint(Deck(main=(1, 2)))


def _report(results, name="opp"):
    """results: (first, winner) per game, from deck a's side."""
    records = [GameRecord(pair=i // 2, seed=i // 2, first=f, winner=w, reason="win") for i, (f, w) in enumerate(results)]
    return summarize(records, agent_a="greedy", agent_b="greedy", deck_a="cand", deck_b=name, seed=0)


def test_labels_from_reports_pool_and_sides():
    r1 = _report([(0, 0), (1, 0), (0, 0), (1, 1)], "x")  # 3/4; first 2/2, second 1/2
    r2 = _report([(0, 1), (1, 1), (0, None), (1, 1)], "y")  # 0.5/4; first 0.5/2, second 0/2
    lab = labels_from_reports([r1, r2])
    assert lab["win_rate"] == pytest.approx((0.75 + 0.125) / 2)
    assert lab["win_rate_first"] == pytest.approx((1.0 + 0.25) / 2)
    assert lab["win_rate_second"] == pytest.approx((0.5 + 0.0) / 2)
    assert lab["games"] == 8 and lab["draws"] == 1 and lab["errors"] == 0
    assert lab["per_opponent"] == {"x": 0.75, "y": 0.125}
    assert 0 < lab["ci_half_width"] < 0.5 and 0 < lab["se"] < 0.5
    weighted = labels_from_reports([r1, r2], weights=[3, 1])
    assert weighted["win_rate"] == pytest.approx(0.75 * 0.75 + 0.25 * 0.125)
    # effective sample size shrinks with unequal weights: wider interval
    assert weighted["ci_half_width"] > lab["ci_half_width"]
    with pytest.raises(ValueError):
        labels_from_reports([r1], weights=[1, 2])


def test_label_config_fingerprint():
    pool = [load_ydk(DECKS / "kashtira.ydk")]
    a = LabelConfig.for_pool(pool, pairs=2, seed=0)
    assert a.fingerprint() == LabelConfig.for_pool(pool, pairs=2, seed=0).fingerprint()
    assert a.fingerprint() != LabelConfig.for_pool(pool, pairs=3, seed=0).fingerprint()
    assert a.fingerprint() != LabelConfig.for_pool(pool, pairs=2, seed=0, agent="random").fingerprint()
    assert a.pool == (("kashtira", deck_fingerprint(pool[0])),)
    assert LabelConfig.from_dict(json.loads(json.dumps(a.to_dict()))) == a


def test_label_cache_round_trip(tmp_path):
    path = tmp_path / "labels" / "cache.jsonl"
    cache = LabelCache(path)
    assert len(cache) == 0 and cache.get("d1", "c1") is None
    cache.put({"deck": "d1", "config": "c1", "labels": {"win_rate": 0.25}, "main": [1, 2], "extra": []})
    cache.put({"deck": "d2", "config": "c1", "labels": {"win_rate": 0.75}, "main": [3], "extra": [4]})
    cache.put({"deck": "d1", "config": "c2", "labels": {"win_rate": 0.5}, "main": [1, 2], "extra": []})
    assert len(path.read_text().splitlines()) == 3
    again = LabelCache(path)
    assert len(again) == 3
    assert again.get("d1", "c1")["labels"] == {"win_rate": 0.25}
    assert again.get("d1", "c2")["labels"]["win_rate"] == 0.5
    assert [r["deck"] for r in again.records("c1")] == ["d1", "d2"]
    with pytest.raises(ValueError, match="deck"):
        cache.put({"config": "c1", "labels": {}})
    # a torn last line (interrupted run) is skipped, earlier lines survive
    with path.open("a") as f:
        f.write('{"deck": "d3", "config"')
    assert len(LabelCache(path)) == 3


def test_label_decks_plays_real_games_and_uses_the_cache(tmp_path, monkeypatch):
    pool = [load_ydk(DECKS / "kashtira.ydk"), load_ydk(DECKS / "yubel.ydk")]
    snake = load_ydk(DECKS / "snake_eye.ydk")
    same = Deck(main=tuple(reversed(snake.main)), extra=snake.extra, name="copy")  # same deck: played once
    cache = LabelCache(tmp_path / "labels.jsonl")
    recs = label_decks([snake, same], pool, pairs=1, seed=3, cache=cache, max_turns=6)
    assert [r["deck"] for r in recs] == [deck_fingerprint(snake)] * 2
    lab = recs[0]["labels"]
    assert lab["games"] == 4 and set(lab["per_opponent"]) == {"kashtira", "yubel"}
    assert 0 <= lab["win_rate"] <= 1 and lab["errors"] == 0
    assert recs[0]["main"] == sorted(snake.main) and len(cache) == 1

    def boom(*a, **k):
        raise AssertionError("cached labels must not be replayed")

    monkeypatch.setattr(Arena, "play", boom)
    again = label_decks([snake], pool, pairs=1, seed=3, cache=LabelCache(tmp_path / "labels.jsonl"), max_turns=6)
    assert again[0]["labels"] == lab
