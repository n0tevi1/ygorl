"""The league (ygorl.train.league): finished games in, snapshot admission / keep-best / evolved-deck scores out,
without engines or a Trainer."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from ygorl.train.league import League  # noqa: E402
from ygorl.train.rollout import Assignment, FinishedGame  # noqa: E402
from ygorl.train.selfplay import SnapshotPool  # noqa: E402

MODEL = torch.nn.Linear(2, 2)


def game(opponent, score, *, truncated=False, info=None, learner_seat=0, reason="win"):
    """A finished game where the learner (on ``learner_seat``) scored ``score`` (1 / 0.5 / 0)."""
    winner = None if score == 0.5 else (learner_seat if score == 1 else 1 - learner_seat)
    return FinishedGame(Assignment(None, opponent, learner_seat, info or {}), winner, reason, truncated, 1, {})


def test_admission_needs_enough_pool_games_above_the_threshold():
    league = League(SnapshotPool(4), min_win_rate=0.55, min_games=4)
    assert league.admit(MODEL, 1) and len(league.pool) == 1  # an empty pool always takes one
    assert not league.admit(MODEL, 2)  # no pool games yet
    sid = league.pool.ids()[0]
    league.record([game(sid, s) for s in (1, 0, 1, 0)])  # 0.5: not above 0.55
    assert not league.admit(MODEL, 3)
    league.record([game(sid, 1) for _ in range(4)])  # 6 / 8 = 0.75
    assert league.admit(MODEL, 4) and len(league.pool) == 2 and league.window == {"games": 0, "points": 0.0}
    info = {i["id"]: i for i in league.pool.info()}
    assert info[sid]["games"] == 8 and info[sid]["learner_win_rate"] == round(6.5 / 9, 3)


def test_a_threshold_without_pool_games_holds_back_even_with_min_games_0():
    league = League(SnapshotPool(4), min_win_rate=0.5, min_games=0)
    assert league.admit(MODEL, 1) and not league.admit(MODEL, 2)  # no division by zero


def test_without_a_threshold_every_due_snapshot_joins():
    league = League(SnapshotPool(2))
    assert all(league.admit(MODEL, u) for u in (1, 2, 3)) and len(league.pool) == 2  # oldest evicted


def test_only_finished_pool_games_count():
    league = League(SnapshotPool(4))
    league.admit(MODEL, 1)
    sid = league.pool.ids()[0]
    league.record([game(None, 1), game(sid, 1, truncated=True), game(sid, 0.5)])  # self-play, truncated, a draw
    assert league.window == {"games": 1, "points": 0.5}


def test_keep_best_takes_only_a_better_valid_score():
    league = League(SnapshotPool(4))
    assert not league.offer_best(MODEL, 1, None) and league.pool.best_id is None  # an invalid evaluation
    assert league.offer_best(MODEL, 2, 0.6) and league.best == {"score": 0.6, "update": 2}
    first = league.pool.best_id
    assert not league.offer_best(MODEL, 3, 0.6) and league.pool.best_id == first
    assert league.offer_best(MODEL, 4, 0.7) and league.pool.best_id != first and league.pool.best_score == 0.7


def test_evolved_decks_score_only_when_the_learner_pilots_them():
    class Evolved:
        def __init__(self):
            self.scores = []

        def record(self, did, score):
            self.scores.append((did, score))

    evolved = Evolved()
    league = League(SnapshotPool(4), evolved)
    league.admit(MODEL, 1)
    sid = league.pool.ids()[0]
    piloted = {"evolved": "evo-1", "evolved_seat": 0}
    opposed = {"evolved": "evo-2", "evolved_seat": 1}
    n = league.record([game(None, 1, info=piloted), game(sid, 1, info=piloted), game(sid, 1, info=opposed),
                       game(sid, 1, info=piloted, reason="error"), game(sid, 1, info=piloted, truncated=True)])  # fmt: skip
    assert n == 3  # error and truncated games are left out
    assert evolved.scores == [("evo-1", 1.0), ("evo-1", 1.0)]  # self-play (both seats learner) and a pool game


def test_state_round_trips():
    league = League(SnapshotPool(4), min_win_rate=0.5, min_games=1)
    league.admit(MODEL, 1)
    league.record([game(league.pool.ids()[0], 1)])
    league.offer_best(MODEL, 2, 0.8)
    state = league.state_dict()
    assert set(state) == {"pool", "best", "league", "evolved"} and state["evolved"] is None
    other = League(SnapshotPool(4))
    other.load_state_dict(state, lambda: torch.nn.Linear(2, 2))
    assert other.best == league.best and other.window == league.window and other.pool.ids() == league.pool.ids()


def test_active_cohort_restricts_pool_deals_and_is_off_by_default():
    import numpy as np

    from ygorl.cards.ydk import load_ydk
    from ygorl.train.selfplay import DeckPool, SelfPlaySchedule

    pool = SnapshotPool(8, sampling="uniform")
    for u in range(6):
        pool.add(torch.nn.Linear(1, 1), update=u)
    decks = [load_ydk(p) for p in sorted(Path("tests/decks").glob("*.ydk"))[:2]]

    def opponents(sched, deals):
        seen = []
        for _ in range(deals):
            a = sched()
            sched()  # the paired game
            seen.append(a.opponent)
        return seen

    base = opponents(SelfPlaySchedule(DeckPool(decks), pool, selfplay_fraction=0.0, seed=1), 40)
    again = opponents(SelfPlaySchedule(DeckPool(decks), pool, selfplay_fraction=0.0, seed=1, active_k=0), 40)
    assert base == again and len(set(base)) > 2
    sched = SelfPlaySchedule(DeckPool(decks), pool, selfplay_fraction=0.0, seed=1, active_k=2, active_deals=10)
    seen = opponents(sched, 40)
    for i in range(0, 40, 10):
        assert len(set(seen[i : i + 10])) <= 2
    assert len(set(seen)) > 2  # the cohort is redrawn
    state = sched.state_dict()
    resumed = SelfPlaySchedule(DeckPool(decks), pool, selfplay_fraction=0.0, seed=1, active_k=2, active_deals=10)
    resumed.load_state_dict(state)
    assert opponents(resumed, 15) == opponents(sched, 15)
    assert np.isfinite(len(seen))
