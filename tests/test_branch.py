"""Branching API: fork a recorded game at any decision point and roll candidates out (T2.9)."""

import json
import random
from pathlib import Path

import pytest

from ygorl.agents import RandomAgent
from ygorl.cards.ydk import load_ydk
from ygorl.data import load_environment
from ygorl.engine import constants as C
from ygorl.engine import messages as M
from ygorl.engine.actions import make_decision
from ygorl.engine.branch import (
    Branch,
    BranchError,
    RecordedAgent,
    actions_for_response,
    fork,
    recorded_actions,
)
from ygorl.engine.duel import Duel, DuelConfig
from ygorl.engine.replay import Replay, ReplayEnvironmentMismatch

DECKS = {p.stem: load_ydk(p) for p in sorted((Path(__file__).parent / "decks").glob("*.ydk"))}


class TrackingAgent(RandomAgent):
    """RandomAgent that notes which steps continue a decision already started (mid multi-select)."""

    def __init__(self, seed, log):
        super().__init__(seed)
        self.log = log

    def act(self, point):
        if point.state is self.log.get("state"):
            self.log["mid"].append(point.index)
        self.log["state"] = point.state
        return super().act(point)


def record(a, b, seed, *, first=0, max_turns=4, record_steps=False, env=None):
    duel = Duel(
        seed, env, DECKS[a], DECKS[b], first=first, config=DuelConfig(max_turns=max_turns), record_steps=record_steps
    )
    log = {"mid": []}
    result = duel.run(TrackingAgent(seed, log), TrackingAgent(seed + 1, log))
    result.mid_steps = log["mid"]
    return Replay.from_duel(duel, result), result


def end(result):
    return result.winner, result.reason, result.turns, result.lp, result.responses


@pytest.fixture(scope="module")
def game():
    """122 agent steps over 119 responses (multi-selects incl. SORT_CARD), ends on the turn limit."""
    rep, result = record("snake_eye", "tearlaments", 2)
    assert result.decisions > len(result.responses)
    return rep, result


@pytest.fixture(scope="module")
def short_game():
    """43 steps over 42 responses (one SELECT_CARD split in two), b moves first."""
    rep, result = record("snake_eye", "tearlaments", 3, first=1, max_turns=3)
    assert result.decisions > len(result.responses)
    return rep, result


# ------------------------------------------------------------------ recorded actions


@pytest.mark.parametrize(
    ("a", "b", "seed", "first"),
    [("snake_eye", "tearlaments", 2, 0), ("fiendsmith_ryzeal", "kashtira", 2, 1), ("tearlaments", "tenpai", 3, 0)],
)
def test_recorded_actions_are_recovered_from_the_responses(a, b, seed, first):
    """SORT_CARD, SELECT_TRIBUTE and ANNOUNCE_CARD among others."""
    rep, result = record(a, b, seed, first=first, max_turns=3 if first else 4)
    assert rep.steps == []  # nothing but responses to go on
    assert recorded_actions(rep) == result.actions


def test_recorded_actions_use_step_records_when_present():
    rep, result = record("snake_eye", "tearlaments", 2, record_steps=True)
    assert recorded_actions(rep) == result.actions


def test_step_records_that_contradict_the_responses_are_an_error():
    rep, result = record("snake_eye", "tearlaments", 2, record_steps=True)
    i = next(i for i, s in enumerate(rep.steps) if len(s["actions"]) > 1)
    rep.steps[i] = {**rep.steps[i], "chosen": (rep.steps[i]["chosen"] + 1) % len(rep.steps[i]["actions"])}
    with pytest.raises(BranchError, match=f"step {i}"):
        recorded_actions(rep)


def test_unmappable_response_is_an_error(short_game):
    rep, _ = short_game
    bad = Replay.from_json(rep.to_json())
    bad.responses[3] = b"\x7f\x7f\x7f\x7f"
    with pytest.raises(BranchError, match="response 3"):
        recorded_actions(bad)


# ------------------------------------------------------------------ inversion of the action model


def card(code, loc=C.LOCATION_HAND, seq=0):
    return M.CardInfo(code, M.Location(0, loc, seq, 0))


def _sum_opt(i, lo, hi=0):
    return M.SumOption(100 + i, M.Location(0, C.LOCATION_HAND, i), lo | (hi << 16))


MULTI_STEP = [
    M.SelectCard(0, True, 1, 3, tuple(card(i, seq=i) for i in range(5))),
    M.SelectTribute(
        0, False, 2, 3, tuple(M.TributeOption(i, M.Location(0, C.LOCATION_MZONE, i), 1 + i % 2) for i in range(4))
    ),
    M.SelectSum(0, True, 8, 1, 3, (), tuple(_sum_opt(i, 1 + i, 2 * i) for i in range(5))),
    M.SelectSum(0, False, 8, 0, 0, (), tuple(_sum_opt(i, 2 + i) for i in range(5))),
    M.SelectCounter(
        0,
        0x1,
        3,
        (
            M.CounterOption(1, M.Location(0, C.LOCATION_SZONE, 0), 3),
            M.CounterOption(2, M.Location(0, C.LOCATION_SZONE, 1), 2),
        ),
    ),  # fmt: skip
    M.SortCard(0, tuple(card(i, seq=i) for i in range(4))),
    M.SortChain(0, tuple(card(i, seq=i) for i in range(3))),
    M.SelectPlace(0, 2, 0xFFFFFFFF & ~0b1011),
    M.SelectDisfield(0, 1, 0xFFFFFFFF & ~(0b11 << 16)),
    M.AnnounceRace(0, 2, C.RACE_DRAGON | C.RACE_ZOMBIE | C.RACE_FIEND),
    M.AnnounceAttrib(0, 1, C.ATTRIBUTE_LIGHT | C.ATTRIBUTE_DARK),
    M.SelectUnselectCard(0, True, False, 1, 3, (card(1), card(2)), (card(3),)),
    M.SelectChain(0, 0, False, 0, 0, (M.ChainOption(7, M.Location(0, C.LOCATION_HAND, 0), 112, 0),)),
]


@pytest.mark.parametrize("msg", MULTI_STEP, ids=lambda m: type(m).__name__)
def test_actions_for_response_inverts_every_path(msg):
    rng = random.Random(0)
    for _ in range(25):
        state, path, response = make_decision(msg), [], None
        while response is None:
            path.append(rng.randrange(len(state.actions())))
            response = state.step(path[-1])
        found = actions_for_response(make_decision(msg), response)
        again, replayed = make_decision(msg), None
        for i in found:
            assert replayed is None
            replayed = again.step(i)
        assert replayed == response
        if not isinstance(msg, (M.SelectCounter, M.AnnounceRace, M.AnnounceAttrib)):  # others: one path per response
            assert found == path


def test_actions_for_response_continues_a_started_decision():
    msg = M.SelectCard(0, False, 2, 2, tuple(card(i, seq=i) for i in range(4)))
    state = make_decision(msg)
    state.step(2)
    assert actions_for_response(state, b"\0\0\0\0\x02\0\0\0\x02\0\0\0\x00\0\0\0") == [0]
    assert state.picked == [2]  # the search works on copies


def test_actions_for_response_returns_none_when_unreachable():
    assert actions_for_response(make_decision(M.SelectYesNo(0, 5)), b"\x05\0\0\0") is None


# ------------------------------------------------------------------ fork


def test_fork_exposes_the_decision_point(short_game):
    rep, result = short_game
    for t in (0, 7, len(result.actions) - 1):
        branch = fork(rep, t)
        assert isinstance(branch, Branch)
        assert branch.t == t and branch.point.index == t
        assert branch.prefix == result.actions[:t]
        assert branch.recorded_action == result.actions[t]
        assert 0 <= branch.recorded_action < len(branch.point.actions)
        assert branch.side == (rep.first + branch.point.player) % 2


def test_fork_mid_multi_select(game):
    rep, result = game
    assert result.mid_steps, "the test game should contain a multi-step decision"
    script = RecordedAgent(result.actions)
    for t in result.mid_steps:
        branch = fork(rep, t)
        first = fork(rep, t - 1)
        assert branch.point.decision == first.point.decision  # same decision, one pick further
        assert branch.point.actions != first.point.actions
        again = branch.rollout(result.actions[t], script, script)
        assert end(again) == end(result), f"t={t}"


def test_fork_at_every_t_and_continue_reproduces_the_end(short_game):
    rep, result = short_game
    script = RecordedAgent(result.actions)
    for t in range(len(result.actions)):
        branch = fork(rep, t)
        again = branch.rollout(result.actions[t], script, script)
        assert end(again) == end(result), f"t={t}"
        assert again.actions == result.actions


def test_fork_and_continue_reproduces_a_longer_game(game):
    rep, result = game
    n = len(result.actions)
    script = RecordedAgent(result.actions)
    for t in (0, 1, n // 3, n // 2, n - 2, n - 1):
        again = fork(rep, t).rollout(result.actions[t], script, script)
        assert end(again) == end(result), f"t={t}"


@pytest.mark.parametrize("t", ["len", "len+5"])
def test_t_out_of_range(short_game, t):
    rep, result = short_game
    n = len(result.actions)
    t = {"len": n, "len+5": n + 5}[t]
    with pytest.raises(BranchError, match=rf"t={t} is out of range: this replay reaches decision points 0\.\.{n - 1}"):
        fork(rep, t)


@pytest.mark.parametrize("t", [-1, 2.0, None])
def test_t_must_be_a_step_index(short_game, t):
    with pytest.raises(BranchError, match="counts agent steps from 0"):
        fork(short_game[0], t)


def test_truncated_replay_can_be_forked_at_its_first_unanswered_decision(short_game):
    rep, result = short_game
    starts = [i for i in range(len(result.actions)) if i not in result.mid_steps]  # first step of every decision
    cut = Replay.from_json(rep.to_json())
    cut.responses = cut.responses[:10]
    s = starts[10]  # the decision after the tenth response
    assert recorded_actions(cut) == result.actions[:s]
    branch = fork(cut, s)
    assert branch.recorded_action is None and branch.prefix == result.actions[:s]
    again = branch.rollout(result.actions[s], RecordedAgent(result.actions), RecordedAgent(result.actions))
    assert end(again) == end(result)
    with pytest.raises(BranchError, match=rf"0\.\.{s}$"):
        fork(cut, s + 1)


def test_fork_needs_the_recorded_environment(tmp_path):
    root = tmp_path / "env-2026-09"
    (root / "meta").mkdir(parents=True)
    (root / "environment.json").write_text(
        json.dumps({"version": "env-2026-09", "format": "md", "rules": {"mode": "MR5"}})
    )
    pool = sorted({c for name in ("kashtira", "labrynth") for c in DECKS[name].main + DECKS[name].extra})
    (root / "pool.json").write_text(json.dumps({"cards": pool}))
    (root / "banlist.lflist.conf").write_text("!none\n")
    (root / "meta.json").write_text(json.dumps({"decks": []}))
    env = load_environment(root)
    rep, result = record("kashtira", "labrynth", 1, max_turns=2, env=env)
    with pytest.raises(ReplayEnvironmentMismatch):
        fork(rep, 0)
    branch = fork(rep, 3, env=env)
    script = RecordedAgent(result.actions)
    assert end(branch.rollout(result.actions[3], script, script)) == end(result)


def test_game_cut_mid_decision_by_the_decision_limit(game):
    """The last decision is incomplete, so no response records it: only step records know those steps."""
    _, full = game
    m = full.mid_steps[0]  # the limit stops the game before step m, inside a multi-select
    config = DuelConfig(max_turns=4, max_decisions=m)
    for record_steps in (True, False):
        duel = Duel(2, None, DECKS["snake_eye"], DECKS["tearlaments"], config=config, record_steps=record_steps)
        result = duel.run(RandomAgent(2), RandomAgent(3))
        assert result.reason == "decision_limit" and result.actions == full.actions[:m]
        rep = Replay.from_duel(duel, result)
        known = m if record_steps else m - 1  # without step records the started decision is unknown
        assert recorded_actions(rep) == result.actions[:known]
        branch = fork(rep, m - 1)
        assert branch.recorded_action == (result.actions[m - 1] if record_steps else None)
        again = branch.rollout(result.actions[m - 1], RecordedAgent(result.actions), RecordedAgent(result.actions))
        assert end(again) == end(result)


# ------------------------------------------------------------------ rollouts


class CountingAgent(RandomAgent):
    def __init__(self, seed):
        super().__init__(seed)
        self.seen = []

    def act(self, point):
        self.seen.append(point.index)
        return super().act(point)


def test_policies_are_asked_only_after_t(short_game):
    rep, _ = short_game
    t = 10
    branch = fork(rep, t)
    pa, pb = CountingAgent(1), CountingAgent(2)
    res = branch.rollout(0, pa, pb)
    seen = sorted(pa.seen + pb.seen)
    assert seen == list(range(t + 1, res.decisions))
    assert res.actions[:t] == branch.prefix and res.actions[t] == 0


def test_rollout_rejects_an_illegal_action(short_game):
    rep, _ = short_game
    branch = fork(rep, 5)
    with pytest.raises(ValueError, match="out of range"):
        branch.rollout(len(branch.point.actions), RandomAgent(0), RandomAgent(1))


def test_try_all_gives_one_result_per_legal_action(game):
    rep, _ = game
    t = next(t for t in range(40) if len(fork(rep, t).point.actions) >= 3)
    branch = fork(rep, t)
    outcomes = branch.try_all(RandomAgent, seed=5)
    assert [o.index for o in outcomes] == list(range(len(branch.point.actions)))
    assert [o.action for o in outcomes] == branch.point.actions
    assert [o.recorded for o in outcomes].count(True) == 1 and outcomes[branch.recorded_action].recorded
    for o in outcomes:
        assert len(o.results) == 1
        r = o.results[0]
        assert r.actions[t] == o.index and r.reason in ("win", "turn_limit", "decision_limit", "end")
        assert o.wins + o.draws + o.losses == 1
    # deterministic, and the same policy seeds are used for every candidate (common random numbers)
    again = branch.try_all(RandomAgent, seed=5)
    assert [end(o.results[0]) for o in again] == [end(o.results[0]) for o in outcomes]


def test_try_all_subset_and_several_rollouts(short_game):
    rep, _ = short_game
    branch = fork(rep, 3)
    seeds = []

    def factory(seed):
        seeds.append(seed)
        return RandomAgent(seed)

    outcomes = branch.try_all(factory, candidates=[0], rollouts=3, seed=10)
    assert len(outcomes) == 1 and len(outcomes[0].results) == 3
    assert seeds == [10, 11, 12, 13, 14, 15]
    with pytest.raises(ValueError, match="candidate"):
        branch.try_all(RandomAgent, candidates=[len(branch.point.actions)])
    with pytest.raises(ValueError, match="rollouts"):
        branch.try_all(RandomAgent, rollouts=0)


# ------------------------------------------------------------------ curriculum modes (T2.6)


@pytest.mark.parametrize("mode", ["solo", "handtrap"])
def test_curriculum_games_can_be_forked(mode):
    """Host-answered decisions are in the response log but not asked of agents; forking must skip them."""
    cfg = DuelConfig(max_turns=4, curriculum=mode, learner=0)
    duel = Duel(2, None, DECKS["snake_eye"], DECKS["tearlaments"], config=cfg)
    result = duel.run(RandomAgent(2), RandomAgent(3))
    assert result.auto_decisions > 0, "the test game should contain host-answered decisions"
    rep = Replay.from_duel(duel, result)
    assert recorded_actions(rep) == result.actions
    script = RecordedAgent(result.actions)
    n = len(result.actions)
    for t in (0, n // 2, n - 1):
        again = fork(rep, t).rollout(result.actions[t], script, script)
        assert end(again) == end(result), f"t={t}"


# ------------------------------------------------------------------ snapshots (T2.8)


def test_rollouts_restore_one_snapshot_instead_of_replaying(short_game, monkeypatch):
    """try_all builds a single snapshot-enabled duel for the fork; results equal the replay-based path."""
    rep, result = short_game
    t = len(result.actions) // 2
    branch = fork(rep, t)
    made = []
    original = Replay.duel

    def counting(self, *args, **kwargs):
        made.append(kwargs.get("snapshots", False))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Replay, "duel", counting)
    outcomes = branch.try_all(RandomAgent, rollouts=2, seed=5)
    assert made == [True]
    for outcome in outcomes:
        for r, res in enumerate(outcome.results):  # record_steps forces the replay path
            again = branch.rollout(outcome.index, RandomAgent(5 + 2 * r), RandomAgent(6 + 2 * r), record_steps=True)
            assert end(res) == end(again) and res.actions == again.actions
    assert made.count(True) == 1


def test_snapshot_rollouts_of_the_recorded_action_reproduce_the_game(game):
    rep, result = game
    script = RecordedAgent(result.actions)
    n = len(result.actions)
    for t in (0, n // 2, n - 1):
        branch = fork(rep, t)
        for _ in range(2):  # repeated rollouts from the same fork
            again = branch.rollout(result.actions[t], script, script)
            assert end(again) == end(result), f"t={t}"
