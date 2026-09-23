"""PPO self-play machinery (T4b.4) on a toy game: collector layout, the PPO update, the league, convergence.

The toy game (``ygorl.train.toy``) is Nim with the EncodedVecEnv interface: a pile of stones, each seat in
turn takes 1-3, whoever takes the last stone wins. The optimal move from ``n`` is to take ``n % 4`` (a pile
that is a multiple of 4 is lost against perfect play), so convergence is checkable exactly.
"""

import copy
import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ygorl.env.encoded import EncodedEvent  # noqa: E402
from ygorl.train.ppo import (  # noqa: E402
    PolicyObjective,
    PPOConfig,
    PPOLearner,
    available_objectives,
    register_objective,
)
from ygorl.train.rollout import Assignment, RolloutCollector  # noqa: E402
from ygorl.train.selfplay import SnapshotPool  # noqa: E402
from ygorl.train.toy import NimEnv, NimModel, optimal_action  # noqa: E402


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def nim_games(rng: np.random.Generator, pool: SnapshotPool | None = None, selfplay: float = 1.0, max_pile=15):
    def next_game() -> Assignment:
        pile = int(rng.integers(1, max_pile + 1))
        opp = pool.sample(rng) if pool is not None and rng.random() >= selfplay else None
        return Assignment(spec=pile, opponent=opp, learner_seat=int(rng.integers(0, 2)))

    return next_game


def greedy_moves(model: NimModel, max_pile: int = 15) -> dict[int, int]:
    obs = [NimEnv.observe(n, player=0, max_pile=max_pile) for n in range(1, max_pile + 1)]
    with torch.no_grad():
        logits = model.policy_logits(model.collate(obs))
    return {n: int(a) + 1 for n, a in zip(range(1, max_pile + 1), logits.argmax(-1).tolist())}


# ------------------------------------------------------------------------------------------ layout


def test_self_play_rollout_layout():
    torch.manual_seed(0)
    env = NimEnv(num_envs=3, max_pile=15)
    model = NimModel(max_pile=15)
    rng = np.random.default_rng(0)
    col = RolloutCollector(env, model, nim_games(rng), num_steps=12, seed=0)
    ro = col.collect()
    T, B = 12, 3
    assert ro.players.shape == (T, B) and ro.action_mask.shape == (T, B, 3) and ro.q.shape == (T, B, 3)
    assert ro.obs["pile"].shape[0] == T * B
    assert ro.action_mask.gather(-1, ro.actions.unsqueeze(-1)).all()  # chosen candidates are legal
    assert torch.allclose(ro.probs.sum(-1), torch.ones(T, B))
    assert torch.allclose(ro.log_probs.exp(), ro.probs.gather(-1, ro.actions.unsqueeze(-1)).squeeze(-1))
    # Nim alternates seats within a game; a new game starts after every done row
    piles = ro.obs["pile"].reshape(T, B)
    for b in range(B):
        for t in range(T - 1):
            if ro.dones[t, b]:
                assert ro.rewards[t, b] == 1.0  # whoever takes the last stone wins
            else:
                assert ro.players[t + 1, b] != ro.players[t, b]
                assert piles[t + 1, b] == piles[t, b] - (ro.actions[t, b] + 1)
    assert ((ro.rewards != 0) <= ro.dones).all()
    # every finished game is recorded with the rows it contributed
    assert sum(g.learner_rows for g in ro.games) <= T * B
    assert all(g.winner in (0, 1) and not g.truncated for g in ro.games)
    # the pending decision of each column is its bootstrap state; collection resumes from it
    assert ro.bootstrap_mask.shape == (B, 3) and ro.bootstrap_player.shape == (B,)
    ro2 = col.collect()
    first = ro2.obs["pile"].reshape(T, B)[0]
    assert torch.equal(ro2.players[0], ro.bootstrap_player)
    assert (first >= 1).all()


def test_snapshot_opponent_rows_are_excluded():
    """Against a snapshot the column holds only the learner's rows: one seat, σ = +1, the terminal reward on
    the learner's last row, and the opponent's decisions are part of the environment."""
    torch.manual_seed(0)
    env = NimEnv(num_envs=4, max_pile=15)
    model = NimModel(max_pile=15)
    pool = SnapshotPool(capacity=2)
    pool.add(model, update=0)
    rng = np.random.default_rng(1)
    col = RolloutCollector(env, model, nim_games(rng, pool, selfplay=0.0), num_steps=10, opponents=pool.get, seed=1)
    ro = col.collect()
    assert all(g.assignment.opponent is not None for g in ro.games)
    for b in range(4):
        seats = ro.players[:, b]
        starts = [0] + [t + 1 for t in range(9) if ro.dones[t, b]]
        for s, e in zip(starts, starts[1:] + [10]):
            assert len(set(seats[s:e].tolist())) <= 1  # one seat per game in the column
    assert set(ro.rewards[ro.dones].tolist()) <= {-1.0, 1.0}
    assert ro.opponent_decisions > 0
    for g in ro.games:
        assert g.learner_score in (0.0, 1.0)


def test_truncated_games_are_flagged_and_carry_no_reward():
    env = NimEnv(num_envs=2, max_pile=15, max_moves=2)  # most games hit the move limit
    model = NimModel(max_pile=15)
    col = RolloutCollector(env, model, nim_games(np.random.default_rng(2), max_pile=15), num_steps=16, seed=2)
    ro = col.collect()
    assert ro.truncated.any()
    assert (ro.truncated <= ro.dones).all()
    assert (ro.rewards[ro.truncated] == 0).all()
    cut = [g for g in ro.games if g.truncated]
    assert cut and all(g.reason == "turn_limit" for g in cut)


# ------------------------------------------------------------------------------------------ update


def test_default_hyperparameters_follow_the_design():
    cfg = PPOConfig()
    assert 0.05 <= cfg.entropy_coef <= 0.2  # design I1
    assert cfg.estimator == "vrpo" and cfg.kl_ref_coef > 0  # design I2 / I8
    with pytest.raises(ValueError):
        PPOConfig(estimator="vtrace")
    assert PPOConfig.from_dict(cfg.to_dict()) == cfg


def test_update_reports_losses_and_moves_the_reference_slowly():
    torch.manual_seed(0)
    env = NimEnv(num_envs=4, max_pile=15)
    model = NimModel(max_pile=15)
    learner = PPOLearner(model, PPOConfig(epochs=2, minibatch_size=32, reference_ema=0.1))
    ro = RolloutCollector(env, model, nim_games(np.random.default_rng(0)), num_steps=16, seed=0).collect()
    ref_before = copy.deepcopy(learner.reference.state_dict())
    stats = learner.update(ro)
    for key in ("loss", "policy_loss", "entropy", "kl_ref", "q_loss", "v_loss", "approx_kl", "clip_frac", "grad_norm"):
        assert np.isfinite(stats[key]), key
    assert stats["kl_ref"] >= 0
    # EMA: reference <- (1 - τ) reference + τ θ
    for name, p in learner.reference.state_dict().items():
        if p.dtype.is_floating_point:
            expect = 0.9 * ref_before[name] + 0.1 * model.state_dict()[name]
            torch.testing.assert_close(p, expect)


def test_kl_to_the_reference_is_zero_for_an_identical_policy_and_pulls_towards_it():
    torch.manual_seed(0)
    model = NimModel(max_pile=15)
    obs = model.collate([NimEnv.observe(n, 0, 15) for n in range(1, 16)])
    learner = PPOLearner(model, PPOConfig())
    with torch.no_grad():
        logits = model.policy_logits(obs)
        ref = learner.reference.policy_logits(obs)
    assert float(PPOLearner.kl(logits, ref, obs["action_mask"])) == pytest.approx(0.0, abs=1e-6)
    # a policy pushed away from the reference pays a positive KL whose gradient points back
    shifted = logits.detach().clone().requires_grad_(True)
    with torch.no_grad():
        shifted[:, 0] += 3.0
    kl = PPOLearner.kl(shifted, ref, obs["action_mask"])
    kl.backward()
    choice = obs["action_mask"].sum(-1) > 1  # a forced move has nothing to regularize
    assert float(kl.detach()) > 0 and (shifted.grad[choice, 0] > 0).all() and (shifted.grad[~choice] == 0).all()


class NimTurnModel(NimModel):
    """Nim with EncodedVecEnv-style ``globals``: column 2 ``is_my_turn`` = 1, column 3 ``turn`` = 1 for piles
    above 8, 5 otherwise (so the turn-restricted prior KL sees a known subset of rows)."""

    @staticmethod
    def collate(observations, device=None):
        out = NimModel.collate(observations, device)
        turn = torch.where(out["pile"] > 8, 1, 5)
        out["globals"] = torch.stack([torch.zeros_like(turn), torch.zeros_like(turn), torch.ones_like(turn), turn], -1)
        return out


def _update_with_prior(cfg: PPOConfig) -> tuple[NimTurnModel, dict]:
    torch.manual_seed(0)
    model = NimTurnModel(max_pile=15)
    torch.manual_seed(1)
    prior = NimTurnModel(max_pile=15)  # a different policy, so the KL is not zero
    learner = PPOLearner(model, cfg, prior)
    ro = RolloutCollector(NimEnv(4, 15), model, nim_games(np.random.default_rng(0)), num_steps=16, seed=0).collect()
    torch.manual_seed(2)
    return model, learner.update(ro)


def test_prior_kl_can_be_restricted_to_first_turn_rows():
    g = torch.tensor([[0, 1, 1, 1], [1, 0, 0, 1], [0, 1, 1, 2], [1, 0, 1, 2], [0, 1, 1, 3]])
    assert PPOLearner.prior_rows({"globals": g}, 0) is None  # default: every row
    assert PPOLearner.prior_rows({"globals": g}, 1).tolist() == [True, False, False, False, False]
    assert PPOLearner.prior_rows({"globals": g}, 2).tolist() == [True, False, True, True, False]
    with pytest.raises(ValueError):
        PPOLearner.prior_rows({"pile": g}, 1)
    with pytest.raises(ValueError):
        PPOConfig(kl_prior_turns=-1)
    # the restricted KL counts the other rows as 0: a selected row weighs as much as in the unrestricted mean
    logits, ref = torch.randn(5, 3), torch.randn(5, 3)
    mask = torch.ones(5, 3, dtype=torch.bool)
    rows = torch.tensor([True, False, True, False, False])
    per_row = [float(PPOLearner.kl(logits[i : i + 1], ref[i : i + 1], mask[i : i + 1])) for i in range(5)]
    assert float(PPOLearner.kl(logits, ref, mask, rows)) == pytest.approx((per_row[0] + per_row[2]) / 5, rel=1e-5)
    assert float(PPOLearner.kl(logits, ref, mask, torch.zeros(5, dtype=torch.bool))) == 0.0
    # in an update: rows past the turn limit feel no prior; with every row selected it matches the default
    base = dict(epochs=1, minibatch_size=1024, kl_ref_coef=0.0)
    no_prior, _ = _update_with_prior(PPOConfig(**base))
    turn1, stats1 = _update_with_prior(PPOConfig(**base, kl_prior_coef=1.0, kl_prior_turns=1))
    every, stats_all = _update_with_prior(PPOConfig(**base, kl_prior_coef=1.0))
    upto5, stats5 = _update_with_prior(PPOConfig(**base, kl_prior_coef=1.0, kl_prior_turns=5))
    assert 0 < stats1["kl_prior_rows"] < stats1["rows"] and stats5["kl_prior_rows"] == stats5["rows"]
    assert "kl_prior_rows" not in stats_all and stats5["kl_prior"] == pytest.approx(stats_all["kl_prior"])
    for a, b in zip(every.parameters(), upto5.parameters()):
        torch.testing.assert_close(a, b)
    assert stats1["kl_prior"] > 0 and any(not torch.equal(a, b) for a, b in zip(turn1.parameters(), no_prior.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(turn1.parameters(), every.parameters()))


def test_learner_state_round_trip():
    torch.manual_seed(0)
    model = NimModel(max_pile=15)
    learner = PPOLearner(model, PPOConfig())
    ro = RolloutCollector(NimEnv(2, 15), model, nim_games(np.random.default_rng(0)), num_steps=8, seed=0).collect()
    learner.update(ro)
    state = learner.state_dict()
    other = PPOLearner(NimModel(max_pile=15), PPOConfig())
    other.load_state_dict(state)
    for a, b in zip(learner.reference.parameters(), other.reference.parameters()):
        assert torch.equal(a, b)
    for a, b in zip(learner.model.parameters(), other.model.parameters()):
        assert torch.equal(a, b)
    assert other.updates == learner.updates == 1


# ------------------------------------------------------------------------------------------ league


def test_snapshot_pool_evicts_the_oldest_but_keeps_the_best():
    model = NimModel(max_pile=15)
    pool = SnapshotPool(capacity=3)
    ids = [pool.add(model, update=u) for u in range(3)]
    best = pool.set_best(model, update=3, score=0.7)
    assert len(pool) == 4 and pool.best_id == best
    pool.add(model, update=4)
    assert ids[0] not in pool.ids() and best in pool.ids() and len(pool) == 4
    pool.set_best(model, update=5, score=0.8)
    assert best not in pool.ids()  # the previous best is replaced
    rng = np.random.default_rng(0)
    assert {pool.sample(rng) for _ in range(200)} == set(pool.ids())
    snap = pool.get(pool.best_id)
    assert not any(p.requires_grad for p in snap.parameters()) and not snap.training
    restored = SnapshotPool(capacity=3)
    restored.load_state_dict(pool.state_dict(), lambda: NimModel(max_pile=15))
    assert restored.ids() == pool.ids() and restored.best_id == pool.best_id and restored.best_score == 0.8
    assert SnapshotPool(capacity=2).sample(rng) is None


# ------------------------------------------------------------------------------------------ convergence


def train_nim(cfg: PPOConfig, updates: int, pool: SnapshotPool | None = None, selfplay: float = 1.0, seed: int = 0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = NimModel(max_pile=15)
    learner = PPOLearner(model, cfg)
    col = RolloutCollector(NimEnv(num_envs=16, max_pile=15), model, nim_games(rng, pool, selfplay), num_steps=16,
                           opponents=pool.get if pool is not None else None, seed=seed)  # fmt: skip
    for u in range(updates):
        learner.update(col.collect())
        if pool is not None and u % 5 == 4:
            pool.add(model, update=u)
    return model


@pytest.mark.parametrize("estimator", ["vrpo", "gae"])
def test_self_play_converges_to_the_optimal_nim_policy(estimator):
    cfg = PPOConfig(estimator=estimator, lr=3e-3, epochs=4, minibatch_size=128)
    model = train_nim(cfg, updates=80)
    moves = greedy_moves(model)
    wrong = {n: a for n, a in moves.items() if n % 4 and a != optimal_action(n)}
    assert not wrong, f"{estimator}: non-optimal moves {wrong}"


def test_training_against_a_snapshot_pool_converges():
    cfg = dataclasses.replace(PPOConfig(), lr=3e-3, epochs=4, minibatch_size=128)
    model = train_nim(cfg, updates=80, pool=SnapshotPool(capacity=4), selfplay=0.5, seed=1)
    moves = greedy_moves(model)
    assert all(a == optimal_action(n) for n, a in moves.items() if n % 4), moves


# ------------------------------------------------------------------------------------------ errors


class FlakyNim(NimEnv):
    """Engine errors as the C++ pool reports them: a failed reset (no decision at all) and a mid-game error."""

    def reset(self, env_id, spec):
        if spec == 7:
            self._ready.append(EncodedEvent(env_id, 0, None, {"winner": None, "reason": "error", "error": "start"}))
            return
        super().reset(env_id, spec)

    def step(self, env_id, action):
        if self._pile[env_id] == 11:
            self._ready.append(EncodedEvent(env_id, 0, None, {"winner": 1, "reason": "error", "error": "no action"}))
            return
        super().step(env_id, action)


def test_engine_errors_are_counted_truncated_and_the_slot_restarts():
    model = NimModel(max_pile=15)
    col = RolloutCollector(FlakyNim(num_envs=3, max_pile=15), model, nim_games(np.random.default_rng(4)), num_steps=40,
                           seed=4)  # fmt: skip
    ro = col.collect()
    errors = [g for g in ro.games if g.reason == "error"]
    assert any(g.learner_rows == 0 for g in errors) and any(g.learner_rows > 0 for g in errors)
    assert all(g.truncated for g in errors)
    assert (ro.rewards[ro.truncated] == 0).all() and ro.truncated.any()  # an error's "winner" is not an outcome
    assert ro.players.shape == (40, 3)


# ------------------------------------------------------------------------------------------ objectives


def test_policy_objective_is_pluggable():
    """A new objective (e.g. MaxRL, docs/eng-plan.md) only registers prepare/loss; collector and league unchanged."""
    calls = {"prepare": 0, "loss": 0}

    class Recording(PolicyObjective):
        def prepare(self, rollout, est):
            calls["prepare"] += 1
            assert est.advantages.shape == rollout.players.shape
            return torch.where(rollout.dones, rollout.rewards, torch.zeros_like(rollout.rewards))  # outcome-only

        def loss(self, x):
            calls["loss"] += 1
            assert x.index.shape == x.actions.shape and x.log_probs.requires_grad
            new = x.log_probs.gather(1, x.actions.unsqueeze(1)).squeeze(1)
            return -(x.advantages * new).mean(), {"custom": torch.ones(())}

    register_objective("recording", lambda cfg: Recording())
    try:
        assert "recording" in available_objectives() and {"ppo_clip", "pg"} <= set(available_objectives())
        with pytest.raises(ValueError, match="already registered"):
            register_objective("recording", lambda cfg: Recording())
        model = NimModel(max_pile=15)
        learner = PPOLearner(model, PPOConfig(objective="recording", epochs=2, minibatch_size=16))
        ro = RolloutCollector(NimEnv(2, 15), model, nim_games(np.random.default_rng(0)), num_steps=16, seed=0).collect()
        stats = learner.update(ro)
        assert calls == {"prepare": 1, "loss": 4} and stats["custom"] == 1.0
        assert np.isfinite(PPOLearner(model, PPOConfig(objective="pg")).update(ro)["loss"])
    finally:
        register_objective("recording", None)
    with pytest.raises(ValueError, match="objective"):
        PPOConfig(objective="recording")
