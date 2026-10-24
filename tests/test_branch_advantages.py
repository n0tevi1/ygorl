"""Window counterfactual advantages in the PPO update (T5; PPOConfig.branch_mode / branch_coef / branch_q_coef,
docs/training.md §8.2 「窗口反事实优势」): on hand-set branch values of a toy rollout, the advantages and Q targets
are the formula's, a row whose taken action is outside the branch set keeps VRPO, and "off" / a zero coefficient
leaves the update bit for bit as it was."""

import copy
import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ygorl.train.advantages import estimate, normalize_advantages  # noqa: E402
from ygorl.train.ppo import PPOConfig, PPOLearner, branch_advantages  # noqa: E402
from ygorl.train.rollout import Assignment, BranchConfig, RolloutCollector  # noqa: E402
from ygorl.train.toy import NimBranchHooks, NimEnv, NimModel  # noqa: E402


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def rollout(branch=True):
    """A Nim rollout against a snapshot with branched windows; then three rows get hand-set branch values."""
    torch.manual_seed(0)
    model = NimModel(15)
    rng = np.random.default_rng(1)

    def next_game():
        return Assignment(int(rng.integers(4, 16)), 0, int(rng.integers(0, 2)))

    env = NimEnv(4, snapshots=True)
    col = RolloutCollector(env, model, next_game, 16, opponents=lambda sid: model, seed=2, min_batch=4,
                           branch=BranchConfig(require_recovery=False) if branch else None,
                           branch_hooks=NimBranchHooks())  # fmt: skip
    ro = col.collect()
    if branch:
        ro.branch_value = torch.full_like(ro.q, float("nan"))
        ro.branch_valid = torch.zeros_like(ro.dones)
        ro.branch_pass = torch.full_like(ro.actions, -1)
        legal = ro.action_mask.sum(-1) >= 2
        rows = legal.nonzero().tolist()
        (t0, b0), (t1, b1), (t2, b2) = rows[0], rows[3], rows[5]
        a0, a1 = int(ro.actions[t0, b0]), int(ro.actions[t1, b1])
        other0 = next(a for a in range(3) if a != a0 and ro.action_mask[t0, b0, a])
        other1 = next(a for a in range(3) if a != a1 and ro.action_mask[t1, b1, a])
        ro.branch_value[t0, b0, a0], ro.branch_value[t0, b0, other0] = 0.8, -0.4  # taken in the set
        ro.branch_value[t1, b1, other1] = 0.5  # a set without the taken action: VRPO stays
        ro.branch_value[t1, b1, next(a for a in range(3) if a not in (a1, other1))] = -0.5
        ro.branch_valid[t0, b0] = ro.branch_valid[t1, b1] = True
        ro.branch_pass[t0, b0] = other0
        ro.branch_value[t2, b2, int(ro.actions[t2, b2])] = 1.0  # values on a row that is not valid: ignored
        ro.marks = ((t0, b0, a0, other0), (t1, b1, a1))
    return model, ro


def vrpo(ro, cfg: PPOConfig):
    return estimate(cfg.estimator, rewards=ro.rewards, dones=ro.dones, players=ro.players, actions=ro.actions,
                    action_mask=ro.action_mask, probs=ro.probs, q=ro.q, values=ro.values, gamma=cfg.gamma, lam=cfg.lam,
                    bootstrap_player=ro.bootstrap_player, bootstrap_value=ro.bootstrap_value,
                    bootstrap_q=ro.bootstrap_q, bootstrap_probs=ro.bootstrap_probs, bootstrap_mask=ro.bootstrap_mask,
                    truncated=ro.truncated, vrpo_mode=cfg.vrpo_mode, normalize="none")  # fmt: skip


def formula(ro, t, b, a, other) -> float:
    """Â = V_b(taken) - Σ π̃ V_b over the set {taken, other}."""
    p, o = float(ro.probs[t, b, a]), float(ro.probs[t, b, other])
    v, w = float(ro.branch_value[t, b, a]), float(ro.branch_value[t, b, other])
    return v - (p * v + o * w) / (p + o)


def test_branch_advantage_is_the_value_minus_the_renormalized_policy_baseline():
    _, ro = rollout()
    (t0, b0, a0, other0), (t1, b1, _) = ro.marks
    adv, used, in_set = branch_advantages(ro)
    assert used.sum() == 1 and used[t0, b0] and not used[t1, b1]  # row 1: taken action outside the set
    assert in_set.sum() == 4  # the two valid rows' sets; the invalid row's value is ignored
    assert adv[t0, b0].item() == pytest.approx(formula(ro, t0, b0, a0, other0), abs=1e-6)
    assert (adv[~used] == 0).all()


@pytest.mark.parametrize("mode", ["replace", "add"])
def test_replace_and_add_put_the_branch_advantage_into_the_raw_advantages(mode):
    model, ro = rollout()
    (t0, b0, a0, other0), (t1, b1, _) = ro.marks
    cfg = PPOConfig(branch_mode=mode, branch_coef=0.7)
    raw = vrpo(ro, cfg).advantages
    a_hat = formula(ro, t0, b0, a0, other0)
    want = raw.clone()
    want[t0, b0] = a_hat if mode == "replace" else raw[t0, b0] + 0.7 * a_hat
    est = PPOLearner(copy.deepcopy(model), cfg).targets(ro)
    torch.testing.assert_close(est.advantages, normalize_advantages(want, mode=cfg.adv_norm), rtol=0, atol=1e-6)
    # the fallback row: its raw advantage is the VRPO one (only the normalization moves it)
    off = PPOLearner(copy.deepcopy(model), PPOConfig()).targets(ro)
    std = want.std(correction=0) + 1e-8
    assert est.advantages[t1, b1].item() == pytest.approx(((raw[t1, b1] - want.mean()) / std).item(), abs=1e-5)
    torch.testing.assert_close(est.q_targets, off.q_targets, rtol=0, atol=0)  # branch_q_coef 0: Q unchanged


def test_branch_q_targets_blend_the_lambda_target_and_the_branch_values():
    model, ro = rollout()
    (t0, b0, a0, other0), (t1, b1, a1) = ro.marks
    w = 0.25
    learner = PPOLearner(copy.deepcopy(model), PPOConfig(branch_q_coef=w))  # Q only: advantages stay VRPO
    est = learner.targets(ro)
    base = vrpo(ro, PPOConfig())
    want = base.q_targets.clone()
    want[t0, b0] = (1 - w) * base.q_targets[t0, b0] + w * ro.branch_value[t0, b0, a0]
    torch.testing.assert_close(est.q_targets, want, rtol=0, atol=1e-6)
    torch.testing.assert_close(est.v_targets, base.v_targets, rtol=0, atol=0)
    torch.testing.assert_close(est.advantages, normalize_advantages(base.advantages), rtol=0, atol=1e-6)
    extra = learner._branch_q
    finite = torch.isfinite(extra)
    assert finite.sum() == 3 and not finite[t0, b0, a0]  # the set's untaken actions of both valid rows
    assert extra[t0, b0, other0].item() == pytest.approx((1 - w) * ro.q[t0, b0, other0].item() - w * 0.4, abs=1e-6)
    for a in finite[t1, b1].nonzero().flatten().tolist():
        assert a != a1
        want_a = (1 - w) * ro.q[t1, b1, a].item() + w * ro.branch_value[t1, b1, a].item()
        assert extra[t1, b1, a].item() == pytest.approx(want_a, abs=1e-6)
    stats = learner.update(ro)
    assert np.isfinite(stats["branch_q_loss"]) and stats["branch_q_loss"] > 0
    assert stats["branch_rows"] == 1 and stats["branch_valid_rows"] == 2 and stats["branch_fallback_rows"] == 1
    assert stats["branch_fire_share"] == float(ro.branch_value[t0, b0, a0] > ro.branch_value[t0, b0, other0])
    assert stats["branch_adv_mean"] == pytest.approx(formula(ro, t0, b0, a0, other0), abs=1e-6)


def _update(cfg: PPOConfig, ro, model):
    m = copy.deepcopy(model)
    learner = PPOLearner(m, dataclasses.replace(cfg, epochs=1, minibatch_size=16, target_kl=None))
    torch.manual_seed(7)
    stats = learner.update(ro)
    return m.state_dict(), stats


@pytest.mark.parametrize("cfg", [PPOConfig(branch_mode="off", branch_q_coef=0.0),
                                 PPOConfig(branch_mode="add", branch_coef=0.0, branch_q_coef=0.0)])  # fmt: skip
def test_off_and_zero_coefficients_update_exactly_as_without_branching(cfg):
    model, ro = rollout()
    _, plain = rollout(branch=False)
    for name in ("obs", "actions", "probs", "q", "rewards"):  # the same rollout, without the branch fields
        a, b = getattr(ro, name), getattr(plain, name)
        assert all(torch.equal(a[k], b[k]) for k in a) if isinstance(a, dict) else torch.equal(a, b)
    ref, _ = _update(PPOConfig(), plain, model)
    got, stats = _update(cfg, ro, model)
    assert all(torch.equal(ref[k], v) for k, v in got.items())
    assert ("branch_rows" in stats) == (cfg.branch_mode != "off")


def test_the_trainer_branches_windows_and_logs_the_branch_advantages(tmp_path):
    from pathlib import Path

    from ygorl.train.trainer import TrainConfig, Trainer

    here = Path(__file__).parent
    data = str(here / "data" / "line_recovery.jsonl")
    pair = (str(here / "decks" / "snake_eye.ydk"), str(here / "decks" / "kashtira.ydk"))
    cfg = TrainConfig(decks=pair, num_envs=2, env_threads=1, steps=24, event_length=16,
                      net={"d_model": 16, "n_heads": 2, "board_layers": 1, "history_layers": 1}, privileged_dim=8,
                      critic_hidden=16, max_decisions=60, eval_every=0, checkpoint_every=0, snapshot_every=0,
                      torch_threads=1, collect_threads=1, seed=4, selfplay_fraction=0.0, line_demos=(data,),
                      line_share=1.0, line_recovery=(data,), branch_windows=True, branch_require_recovery=False,
                      ppo=PPOConfig(branch_mode="replace", branch_q_coef=0.5))  # fmt: skip
    logs = []
    trainer = Trainer(cfg, tmp_path, log=logs.append)
    assert trainer.env.snapshots and trainer.collector.branch == cfg.branch_config()
    trainer.pool.add(trainer.model, update=0)
    rec = trainer.step()
    assert "branch_windows" in rec and "branched windows" in logs[-1]
    assert "branch_rows" in rec and "branch_q_loss" in rec and trainer.env.snapshot_count() == 0
    assert TrainConfig.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(ValueError, match="branch_windows"):
        TrainConfig(decks=pair, ppo=PPOConfig(branch_mode="add"))
    with pytest.raises(ValueError, match="line_recovery"):
        TrainConfig(decks=pair, line_demos=(data,), line_share=1.0, branch_windows=True)
    with pytest.raises(ValueError, match="line_demos"):
        TrainConfig(decks=pair, branch_windows=True, branch_require_recovery=False)


def test_branch_modes_change_the_update_and_bad_values_are_refused():
    model, ro = rollout()
    ref, _ = _update(PPOConfig(), ro, model)
    for cfg in (PPOConfig(branch_mode="replace"), PPOConfig(branch_q_coef=0.5)):
        got, _ = _update(cfg, ro, model)
        assert any(not torch.equal(ref[k], v) for k, v in got.items())
    for bad in ({"branch_mode": "mix"}, {"branch_coef": -1.0}, {"branch_q_coef": 1.5}):
        with pytest.raises(ValueError):
            PPOConfig(**bad)
