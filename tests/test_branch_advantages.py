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
from ygorl.train.ppo import PPOConfig, PPOLearner, branch_advantages, branch_ce_targets  # noqa: E402
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
                                 PPOConfig(branch_mode="add", branch_coef=0.0, branch_q_coef=0.0),
                                 PPOConfig(branch_ce_coef=0.0, branch_ce_margin=0.0, branch_ce_weight="uniform")])  # fmt: skip
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
    for bad in ({"branch_mode": "mix"}, {"branch_coef": -1.0}, {"branch_q_coef": 1.5}, {"branch_ce_coef": -0.1},
                {"branch_ce_margin": -1.0}, {"branch_ce_weight": "abs"}):  # fmt: skip
        with pytest.raises(ValueError):
            PPOConfig(**bad)


# ------------------------------------------------------------------------- the supervised term (branch_ce_coef)


def ce_rollout():
    """Four valid branched rows of the toy rollout with hand-set values (the second candidate is the pass): a decided
    fire (gap 1.2), a tie (gap 0.05), a decided hold (gap 0.5), a decided fire (gap 0.3); and values on a fifth row
    that is not valid (ignored)."""
    model, ro = rollout()
    ro.branch_value = torch.full_like(ro.q, float("nan"))
    ro.branch_valid = torch.zeros_like(ro.dones)
    ro.branch_pass = torch.full_like(ro.actions, -1)
    legal = (ro.action_mask.sum(-1) >= 2).nonzero().tolist()
    marks = []
    for (t, b), (fire, hold) in zip(legal[:5], [(0.8, -0.4), (0.5, 0.45), (-0.2, 0.3), (0.1, -0.2), (1.0, -1.0)]):
        a, p = [int(x) for x in ro.action_mask[t, b].nonzero().flatten()[:2]]
        ro.branch_value[t, b, a], ro.branch_value[t, b, p] = fire, hold
        ro.branch_pass[t, b] = p
        ro.branch_valid[t, b] = len(marks) < 4
        marks.append((t, b, a, p))
    return model, ro, marks


def test_branch_ce_targets_take_the_best_candidate_drop_ties_and_weigh_by_the_gap():
    _, ro, marks = ce_rollout()
    (t0, b0, a0, _), (t1, b1, _, _), (t2, b2, _, p2), (t3, b3, a3, _), (t4, b4, _, _) = marks
    target, weight, gap, tie = branch_ce_targets(ro, margin=0.1)
    assert int((target >= 0).sum()) == 3 and int(tie.sum()) == 1 and tie[t1, b1]
    assert [int(target[t, b]) for t, b in ((t0, b0), (t1, b1), (t2, b2), (t3, b3))] == [a0, -1, p2, a3]
    assert target[t4, b4] == -1 and weight[t4, b4] == 0  # not a valid row
    mean = (1.2 + 0.5 + 0.3) / 3
    for (t, b), g in (((t0, b0), 1.2), ((t2, b2), 0.5), ((t3, b3), 0.3)):
        assert gap[t, b].item() == pytest.approx(g, abs=1e-6)
        assert weight[t, b].item() == pytest.approx(g / mean, abs=1e-6)
    assert weight[t1, b1] == 0 and float(weight.sum()) == pytest.approx(3.0, abs=1e-5)
    uniform = branch_ce_targets(ro, margin=0.1, weight="uniform")[1]
    assert torch.equal(uniform, (target >= 0).to(uniform))
    wide, _, _, wide_tie = branch_ce_targets(ro, margin=0.3)  # the margin is inclusive: gap 0.3 is a tie now
    assert wide[t3, b3] == -1 and wide_tie[t3, b3] and int(wide_tie.sum()) == 2


def test_the_learner_puts_the_margin_in_board_points_and_reports_tie_and_hold_shares():
    model, ro, _ = ce_rollout()
    cfg = PPOConfig(branch_ce_coef=0.5, branch_ce_margin=1.0)
    learner = PPOLearner(copy.deepcopy(model), cfg, branch_board_scale=0.1)  # margin 0.1 in value units
    learner.targets(ro)
    target, weight = learner._branch_ce
    want_t, want_w, _, _ = branch_ce_targets(ro, margin=0.1)
    assert torch.equal(target, want_t) and torch.equal(weight, want_w)
    stats = learner._branch_stats
    assert stats["branch_ce_rows"] == 3 and stats["branch_valid_rows"] == 4
    assert stats["branch_tie_share"] == pytest.approx(0.25) and stats["branch_hold_share"] == pytest.approx(0.25)
    assert stats["branch_fire_share"] == pytest.approx(0.75)  # no margin there: the tie's 0.5 > 0.45 counts as fire
    wide = PPOLearner(copy.deepcopy(model), cfg, branch_board_scale=1.0)  # margin 1.0: only the 1.2 gap is decided
    wide.targets(ro)
    assert int((wide._branch_ce[0] >= 0).sum()) == 1 and wide._branch_stats["branch_tie_share"] == pytest.approx(0.75)


class FakeLabels:
    """``line_ce_coef`` targets on other rows (the interface of LineLabels.ce_targets)."""

    def __init__(self, ro, rows):
        self.target = torch.full_like(ro.actions, -1)
        self.weight = torch.zeros_like(ro.rewards)
        for t, b in rows:
            self.target[t, b] = int(ro.action_mask[t, b].nonzero()[-1])
            self.weight[t, b] = 2.0

    def ce_targets(self, ro):
        return self.target, self.weight, {"line_ce_rows": int((self.target >= 0).sum())}


def test_branch_ce_and_line_ce_add_up_in_the_loss():
    from ygorl.nets.batch import policy_logits

    model, ro, _ = ce_rollout()
    labels = FakeLabels(ro, (ro.action_mask.sum(-1) >= 2).nonzero().tolist()[6:8])
    one = {"epochs": 1, "minibatch_size": ro.actions.numel(), "target_kl": None, "branch_ce_margin": 1.0}  # 1 minibatch

    def run(**kw):
        learner = PPOLearner(copy.deepcopy(model), PPOConfig(**one, **kw), line_labels=labels, branch_board_scale=0.1)
        torch.manual_seed(7)
        return learner.update(ro)

    base, line, branch = run(), run(line_ce_coef=0.2), run(branch_ce_coef=0.3)
    both = run(branch_ce_coef=0.3, line_ce_coef=0.2)
    assert both["branch_ce_loss"] == pytest.approx(branch["branch_ce_loss"], abs=1e-6)
    assert both["line_ce_loss"] == pytest.approx(line["line_ce_loss"], abs=1e-6)
    want = base["loss"] + 0.2 * line["line_ce_loss"] + 0.3 * branch["branch_ce_loss"]
    assert both["loss"] == pytest.approx(want, abs=1e-5)
    # the branch term: the weighted NLL of the targets, averaged over the targeted rows
    with torch.no_grad():
        logp = torch.log_softmax(policy_logits(model, ro.obs).float(), -1)
    target, weight, _, _ = branch_ce_targets(ro, margin=0.1)
    rows = (target >= 0).reshape(-1)
    nll = -logp[rows].gather(1, target.reshape(-1)[rows].unsqueeze(1)).squeeze(1)
    want_ce = float((weight.reshape(-1)[rows] * nll).sum() / rows.sum())
    assert branch["branch_ce_loss"] == pytest.approx(want_ce, abs=1e-5)
    assert branch["branch_ce_rows"] == 3 and "branch_tie_share" in branch  # branch_mode off: the stats still come
    assert "branch_ce_loss" not in base and "branch_rows" not in base


def test_branch_ce_raises_the_probability_of_the_best_nim_take():
    """Positive control: against a perfect Nim opponent the best take from a winning pile is the only one that does
    not lose (tests/test_window_branching.py); a few updates with branch_ce_coef on the same rollout raise its
    probability on the targeted rows well above the same updates without it."""
    from ygorl.nets.batch import policy_logits
    from ygorl.nets.heads import MASKED_LOGIT
    from ygorl.train.toy import optimal_action

    class Perfect(torch.nn.Module):  # the best take where there is one, else 1 (test_window_branching.FixedNim)
        def policy_logits(self, obs):
            pile = obs["pile"].long()
            take = torch.where(pile % 4 != 0, pile % 4, torch.ones_like(pile)).clamp(max=pile)
            logits = torch.full(obs["action_mask"].shape, -30.0)
            logits[torch.arange(len(pile)), take - 1] = 30.0
            return logits.masked_fill(~obs["action_mask"], MASKED_LOGIT)

    torch.manual_seed(0)
    model, rng = NimModel(15), np.random.default_rng(3)
    branch = BranchConfig(k=3, rollouts=1, require_recovery=False, board_scale=0.5)
    col = RolloutCollector(NimEnv(8, snapshots=True), model,
                           lambda: Assignment(int(rng.integers(1, 16)), 0, int(rng.integers(0, 2))), 16,
                           opponents=lambda sid: Perfect(), seed=3, min_batch=8, branch=branch,
                           branch_hooks=NimBranchHooks())  # fmt: skip
    ro = col.collect()
    target = branch_ce_targets(ro, margin=0.5 * 0.5)[0]
    rows = (target >= 0).reshape(-1).nonzero().squeeze(1)
    piles = ro.obs["pile"].reshape(-1)[rows]
    best = target.reshape(-1)[rows]
    assert len(rows) > 10 and bool((piles % 4 != 0).all())
    assert best.tolist() == [optimal_action(int(p)) - 1 for p in piles]  # every target is the best take

    def p_best(m):
        with torch.no_grad():
            return float(torch.softmax(policy_logits(m, ro.obs).float(), -1)[rows, best].mean())

    def train(coef):
        m = copy.deepcopy(model)
        cfg = PPOConfig(branch_ce_coef=coef, epochs=2, minibatch_size=32, target_kl=None, lr=1e-2)
        learner = PPOLearner(m, cfg, branch_board_scale=0.5)
        torch.manual_seed(11)
        for _ in range(5):
            learner.update(ro)
        return m

    before, control, ce = p_best(model), p_best(train(0.0)), p_best(train(1.0))
    print(f"P(best take) on {len(rows)} targeted rows: before {before:.3f}, control {control:.3f}, CE {ce:.3f}")
    assert ce > before + 0.1 and ce > control + 0.05


def test_branch_ce_trains_with_branch_mode_off_in_the_trainer(tmp_path):
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
                      ppo=PPOConfig(branch_ce_coef=0.05, branch_ce_margin=0.0))  # fmt: skip
    logs = []
    trainer = Trainer(cfg, tmp_path, log=logs.append)
    assert trainer.learner.branch_board_scale == cfg.branch_board_scale
    trainer.pool.add(trainer.model, update=0)
    rec = trainer.step()
    for key in ("branch_ce_rows", "branch_ce_loss", "branch_tie_share", "branch_hold_share", "branch_fire_share"):
        assert key in rec, key
    assert "CE rows" in logs[-1] and "tie share" in logs[-1]
    assert "branch_ce_loss" in (tmp_path / "metrics.jsonl").read_text().splitlines()[-1]
    with pytest.raises(ValueError, match="branch_windows"):
        TrainConfig(decks=pair, ppo=PPOConfig(branch_ce_coef=0.1))
