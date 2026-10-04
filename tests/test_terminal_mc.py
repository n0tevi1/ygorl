"""True terminal supervision, collection boundaries and matched policy advantages (#61)."""

import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ygorl.train.advantages import terminal_returns  # noqa: E402
from ygorl.train.ppo import PPOConfig, PPOLearner  # noqa: E402
from ygorl.train.rollout import Assignment, RolloutCollector  # noqa: E402
from ygorl.train.toy import NimEnv, NimModel  # noqa: E402
from ygorl.train.trainer import TrainConfig  # noqa: E402


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def collect(piles, **env_kw):
    env = NimEnv(len(piles), **env_kw)
    model = NimModel()
    specs = iter(piles * 2)
    collector = RolloutCollector(env, model, lambda: Assignment(next(specs)), 1, complete_games=True)
    return collector, model


def test_terminal_z_flips_seats_and_stops_at_every_game_including_draw():
    players = torch.tensor([[0, 1, 0, 1, 0, 0, 1]]).T
    rewards = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0]]).T
    dones = torch.tensor([[False, False, True, False, True, False, True]]).T
    z = terminal_returns(rewards, dones, players, torch.zeros_like(dones))
    torch.testing.assert_close(z[:, 0], torch.tensor([1.0, -1.0, 1.0, 0.0, 0.0, 1.0, -1.0]))
    with pytest.raises(ValueError, match="unfinished"):
        terminal_returns(rewards[:-1], dones[:-1], players[:-1], torch.zeros_like(dones[:-1]))
    with pytest.raises(ValueError, match="non-truncated"):
        terminal_returns(rewards, dones, players, dones)
    with pytest.raises(ValueError, match="unscaled"):
        terminal_returns(rewards * 0.9, dones, players, torch.zeros_like(dones))


def test_complete_games_pack_unequal_lengths_no_padding_and_restart():
    collector, model = collect([1, 15, 7])
    for _ in range(2):
        ro = collector.collect()
        assert ro.complete_games and ro.shape[1] == 1
        assert len(ro.games) == 3 and ro.dones.sum() == 3 and ro.dones[-1].all()
        assert ro.players.numel() == sum(g.learner_rows for g in ro.games)
        assert ro.obs["pile"][0] == 1 and not collector._held
        assert all(s is None for s in collector._slots)
        # Compare each actual game's winner to every acting seat (independent of recursion).
        z = terminal_returns(ro.rewards, ro.dones, ro.players, ro.truncated).flatten()
        start = 0
        for end in torch.where(ro.dones[:, 0])[0].tolist():
            winner = int(ro.players[end])  # taking the last stone wins
            assert torch.equal(z[start : end + 1], torch.where(ro.players[start : end + 1, 0] == winner, 1.0, -1.0))
            start = end + 1
        assert model.training
    assert collector.games_started == 6


def test_discard_entire_truncated_game_and_charge_its_cost():
    collector, _ = collect([1, 15], max_moves=1)
    ro = collector.collect()
    assert len(ro.games) == 2 and sum(g.truncated for g in ro.games) == 1
    assert ro.shape == (1, 1) and ro.rewards.item() == 1
    assert ro.discarded_rows == 1 and ro.decisions == 2 and not ro.truncated.any()
    collector, _ = collect([15, 15], max_moves=1)
    with pytest.raises(ValueError, match="no valid learner rows"):
        collector.collect()
    assert collector.games_started == 2  # no retries that favour short games


def test_complete_pool_games_keep_only_learner_rows_and_terminal_loser_reward():
    env = NimEnv(2)
    model = NimModel()
    # pile 1: opponent wins immediately, the learner contributed zero rows.
    specs = iter([Assignment(1, opponent=0, learner_seat=1), Assignment(15, opponent=0, learner_seat=1)])
    collector = RolloutCollector(env, model, lambda: next(specs), 1, complete_games=True, opponents=lambda _: model)
    ro = collector.collect()
    assert len(ro.games) == 2 and any(g.learner_rows == 0 for g in ro.games)
    assert (ro.players == 1).all() and ro.opponent_decisions > 0
    game = next(g for g in ro.games if g.learner_rows)
    assert (terminal_returns(ro.rewards, ro.dones, ro.players, ro.truncated) == (1 if game.winner == 1 else -1)).all()


def test_terminal_supervision_is_critic_independent_but_advantages_stay_matched():
    collector, model = collect([15, 7])
    ro = collector.collect()
    control = PPOLearner(model, PPOConfig(adv_norm="none"))
    mc = PPOLearner(model, PPOConfig(critic_target="terminal", adv_norm="none", minibatch_size=8))
    a, b = control.targets(ro), mc.targets(ro)
    torch.testing.assert_close(a.advantages, b.advantages)
    assert not torch.allclose(a.q_targets, b.q_targets)
    altered = dataclasses.replace(
        ro, q=ro.q + 20, values=ro.values - 10, bootstrap_q=ro.bootstrap_q + 40, bootstrap_value=ro.bootstrap_value + 30
    )
    c = mc.targets(altered)
    torch.testing.assert_close(b.q_targets, c.q_targets)
    torch.testing.assert_close(b.v_targets, c.v_targets)
    with pytest.raises(ValueError, match="complete-game"):
        mc.targets(dataclasses.replace(ro, complete_games=False))
    stats = mc.update(ro)
    assert all(np.isfinite(stats[k]) for k in ("q_terminal_ev", "v_terminal_ev", "q_terminal_mse", "loss"))


def test_terminal_config_requires_unscaled_complete_synchronous_games():
    assert not TrainConfig().complete_games and PPOConfig().critic_target == "lambda"
    with pytest.raises(ValueError, match="complete_games"):
        TrainConfig(ppo=PPOConfig(critic_target="terminal"))
    for change in ({"gamma": 0.9}, {"critic_lam": 1.0}):
        with pytest.raises(ValueError):
            PPOConfig(critic_target="terminal", **change)
    for change in ({"turn_discount": 0.9}, {"overlap_collect": True}):
        with pytest.raises(ValueError):
            TrainConfig(complete_games=True, **change)
    cfg = TrainConfig(complete_games=True, ppo=PPOConfig(critic_target="terminal"))
    assert TrainConfig.from_dict(cfg.to_dict()) == cfg
