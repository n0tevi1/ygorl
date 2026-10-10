"""Guarded training must receive the same native mask as the inference ablation."""

import json
from pathlib import Path

import numpy as np
import pytest

from ygorl import _core
from ygorl.cards.cdb import CardVocab
from ygorl.cards.ydk import Deck
from ygorl.engine.duel import Duel, DuelConfig, PlayerRules
from ygorl.env.encoded import EncodedVecEnv
from ygorl.env.encoding import ACTION_KINDS
from ygorl.env.pool import GameSpec
from ygorl.train.trainer import TrainConfig


@pytest.mark.parametrize("budget", [0, 1, 4, 32])
def test_budget_config_round_trip_and_legacy_default(budget):
    cfg = TrainConfig(cancel_budget=budget)
    assert TrainConfig.from_dict(cfg.to_dict()).cancel_budget == budget
    assert TrainConfig.from_dict({}).cancel_budget == 32


@pytest.mark.parametrize("bad", [-1, 2, 33, True, 1.0])
def test_invalid_budget_rejected_before_environment_creation(bad):
    with pytest.raises(ValueError, match="cancel_budget"):
        TrainConfig(cancel_budget=bad)
    with pytest.raises(ValueError, match="cancel_budget"):
        EncodedVecEnv(1, cancel_budget=bad)


@pytest.mark.parametrize("fixture", ["target_cancel_400", "material_cancel_loop"])
@pytest.mark.parametrize("budget", [1, 32])
def test_pool_masks_match_host_through_a_real_cancel_cycle(fixture, budget):
    data = json.loads((Path(__file__).parent / "data" / (fixture + ".json")).read_text())
    cfg = DuelConfig(**{**data["config"], "player": PlayerRules(**data["config"]["player"])})
    a, b = Deck(**data["deck_a"]), Deck(**data["deck_b"])
    duel = Duel(data["seed"], None, a, b, config=cfg, first=data["first"])
    vocab = CardVocab.from_db(duel.cards)
    host = _core.HostDuel(
        duel.cards.to_core(),
        duel.scripts,
        [vocab.password(i) for i in range(vocab.FIRST_INDEX, len(vocab))],
        cancel_budget=budget,
    )
    p = cfg.player
    player = (p.starting_lp, p.starting_hand, p.draw_per_turn)
    host.start(
        list(duel.core_seed),
        cfg.rule_flags,
        player,
        player,
        [(list(m), list(e)) for m, e in duel.loaded_decks()],
        cfg.max_turns,
        cfg.max_decisions,
    )
    env = EncodedVecEnv(
        1,
        1,
        cards=duel.cards,
        scripts=duel.scripts,
        vocab=vocab,
        event_length=0,
        skip_forced=False,
        cancel_budget=budget,
    )
    env.reset(0, GameSpec(data["seed"], a, b, first=data["first"], config=cfg))

    def ready():
        (ev,) = env.recv(timeout=10)
        assert ev.result is None and ev.player == host.player()
        for k, v in host.observe().items():
            np.testing.assert_array_equal(ev.obs[k], v)
        return ev

    for index in data["actions"]:
        ready()
        env.step(0, index)
        host.act(index)
    before = ready()
    cancel = next(i for i, a in enumerate(host.actions()) if a[0] == ACTION_KINDS.index("cancel"))
    assert before.obs["action_mask"][cancel]
    env.step(0, cancel)
    host.act(cancel)
    ready()
    env.step(0, 0)
    host.act(0)
    after = ready()
    cancel = next(i for i, a in enumerate(host.actions()) if a[0] == ACTION_KINDS.index("cancel"))
    assert bool(after.obs["action_mask"][cancel]) == (budget == 32)


def test_trainer_stores_guarded_mask_and_restores_budget(tmp_path):
    import torch
    from ygorl.train.ppo import PPOConfig
    from ygorl.train.trainer import Trainer

    decks = Path(__file__).parent / "decks"
    cfg = TrainConfig(
        decks=(str(decks / "snake_eye.ydk"), str(decks / "kashtira.ydk")),
        cancel_budget=1,
        num_envs=2,
        steps=4,
        event_length=8,
        eval_every=0,
        net={"d_model": 16, "n_heads": 2, "board_layers": 1, "history_layers": 1},
        ppo=PPOConfig(epochs=1, minibatch_size=8),
    )
    trainer = Trainer(cfg, tmp_path / "train", log=None)
    assert trainer.env.cancel_budget == 1
    collect = trainer._collect
    checked = []

    def audited_collect():
        ro = collect()
        torch.testing.assert_close(ro.obs["action_mask"].reshape_as(ro.action_mask), ro.action_mask)
        assert (ro.probs[~ro.action_mask.bool()] == 0).all()
        chosen = ro.probs.gather(-1, ro.actions.unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(ro.log_probs, chosen.log())
        checked.append(True)
        return ro

    trainer._collect = audited_collect
    assert trainer.step()["minibatches"] > 0 and checked
    path = trainer.save()
    resumed = Trainer.resume(path, tmp_path / "resumed", log=None)
    assert resumed.cfg.cancel_budget == resumed.env.cancel_budget == 1
    assert resumed.counters == trainer.counters
    assert resumed.learner.optimizer.state
    for k, v in trainer.model.state_dict().items():
        torch.testing.assert_close(resumed.model.state_dict()[k], v, rtol=0, atol=0)
    assert torch.equal(resumed.collector.generator.get_state(), trainer.collector.generator.get_state())
