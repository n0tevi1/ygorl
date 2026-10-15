"""Real native prefix: selected attacker must be known before target selection."""

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ygorl import _core
from ygorl.cards.cdb import CardVocab
from ygorl.cards.ydk import Deck
from ygorl.engine.duel import Duel, DuelConfig, DuelSession, PlayerRules
from ygorl.env.events import ATTACK_CHOICE_EVENT
from ygorl.env.observer import PointObserver

CASE = json.loads((Path(__file__).parent / "data/attack_history_case.json").read_text())


def setup(attack, selection=False):
    cfg = DuelConfig(**{**CASE["config"], "player": PlayerRules(**CASE["config"]["player"])})
    duel = Duel(
        CASE["seed"],
        None,
        Deck(**CASE["deck_a"]),
        Deck(**CASE["deck_b"]),
        first=CASE["first"],
        config=cfg,
        snapshots=True,
    )
    vocab = CardVocab.from_db(duel.cards)
    observer = PointObserver(duel.cards, vocab, 64, selection_history=selection, attack_history=attack)
    host = _core.HostDuel(
        duel.cards.to_core(),
        duel.scripts,
        list(vocab.passwords()),
        event_length=64,
        selection_history=selection,
        attack_history=attack,
    )
    p = cfg.player
    player = (p.starting_lp, p.starting_hand, p.draw_per_turn)
    args = (
        list(duel.core_seed),
        cfg.rule_flags,
        player,
        player,
        [(list(m), list(e)) for m, e in duel.loaded_decks()],
        cfg.max_turns,
        cfg.max_decisions,
    )
    host.start(*args)
    return DuelSession(duel), host, observer, args


def step(session, host, observer, index):
    point = session.point
    observer.observe(point)
    session.act(index)
    observer.on_decision(point, index)
    host.act(index)


def parity(session, host, observer):
    obs = observer.encode(session.point, session.core)
    native = host.observe()
    assert obs.keys() == native.keys()
    for key in obs:
        np.testing.assert_array_equal(obs[key], native[key], err_msg=key)
    return obs


@pytest.mark.parametrize("selection", [False, True])
@pytest.mark.parametrize("attack", [False, True])
def test_actual_attack_target_distinguishability_and_snapshot_reset(attack, selection):
    session, host, observer, args = setup(attack, selection)
    try:
        for index in CASE["actions"]:
            parity(session, host, observer)
            step(session, host, observer, index)
        point = session.point
        assert point.index == 51
        assert [a.kind for a in point.actions[:2]] == ["attack", "attack"]
        parity(session, host, observer)
        snap = session.snapshot()
        saved = copy.deepcopy(observer, {id(observer.cards): observer.cards, id(observer.vocab): observer.vocab})
        observations = []
        for chosen in (0, 1):
            if chosen:
                session.restore(snap)
                observer = copy.deepcopy(saved, {id(saved.cards): saved.cards, id(saved.vocab): saved.vocab})
                # Native host has no snapshot API; cold replay verifies the same branch.
                host.start(*args)
                for i in CASE["actions"]:
                    host.act(i)
            before = host.observe()
            with pytest.raises((IndexError, RuntimeError)):
                host.act(len(point.actions))
            for key in before:
                np.testing.assert_array_equal(before[key], host.observe()[key])
            step(session, host, observer, chosen)
            obs = parity(session, host, observer)
            observations.append(obs)
            assert type(session.point.decision).__name__ == "SelectCard"
            own = observer.history.encode(point.player)
            other = observer.history.encode(1 - point.player)
            tokens = own["events"][own["event_mask"] != 0]
            choices = tokens[tokens[:, 0] == ATTACK_CHOICE_EVENT]
            assert len(choices) == int(attack)
            assert not (other["events"][:, 0] == ATTACK_CHOICE_EVENT).any()
            if attack:
                assert choices[-1, 2] == observer.vocab.index(point.actions[chosen].card.code)
                assert choices[-1, 6] == point.actions[chosen].card.loc.sequence
            cancel = next(i for i, a in enumerate(session.point.actions) if a.kind == "cancel")
            step(session, host, observer, cancel)
            parity(session, host, observer)
        for key in observations[0]:
            assert np.array_equal(observations[0][key], observations[1][key]) == (not attack or key != "events")
        host.start(*args)
        fresh, other, fresh_observer, _ = setup(attack, selection)
        try:
            for key, value in host.observe().items():
                np.testing.assert_array_equal(value, other.observe()[key])
        finally:
            fresh.close()
    finally:
        session.close()


@pytest.mark.parametrize("selection", [False, True])
def test_attack_warm_start_preserves_old_logits_and_enforces_schema(tmp_path, selection):
    torch = pytest.importorskip("torch")
    from ygorl.nets import NetConfig, PolicyNet, collate
    from ygorl.train.checkpoint import Signature, warm_start

    torch.set_num_threads(1)
    session, host, observer, _ = setup(False, selection)
    try:
        vocab = observer.vocab
        cfg = NetConfig(
            len(vocab), d_model=16, n_heads=2, board_layers=1, history_layers=1, selection_history=selection
        )
        old = PolicyNet(cfg).eval()
        new = PolicyNet(replace(cfg, attack_history=True)).eval()
        assert warm_start(new, SimpleNamespace(net_config=cfg, net=old, path=tmp_path / "old.pt")) == ["attack_history"]
        obs = parity(session, host, observer)
        batch = collate([obs])
        with torch.no_grad():
            torch.testing.assert_close(
                old(batch).logits, new({**batch, "attack_history": torch.ones(1, 1)}).logits, atol=0, rtol=0
            )
        with pytest.raises(ValueError, match="attack-history"):
            new(batch)
        with pytest.raises(ValueError, match="attack-history"):
            collate([obs, {**obs, "attack_history": np.ones(1, dtype=np.int32)}])
        assert Signature.of(vocab, 64, selection).mismatches(Signature.of(vocab, 64, selection, True))
        with pytest.raises(ValueError):
            warm_start(old, SimpleNamespace(net_config=new.cfg, net=new, path=tmp_path / "new.pt"))
    finally:
        session.close()


@pytest.mark.parametrize("battle_context", [False, True])
def test_attack_training_checkpoint_and_legacy_resume(tmp_path, battle_context):
    pytest.importorskip("torch")
    from ygorl.train.trainer import TrainConfig, Trainer
    from ygorl.train.ppo import PPOConfig
    from ygorl.train.checkpoint import load_policy

    cfg = TrainConfig(
        decks=tuple(str(Path(__file__).parent / "decks" / f"{n}.ydk") for n in ("branded_despia", "snake_eye")),
        num_envs=2,
        env_threads=1,
        steps=128,
        event_length=64,
        net=dict(
            d_model=16, n_heads=2, board_layers=1, history_layers=1, attack_history=True, battle_context=battle_context
        ),
        privileged_dim=8,
        critic_hidden=16,
        ppo=PPOConfig(epochs=1, minibatch_size=32),
        checkpoint_every=0,
        eval_every=0,
        snapshot_every=1,
        torch_threads=1,
        collect_threads=1,
        seed=3,
    )
    trainer = Trainer(cfg, tmp_path, log=None)
    try:
        assert trainer.env.attack_history
        assert np.isfinite(trainer.step()["loss"])
        checkpoint = trainer.save()
    finally:
        del trainer
    policy = load_policy(checkpoint)
    assert policy.signature.attack_history and policy.net_config.attack_history
    assert policy.net_config.battle_context == battle_context
    resumed = Trainer.resume(checkpoint, log=None)
    try:
        assert resumed.net_config.attack_history and np.isfinite(resumed.step()["loss"])
        assert resumed.net_config.battle_context == battle_context
    finally:
        del resumed


def test_attack_bc_schema_roundtrip(tmp_path):
    pytest.importorskip("torch")
    from ygorl.solver import read_jsonl
    from ygorl.train.bc import build_dataset
    from ygorl.train.heuristic_demos import save_data, data_identity, load_compatible_data
    from ygorl.engine.duel import default_cards

    cards = default_cards()
    vocab = CardVocab.from_db(cards)
    demos = list(read_jsonl(Path(__file__).parent / "data/bc_demo.jsonl"))
    data = build_dataset(demos, vocab, cards=cards, event_length=32, attack_history=True)
    assert "attack_history" in data.obs and "selection_history" not in data.obs
    path = save_data(tmp_path / "samples.npz", data, identity=data_identity(vocab, 32))
    loaded, _ = load_compatible_data(path, vocab=vocab, event_length=32, attack_history=True)
    assert len(loaded) == len(data)
    with pytest.raises(ValueError, match="attack-history"):
        load_compatible_data(path, vocab=vocab, event_length=32)
