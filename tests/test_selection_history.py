"""Regression cases: actual fusion-target aliasing and policy cancellation loops."""

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
from ygorl.engine import messages as M
from ygorl.engine.actions import Action
from ygorl.engine.duel import Duel, DuelConfig, DuelSession, PlayerRules
from ygorl.env.events import EventHistory, SELECTION_EVENT
from ygorl.env.observer import PointObserver

CASES = json.loads((Path(__file__).parent / "data/selection_history_cases.json").read_text())


def setup(case, enabled=True):
    cfg = DuelConfig(**{**case["config"], "player": PlayerRules(**case["config"]["player"])})
    duel = Duel(
        case["seed"],
        None,
        Deck(**case["deck_a"]),
        Deck(**case["deck_b"]),
        first=case["first"],
        config=cfg,
        snapshots=True,
    )
    vocab = CardVocab.from_db(duel.cards)
    observer = PointObserver(duel.cards, vocab, 64, selection_history=enabled)
    host = _core.HostDuel(
        duel.cards.to_core(),
        duel.scripts,
        [vocab.password(i) for i in range(vocab.FIRST_INDEX, len(vocab))],
        event_length=64,
        selection_history=enabled,
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


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
@pytest.mark.parametrize("enabled", [False, True])
def test_real_selection_history_native_reference_parity(case, enabled):
    session, host, observer, args = setup(case, enabled)
    windows = []
    try:
        for index in case["actions"]:
            point = session.point
            py = observer.encode(point, session.core)
            native = host.observe()
            assert host.player() == point.player
            for key in py:
                np.testing.assert_array_equal(py[key], native[key], err_msg=f"{key} at {point.index}")
            tokens = py["events"][py["event_mask"] != 0]
            choices = tokens[tokens[:, 0] == SELECTION_EVENT]
            assert enabled or len(choices) == 0
            # Even when the other player just selected private cards, every
            # selection token received by the current viewer belongs to them.
            assert (choices[:, 1] == 1).all()
            if point.index in range(case["start"], case["end"] + 1, 2):
                windows.append(py)
            step(session, host, observer, index)
        assert host.result()["responses"] == session.tracker.result.responses
        if enabled:
            assert all(not np.array_equal(a["events"], b["events"]) for a, b in zip(windows, windows[1:]))
            if case["name"] == "initial-128x2-168":
                previous = [case["actions"][i - 1] for i in range(case["start"], case["end"] + 1, 2)]
                # Three distinct chosen fusion cards must survive into their
                # material windows, not merely a monotonically increasing clock.
                codes = []
                for obs in windows:
                    selections = obs["events"][(obs["event_mask"] != 0) & (obs["events"][:, 0] == SELECTION_EVENT)]
                    codes.append(int(selections[-1, 2]))
                assert len(set(codes)) == 3
                assert all((a == b) == (ca == cb) for a, ca in zip(previous, codes) for b, cb in zip(previous, codes))
        else:
            assert all(np.array_equal(windows[0]["events"], x["events"]) for x in windows)
        # Host reuse must reset private ordinals/history at the new duel.
        host.start(*args)
        fresh, other, fresh_observer, _ = setup(case, enabled)
        try:
            for key, value in host.observe().items():
                np.testing.assert_array_equal(value, other.observe()[key])
        finally:
            fresh.close()
    finally:
        session.close()


def test_observer_snapshot_invalid_action_and_idempotent_hook():
    case = CASES[2]
    session, host, observer, _ = setup(case)
    try:
        for index in case["actions"][:168]:
            step(session, host, observer, index)
        point = session.point
        observer.observe(point)
        snap = session.snapshot()
        saved = copy.deepcopy(observer, {id(observer.cards): observer.cards, id(observer.vocab): observer.vocab})
        before = host.observe()
        with pytest.raises((IndexError, RuntimeError)):
            host.act(len(point.actions))
        for key in before:
            np.testing.assert_array_equal(before[key], host.observe()[key])
        session.act(0)
        observer.on_decision(point, 0)
        first = observer.encode(session.point, session.core)
        observer.on_decision(point, 0)  # identical duplicate is harmless
        with pytest.raises(ValueError, match="conflicting"):
            observer.on_decision(point, 1)
        session.restore(snap)
        observer = copy.deepcopy(saved, {id(saved.cards): saved.cards, id(saved.vocab): saved.vocab})
        session.act(1)
        observer.on_decision(point, 1)
        second = observer.encode(session.point, session.core)
        assert not np.array_equal(first["events"], second["events"])
        for key in ("cards", "actions", "globals", "action_mask"):
            np.testing.assert_array_equal(first[key], second[key])
    finally:
        session.close()


def test_actor_private_choice_and_unknown_identity():
    vocab = CardVocab([11765832])
    history = EventHistory({}, vocab, 8, selection_history=True)
    before = history.encode(1)
    history.on_action(0, M.SelectCard.TYPE, Action("select", card=M.CardInfo(11765832, M.Location(0, 64, 1, 8))))
    history.on_action(0, M.SelectCard.TYPE, Action("select", card=M.CardInfo(0, M.Location(1, 2, 1, 10))))
    history.on_action(0, M.SelectCard.TYPE, Action("cancel"))
    for key in before:
        np.testing.assert_array_equal(before[key], history.encode(1)[key])
    rows = history.encode(0)["events"][:3]
    assert rows[:, 2].tolist() == [vocab.index(11765832), CardVocab.UNKNOWN, 0]
    assert rows[:, 14].tolist() == [1, 2, 3]
    with pytest.raises(ValueError, match="nonempty"):
        EventHistory({}, vocab, 0, selection_history=True)


def test_selection_signature_and_warm_start_preserve_legacy_logits(tmp_path):
    torch = pytest.importorskip("torch")
    from ygorl.nets import NetConfig, PolicyNet, collate
    from ygorl.train.checkpoint import Signature, warm_start

    torch.set_num_threads(1)
    session, host, observer, _ = setup(CASES[2], False)
    try:
        vocab = observer.vocab
        cfg = NetConfig(len(vocab), d_model=16, n_heads=2, board_layers=1, history_layers=1)
        old = PolicyNet(cfg).eval()
        new = PolicyNet(replace(cfg, selection_history=True)).eval()
        source = SimpleNamespace(net_config=cfg, net=old, path=tmp_path / "legacy.pt")
        assert warm_start(new, source) == ["selection_history"]
        for index in CASES[2]["actions"][:170]:
            point = session.point
            if point.index in (0, 168, 169):
                batch = collate([observer.encode(point, session.core)])
                with torch.no_grad():
                    torch.testing.assert_close(
                        old(batch).logits, new({**batch, "selection_history": torch.ones(1, 1)}).logits, atol=0, rtol=0
                    )
            step(session, host, observer, index)
        a, b = Signature.of(vocab, 64), Signature.of(vocab, 64, True)
        assert a != b and "selection-history encoding differs" in a.mismatches(b, event_length=False)
        with pytest.raises(ValueError):
            warm_start(old, SimpleNamespace(net_config=new.cfg, net=new, path=tmp_path / "new.pt"))
    finally:
        session.close()


def test_selection_training_checkpoint_resume_and_schema_rejection(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from ygorl.nets import collate
    from ygorl.train.trainer import TrainConfig, Trainer
    from ygorl.train.ppo import PPOConfig
    from ygorl.train.checkpoint import load_policy

    decks = tuple(str(Path(__file__).parent / "decks" / f"{name}.ydk") for name in ("branded_despia", "snake_eye"))
    cfg = TrainConfig(
        decks=decks,
        num_envs=2,
        env_threads=1,
        steps=64,
        event_length=32,
        net=dict(d_model=16, n_heads=2, board_layers=1, history_layers=1, selection_history=True),
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
    original = trainer.learner.update
    seen = []

    def update(ro, *args, **kwargs):
        assert "selection_history" in ro.obs
        seen.append(bool((ro.obs["events"][..., 0] == SELECTION_EVENT).any()))
        return original(ro, *args, **kwargs)

    monkeypatch.setattr(trainer.learner, "update", update)
    rec = trainer.step()
    assert seen == [True] and np.isfinite(rec["loss"])
    path = trainer.save()
    policy = load_policy(path)
    assert policy.net_config.selection_history and policy.signature.selection_history
    resumed = Trainer.resume(path, log=None)
    assert resumed.net_config.selection_history
    assert np.isfinite(resumed.step()["loss"])
    session, host, observer, _ = setup(CASES[2])
    try:
        obs = host.observe()
        policy.net(collate([obs]))
        legacy = {k: v for k, v in obs.items() if k != "selection_history"}
        with pytest.raises(ValueError, match="selection"):
            policy.net(collate([legacy]))
        with pytest.raises(ValueError, match="schemas"):
            collate([obs, legacy])
    finally:
        session.close()


def test_selection_bc_demonstrations_and_cached_schema(tmp_path):
    pytest.importorskip("torch")
    from ygorl.solver import read_jsonl
    from ygorl.train.bc import build_dataset
    from ygorl.train.heuristic_demos import save_data, data_identity, load_compatible_data
    from ygorl.engine.duel import default_cards

    cards = default_cards()
    vocab = CardVocab.from_db(cards)
    demos = list(read_jsonl(Path(__file__).parent / "data/bc_demo.jsonl"))
    data = build_dataset(demos, vocab, cards=cards, event_length=32, selection_history=True)
    assert "selection_history" in data.obs
    assert (data.obs["events"][..., 0] == SELECTION_EVENT).any()
    path = save_data(tmp_path / "samples.npz", data, identity=data_identity(vocab, 32))
    loaded, _ = load_compatible_data(path, vocab=vocab, event_length=32, selection_history=True)
    assert len(loaded) == len(data)
    with pytest.raises(ValueError, match="selection-history"):
        load_compatible_data(path, vocab=vocab, event_length=32)


def test_checkpoint_before_selection_flag_still_resumes(tmp_path):
    torch = pytest.importorskip("torch")
    from ygorl.train.trainer import TrainConfig, Trainer
    from ygorl.train.ppo import PPOConfig
    from ygorl.train.checkpoint import save_checkpoint

    decks = tuple(str(Path(__file__).parent / "decks" / f"{name}.ydk") for name in ("branded_despia", "snake_eye"))
    cfg = TrainConfig(
        decks=decks,
        num_envs=2,
        env_threads=1,
        steps=8,
        event_length=16,
        net=dict(d_model=16, n_heads=2, board_layers=1, history_layers=1),
        ppo=PPOConfig(epochs=1, minibatch_size=16),
        eval_every=0,
        torch_threads=1,
        collect_threads=1,
    )
    old = Trainer(cfg, tmp_path / "old", log=None)
    old.step()
    state = old.state_dict()
    state["net_config"].pop("selection_history")
    state["config"]["net"].pop("selection_history", None)
    path = save_checkpoint(state, tmp_path / "legacy.pt")
    resumed = Trainer.resume(path, tmp_path / "resumed", log=None)
    assert not resumed.net_config.selection_history
    for key, value in old.model.state_dict().items():
        torch.testing.assert_close(value, resumed.model.state_dict()[key], atol=0, rtol=0)
    assert np.isfinite(resumed.step()["loss"])
