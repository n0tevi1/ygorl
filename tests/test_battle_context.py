"""Explicit attacker/target context: native identity, abstention and trainability."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ygorl.engine import constants as C
from ygorl.env.encoding import ACTION_KINDS
from ygorl.env.events import ATTACK_CHOICE_EVENT
from ygorl.nets import NetConfig, PolicyNet, collate
from ygorl.nets.battle_context import BattleContext, battle_context_rows
from ygorl.train.checkpoint import warm_start


def observation():
    cards = torch.zeros(1, 4, 23, dtype=torch.long)
    for i in range(4):
        cards[0, i, [0, 1, 2, 4, 6, 7]] = torch.tensor([7 if i < 2 else 8, 3, i % 2, i // 2, 1, 1])
        cards[0, i, 17] = 1000 * (i + 1)
    actions = torch.zeros(1, 4, 10, dtype=torch.long)
    for i in range(2):
        actions[0, i, :3] = torch.tensor([ACTION_KINDS.index("select") + 1, i + 3, 8])
    actions[0, 2, 0] = ACTION_KINDS.index("cancel") + 1
    event = torch.zeros(1, 2, 20, dtype=torch.long)
    event[0, 0, [0, 1, 2, 4, 5, 6, 12, 13, 15, 16, 17]] = torch.tensor(
        [
            ATTACK_CHOICE_EVENT,
            1,
            7,
            1,
            3,
            1,
            ACTION_KINDS.index("attack") + 1,
            C.MSG_SELECT_BATTLECMD,
            3,
            C.PHASE_BATTLE_STEP.bit_length(),
            1,
        ]
    )
    glob = torch.zeros(1, 22, dtype=torch.long)
    glob[0, [2, 3, 4, 18]] = torch.tensor([1, 3, C.PHASE_BATTLE_STEP.bit_length(), C.MSG_SELECT_CARD])
    return dict(
        cards=cards,
        actions=actions,
        globals=glob,
        events=event,
        event_mask=torch.tensor([[True, False]]),
        action_mask=torch.tensor([[True, True, True, False]]),
        attack_history=torch.ones(1, 1, dtype=torch.long),
    )


def test_instance_coordinates_and_current_stats():
    obs = observation()
    attacker, targets, valid = battle_context_rows(obs)
    assert attacker.tolist() == [1]  # same identity, distinct instances
    assert targets[0, :2].tolist() == [2, 3]
    assert valid.tolist() == [[True, True, False, False]]
    obs["events"][0, 0, 6] = 0
    assert battle_context_rows(obs)[0].tolist() == [0]
    layer = BattleContext(8)
    with torch.no_grad():
        layer.pair[-1].weight.fill_(0.1)
    features = torch.randn(1, 4, 8)
    original = layer(obs, features)
    features[:, 0] = torch.arange(8).float()
    changed = layer(obs, features)
    assert not torch.equal(original[:, :2], changed[:, :2])
    assert torch.equal(original[:, 2:], changed[:, 2:])


@pytest.mark.parametrize(
    "key,index,value",
    [
        ("globals", (0, 2), 0),
        ("globals", (0, 3), 4),
        ("globals", (0, 4), 1),
        ("globals", (0, 17), 1),
        ("globals", (0, 18), C.MSG_SELECT_BATTLECMD),
        ("events", (0, 0, 0), 1),
        ("events", (0, 0, 1), 2),
        ("events", (0, 0, 2), 1),
        ("events", (0, 0, 4), 2),
        ("events", (0, 0, 13), C.MSG_SELECT_IDLECMD),
        ("cards", (0, 1, 0), 9),
        ("cards", (0, 1, 7), 0),
    ],
)
def test_stale_unknown_wrong_owner_and_nonbattle_abstain(key, index, value):
    obs = observation()
    obs[key][index] = value
    assert not battle_context_rows(obs)[2].any()


def test_missing_truncated_intervening_and_ambiguous_context():
    for kind in ("missing", "empty", "intervening", "ambiguous"):
        obs = observation()
        if kind == "missing":
            del obs["events"]
        elif kind == "empty":
            obs["events"] = obs["events"][:, :0]
            obs["event_mask"] = obs["event_mask"][:, :0]
        elif kind == "intervening":
            obs["event_mask"][0, 1] = True
            obs["events"][0, 1, 0] = 1
        else:
            obs["cards"][0, 0] = obs["cards"][0, 1]
        assert not battle_context_rows(obs)[2].any()


def test_hidden_target_invalid_reference_and_mask_abstain():
    for key, index, value in [
        ("cards", (0, 2, 7), 0),
        ("cards", (0, 2, 6), 4),
        ("cards", (0, 2, 4), 0),
        ("actions", (0, 0, 1), 0),
        ("actions", (0, 0, 1), 99),
        ("actions", (0, 0, 2), 9),
        ("action_mask", (0, 0), False),
    ]:
        obs = observation()
        obs[key][index] = value
        assert battle_context_rows(obs)[2].tolist() == [[False, True, False, False]]


def test_zero_initialization_and_gradient_path():
    torch.manual_seed(41)
    obs = observation()
    layer = BattleContext(8)
    features = torch.randn(1, 4, 8)
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
    assert torch.count_nonzero(layer(obs, features)) == 0
    for _ in range(2):
        optimizer.zero_grad()
        loss = (layer(obs, features)[:, 0] - 1).square().mean()
        loss.backward()
        assert layer.pair[-1].weight.grad.abs().sum() > 0
        optimizer.step()
    assert layer.pair[1].weight.grad.abs().sum() > 0
    assert torch.count_nonzero(layer(obs, features)[:, 2:]) == 0


def test_warm_start_exact_logits_and_roundtrip(tmp_path):
    torch.set_num_threads(1)
    cfg = NetConfig(12, d_model=8, n_heads=2, board_layers=1, history_layers=1, attack_history=True)
    old, new = PolicyNet(cfg).eval(), PolicyNet(replace(cfg, battle_context=True)).eval()
    source = SimpleNamespace(net_config=cfg, net=old, path=tmp_path / "old.pt")
    assert warm_start(new, source) == ["battle_context"]
    obs = observation()
    with torch.no_grad():
        torch.testing.assert_close(old(obs).logits, new(obs).logits, atol=0, rtol=0)
    for name, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[name])
    assert new.battle_context.pair[1].weight.abs().sum() > 0
    clone = PolicyNet(NetConfig.from_dict(new.cfg.to_dict())).eval()
    clone.load_state_dict(new.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(new(obs).logits, clone(obs).logits, atol=0, rtol=0)
    with pytest.raises(ValueError):
        NetConfig(12, battle_context=True)
    with pytest.raises(ValueError):
        warm_start(old, SimpleNamespace(net_config=new.cfg, net=new, path=source.path))


@pytest.mark.parametrize("selection", [False, True])
def test_native_attacker_cancel_and_reattack(selection):
    from tests.test_attack_history import CASE, parity, setup, step

    session, host, observer, _ = setup(True, selection)
    try:
        for index in CASE["actions"]:
            parity(session, host, observer)
            step(session, host, observer, index)
        snapshots = []
        for chosen in (0, 1):
            attack = session.point.actions[chosen].card
            step(session, host, observer, chosen)
            obs = parity(session, host, observer)
            attacker, targets, valid = battle_context_rows(collate([obs]))
            assert valid.any()
            row = obs["cards"][int(attacker[0])]
            assert row[0] == observer.vocab.index(attack.code)
            assert row[2] == attack.loc.sequence
            snapshots.append(row.copy())
            cancel = next(i for i, a in enumerate(session.point.actions) if a.kind == "cancel")
            step(session, host, observer, cancel)
            assert not battle_context_rows(collate([parity(session, host, observer)]))[2].any()
        assert not np.array_equal(*snapshots)
    finally:
        session.close()
