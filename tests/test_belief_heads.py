"""Belief heads and their masked losses (T4c.1, docs/belief-heads.md)."""

from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ygorl.env.encoding import A_ACTION, F_CARD, G_GLOBAL, N_CARDS  # noqa: E402
from ygorl.eval.beliefs import evaluate_beliefs  # noqa: E402
from ygorl.nets import NetConfig, PolicyNet, collate  # noqa: E402
from ygorl.nets.belief import (  # noqa: E402
    HEADS,
    BeliefConfig,
    BeliefHeads,
    BeliefPolicy,
    belief_losses,
    evaluation_batch,
    loss_weights,
    prior_tensors,
)

B, K1, C, R, S, F_DIM, A = 6, 5, 12, 3, 15, 20, 7


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def dirichlet(rng, *shape):
    return rng.dirichlet(np.ones(shape[-1]), size=shape[:-1])


def random_prior(seed=0, b=B):
    rng = np.random.default_rng(seed)
    max_copies = rng.integers(0, 4, (b, C))
    copies = dirichlet(rng, b, C, 4) * (np.arange(4) <= max_copies[..., None])
    copies /= copies.sum(-1, keepdims=True)
    public_hand = rng.random((b, C)) < 0.2
    public_roles = rng.random((b, R)) < 0.2
    return {
        "deck_type": dirichlet(rng, b, K1),
        "remaining_copies": copies,
        "hand": np.where(public_hand, 1.0, rng.random((b, C))),
        "hand_roles": np.where(public_roles, 1.0, rng.random((b, R))),
        "set_cards": dirichlet(rng, b, S, C),
        "features": rng.standard_normal((b, F_DIM)).astype(np.float32),
        "public_hand": public_hand,
        "public_roles": public_roles,
        "max_copies": max_copies,
    }


def random_targets(prior, seed=1):
    rng = np.random.default_rng(seed)
    b = len(prior["deck_type"])
    copies = np.minimum(rng.integers(0, 4, (b, C)), prior["max_copies"])
    return {
        "deck_type": torch.as_tensor(rng.integers(0, K1, b)),
        "remaining_copies": torch.as_tensor(copies),
        "remaining_copies_mask": torch.ones(b, C, dtype=torch.bool),
        "hand": torch.as_tensor(np.where(prior["public_hand"], 1, rng.integers(0, 2, (b, C)))),
        "hand_mask": torch.as_tensor(~prior["public_hand"]),
        "hand_roles": torch.as_tensor(rng.integers(0, 2, (b, R))),
        "hand_roles_mask": torch.as_tensor(~prior["public_roles"]),
        "set_cards": torch.as_tensor(np.where(rng.random((b, S)) < 0.3, rng.integers(0, C, (b, S)), -1)),
        "set_cards_mask": torch.as_tensor(rng.random((b, S)) < 0.3),
        "responded": torch.as_tensor(rng.integers(-1, 2, b)),
        "responded_action": torch.as_tensor(rng.integers(0, A, b)),
    }


def config(**kw):
    return replace(BeliefConfig(K1, C, R, F_DIM, hidden=32, card_dim=8), **kw)


def perturb(heads, scale=0.3, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in heads.parameters():
            p.add_(torch.randn(p.shape, generator=g) * scale)


# ------------------------------------------------------------------ initialisation and constraints


def test_untrained_heads_output_the_prior():
    """Zero-initialised residuals: the heads start exactly at the meta prior / HDT posterior."""
    prior = random_prior()
    heads = BeliefHeads(config(prior_eps=1e-6))
    probs = heads(prior_tensors(prior)).probs()
    for name in ("deck_type", "remaining_copies", "hand", "hand_roles", "set_cards"):
        want = np.clip(prior[name], 1e-6, 1 - 1e-6) if name in ("hand", "hand_roles") else prior[name]
        np.testing.assert_allclose(probs[name].detach().numpy(), want, atol=1e-4, err_msg=name)
    assert probs["responded"].shape == (B,)


def test_constraints_hold_after_training_moves_the_heads():
    prior = random_prior()
    heads = BeliefHeads(config())
    perturb(heads, 1.0)
    probs = heads(prior_tensors(prior)).probs()
    assert (probs["hand"][torch.as_tensor(prior["public_hand"])] == 1).all()  # public in hand -> 1
    assert (probs["hand_roles"][torch.as_tensor(prior["public_roles"])] == 1).all()
    too_many = torch.arange(4) > torch.as_tensor(prior["max_copies"]).unsqueeze(-1)
    assert (probs["remaining_copies"][too_many] == 0).all()  # copies seen are deducted
    torch.testing.assert_close(probs["remaining_copies"].sum(-1), torch.ones(B, C))


def test_policy_features_width():
    cfg = config()
    out = BeliefHeads(cfg)(prior_tensors(random_prior()))
    f = out.policy_features()
    assert f.shape == (B, cfg.policy_dim) == (B, K1 + 2 * C + R)
    assert torch.isfinite(f).all() and (f >= 0).all() and (f <= 1).all()


# ------------------------------------------------------------------ losses and masks


def manual_losses(out, tg):
    logp = torch.log_softmax(out.deck_type, -1)
    deck = -logp[torch.arange(B), tg["deck_type"]].mean()
    lp = torch.log_softmax(out.remaining_copies, -1).gather(-1, tg["remaining_copies"].unsqueeze(-1)).squeeze(-1)
    copies = -lp.mean()

    def bce(logit, y, m):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y.float(), reduction="none")
        return loss[m].mean()

    hand = bce(out.hand, tg["hand"], tg["hand_mask"])
    roles = bce(out.hand_roles, tg["hand_roles"], tg["hand_roles_mask"])
    m = tg["set_cards_mask"] & (tg["set_cards"] >= 0)
    ls = torch.log_softmax(out.set_cards, -1)
    zb, zs = torch.nonzero(m, as_tuple=True)
    set_cards = -ls[zb, zs, tg["set_cards"][zb, zs]].mean()
    rm = tg["responded"] >= 0
    logit = out.responded.gather(1, tg["responded_action"].unsqueeze(1)).squeeze(1)
    resp = bce(logit, tg["responded"].clamp(min=0), rm)
    return {"deck_type": deck, "remaining_copies": copies, "hand": hand, "hand_roles": roles, "set_cards": set_cards,
            "responded": resp}  # fmt: skip


def test_losses_match_manual_cross_entropy():
    prior = random_prior()
    heads = BeliefHeads(config(action_dim=16))
    perturb(heads)
    actions = torch.randn(B, A, 16)
    out = heads(prior_tensors(prior), actions=actions, action_mask=torch.ones(B, A, dtype=torch.bool))
    assert out.responded.shape == (B, A)
    tg = random_targets(prior)
    got = belief_losses(out, tg, prior=prior_tensors(prior))
    for name, value in manual_losses(out, tg).items():
        torch.testing.assert_close(got[name], value, msg=name)
    torch.testing.assert_close(got["total"], sum(got[h] for h in HEADS))
    w = loss_weights(1.0, late=3.0)
    assert w["hand"] == w["set_cards"] == w["hand_roles"] == 3.0 and w["deck_type"] == 1.0
    weighted = belief_losses(out, tg, w, prior=prior_tensors(prior))
    torch.testing.assert_close(weighted["total"], sum(w[h] * got[h] for h in HEADS))


def test_masked_entries_do_not_matter():
    """Changing a masked target (public card, empty zone, unknown label) changes neither loss nor gradients."""
    prior = random_prior()
    pt = prior_tensors(prior)
    heads = BeliefHeads(config()).eval()  # no dropout: both passes must be deterministic
    perturb(heads)
    tg = random_targets(prior)
    tg["responded_action"] = None
    tg2 = dict(tg)
    tg2["hand"] = torch.where(tg["hand_mask"], tg["hand"], 1 - tg["hand"])
    tg2["hand_roles"] = torch.where(tg["hand_roles_mask"], tg["hand_roles"], 1 - tg["hand_roles"])
    tg2["set_cards"] = torch.where(tg["set_cards_mask"], tg["set_cards"], (tg["set_cards"] + 3) % C)
    tg2["responded"] = torch.where(tg["responded"] >= 0, tg["responded"], torch.full_like(tg["responded"], -1))
    grads = []
    for t in (tg, tg2):
        heads.zero_grad()
        loss = belief_losses(heads(pt), t, prior=pt)
        loss["total"].backward()
        grads.append([p.grad.clone() for p in heads.parameters() if p.grad is not None])
    for a, b in zip(*grads):
        torch.testing.assert_close(a, b)


def test_impossible_copy_targets_are_dropped():
    prior = random_prior()
    pt = prior_tensors(prior)
    heads = BeliefHeads(config())
    tg = random_targets(prior)
    bad = dict(tg)
    bad["remaining_copies"] = torch.full_like(tg["remaining_copies"], 3)  # above max_copies where it is < 3
    loss = belief_losses(heads(pt), bad, prior=pt)["remaining_copies"]
    assert torch.isfinite(loss) and loss < 20  # masked-out classes (-1e9 logits) never reach the loss


def test_all_masked_heads_give_zero_not_nan():
    prior = random_prior()
    pt = prior_tensors(prior)
    tg = {"hand": torch.zeros(B, C, dtype=torch.long), "hand_mask": torch.zeros(B, C, dtype=torch.bool),
          "responded": torch.full((B,), -1), "deck_type": torch.full((B,), -1)}  # fmt: skip
    losses = belief_losses(BeliefHeads(config())(pt), tg)
    assert set(losses) == {"hand", "responded", "deck_type", "total"}
    assert all(float(v.detach()) == 0 for v in losses.values())


def test_heads_learn_from_features():
    """A few steps on a toy task where the deck type is written in the features beat the (flat) prior."""
    rng = np.random.default_rng(0)
    prior = random_prior(b=64)
    prior["deck_type"] = np.full((64, K1), 1 / K1)
    label = rng.integers(0, K1, 64)
    prior["features"] = np.eye(F_DIM, dtype=np.float32)[label]
    pt = prior_tensors(prior)
    heads = BeliefHeads(config())
    opt = torch.optim.Adam(heads.parameters(), lr=1e-2)
    tg = {"deck_type": torch.as_tensor(label)}
    first = None
    for _ in range(60):
        loss = belief_losses(heads(pt), tg)["total"]
        first = float(loss.detach()) if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert first == pytest.approx(np.log(K1), rel=1e-3) and float(loss.detach()) < 0.2


# ------------------------------------------------------------------ evaluation and wiring


def test_evaluation_batch_feeds_the_t35_report():
    prior = random_prior()
    heads = BeliefHeads(config(action_dim=16))
    out = heads(prior_tensors(prior), actions=torch.randn(B, A, 16))
    tg = random_targets(prior)
    tg["remaining_copies_mask"] = tg["remaining_copies_mask"].numpy()
    batch = evaluation_batch(out.probs(), tg)
    assert set(batch.heads()) == set(HEADS)
    assert batch.responded.probs.shape == (B,)
    report = evaluate_beliefs(batch)
    assert report["responded/n"] == int((tg["responded"] >= 0).sum())
    assert report["set_cards/n"] == int((tg["set_cards_mask"] & (tg["set_cards"] >= 0)).sum())
    assert report["hand/n"] == int(tg["hand_mask"].sum())


def random_obs(rng, vocab=50, length=8):
    n = int(rng.integers(3, 20))
    cards = np.zeros((N_CARDS, F_CARD), dtype=np.int32)
    cards[:n, 0] = rng.integers(1, vocab, n)
    cards[:n, 1] = rng.integers(1, 9, n)
    globals_ = rng.integers(0, 5, G_GLOBAL).astype(np.int32)
    actions = np.zeros((128, A_ACTION), dtype=np.int32)
    actions[:4, 0] = rng.integers(1, 10, 4)
    mask = np.zeros(128, dtype=np.int32)
    mask[:4] = 1
    events = np.zeros((length, 20), dtype=np.int32)
    events[:3, 0] = rng.integers(1, 40, 3)
    return {"cards": cards, "globals": globals_, "actions": actions, "action_mask": mask, "events": events,
            "event_mask": (np.arange(length) < 3).astype(np.int32)}  # fmt: skip


def test_belief_policy_wiring():
    """Heads on the trunk: belief losses train the trunk, the actor gets the beliefs detached."""
    cfg = config(context_dim=32, action_dim=32)
    net = PolicyNet(NetConfig(vocab_size=50, d_model=32, n_heads=4, board_layers=1, history_layers=1,
                              belief_dim=cfg.policy_dim))  # fmt: skip
    model = BeliefPolicy(net, BeliefHeads(cfg))
    rng = np.random.default_rng(0)
    obs = collate([random_obs(rng) for _ in range(B)])
    prior = prior_tensors(random_prior())
    policy, beliefs = model(obs, prior)
    assert policy.logits.shape == (B, 128) and beliefs.responded.shape == (B, 128)
    # policy loss: no gradient reaches the belief heads (detached input)
    policy.log_probs()[:, 0].sum().backward()
    assert all(p.grad is None or not p.grad.any() for p in model.heads.parameters())
    assert net.head.belief.weight.grad is not None and net.head.belief.weight.grad.any()
    model.zero_grad()
    # belief loss: reaches the heads and the shared trunk (auxiliary-loss channel)
    tg = random_targets(random_prior())
    tg["responded_action"] = torch.zeros(B, dtype=torch.long)
    policy, beliefs = model(obs, prior)
    belief_losses(beliefs, tg, prior=prior)["total"].backward()
    assert net.context[0].weight.grad is not None and net.context[0].weight.grad.any()
    assert net.head.query[0].weight.grad is None or not net.head.query[0].weight.grad.any()
    # detach_context: the trunk is not trained by the belief losses
    detached = BeliefPolicy(net, BeliefHeads(replace(cfg, detach_context=True)))
    net.zero_grad()
    _, beliefs = detached(obs, prior)
    belief_losses(beliefs, tg, prior=prior)["total"].backward()
    assert net.context[0].weight.grad is None or not net.context[0].weight.grad.any()


def test_belief_policy_checks_widths():
    cfg = config(context_dim=32)
    with pytest.raises(ValueError, match="belief_dim"):
        BeliefPolicy(PolicyNet(NetConfig(vocab_size=50, d_model=32, n_heads=4)), BeliefHeads(cfg))
    with pytest.raises(ValueError, match="d_model"):
        BeliefPolicy(PolicyNet(NetConfig(vocab_size=50, d_model=16, n_heads=4)), BeliefHeads(cfg), feed_policy=False)
    # a policy without the belief input keeps working (existing callers)
    model = BeliefPolicy(PolicyNet(NetConfig(vocab_size=50, d_model=32, n_heads=4)), BeliefHeads(cfg), feed_policy=False)
    obs = collate([random_obs(np.random.default_rng(1)) for _ in range(2)])
    policy, _ = model(obs, prior_tensors(random_prior(b=2)))
    assert torch.isfinite(policy.log_probs()[:, :4]).all()
    with pytest.raises(ValueError, match="context"):
        BeliefHeads(cfg)(prior_tensors(random_prior()))


def test_config_roundtrip():
    cfg = config(context_dim=64)
    assert BeliefConfig.from_dict(cfg.to_dict()) == cfg
