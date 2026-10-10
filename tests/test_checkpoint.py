"""The checkpoint module (#122): every network rebuilt from PPO and policy (BC) checkpoints, the frozen-table check,
signatures (vocab, event length) and warm starts, on small hand-made checkpoints (no Trainer, no duels)."""

from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ygorl.cards.cdb import CardVocab  # noqa: E402
from ygorl.nets import NetConfig, PolicyNet, agent  # noqa: E402
from ygorl.nets.text import TextFeatures  # noqa: E402
from ygorl.train.checkpoint import (  # noqa: E402
    CriticConfig,
    Signature,
    build_actor_critic,
    load_actor,
    load_actor_critic,
    load_policy,
    save_checkpoint,
    vocab_to_text,
    warm_start,
)
from ygorl.train.trainer import TrainConfig  # noqa: E402

TINY = {"d_model": 16, "n_heads": 2, "board_layers": 1, "history_layers": 1}
VOCAB = CardVocab([10_000_000 + i for i in range(12)])


def _facts_dir(tmp_path, vocab=VOCAB):
    """A card-feature directory with only card facts (every card in archetype 0x1)."""
    feat = tmp_path / "features"
    feat.mkdir(exist_ok=True)
    pws = [vocab.password(i) for i in range(CardVocab.FIRST_INDEX, len(vocab))]
    n = len(pws)
    np.savez_compressed(feat / "card_facts.npz", passwords=np.asarray(pws, np.int64),
                        setcodes=np.tile([[0x1, 0, 0, 0]], (n, 1)), references=np.zeros((n, 2), np.int64),
                        categories=np.zeros(n, np.uint64), queries=np.zeros((n, 3), np.uint8),
                        query_names=np.array(["a", "b", "c"]))  # fmt: skip
    return feat


def _net_config(text=None, **kw):
    return NetConfig(vocab_size=len(VOCAB), **{**TINY, **kw}).with_text(text)


def _ppo(tmp_path, critic=CriticConfig(), *, net=None, text=None, text_dir=None, event_length=16, name="ppo.pt"):
    """A PPO checkpoint as the trainer writes it (the keys the loaders read), around a fresh actor-critic."""
    cfg = net or _net_config(text)
    torch.manual_seed(1)
    model = build_actor_critic(cfg, text, critic)
    train = TrainConfig(decks=("x.ydk",), event_length=event_length, text_dir=text_dir, privileged_critic=critic.privileged,
                        privileged_dim=critic.privileged_dim, critic_hidden=critic.hidden,
                        shared_backbone=critic.shared_backbone, critic_deck_order=critic.deck_order)  # fmt: skip
    state = {"config": train.to_dict(), "net_config": cfg.to_dict(), "vocab": vocab_to_text(VOCAB, tmp_path / "v.json"),
             "environment": None, "learner": {"model": model.state_dict(), "updates": 7}}  # fmt: skip
    return save_checkpoint(state, tmp_path / name), model


def _bc(tmp_path, *, net=None, text=None, event_length=16, vocab=VOCAB, name="bc.pt"):
    torch.manual_seed(2)
    policy = PolicyNet(net or _net_config(text), text)
    return agent.save_checkpoint(tmp_path / name, policy, vocab, event_length=event_length), policy


def _same(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


@pytest.mark.parametrize("critic", [CriticConfig(), CriticConfig(deck_order=True, privileged_dim=8, hidden=16),
                                    CriticConfig(shared_backbone=False), CriticConfig(privileged=False)])  # fmt: skip
def test_actor_critic_round_trips_with_its_critic_options(tmp_path, critic):
    path, model = _ppo(tmp_path, critic)
    ac = load_actor_critic(path)
    assert ac.critic == critic and ac.update == 7 and ac.event_length == 16 and not ac.model.training
    assert _same(ac.model.state_dict(), model.state_dict())
    assert not any(p.requires_grad for p in ac.model.parameters())
    assert (ac.model.privileged is not None) == critic.privileged
    assert critic.deck_order == (critic.privileged and ac.model.privileged.deck_order)
    actor = load_actor(path)  # the same actor alone
    assert _same(actor.net.state_dict(), model.actor.state_dict()) and actor.signature == ac.signature


def test_critic_config_reads_the_train_config_fields():
    cfg = TrainConfig(
        decks=("x.ydk",), privileged_dim=8, critic_hidden=16, shared_backbone=False, critic_deck_order=True
    )
    assert cfg.critic == CriticConfig(True, 8, 16, False, True)
    old = cfg.to_dict()
    del old["critic_deck_order"]  # checkpoints from before the deck-order critic
    assert not CriticConfig.from_train_config(old).deck_order


def test_a_policy_checkpoint_loads_as_an_actor_but_has_no_critic(tmp_path):
    path, policy = _bc(tmp_path, event_length=32)
    actor = load_actor(path)
    assert _same(actor.net.state_dict(), policy.state_dict()) and actor.event_length == 32 and actor.update == 0
    with pytest.raises(ValueError, match="no critic"):
        load_actor_critic(path)
    with pytest.raises(ValueError, match="not a ygorl PPO checkpoint"):
        load_policy(path)


def test_one_frozen_table_check_for_every_loader(tmp_path):
    """A network with card facts needs its feature directory: every loader says so the same way, and a PPO
    checkpoint finds it through its run's text_dir."""
    feat = _facts_dir(tmp_path)
    text = TextFeatures.load(feat, VOCAB)
    bc, _ = _bc(tmp_path, net=_net_config(text, card_facts=True), text=text)
    ppo, model = _ppo(tmp_path, net=_net_config(text, card_facts=True), text=text, text_dir=None)
    for load in (lambda: agent.load_checkpoint(bc), lambda: load_actor(bc), lambda: load_actor(ppo),
                 lambda: load_policy(ppo), lambda: load_actor_critic(ppo)):  # fmt: skip
        with pytest.raises(ValueError, match="frozen tables \\(card_facts\\)"):
            load()
    assert load_actor(bc, feat).net_config.n_archetypes == 1
    ac = load_actor_critic(ppo, feat)
    assert _same(ac.model.state_dict(), model.state_dict())
    from_run, _ = _ppo(tmp_path, net=_net_config(text, card_facts=True), text=text, text_dir=str(feat), name="r.pt")
    assert load_policy(from_run).net.identity.archetype is not None


def test_signatures_explain_why_two_checkpoints_do_not_fit(tmp_path):
    a = load_actor(_bc(tmp_path, name="a.pt")[0])
    assert a.signature.mismatches(load_actor(_bc(tmp_path, name="b.pt")[0]).signature) == []
    pws = [VOCAB.password(i) for i in range(CardVocab.FIRST_INDEX, len(VOCAB))]
    shuffled = CardVocab(pws[1:] + pws[:1])
    other = load_actor(_bc(tmp_path, vocab=shuffled, event_length=64, name="c.pt")[0]).signature
    why = a.signature.mismatches(other)
    assert len(why) == 2
    assert why[0] == f"the card vocab differs (index 2 is card {pws[0]} vs {pws[1]})"
    assert why[1] == "16 vs 64 event tokens per observation"
    assert a.signature.mismatches(other, event_length=False) == why[:1]
    shorter = Signature.of(CardVocab(pws[:-1]), 16)
    assert a.signature.mismatches(shorter) == [f"the card vocab differs ({len(pws)} vs {len(pws) - 1} cards)"]
    assert len({a.signature, Signature.of(VOCAB, 16), other}) == 2  # hashable: groups checkpoints


def test_warm_start_copies_ppo_and_bc_actors(tmp_path):
    for path in (_ppo(tmp_path)[0], _bc(tmp_path)[0]):
        source = load_actor(path)
        torch.manual_seed(5)
        actor = PolicyNet(_net_config())
        assert warm_start(actor, source) == []
        assert _same(actor.state_dict(), source.net.state_dict())
        assert all(p.requires_grad for p in actor.parameters())  # the copy trains


def test_warm_start_may_add_card_views_only(tmp_path):
    source = load_actor(_bc(tmp_path)[0])
    text = TextFeatures.load(_facts_dir(tmp_path), VOCAB)
    grown = PolicyNet(_net_config(text, card_facts=True, id_dropout=0.2), text)
    added = warm_start(grown, source)
    assert "n_archetypes" in added and "id_dropout" in added and "d_model" not in added
    old = source.net.state_dict()
    new = grown.state_dict()
    assert all(torch.equal(new[k], v) for k, v in old.items()) and set(new) > set(old)
    with pytest.raises(ValueError, match="network"):
        warm_start(PolicyNet(replace(_net_config(), d_model=32)), source)


@pytest.mark.parametrize("card_text,effect_text", [(True, False), (False, True), (True, True)])
def test_new_text_views_preserve_logits_and_can_learn(tmp_path, card_text, effect_text):
    from tests.test_nets import random_obs
    from ygorl.nets import collate

    source = load_actor(_bc(tmp_path)[0])
    text = TextFeatures.random(len(VOCAB), 8, 8, np.random.default_rng(7))
    cfg = _net_config(text, card_text=card_text, effect_text=effect_text)
    grown = PolicyNet(cfg, text).eval()
    warm_start(grown, source)
    obs = collate([random_obs(np.random.default_rng(12), vocab=len(VOCAB))])
    before = source.net(obs).logits.detach()
    after = grown(obs).logits
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    loss = -grown(obs).log_probs()[0, before.argmax(-1).item()]
    loss.backward()
    new = [(n, p) for n, p in grown.named_parameters() if n not in dict(source.net.named_parameters())]
    assert new and any(p.grad is not None and p.grad.abs().sum() > 0 for _, p in new)


def test_warm_start_keeps_learned_text_weights(tmp_path):
    text = TextFeatures.random(len(VOCAB), 8, 8, np.random.default_rng(7))
    cfg = _net_config(text)
    path, trained = _bc(tmp_path, net=cfg, text=text)
    from types import SimpleNamespace

    source = SimpleNamespace(net=trained, net_config=cfg, path=path)
    grown = PolicyNet(cfg, text)
    assert warm_start(grown, source) == []
    assert _same(grown.state_dict(), trained.state_dict())
