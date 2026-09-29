"""Batched policy-vs-policy games (ygorl.eval.batched)."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from ygorl.cards.cdb import CardVocab  # noqa: E402
from ygorl.cards.ydk import load_ydk  # noqa: E402
from ygorl.engine.duel import DuelConfig, default_cards  # noqa: E402
from ygorl.env.encoded import EncodedVecEnv  # noqa: E402
from ygorl.eval.arena import summarize  # noqa: E402
from ygorl.eval.batched import paired_specs, play_policies  # noqa: E402
from ygorl.nets import NetConfig, PolicyNet  # noqa: E402

DECKS = Path(__file__).parent / "decks"


@pytest.fixture(scope="module")
def setup():
    torch.set_num_threads(1)
    cards = default_cards()
    vocab = CardVocab.from_db(cards)
    torch.manual_seed(0)
    net = PolicyNet(NetConfig(vocab_size=len(vocab), d_model=16, n_heads=2, board_layers=1, history_layers=1))
    specs = paired_specs(load_ydk(DECKS / "snake_eye.ydk"), load_ydk(DECKS / "kashtira.ydk"), 3, 5,
                         DuelConfig(max_decisions=300))  # fmt: skip
    return cards, vocab, net, specs


def _env(n, cards, vocab):
    return EncodedVecEnv(n, 1, cards=cards, vocab=vocab, event_length=16, skip_forced=True)


def test_results_do_not_depend_on_batching(setup):
    cards, vocab, net, specs = setup
    one, st1 = play_policies(_env(1, cards, vocab), specs, net)
    many, st4 = play_policies(_env(4, cards, vocab), specs, net, min_batch=2)
    assert one == many and len(one) == len(specs) == 6
    assert st4["forwards"] < st1["forwards"] and st1["decisions"] == st4["decisions"] > 0
    # arena pairing: two games per seed, deck a first then deck b first; records from deck a's side
    assert [r.first for r in one] == [0, 1] * 3 and [r.pair for r in one] == [0, 0, 1, 1, 2, 2]
    assert all(r.winner in (0, 1, None) and r.reason for r in one)
    rep = summarize(one, agent_a="a", agent_b="b", deck_a="snake_eye", deck_b="kashtira", seed=5)
    assert rep.games == 6 and rep.wins + rep.losses + rep.draws == 6


def test_greedy_play_is_reproducible_and_seats_follow_the_decks(setup):
    cards, vocab, net, specs = setup
    g1, _ = play_policies(_env(2, cards, vocab), specs, net, greedy=True)
    g2, _ = play_policies(_env(3, cards, vocab), specs, net, net, greedy=True)  # same module on both decks
    assert g1 == g2
    # a different module on deck b changes deck b's decisions, not the record layout
    torch.manual_seed(1)
    other = PolicyNet(net.cfg)
    g3, _ = play_policies(_env(2, cards, vocab), specs, net, other, greedy=True)
    assert [r.first for r in g3] == [r.first for r in g1] and g3 != g1


def test_a_deck_that_cannot_start_is_recorded_and_the_rest_play(setup):
    cards, vocab, net, specs = setup
    from dataclasses import replace

    # EncodedVecEnv.reset refuses curriculum modes: a game that cannot start
    bad = [replace(specs[0], config=replace(specs[0].config, curriculum="solo")), *specs[1:3]]
    recs, _ = play_policies(_env(1, cards, vocab), bad, net)
    assert recs[0].reason == "exception" and recs[0].winner is None and recs[0].error
    assert all(r.reason != "exception" for r in recs[1:])
    rep = summarize(recs, agent_a="a", agent_b="b", deck_a="x", deck_b="y", seed=5)
    assert rep.errors == 1


def test_same_seeds_give_the_same_records_with_per_side_sampling(setup):
    cards, vocab, net, specs = setup
    torch.manual_seed(1)
    other = PolicyNet(net.cfg)
    seeds = [(11 + i, 99 - i) for i in range(len(specs))]
    kw = dict(sampling=((True, 1.0), (False, 0.5)), sample_seeds=seeds)
    one, st1 = play_policies(_env(1, cards, vocab), specs, net, other, **kw)
    three, st3 = play_policies(_env(3, cards, vocab), specs, net, other, min_batch=2, **kw)
    again, _ = play_policies(_env(3, cards, vocab), specs, net, other, min_batch=3, **kw)
    assert one == three == again and st1["decisions"] == st3["decisions"]
    # the sample seeds are the only randomness: other seeds, other games
    moved, _ = play_policies(_env(3, cards, vocab), specs, net, other, sampling=kw["sampling"],
                             sample_seeds=[(a + 1, b + 1) for a, b in seeds])  # fmt: skip
    assert [r.first for r in moved] == [r.first for r in one] and moved != one
