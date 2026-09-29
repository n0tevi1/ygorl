"""Public evidence, meta prior and HDT-style filter of the belief heads (T4c.1, docs/belief-heads.md)."""

import json
from math import comb
from pathlib import Path

import numpy as np
import pytest

from ygorl.cards.cdb import CardDB, CardVocab
from ygorl.cards.ydk import load_ydk
from ygorl.engine.duel import DuelConfig
from ygorl.env import GameSpec
from ygorl.env import events as E
from ygorl.env.belief_prior import (
    EV_DECK,
    EV_EXTRA,
    EV_HAND,
    EV_REMOVED,
    EV_SET,
    N_SET_ZONES,
    Evidence,
    EvidenceTracker,
    MetaTable,
    default_roles,
    hdt_prior,
    hypergeometric,
    observe,
    password_bucket,
    responded_labels,
    role_targets,
)
from ygorl.env.encoded import EncodedVecEnv
from ygorl.env.privileged import (
    COUNT_DECK,
    COUNT_EXTRA,
    COUNT_HAND,
    COUNT_REMOVED,
    COUNT_SET,
    belief_targets,
    copy_counts,
)
from ygorl.eval.beliefs import BeliefBatch, Head, evaluate_beliefs

DATA = Path(__file__).parent / "data"
DECKS = {p.stem: load_ydk(p) for p in sorted((Path(__file__).parent / "decks").glob("*.ydk"))}
NAMES = sorted(DECKS)
META = NAMES[:8]  # the last two decks play the rogue ("other") role
GENERIC = json.loads((DATA / "generic_pool.json").read_text())["cards"]
ASH, MAXX_C = 14558127, 23434538


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


@pytest.fixture(scope="module")
def vocab(db):
    return CardVocab.from_db(db)


@pytest.fixture(scope="module")
def meta(db, vocab):
    return MetaTable(vocab, db, {n: DECKS[n] for n in META}, shares=np.arange(8, 0, -1), other_share=0.2,
                     generic=[c["password"] for c in GENERIC], roles=default_roles(GENERIC), n_hash=16)  # fmt: skip


def empty_evidence(meta, **kw):
    ev = Evidence(np.zeros(meta.n_cards, dtype=np.int64), np.zeros(meta.n_hash, dtype=np.int64),
                  np.zeros(meta.n_cards, dtype=np.int64), np.zeros(meta.n_cards, dtype=np.int64),
                  np.array([35, 5, 0, 0, 15]), np.zeros(N_SET_ZONES, dtype=bool))  # fmt: skip
    for k, v in kw.items():
        setattr(ev, k, v)
    return ev


def col(meta, password):
    return meta.candidates.passwords.index(password)


# ------------------------------------------------------------------ meta table


def test_meta_table_shapes_and_classes(meta):
    assert meta.names[-1] == "other" and meta.n_deck_types == 9
    assert meta.shares.sum() == pytest.approx(1) and meta.shares[-1] == pytest.approx(0.2)
    assert meta.shares[0] > meta.shares[7]
    union = {p for n in META for p in DECKS[n].counts()}
    assert set(meta.candidates.passwords) == union | {
        c["password"] for c in GENERIC
    }  # class count = meta union + generic
    for i, n in enumerate(META):
        assert {p: int(m) for p, m in zip(meta.candidates.passwords, meta.counts[i]) if m} == dict(DECKS[n].counts())
        assert meta.deck_type(DECKS[n]) == i
    assert meta.deck_type(DECKS[NAMES[9]]) == 8
    assert meta.role_names == ["hand_trap", "ash", "maxx_c", "nibiru", "extender"]
    assert meta.roles[1].sum() == 1 and meta.roles[1, col(meta, ASH)] and meta.roles[0, col(meta, ASH)]
    # non-candidate cards hash to buckets; candidates and tokens do not
    rogue_only = next(p for p in DECKS[NAMES[9]].main if p not in meta.candidates.passwords)
    assert meta.hash_bucket[meta.vocab.index(rogue_only)] == password_bucket(rogue_only, 16)
    assert meta.hash_bucket[meta.vocab.index(ASH)] == -1
    assert not (meta.hash_bucket[meta.is_token] >= 0).any() and meta.is_token.any()
    assert meta.zone_compat[:7].any() and meta.zone_compat[7:12].any()
    assert meta.feature_dim == 9 + 1 + 4 * meta.n_cards + 5 + 16 + 5


# ------------------------------------------------------------------ HDT filter


def test_hypergeometric_matches_combinatorics():
    for total, success, draws in [(40, 3, 5), (10, 0, 4), (6, 3, 6), (0, 0, 0), (35, 2, 30)]:
        p = hypergeometric(total, success, draws, np.arange(4))
        want = [comb(success, j) * comb(total - success, draws - j) / comb(total, draws) if j <= draws else 0
                for j in range(4)]  # fmt: skip
        np.testing.assert_allclose(p, want, atol=1e-12)
    assert hypergeometric(5, 7, 2, 0) == 0  # impossible configurations get probability 0


def test_no_evidence_gives_the_meta_prior(meta):
    p = hdt_prior(empty_evidence(meta), meta, eps=0.0)
    np.testing.assert_allclose(p.deck_type, meta.shares)
    assert p.n_consistent == 8
    np.testing.assert_allclose(p.remaining_copies.sum(-1), 1)
    np.testing.assert_allclose(p.set_cards.sum(-1), 1)
    assert p.features.shape == (meta.feature_dim,) and np.isfinite(p.features).all()
    # a main-deck card: mixture over the deck types of hypergeometric (pool 40, 35 left in the deck, 5 in hand)
    c = col(meta, ASH)
    comp = np.append(meta.counts[:, c], meta.other_counts[c])
    want = sum(w * hypergeometric(40, m, 35, np.arange(4)) for w, m in zip(meta.shares, comp))
    np.testing.assert_allclose(p.remaining_copies[c], want)
    assert p.hand[c] == pytest.approx(sum(w * (1 - hypergeometric(40, m, 5, 0)) for w, m in zip(meta.shares, comp)))
    # extra-deck cards never are in hand and stay in the extra deck
    e = int(np.nonzero(meta.is_extra & (meta.counts[0] > 0))[0][0])
    assert p.hand[e] == 0
    assert p.remaining_copies[e, meta.counts[0, e]] >= meta.shares[0]


def test_filter_and_constraints(meta):
    k = 3
    unique = int(np.nonzero((meta.counts[k] > 0) & ((meta.counts > 0).sum(0) == 1) & ~meta.is_extra)[0][0])
    seen = np.zeros(meta.n_cards, dtype=np.int64)
    seen[unique] = 1
    hand = seen.copy()
    p = hdt_prior(empty_evidence(meta, seen=seen, visible=seen, public_hand=hand), meta, eps=0.0)
    assert p.n_consistent == 1
    # only deck k and "other" survive, in proportion to their shares
    assert p.deck_type[k] == pytest.approx(meta.shares[k] / (meta.shares[k] + meta.shares[8]))
    assert p.deck_type[[k, 8]].sum() == pytest.approx(1)
    assert p.hand[unique] == 1 and p.public_hand[unique]  # public in hand -> 1 by construction
    assert p.max_copies[unique] == 2 and p.remaining_copies[unique, 3] == 0  # the copy seen is deducted
    # an off-list card rules out every meta deck
    other = np.zeros(meta.n_hash, dtype=np.int64)
    other[3] = 1
    p = hdt_prior(empty_evidence(meta, seen_other=other), meta)
    assert p.n_consistent == 0 and p.deck_type[8] > 0.99
    # public hand role -> role bit 1; three visible copies -> nothing can remain
    three = np.zeros(meta.n_cards, dtype=np.int64)
    three[col(meta, MAXX_C)] = 3
    p = hdt_prior(empty_evidence(meta, visible=three, public_hand=three), meta)
    assert p.hand_roles[meta.role_names.index("maxx_c")] == 1 and p.public_roles[meta.role_names.index("maxx_c")]
    assert p.max_copies[col(meta, MAXX_C)] == 0 and p.remaining_copies[col(meta, MAXX_C), 0] > 0.99


def test_set_zone_prior_respects_zone_kinds(meta):
    zones = np.zeros(N_SET_ZONES, dtype=bool)
    zones[[0, 8]] = True
    p = hdt_prior(empty_evidence(meta, set_zones=zones), meta, eps=0.0)
    assert p.set_cards[0][~meta.zone_compat[0]].sum() == 0  # a face-down monster is a main-deck monster
    assert p.set_cards[8][~meta.zone_compat[8]].sum() == 0  # a set spell / trap is a spell / trap
    assert p.set_cards[0].sum() == pytest.approx(1) and p.set_cards[8].sum() == pytest.approx(1)


def test_batched_matches_single(meta):
    rng = np.random.default_rng(0)
    evs = []
    for _ in range(4):
        seen = rng.integers(0, 2, meta.n_cards) * (rng.random(meta.n_cards) < 0.05)
        evs.append(empty_evidence(meta, seen=seen, visible=seen, set_zones=rng.random(N_SET_ZONES) < 0.3))
    batch = hdt_prior(Evidence.stack(evs), meta)
    for i, ev in enumerate(evs):
        single = hdt_prior(ev, meta)
        for key, value in single.arrays().items():
            np.testing.assert_allclose(getattr(batch, key)[i], value, err_msg=key)


# ------------------------------------------------------------------ real games: evidence vs ground truth


def test_evidence_agrees_with_privileged_ground_truth(db, vocab, meta):
    """Public counts equal the true hidden counts; the true meta deck is never ruled out; constraints hold."""
    env = EncodedVecEnv(2, 1, cards=db, vocab=vocab, privileged=True, event_length=0)
    games = [(META[0], META[5]), (META[2], NAMES[9])]  # the second game has a rogue opponent for seat 0
    trackers = {(e, p): EvidenceTracker(meta) for e in range(2) for p in (0, 1)}
    for e, (a, b) in enumerate(games):
        env.reset(e, GameSpec(seed=500 + e, deck_a=DECKS[a], deck_b=DECKS[b], config=DuelConfig(max_turns=10)))
    done, points, facedown, ruled_out_rogue = 0, 0, 0, 0
    rng = np.random.default_rng(1)
    while done < 2:
        for ev in env.recv(1):
            if ev.result is not None:
                done += 1
                continue
            priv, obs, p = ev.privileged, ev.obs, ev.player
            evidence = trackers[ev.env_id, p].update(obs["cards"], obs["globals"])
            hidden = lambda key, n: int((priv[key][:n][:, 1] == 0).sum())  # noqa: E731
            assert evidence.counts[EV_DECK] == priv["counts"][COUNT_DECK]
            assert evidence.counts[EV_HAND] == hidden("op_hand", priv["counts"][COUNT_HAND])
            assert evidence.counts[EV_SET] == priv["counts"][COUNT_SET]
            assert evidence.counts[EV_REMOVED] == priv["counts"][COUNT_REMOVED]
            assert evidence.counts[EV_EXTRA] == hidden("op_extra", priv["counts"][COUNT_EXTRA])
            np.testing.assert_array_equal(evidence.set_zones, priv["op_set"][:, 0] != 0)
            opponent = games[ev.env_id][1 - p]  # first=0: engine player p holds deck p
            prior = hdt_prior(evidence, meta)
            if opponent in META:
                assert prior.deck_type[META.index(opponent)] > 1e-3  # an unmodified meta deck is never filtered out
            elif prior.n_consistent == 0:
                ruled_out_rogue += 1
            targets = belief_targets(priv, meta.candidates)
            assert (targets["remaining_copies"].targets <= prior.max_copies).all()  # "3 - visible" is a valid bound
            public = copy_counts(priv["op_hand"], meta.candidates, where=priv["op_hand"][:, 1] == 1) > 0
            np.testing.assert_array_equal(prior.public_hand, public)
            roles = role_targets(priv, meta)
            assert (roles.targets[prior.public_roles] == 1).all()
            facedown += int(evidence.set_zones.any())
            points += 1
            legal = np.nonzero(obs["action_mask"])[0]
            env.step(ev.env_id, int(legal[rng.integers(len(legal))]))
    assert points > 200 and facedown > 10 and ruled_out_rogue > 0


def test_observe_is_the_tracker_first_step(db, vocab, meta):
    env = EncodedVecEnv(1, 1, cards=db, vocab=vocab, event_length=0)
    env.reset(0, GameSpec(seed=3, deck_a=DECKS[META[1]], deck_b=DECKS[META[4]]))
    (ev,) = env.recv(1)
    a = observe(ev.obs["cards"], ev.obs["globals"], meta)
    b = EvidenceTracker(meta).update(ev.obs["cards"], ev.obs["globals"])
    for key in ("seen", "visible", "counts", "set_zones"):
        np.testing.assert_array_equal(getattr(a, key), getattr(b, key))


def test_hdt_batch_evaluates(meta):
    """The HDT output has the shapes of ygorl.eval.beliefs and passes its validation."""
    ev = Evidence.stack([empty_evidence(meta) for _ in range(3)])
    p = hdt_prior(ev, meta)
    batch = BeliefBatch(
        deck_type=Head(p.deck_type, np.array([0, 1, 8])),
        remaining_copies=Head(p.remaining_copies, np.zeros((3, meta.n_cards), dtype=int)),
        hand=Head(p.hand, np.zeros((3, meta.n_cards), dtype=int)),
        hand_roles=Head(p.hand_roles, np.ones((3, meta.n_roles), dtype=int)),
        set_cards=Head(p.set_cards, np.zeros((3, N_SET_ZONES), dtype=int), np.zeros((3, N_SET_ZONES), dtype=bool)),
    )
    report = evaluate_beliefs(batch)
    assert report["deck_type/n"] == 3 and np.isfinite(report["hand/ece"])


# ------------------------------------------------------------------ extra targets


def test_role_targets(meta, vocab):
    hand = np.zeros((2, 32, 3), dtype=np.int32)
    hand[0, 0] = (vocab.index(ASH), 0, 0)  # hidden Ash
    hand[1, 0] = (vocab.index(MAXX_C), 1, 0)  # public Maxx "C"
    hand[1, 1] = (vocab.index(DECKS[META[0]].main[0]), 0, 1)
    t = role_targets({"op_hand": hand}, meta)
    r = {n: i for i, n in enumerate(meta.role_names)}
    assert t.targets.shape == (2, meta.n_roles)
    assert t.targets[0, r["ash"]] == 1 and t.targets[0, r["hand_trap"]] == 1 and t.targets[0, r["maxx_c"]] == 0
    assert t.mask[0].all()
    assert t.targets[1, r["maxx_c"]] == 1 and not t.mask[1, r["maxx_c"]] and not t.mask[1, r["hand_trap"]]
    assert t.mask[1, r["ash"]]


def token(kind, player=0, value1=0):
    row = np.zeros(E.E_EVENT, dtype=np.int32)
    row[E.TYPE], row[E.PLAYER], row[E.VALUE1] = E.EV[kind], player, value1
    return row


def test_responded_labels():
    stream = np.stack([
        token("draw", 1),                    # 0
        token("chaining", 1, 1),             # 1  my link 1
        token("chaining", 2, 2),             # 2  opponent responds
        token("chain_solving", 2, 2),        # 3
        token("chain_solving", 1, 1),        # 4
        token("chain_end"),                  # 5
        token("chaining", 1, 1),             # 6  my link 1
        token("abstain", 2, 32),             # 7
        token("chain_solving", 1, 1),        # 8  resolved without a response
        token("chain_end"),                  # 9
        token("chaining", 2, 1),             # 10 opponent link 1
        token("chaining", 1, 2),             # 11 my link 2 on top: responded when the opponent adds link 3
        token("chaining", 2, 3),             # 12
        token("chain_solving", 2, 3),        # 13
        token("chaining", 1, 1),             # 14 my activation, stream ends before it resolves
    ])  # fmt: skip
    got = responded_labels(stream, [0, 1, 2, 6, 7, 10, 12, 14, 15])
    assert got.tolist() == [1, 1, 0, 0, 1, 1, -1, -1, -1]
