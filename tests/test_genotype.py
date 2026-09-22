"""Tests for deck genotypes, hard constraints and variation operators (T5.5)."""

import json
from pathlib import Path

import numpy as np
import pytest

from ygorl.build.genotype import OPERATORS, Genotype, GenotypeSpace
from ygorl.cards.cdb import Card
from ygorl.cards.legality import DeckRules, validate_deck
from ygorl.cards.lflist import Banlist
from ygorl.cards.ydk import Deck
from ygorl.engine import constants as C

DATA = Path(__file__).parent / "data"
MONSTER = C.TYPE_MONSTER | C.TYPE_EFFECT
LINK = C.TYPE_MONSTER | C.TYPE_LINK | C.TYPE_EFFECT


def mk(pw, name=None, type_=MONSTER, alias=0):
    return Card(pw, name or f"Card {pw}", "", ("",) * 16, alias, 3, (), type_, 0, 0, 4, 0, 0, C.RACE_DRAGON, C.ATTRIBUTE_DARK, 0, 0)


# main-deck cards 100..199, extra-deck monsters 300..339
CARDS = {c.password: c for c in [mk(i) for i in range(100, 200)] + [mk(i, type_=LINK) for i in range(300, 340)]}
CARDS.update(
    {
        1: mk(1, "Ash Blossom"),
        4: mk(4, "Ash Blossom", alias=1),  # alternate artwork of 1
        2: mk(2, "Forbidden One"),
        3: mk(3, "Semi Card"),
        5: mk(5, "Spell", C.TYPE_SPELL),
        6: mk(6, "Alt Pool Card"),
        7: mk(7, "Alt Pool Card", alias=6),  # only the alternate artwork is in the pool
        12: mk(12, "Token", C.TYPE_MONSTER | C.TYPE_TOKEN),
    }
)
BANLIST = Banlist("test", {1: 1, 2: 0, 3: 2, 105: 1, 301: 0})
POOL = frozenset(CARDS) - {6, 199, 198, 197, 196, 195}
PACKAGES = [
    [100, 101, 102, 103, 104, 105, 106, 107, 300, 301, 302],  # 105 limited, 301 forbidden
    [110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 303, 304, 305, 306],
    [120, 121, 122, 123, 3, 2],  # 3 semi-limited, 2 forbidden
    [130, 131, 132, 133, 134, *range(310, 330)],  # 20 extra-deck members: must be trimmed to <= 15
    [195, 196, 197, 198, 199],  # nothing in the pool: dropped
    [140, 141, 142, 143, 144, 145, 146],
    [300, 302],  # no Main Deck member: dropped
]
GENERIC = [150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 4, 7, 5, 12, 330, 331, 332, 333, 334]


def space(**kw):
    kw.setdefault("banlist", BANLIST)
    kw.setdefault("pool", POOL)
    return GenotypeSpace(CARDS, PACKAGES, GENERIC, **kw)


def legal(sp, g, cards=CARDS, banlist=BANLIST, pool=POOL, rules=DeckRules()):
    deck = sp.decode(g)
    return validate_deck(deck, cards=cards, banlist=banlist, pool=pool, rules=rules)


# ---------------------------------------------------------------- the space


def test_space_index_filters_and_caps():
    sp = space()
    idx = {pw: i for i, pw in enumerate(sp.passwords)}
    assert 2 not in idx and 301 not in idx  # forbidden
    assert 12 not in idx  # token
    assert not {195, 196, 197, 198, 199} & set(idx)  # not in the pool
    assert 4 not in idx and 1 in idx  # alternate artwork folds into the original ...
    assert 7 in idx and 6 not in idx  # ... unless only the alternate artwork is in the pool
    assert sp.cap[idx[1]] == 1 and sp.cap[idx[3]] == 2 and sp.cap[idx[105]] == 1 and sp.cap[idx[100]] == 3
    # Main Deck cards first, then Extra Deck monsters, each block sorted by password
    n_main = sp.n_main
    assert all(not CARDS[p].type & C.TYPE_LINK for p in sp.passwords[:n_main])
    assert all(CARDS[p].type & C.TYPE_LINK for p in sp.passwords[n_main:])
    assert list(sp.passwords[:n_main]) == sorted(sp.passwords[:n_main])
    assert list(sp.passwords[n_main:]) == sorted(sp.passwords[n_main:])
    assert sp.is_extra.tolist() == [False] * n_main + [True] * (len(sp) - n_main)
    # packages restricted to the space; empty / extra-only packages dropped
    assert len(sp.packages) == 5
    assert sp.packages[0] == (100, 101, 102, 103, 104, 105, 106, 107, 300, 302)
    assert sp.packages[2] == (3, 120, 121, 122, 123)
    assert sp.package_source == (0, 1, 2, 3, 5)
    assert sp.index_of(4) == idx[1] and sp.index_of(1) == idx[1]
    with pytest.raises(KeyError):
        sp.index_of(199)


def test_space_roles():
    sp = GenotypeSpace(CARDS, PACKAGES, {150: "hand_trap", 151: "hand_trap", 152: "staple", 330: "extra"},
                       banlist=BANLIST, pool=POOL)  # fmt: skip
    g = sp.sample(0)
    roles = sp.role_counts(g)
    assert set(roles) <= {"hand_trap", "staple", "extra"}
    hand_traps = sum(g.counts[sp.index_of(p)] for p in (150, 151))
    assert roles.get("hand_trap", 0) == hand_traps


def test_infeasible_space_raises():
    with pytest.raises(ValueError, match="Main Deck"):
        GenotypeSpace(CARDS, [[100, 101, 102]], [150, 151], banlist=BANLIST, pool=POOL)
    with pytest.raises(ValueError, match="range"):
        space(main_range=(35, 45))
    with pytest.raises(ValueError, match="package"):
        GenotypeSpace(CARDS, [[195, 196]], GENERIC, banlist=BANLIST, pool=POOL)


# ---------------------------------------------------------------- sampling and decoding


def test_sample_is_legal_and_deterministic():
    sp = space()
    a, b = sp.sample(7), sp.sample(np.random.default_rng(7))
    assert a == b and hash(a) == hash(b)
    assert sp.sample(8) != a
    rng = np.random.default_rng(0)
    for g in sp.sample_many(2000, rng):
        assert sp.check(g) == []
        assert legal(sp, g) == []
        main = int(g.counts[: sp.n_main].sum())
        assert sp.main_range[0] <= main <= sp.main_range[1]
        assert int(g.counts[sp.n_main :].sum()) <= 15
        assert sp.min_packages <= len(g.packages)
    assert [g.key() for g in sp.sample_many(5, 3)] == [g.key() for g in sp.sample_many(5, 3)]


def test_decode_places_cards_by_type():
    sp = space()
    g = sp.sample(1)
    deck = sp.decode(g, name="x")
    assert isinstance(deck, Deck) and deck.name == "x" and deck.side == ()
    assert all(not CARDS[p].type & C.TYPE_LINK for p in deck.main)
    assert all(CARDS[p].type & C.TYPE_LINK for p in deck.extra)
    assert list(deck.main) == sorted(deck.main) and len(deck.main) == int(g.counts[: sp.n_main].sum())
    assert 4 not in deck.main  # alternate artworks are emitted as the pool's password


def test_full_main_range():
    sp = space(main_range=(40, 60), max_packages=5)
    sizes = []
    for g in sp.sample_many(500, 11):
        assert legal(sp, g) == []
        sizes.append(int(g.counts[: sp.n_main].sum()))
    assert min(sizes) == 40 and max(sizes) > 55


def test_custom_rules():
    rules = DeckRules(main_min=20, main_max=30, extra_max=5, max_copies=2)
    sp = space(rules=rules)
    assert sp.main_range == (20, 25) and sp.extra_size == 5 and int(sp.cap.max()) == 2
    for g in sp.sample_many(300, 2):
        assert legal(sp, g, rules=rules) == []


# ---------------------------------------------------------------- operators


@pytest.mark.parametrize("op", OPERATORS)
def test_each_mutation_operator_keeps_legality(op):
    sp = space(max_packages=4)
    rng = np.random.default_rng(5)
    g = sp.sample(rng)
    changed = 0
    for _ in range(300):
        h = sp.mutate(g, rng, ops=(op,))
        assert sp.check(h) == [] and legal(sp, h) == []
        changed += h != g
        g = h
    assert changed > 150


def test_operator_semantics():
    sp = space(max_packages=4)
    rng = np.random.default_rng(3)
    g = sp.sample(rng)
    while len(g.packages) >= 4:
        g = sp.mutate(g, rng, ops=("drop_package",))
    h = sp.mutate(g, rng, ops=("add_package",))
    assert len(h.packages) == len(g.packages) + 1 and set(g.packages) < set(h.packages)
    added = (set(h.packages) - set(g.packages)).pop()
    assert any(h.counts[i] > 0 for i in sp.package_indices(added))
    while len(h.packages) < 2:
        h = sp.mutate(h, rng, ops=("add_package",))
    d = sp.mutate(h, rng, ops=("drop_package",))
    assert len(d.packages) == len(h.packages) - 1 and set(d.packages) < set(h.packages)
    # swap_extra keeps the Extra Deck size; swap_generic touches only generic cards
    e = sp.mutate(h, rng, ops=("swap_extra",))
    assert e.counts[sp.n_main :].sum() == h.counts[sp.n_main :].sum()
    s = sp.mutate(h, rng, ops=("swap_generic",))
    generic_main = sp.is_generic & ~sp.is_extra
    assert np.any(generic_main & (h.counts > 0) & (s.counts == 0))
    assert np.any(generic_main & (h.counts == 0) & (s.counts > 0))


def test_mutate_is_deterministic():
    sp = space()
    g = sp.sample(0)
    assert sp.mutate(g, 42, n_ops=3) == sp.mutate(g, 42, n_ops=3)


def test_crossover_is_package_level_and_legal():
    sp = space(max_packages=4)
    rng = np.random.default_rng(9)
    pop = sp.sample_many(60, rng)
    for _ in range(600):
        i, j = rng.integers(len(pop), size=2)
        a, b = pop[i], pop[j]
        c = sp.crossover(a, b, rng)
        assert sp.check(c) == [] and legal(sp, c) == []
        assert set(c.packages) <= set(a.packages) | set(b.packages)
        assert sp.min_packages <= len(c.packages) <= sp.max_packages
        pop[rng.integers(len(pop))] = sp.mutate(c, rng)
    a, b = pop[0], pop[1]
    assert sp.crossover(a, b, 1) == sp.crossover(a, b, 1)


def test_crossover_of_identical_parents_is_identity():
    sp = space()
    g = sp.sample(4)
    assert sp.crossover(g, g, 0) == g


# ---------------------------------------------------------------- encodings


def test_vector_roundtrip_and_repair_from_junk():
    sp = space()
    rng = np.random.default_rng(1)
    for g in sp.sample_many(100, rng):
        x = sp.vector(g)
        assert x.dtype == np.int8 and x.shape == (len(sp),)
        assert sp.decode(sp.from_vector(x, rng)) == sp.decode(g)
    for _ in range(300):
        junk = rng.normal(0.5, 2.0, size=len(sp))
        h = sp.from_vector(junk, rng)
        assert sp.check(h) == [] and legal(sp, h) == []


def test_from_deck():
    sp = space()
    deck = Deck(main=(4, 100, 100, 110, 199, 5) + (150,) * 3, extra=(330, 331))
    g = sp.from_deck(deck, 0)
    assert legal(sp, g) == []
    assert g.counts[sp.index_of(1)] == 1 and g.counts[sp.index_of(100)] >= 2 and g.counts[sp.index_of(330)] >= 1
    assert 0 in g.packages and 1 in g.packages  # inferred from the engine cards present


def test_genotype_json_roundtrip():
    sp = space()
    g = sp.sample(12)
    data = json.loads(json.dumps(sp.genotype_to_json(g)))
    assert data["space"] == sp.fingerprint
    assert sp.genotype_from_json(data) == g
    other = space(max_packages=5)
    assert other.fingerprint != sp.fingerprint
    with pytest.raises(ValueError, match="space"):
        other.genotype_from_json(data)


def test_genotype_is_immutable():
    g = space().sample(0)
    assert isinstance(g, Genotype)
    with pytest.raises(ValueError):
        g.counts[0] = 3


# ---------------------------------------------------------------- real packages (acceptance)


@pytest.fixture(scope="module")
def real_space(real_graph):
    from ygorl.build.packages import enumerate_packages, setcodes_from_db
    from ygorl.cards.cdb import CardDB
    from ygorl.data.environment import Environment

    db = CardDB.load()
    pkgs = enumerate_packages(real_graph, setcodes=setcodes_from_db(db))
    generic = {c["password"]: c["role"] for c in json.loads((DATA / "generic_pool.json").read_text())["cards"]}
    generic[14558128] = "hand_trap"  # alternate artwork of Ash Blossom: must fold into 14558127
    # a 60-package format; hard constraints come from a banlist that hits package and generic cards
    chosen = pkgs[:60]
    pool = frozenset(m for p in chosen for m in p.members) | frozenset(generic) - {14558128}
    members = sorted({m for p in chosen for m in p.members})
    limits = {pw: i % 3 for i, pw in enumerate(members[::7])}
    limits.update({14558127: 1, 23434538: 0, 54693926: 1, 29301450: 0})
    env = Environment(
        version="test-genotype", format="md", card_pool=pool, banlist=Banlist("test", limits), rule_flags=0,
        meta_decks=(),
    )  # fmt: skip
    sp = GenotypeSpace.from_environment(env, db, pkgs, generic)
    return sp, env, db


def test_real_space_shape(real_space):
    sp, env, db = real_space
    assert 50 <= len(sp.packages) <= len(env.card_pool)
    assert sp.stamp == env.stamp()
    assert db.canonical(14558128) == 14558127 and sp.index_of(14558128) == sp.index_of(14558127)
    assert 23434538 not in sp.passwords and 29301450 not in sp.passwords  # forbidden
    assert all(pw in env.card_pool for pw in sp.passwords)
    assert {1, 2, 3} <= set(sp.cap.tolist())
    # negative control: the validator catches a genotype that bypasses repair
    g = sp.sample(0)
    counts = g.counts.copy()
    limited = int(np.flatnonzero((sp.cap == 1) & ~sp.is_extra)[0])
    counts[limited] = 3
    raw = Genotype(g.packages, counts)
    assert sp.check(raw) and "over_limit" in {v.code for v in env.validate_deck(sp.decode(raw), db)}


def test_real_10k_random_genotypes_are_legal(real_space):
    """Acceptance (T5.5): 10k randomly sampled genotypes are all legal."""
    sp, env, db = real_space
    rng = np.random.default_rng(2026)
    bad = []
    for n, g in enumerate(sp.sample_many(10_000, rng)):
        vs = env.validate_deck(sp.decode(g), db)
        if vs:
            bad.append((n, vs))
    assert bad == []


def test_real_mutation_crossover_chains_are_legal(real_space):
    sp, env, db = real_space
    rng = np.random.default_rng(7)
    pop = sp.sample_many(100, rng)
    for _ in range(2000):
        i, j = rng.integers(len(pop), size=2)
        child = sp.mutate(sp.crossover(pop[i], pop[j], rng), rng, n_ops=int(rng.integers(1, 4)))
        assert env.validate_deck(sp.decode(child), db) == []
        pop[int(rng.integers(len(pop)))] = child
