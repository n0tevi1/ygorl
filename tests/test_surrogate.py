"""Tests for the deck surrogate model (T5.7): features, ridge ensemble, online updates, acquisition."""

import numpy as np
import pytest

from ygorl.build.genotype import GenotypeSpace
from ygorl.build.surrogate import (
    DEFAULT_TARGETS,
    FeatureMap,
    RidgeEnsemble,
    Surrogate,
    TextEmbeddings,
    acquire,
    card_feature_matrix,
)
from ygorl.cards.cdb import Card, CardVocab
from ygorl.cards.ydk import Deck
from ygorl.engine import constants as C

MONSTER = C.TYPE_MONSTER | C.TYPE_EFFECT
TUNER = MONSTER | C.TYPE_TUNER
XYZ = C.TYPE_MONSTER | C.TYPE_XYZ | C.TYPE_EFFECT
LINK = C.TYPE_MONSTER | C.TYPE_LINK | C.TYPE_EFFECT
SPELL, QUICK = C.TYPE_SPELL, C.TYPE_SPELL | C.TYPE_QUICKPLAY
TRAP = C.TYPE_TRAP


def mk(pw, type_=MONSTER, level=4, attribute=C.ATTRIBUTE_DARK, race=C.RACE_DRAGON, attack=1000):
    kind = type_ & C.TYPE_MONSTER
    return Card(
        pw, f"Card {pw}", "", ("",) * 16, 0, 3, (), type_, attack if kind else 0, 0, level if kind else 0, 0, 0,
        race if kind else 0, attribute if kind else 0, 0, 0,
    )  # fmt: skip


def _cards():
    cards = []
    for i in range(100, 160):  # main-deck monsters with varied stats
        cards.append(mk(i, TUNER if i % 5 == 0 else MONSTER, level=1 + i % 8,
                        attribute=(C.ATTRIBUTE_LIGHT, C.ATTRIBUTE_DARK, C.ATTRIBUTE_FIRE)[i % 3],
                        race=(C.RACE_DRAGON, C.RACE_FIEND, C.RACE_WARRIOR)[i % 3], attack=100 * (i % 30)))  # fmt: skip
    cards += [mk(i, SPELL if i % 2 else QUICK) for i in range(160, 180)]
    cards += [mk(i, TRAP) for i in range(180, 190)]
    cards += [mk(i, LINK if i % 2 else XYZ, level=2) for i in range(300, 330)]
    cards += [mk(i, MONSTER) for i in range(1, 6)]  # hand traps
    cards += [mk(i, SPELL) for i in range(6, 10)]  # board breakers
    return {c.password: c for c in cards}


CARDS = _cards()
PACKAGES = [
    [*range(100, 112), 160, 161, 180, 300, 301, 302],
    [*range(112, 124), 162, 163, 181, 303, 304, 305],
    [*range(124, 136), 164, 165, 182, 306, 307],
    [*range(136, 148), 166, 167, 183, 308, 309, 310],
    [*range(148, 160), *range(168, 180), *range(184, 190), *range(311, 318)],
]
GENERIC = {1: "hand_trap", 2: "hand_trap", 3: "hand_trap", 4: "hand_trap", 5: "hand_trap",
           6: "board_breaker", 7: "board_breaker", 8: "board_breaker", 9: "board_breaker",
           320: "extra", 321: "extra", 322: "extra", 323: "extra"}  # fmt: skip


@pytest.fixture(scope="module")
def space():
    return GenotypeSpace(CARDS, PACKAGES, GENERIC, max_packages=2)


@pytest.fixture(scope="module")
def genotypes(space):
    return space.sample_many(400, np.random.default_rng(0))


# ---------------------------------------------------------------- features


def test_card_feature_matrix_rows_and_names(space):
    m, names = card_feature_matrix(space.passwords, CARDS)
    assert m.shape == (len(space), len(names)) and len(set(names)) == len(names)
    i_spell, i_trap, i_monster = (space.index_of(p) for p in (160, 180, 100))
    col = {n: j for j, n in enumerate(names)}
    assert m[i_spell, col["spell"]] == 1 and m[i_spell, col["monster"]] == 0
    assert m[i_trap, col["trap"]] == 1
    assert m[i_monster, col["monster"]] == 1 and m[i_monster, col["tuner"]] == 1  # 100 % 5 == 0
    assert m[space.index_of(300), col["xyz"]] == 1 and m[space.index_of(301), col["link"]] == 1


def test_feature_map_shapes_and_blocks(space, genotypes):
    fm = FeatureMap(space, CARDS)
    x = fm(genotypes[:10])
    assert x.shape == (10, fm.dim) and x.dtype == np.float64 and len(fm.names) == fm.dim
    assert list(fm.groups) == ["counts", "packages", "structure", "roles"]
    counts = x[:, fm.groups["counts"]]
    assert np.array_equal(counts, np.stack([space.vector(g) for g in genotypes[:10]]))
    roles = x[:, fm.groups["roles"]]
    names = [fm.names[i] for i in range(fm.dim)][fm.groups["roles"]]
    for g, row in zip(genotypes[:10], roles):
        assert dict(zip((n.split(":", 1)[1] for n in names), row.tolist())) == pytest.approx(
            {r: float(v) for r, v in space.role_counts(g).items()}
        )
    pk = x[:, fm.groups["packages"]]
    assert pk.shape[1] == len(space.packages)
    assert np.array_equal(
        pk[0], [space.vector(genotypes[0])[space.package_indices(p)].sum() for p in range(len(space.packages))]
    )
    st = x[:, fm.groups["structure"]]
    assert np.isfinite(st).all()
    # main-deck type fractions sum to 1 per deck
    col = {n: j for j, n in enumerate(fm.names)}
    frac = x[:, col["main:monster"]] + x[:, col["main:spell"]] + x[:, col["main:trap"]]
    assert frac == pytest.approx(np.ones(10))
    assert x[:, col["main_size"]] == pytest.approx([space.vector(g)[: space.n_main].sum() for g in genotypes[:10]])


def test_feature_map_groups_can_be_switched_off(space, genotypes):
    fm = FeatureMap(space, groups=("counts",))
    assert fm.dim == len(space) and list(fm.groups) == ["counts"]
    assert fm(genotypes[:3]).shape == (3, len(space))
    with pytest.raises(ValueError, match="cards"):
        FeatureMap(space, groups=("counts", "structure"))
    with pytest.raises(ValueError, match="unknown"):
        FeatureMap(space, CARDS, groups=("counts", "nonsense"))


def test_feature_map_accepts_count_matrices_and_decks(space, genotypes):
    fm = FeatureMap(space, CARDS)
    counts = np.stack([space.vector(g) for g in genotypes[:5]])
    assert np.array_equal(fm.transform(counts), fm(genotypes[:5]))
    decks = [space.decode(g) for g in genotypes[:5]]
    assert np.array_equal(fm.deck_counts(decks), counts)
    outside = Deck(main=decks[0].main + (999999,), extra=decks[0].extra)  # unknown card: ignored
    assert np.array_equal(fm.deck_counts([outside])[0], counts[0])
    assert np.array_equal(fm(decks), fm(genotypes[:5]))


def test_text_embedding_mean_is_copy_weighted_and_skips_missing(space, genotypes, tmp_path):
    vocab = CardVocab(sorted(CARDS))
    rng = np.random.default_rng(1)
    table = rng.normal(size=(len(vocab), 4))
    missing = space.passwords[3]
    emb = TextEmbeddings(table, vocab, missing={missing})
    fm = FeatureMap(space, CARDS, text=emb)
    assert list(fm.groups)[-1] == "text" and fm.groups["text"].stop - fm.groups["text"].start == 4
    g = genotypes[0]
    c = space.vector(g).astype(float)
    rows = np.array([table[vocab.index(p)] for p in space.passwords])
    keep = np.array([p != missing for p in space.passwords])
    want = (c[keep] @ rows[keep]) / c[keep].sum()
    assert fm([g])[0, fm.groups["text"]] == pytest.approx(want)
    assert emb.coverage(space.passwords) == pytest.approx(1 - 1 / len(space))

    # .npy + vocab JSON round trip (the T5.2 artefact layout)
    np.save(tmp_path / "emb.npy", table)
    vocab.save(tmp_path / "vocab.json")
    loaded = TextEmbeddings.load(tmp_path / "emb.npy", tmp_path / "vocab.json")
    assert loaded.dim == 4 and loaded.coverage(space.passwords) == 1.0
    assert np.allclose(FeatureMap(space, CARDS, text=loaded)([g])[0, fm.groups["text"]], (c @ rows) / c.sum())
    with pytest.raises(ValueError, match="rows"):
        TextEmbeddings(table[:-1], vocab)


# ---------------------------------------------------------------- ridge ensemble


def _linear(n, d=30, noise=0.05, seed=0, targets=2):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(d, targets)) / np.sqrt(d)
    x = rng.normal(size=(n, d))
    return x, x @ w + noise * rng.normal(size=(n, targets)), w


def test_ridge_ensemble_fits_a_linear_target():
    x, y, w = _linear(400)
    xt = np.random.default_rng(9).normal(size=(200, 30))
    model = RidgeEnsemble(n_members=8, seed=0).fit(x, y)
    mean, std = model.predict(xt)
    assert mean.shape == std.shape == (200, 2)
    assert np.abs(mean - xt @ w).mean() < 0.03
    assert (std > 0).all()
    assert model.residual_std_ == pytest.approx([0.05, 0.05], abs=0.015)


def test_ridge_ensemble_handles_missing_targets_and_weights():
    x, y, w = _linear(300)
    y[::3, 1] = np.nan  # the second target is only labelled on 2/3 of the rows
    model = RidgeEnsemble(n_members=4, seed=0).fit(x, y, weights=np.linspace(1, 3, 300))
    xt = np.random.default_rng(9).normal(size=(50, 30))
    mean, _ = model.predict(xt)
    assert np.abs(mean - xt @ w).mean() < 0.05
    y[:, 1] = np.nan
    y[:5, 1] = 0.3  # too few labels: that target is not fitted
    mean, std = RidgeEnsemble(n_members=4, min_rows=10).fit(x, y).predict(xt)
    assert np.isnan(mean[:, 1]).all() and np.isnan(std[:, 1]).all() and np.isfinite(mean[:, 0]).all()


def test_ensemble_uncertainty_shrinks_with_data():
    xt = np.random.default_rng(9).normal(size=(200, 30))
    stds = []
    for n in (40, 160, 640):
        x, y, _ = _linear(n, noise=0.2, seed=n)
        stds.append(RidgeEnsemble(n_members=16, seed=0).fit(x, y).predict(xt)[1].mean())
    assert stds[0] > stds[1] > stds[2]


def test_ensemble_is_less_certain_far_from_the_data():
    x, y, _ = _linear(200, noise=0.1)
    model = RidgeEnsemble(n_members=16, seed=0).fit(x, y)
    near = model.predict(x[:50])[1].mean()
    far = model.predict(5 * np.random.default_rng(3).normal(size=(50, 30)))[1].mean()
    assert far > 2 * near


def test_ridge_ensemble_is_deterministic_and_validates_input():
    x, y, _ = _linear(100)
    a = RidgeEnsemble(n_members=4, seed=7).fit(x, y).predict(x[:5])
    b = RidgeEnsemble(n_members=4, seed=7).fit(x, y).predict(x[:5])
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    with pytest.raises(RuntimeError, match="fit"):
        RidgeEnsemble().predict(x)
    with pytest.raises(ValueError, match="rows"):
        RidgeEnsemble().fit(x, y[:-1])


# ---------------------------------------------------------------- online surrogate


def _truth(space, genotypes):
    """A synthetic 'win rate': package 0 and hand traps are good, package 3 is bad."""
    c = np.stack([space.vector(g) for g in genotypes]).astype(float)
    p0 = c[:, space.package_indices(0)].sum(1) / 40
    p3 = c[:, space.package_indices(3)].sum(1) / 40
    ht = np.array([space.role_counts(g).get("hand_trap", 0) for g in genotypes]) / 15
    base = np.clip(0.4 + 0.35 * p0 - 0.3 * p3 + 0.2 * ht, 0, 1)
    return base, np.clip(base + 0.1, 0, 1), np.clip(base - 0.1, 0, 1)


def _labels(space, genotypes, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    cols = _truth(space, genotypes)
    return [
        {t: float(np.clip(v[i] + noise * rng.normal(), 0, 1)) for t, v in zip(DEFAULT_TARGETS, cols)}
        for i in range(len(genotypes))
    ]


def test_surrogate_online_update_improves_held_out_error(space, genotypes):
    sur = Surrogate(FeatureMap(space, CARDS), n_members=8, seed=0)
    assert sur.targets == DEFAULT_TARGETS and sur.n_observations == 0
    with pytest.raises(RuntimeError, match="no labelled"):
        sur.predict(genotypes[:3])
    test = genotypes[300:]
    truth = _truth(space, test)[0]
    errors = []
    for lo, hi in ((0, 20), (20, 80), (80, 250)):
        sur.update(genotypes[lo:hi], _labels(space, genotypes[lo:hi], noise=0.05, seed=lo))
        assert sur.n_observations == hi
        pred = sur.predict(test)
        assert set(pred.mean) == set(DEFAULT_TARGETS)
        assert pred.mean["win_rate"].shape == pred.std["win_rate"].shape == (len(test),)
        assert ((pred.mean["win_rate"] >= 0) & (pred.mean["win_rate"] <= 1)).all()
        errors.append(np.abs(pred.mean["win_rate"] - truth).mean())
    assert errors[0] > errors[-1] and errors[-1] < 0.03


def test_surrogate_merges_repeated_evaluations(space, genotypes):
    sur = Surrogate(FeatureMap(space, CARDS), n_members=2)
    g = genotypes[0]
    sur.add([g], [{"win_rate": 0.2}], weights=[40])
    sur.add([g], [{"win_rate": 0.6, "win_rate_first": 0.5}], weights=[120])
    assert sur.n_observations == 1
    row = sur.observation(g)
    assert row["win_rate"] == pytest.approx(0.5) and row["win_rate_first"] == pytest.approx(0.5)
    assert row["weight"] == 160 and np.isnan(row["win_rate_second"])
    # the same deck twice in one call is one observation too, and the model still fits
    h = genotypes[1]
    sur.add([h, h], [{"win_rate": 0.4}, {"win_rate": 0.8}], weights=[1, 3])
    x, y, w = sur.dataset()
    assert sur.n_observations == 2 and len(x) == len(y) == len(w) == 2
    assert sur.observation(h)["win_rate"] == pytest.approx(0.7) and sur.observation(h)["weight"] == 4
    sur.fit()


def test_surrogate_descriptors_mix_exact_and_predicted(space, genotypes):
    sur = Surrogate(FeatureMap(space, CARDS), n_members=4, targets=("win_rate", "win_rate_first", "win_rate_second"))
    sur.update(genotypes[:100], _labels(space, genotypes[:100]))
    d = sur.descriptors(genotypes[100:110])
    assert set(d) == {"win_rate_first", "win_rate_second", "hand_traps"}
    assert d["hand_traps"].tolist() == [space.role_counts(g).get("hand_trap", 0) for g in genotypes[100:110]]
    custom = Surrogate(FeatureMap(space, CARDS), n_members=2, targets=("win_rate", "combo_length"),
                       descriptors=("combo_length", "hand_traps"))  # fmt: skip
    labels = [{"win_rate": 0.5, **({"combo_length": float(i % 7)} if i % 2 else {})} for i in range(60)]
    custom.update(genotypes[:60], labels)
    assert set(custom.descriptors(genotypes[:3])) == {"combo_length", "hand_traps"}
    with pytest.raises(ValueError, match="descriptor"):
        Surrogate(FeatureMap(space, CARDS), descriptors=("brick_rate",))


def test_surrogate_select_prefers_promising_uncertain_candidates(space, genotypes):
    sur = Surrogate(FeatureMap(space, CARDS), n_members=8, seed=0)
    sur.update(genotypes[:60], _labels(space, genotypes[:60], noise=0.02))
    cand = genotypes[60:300]
    picks = sur.select(cand, 10, beta=0.0)
    mean = sur.predict(cand).mean["win_rate"]
    assert sorted(picks.tolist()) == sorted(np.argsort(-mean)[:10].tolist())


# ---------------------------------------------------------------- acquisition


def test_acquire_ucb_ranking():
    mean = np.array([0.5, 0.6, 0.4, 0.55])
    std = np.array([0.0, 0.01, 0.3, 0.02])
    assert acquire(mean, std, 2, beta=0.0).tolist() == [1, 3]
    assert acquire(mean, std, 2, beta=1.0).tolist() == [2, 1]
    assert acquire(mean, std, 10, beta=1.0).tolist() == [2, 1, 3, 0]  # k > n: everything, best first


def test_acquire_one_per_cell_first_then_fill():
    mean = np.array([0.9, 0.8, 0.7, 0.3, 0.2])
    std = np.zeros(5)
    cells = np.array([0, 0, 0, 1, 2])
    assert acquire(mean, std, 3, cells=cells).tolist() == [0, 3, 4]  # each cell's best before a second from cell 0
    assert acquire(mean, std, 4, cells=cells).tolist() == [0, 3, 4, 1]


def test_acquire_explore_share_takes_the_most_uncertain():
    mean = np.array([0.9, 0.8, 0.7, 0.1, 0.1])
    std = np.array([0.0, 0.0, 0.0, 0.2, 0.1])
    assert acquire(mean, std, 3, beta=0.0, explore=1 / 3).tolist() == [0, 1, 3]
    with pytest.raises(ValueError):
        acquire(mean, std, 2, explore=1.5)
