"""Edit-value model (ygorl.build.value_model, #151), its labels (ygorl.build.edit_labels) and its use in the evolution
step, on synthetic data with known effects."""

import gzip
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ygorl.build.edit_labels import (  # noqa: E402
    CheckpointRef,
    Clock,
    PairedLabel,
    paired_labels,
    read_game_log,
    unique,
)
from ygorl.build.evolve import Evolution, RoundConfig, blend_prior  # noqa: E402
from ygorl.build.signals import Child, spearman  # noqa: E402
from ygorl.build.tuner import Edit  # noqa: E402
from ygorl.build.value_model import (  # noqa: E402
    SCORES,
    DeckModelFeatures,
    ValueModelConfig,
    cross_validate,
    edit_features,
    fit,
    group_folds,
    load_value_model,
    summarize,
)
from ygorl.cards.ydk import Deck  # noqa: E402

CARDS = list(range(1, 61))
DIM = 6


class Features:
    """Stand-in deck-model features: every card has a fixed random vector; a deck is the copy-weighted mean."""

    def __init__(self, seed=0):
        self.vec = {c: np.random.default_rng([seed, c]).normal(size=DIM).astype(np.float32) for c in CARDS}
        self.dim = DIM

    def card_vectors(self, cards):
        return np.stack([self.vec[c] for c in cards]) if cards else np.zeros((0, DIM), np.float32)

    def deck_vectors(self, decks):
        return np.stack([self.card_vectors(list(d.main)).mean(0) for d in decks]) if decks else np.zeros((0, DIM))

    def edit_scores(self, parent, edits):
        return np.zeros(len(SCORES), np.float32)


W = np.random.default_rng(99).normal(size=DIM)


def value(feats, card):
    """The planted per-copy value of a card: linear in its vector, ±2-3 pp."""
    return 0.01 * float(feats.vec[card] @ W)


def make_labels(feats, n, *, seed, run="/r", update=400, flip=False, se=0.004, decks=6):
    """``n`` single-swap labels on ``decks`` random parents: Δ = value(in) − value(out) (reversed with ``flip``:
    a policy that played the cards the other way) plus noise of ``se``."""
    rng = np.random.default_rng(seed)
    parents = [Deck(main=tuple(int(c) for c in np.random.default_rng([7, k]).choice(CARDS[:40], 40)), extra=())
               for k in range(decks)]  # fmt: skip
    out = []
    for i in range(n):
        p = parents[i % decks]
        o = int(p.main[rng.integers(40)])
        into = int(rng.choice(CARDS))
        while into == o:
            into = int(rng.choice(CARDS))
        d = value(feats, into) - value(feats, o)
        d = -d if flip else d
        out.append(PairedLabel(f"{seed}:{i}", p, "T", (Edit(o, into, "main"),), d + rng.normal(0, se), se,
                               CheckpointRef(run, update), "synthetic"))  # fmt: skip
    return out


def truth(feats, labels):
    return np.array([value(feats, lab.edits[0].into) - value(feats, lab.edits[0].out) for lab in labels])


SMALL = ValueModelConfig(hidden=16, members=4, steps=400, lr=1e-2, weight_decay=1e-2, dropout=0.0)


def test_the_model_recovers_known_edit_values_on_held_out_edits():
    feats = Features()
    train = make_labels(feats, 240, seed=1)
    test = make_labels(feats, 80, seed=2)
    model, rep = fit(feats, train, Clock("/r", 400), SMALL)
    m, sd = model.predict_batch(edit_features(feats, [t.parent for t in test], [t.edits for t in test]))
    assert spearman(m, truth(feats, test)) > 0.8
    assert np.all(sd > 0) and rep.members == 4 and rep.labels == 240
    # predict() is the evolution step's interface: (mean, sd) per edit list of one parent
    p = test[0].parent
    got = model.predict(p, [t.edits for t in test[:3] if t.parent == p] or [test[0].edits])
    assert all(len(x) == 2 and x[1] > 0 for x in got)


def test_held_out_parent_decks_and_the_baselines_are_reported():
    feats = Features()
    labels = make_labels(feats, 180, seed=3, decks=3)
    folds = group_folds(labels, "deck")
    assert len(set(folds)) == 3 and all(f == folds[i % 3] for i, f in enumerate(folds))
    preds = cross_validate(feats, labels, Clock("/r", 400), SMALL, folds=folds)
    table = summarize(labels, preds)
    assert set(table) == {"value_model", "card_value", "add", "removal", "add_removal"}
    assert table["value_model"]["pooled"] > 0.6  # the value lives in the card vectors: it carries over to new decks
    # random folds put every measurement of one edit in the same fold
    twice = labels + [PairedLabel("again", labels[0].parent, "T", labels[0].edits, 0.0, 0.01, labels[0].checkpoint)]
    f = group_folds(twice, "random", k=5)
    assert f[0] == f[-1]


def test_older_labels_are_down_weighted_and_their_age_is_a_feature():
    """Synthetic drift: the old policy (update 0) measured every edit reversed; the current one (update 400) has
    few, noisy labels. Conditioning on the checkpoint follows the current policy; ignoring it does not."""
    feats = Features()
    old = make_labels(feats, 300, seed=4, update=0, flip=True)
    new = make_labels(feats, 60, seed=5, update=400, se=0.01)
    test = make_labels(feats, 80, seed=6)
    x = edit_features(feats, [t.parent for t in test], [t.edits for t in test])
    clocked, _ = fit(feats, old + new, Clock("/r", 400, half_life=100.0), SMALL)
    blind, _ = fit(feats, old + new, Clock("/r", 400, half_life=float("inf")), SMALL)
    rho_c = spearman(clocked.predict_batch(x)[0], truth(feats, test))
    rho_b = spearman(blind.predict_batch(x)[0], truth(feats, test))
    assert rho_c > 0.6 and rho_b < 0.2, (rho_c, rho_b)
    clock = Clock("/r", 400, half_life=100.0, foreign_age=300.0, same=("/other",))
    assert clock.age(CheckpointRef("/r", 100)) == 300 and clock.weight(CheckpointRef("/r", 300)) == pytest.approx(0.5)
    assert clock.age(CheckpointRef("/other", 350)) == 50  # a continuation of the run in another directory
    assert clock.age(CheckpointRef("/elsewhere", 400)) == 300 and clock.age(CheckpointRef("/r", None)) == 300


def test_training_games_train_the_strength_head(tmp_path):
    """The auxiliary loss: deck a beats deck b with P = σ(s(a) − s(b)); the head learns which decks are strong."""
    feats = Features()
    rng = np.random.default_rng(8)
    decks = {f"d{k}": Deck(main=tuple(int(c) for c in rng.choice(CARDS, 40)), extra=()) for k in range(12)}
    strength = {n: sum(value(feats, c) for c in d.main) * 3 for n, d in decks.items()}
    from ygorl.build.edit_labels import GameData, GameRecord

    games = []
    names = sorted(decks)
    for _ in range(4000):
        a, b = rng.choice(names, 2, replace=False)
        p = 1 / (1 + np.exp(-(strength[a] - strength[b])))
        games.append(GameRecord(str(a), str(b), int(rng.integers(2)), 0 if rng.random() < p else 1,
                                CheckpointRef("/r", 399)))  # fmt: skip
    labels = make_labels(feats, 60, seed=9)
    cfg = ValueModelConfig(**{**SMALL.__dict__, "aux_weight": 1.0, "members": 2})
    model, rep = fit(feats, labels, Clock("/r", 400), cfg, games=GameData(decks, games))
    assert rep.games == 4000
    z = model.zs(feats.deck_vectors([decks[n] for n in names]))
    with torch.no_grad():
        s = model.nets[0].strength(z).squeeze(-1).numpy()
    assert spearman(s, [strength[n] for n in names]) > 0.8


def test_a_run_game_log_resolves_corpus_and_evolved_decks(tmp_path):
    deck = Deck(main=(1,) * 3 + tuple(range(2, 39)), extra=())
    for name in ("alpha", "beta"):
        (tmp_path / f"{name}.ydk").write_text(deck.to_ydk())
    (tmp_path / "pool" / "decks").mkdir(parents=True)
    (tmp_path / "pool" / "decks" / "evo-1.ydk").write_text(deck.to_ydk())
    (tmp_path / "pool" / "manifest.json").write_text(json.dumps(
        {"format": "ygorl-deck-pool", "version": 1, "decks": [{"id": "evo-1", "file": "decks/evo-1.ydk"}]}))  # fmt: skip
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({"decks": [str(tmp_path / "alpha.ydk"), str(tmp_path / "beta.ydk")],
                                                 "deck_pool": str(tmp_path / "pool" / "manifest.json"),
                                                 "overlap_collect": False}))  # fmt: skip
    rec = {"update": 12, "decks": ["alpha", "evo-1"], "evolved": 1, "first": 0, "winner": 1, "turns": 7,
           "reason": "win", "truncated": False, "opponent": None, "learner": None, "seed": 1}  # fmt: skip
    lines = [rec, rec, {**rec, "seed": 2, "truncated": True}, {**rec, "seed": 3, "winner": None},
             {**rec, "seed": 4, "decks": ["alpha", "gone"]}, {**rec, "seed": 5, "decks": ["beta", "alpha"]}]  # fmt: skip
    with gzip.open(run / "games.jsonl.gz", "at") as f:  # two gzip members, as the trainer writes one per update
        f.write("\n".join(json.dumps(x) for x in lines[:3]) + "\n")
    with gzip.open(run / "games.jsonl.gz", "at") as f:
        f.write("\n".join(json.dumps(x) for x in lines[3:]) + "\n")
    g = read_game_log(run)
    assert [(x.a, x.b, x.winner) for x in g.games] == [("alpha", "evo-1", 1), ("beta", "alpha", 1)]
    assert g.skipped == 4 and set(g.decks) >= {"alpha", "beta", "evo-1"}
    assert g.games[0].checkpoint == CheckpointRef(str(run.resolve()), 11)  # played by the policy after 11 updates


def test_save_and_load_keep_the_predictions(tmp_path):
    feats = Features()
    labels = make_labels(feats, 60, seed=10)
    model, _ = fit(feats, labels, Clock("/r", 400), ValueModelConfig(members=2, steps=50, card_terms=True))
    model.save(tmp_path / "vm.pt")
    again = load_value_model(tmp_path / "vm.pt")
    again.features = feats
    x = edit_features(feats, [t.parent for t in labels[:10]], [t.edits for t in labels[:10]])
    assert np.allclose(model.predict_batch(x)[0], again.predict_batch(x)[0], atol=1e-6)
    assert again.cards == model.cards and again.clock.to_dict() == model.clock.to_dict()


def test_deck_model_features_have_the_documented_layout():
    from ygorl.build.deck_model import DeckModel, DeckModelConfig

    vocab = CARDS + [900]
    dm = DeckModel(vocab, np.random.default_rng(0).normal(size=(len(vocab), 8)), DeckModelConfig(dim=16, layers=1,
                   heads=2, ff=32, dropout=0.0))  # fmt: skip
    feats = DeckModelFeatures(dm)
    parent = Deck(main=tuple(range(1, 41)), extra=(900,))
    b = edit_features(feats, [parent, parent], [[Edit(3, 50, "main")], [Edit(3, 50, "main"), Edit(4, 51, "main")]])
    assert b.x.shape == (2, 3 * 16 + len(SCORES) + 2) and b.zp.shape == (2, 16)
    assert b.x[1, -2] == 2 and b.x[0, -1] == 0  # edits, age
    assert b.cards[1] == {50: 1, 51: 1, 3: -1, 4: -1}


# ------------------------------------------------------------------ labels from files


def test_paired_labels_read_lineage_and_factorial_files_and_drop_replayed_measurements(tmp_path):
    from tests.test_evolve import BASE, run

    ev, report = run(tmp_path / "state")
    labels, skipped = paired_labels(tmp_path / "state", tmp_path)
    learned = sum(len(r["learned"]) for r in ev.lineage())
    assert len(labels) == learned and skipped == 0
    assert all(lab.parent.main == BASE.main and lab.checkpoint.update == 400 for lab in labels)
    assert all(lab.source == "lineage" for lab in labels)
    # a second state replaying the same games (same seed, as #145 and #150) counts once; equal numbers from another
    # seed (its round.json) are other games and count again
    run(tmp_path / "again")
    again, _ = paired_labels(tmp_path / "again", tmp_path)
    assert [x.games for x in again] == [x.games for x in labels] and len(unique(labels + again)) == len(labels)
    rj = tmp_path / "again" / "rounds" / "0001" / "round.json"
    rj.write_text(json.dumps({**json.loads(rj.read_text()), "config": {**json.loads(rj.read_text())["config"],
                                                                       "seed": 8}}))  # fmt: skip
    other, _ = paired_labels(tmp_path / "again", tmp_path)
    assert len(unique(labels + other)) == 2 * len(labels)
    # the factorial measurement: main effects and one-at-a-time differences of the same edits
    (tmp_path / "decks").mkdir()
    (tmp_path / "decks" / "base.ydk").write_text(BASE.to_ydk())
    fac = {"checkpoint": "out/x/checkpoints/update_000400.pt", "parents": [{
        "type": "Base", "file": "decks/base.ydk", "edits": ["main: -Ten +Fifty", "main: -Eleven +Fiftyone"],
        "factorial": {"edits": [{"out": 10, "into": 50, "section": "main", "letter": "A"},
                                {"out": 11, "into": 51, "section": "main", "letter": "B"}],
                      "effects": [{"term": [0], "label": "A", "effect": 0.03, "stderr": 0.01},
                                  {"term": [1], "label": "B", "effect": -0.01, "stderr": 0.01},
                                  {"term": [0, 1], "label": "AB", "effect": 0.0, "stderr": 0.01}]},
        "one_at_a_time": {"effects": [{"edit": "main: -Ten +Fifty", "diff": 0.02, "stderr": 0.012},
                                      {"edit": "main: -Eleven +Fiftyone", "diff": 0.0, "stderr": 0.012}]}}]}  # fmt: skip
    (tmp_path / "factorial.json").write_text(json.dumps(fac))
    got, _ = paired_labels(tmp_path / "factorial.json", tmp_path)
    assert [(g.edits[0].into, g.diff) for g in got] == [(50, 0.03), (51, -0.01), (50, 0.02), (51, 0.0)]
    assert got[0].checkpoint.update == 400 and got[0].checkpoint.run.endswith("out/x")
    assert got[0].edit_key == got[2].edit_key  # two measurements of one edit


# ------------------------------------------------------------------ the evolution step


class FakeValueModel:
    """Predicts +20 pp for putting in card 51 and −5 pp otherwise."""

    def __init__(self):
        self.calls = 0

    def predict(self, parent, edits):
        self.calls += 1
        return [(0.2 if any(e.into == 51 for e in es) else -0.05, 0.01) for es in edits]


def _round(tmp, **kw):
    from tests.test_evolve import CONFIG, OPPONENTS, PARENTS, Env, make_lab

    lab = make_lab()
    vm = kw.pop("value_model", None)
    lab.value_model = vm
    ev = Evolution(tmp, Env())
    for p, m in kw.pop("calibration", []):
        ev.calibration.record("value_model", p, m)
    config = RoundConfig(**{**CONFIG.__dict__, **kw})
    return ev, ev.run_round(PARENTS, lab, config, OPPONENTS)


def test_the_value_model_is_the_thompson_prior_by_its_weight_and_is_calibrated(tmp_path):
    ev0, rep0 = _round(tmp_path / "none")
    vm = FakeValueModel()
    evz, repz = _round(tmp_path / "zero", value_model=vm)  # weight 0: shadow only, the round is unchanged
    assert vm.calls > 0
    strip = lambda rows: [{k: r[k] for k in ("child", "edits", "search", "validation", "accepted")} for r in rows]  # noqa: E731
    assert strip(evz.lineage()) == strip(ev0.lineage())
    assert all(r["prior"]["mean"] == r["predicted"]["mean"] and r["prior"]["weight"] == 0 for r in evz.lineage())
    assert all(r["value_model"]["mean"] in (0.2, -0.05) for r in evz.lineage())
    # every child's first batch calibrates the value model's prediction, as for the card-value model
    assert evz.calibration.report()["value_model"]["pairs"] == rep0["children"]
    ev1, _ = _round(tmp_path / "one", value_model=FakeValueModel(), value_model_weight=1.0)
    for r in ev1.lineage():
        assert r["prior"]["weight"] == 1.0
        assert (r["prior"]["mean"], r["prior"]["sd"]) == pytest.approx((r["value_model"]["mean"], 0.01))


def test_the_calibration_table_switches_a_poor_value_model_off(tmp_path):
    bad = [(float(k), -float(k)) for k in range(30)]  # predictions that rank the measurements backwards
    ev, _ = _round(tmp_path / "s", value_model=FakeValueModel(), value_model_weight=1.0, calibration=bad)
    assert all(r["prior"]["weight"] == 0 and r["prior"]["mean"] == r["predicted"]["mean"] for r in ev.lineage())
    good = [(float(k), float(k)) for k in range(30)]
    ev, _ = _round(tmp_path / "g", value_model=FakeValueModel(), value_model_weight=0.0, calibration=good)
    assert all(r["prior"]["weight"] == pytest.approx(1.0) for r in ev.lineage())  # calibrated: Spearman 1


def test_blend_and_the_learned_ranking_follow_the_weight(tmp_path):
    assert blend_prior((0.0, 0.04), (0.1, 0.02), 0.0) == (0.0, 0.04)
    assert blend_prior((0.0, 0.04), None, 1.0) == (0.0, 0.04)
    m, sd = blend_prior((0.0, 0.04), (0.1, 0.02), 0.5)
    assert m == pytest.approx(0.05) and sd == pytest.approx(np.sqrt(0.5 * 0.02**2 + 0.5 * 0.04**2))
    from tests.test_evolve import BASE, Env, make_lab

    ev = Evolution(tmp_path, Env())
    lab = make_lab()
    lab.value_model = FakeValueModel()
    kids = [Child((Edit(10 + k, 50 + (k == 2), "main"),), BASE, "learned", 0.0) for k in range(4)]
    rng = np.random.default_rng(0)
    assert ev._rank_learned(lab, BASE, kids, 2, 0.0, rng) == kids[:2]  # weight 0: as drawn
    best = ev._rank_learned(lab, BASE, kids, 1, 0.5, rng)
    assert best == [kids[2]]  # the value model's favourite (card 51)
    assert Evolution._oversample(RoundConfig(value_model_oversample=3), 0.5) == 3
    assert Evolution._oversample(RoundConfig(value_model_oversample=3), 0.0) == 1
