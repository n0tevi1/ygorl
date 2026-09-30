"""Masked deck model (ygorl.build.deck_model) and its learned children (ygorl.build.learned), on a synthetic corpus."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ygorl.build.deck_model import (  # noqa: E402
    DeckModel,
    DeckModelConfig,
    fill_in_tasks,
    held_out,
    load_deck_model,
    ranks,
    split_by_type,
    topk,
    train_deck_model,
)
from ygorl.build.evolve import Evolution, Parent, RoundConfig  # noqa: E402
from ygorl.build.learned import learned_children  # noqa: E402
from ygorl.build.signals import CardValueModel  # noqa: E402
from ygorl.cards.ydk import Deck  # noqa: E402

VOCAB = list(range(1, 61)) + [900, 901]  # 900 / 901: Extra Deck cards
CORE = {"X": (1, 2, 3, 4), "Y": (10, 11, 12, 13)}  # the planted co-occurrence: a type's core runs together


def corpus(n=120, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        t = "X" if i % 2 else "Y"
        filler = rng.choice(np.arange(20, 61), size=10, replace=False)
        main = [c for c in CORE[t] for _ in range(3)] + [int(c) for c in filler]
        out.append((t, Deck(main=tuple(main), extra=(900,) if t == "X" else (901,))))
    return out


@pytest.fixture(scope="module")
def trained():
    rng = np.random.default_rng(1)
    text = rng.normal(size=(len(VOCAB), 8)).astype(np.float32)
    cfg = DeckModelConfig(dim=32, layers=1, heads=2, ff=64, dropout=0.0)
    model = DeckModel(VOCAB, text, cfg, environment={"environment": "md-test"})
    return train_deck_model(model, corpus(), epochs=60, batch=32, lr=3e-3, seed=0)


def test_the_model_learns_a_planted_co_occurrence(trained):
    deck = corpus(2, seed=5)[1][1]  # an X list
    assert deck.counts()[3] == 3
    rest = Deck(main=tuple(c for c in deck.main if c != 3), extra=deck.extra)
    add = trained.addition_scores(rest, VOCAB)
    missing = sorted((c for c in add if c not in rest.counts()), key=lambda c: -add[c])
    assert missing[0] == 3  # the missing core card of X, not Y's core
    assert add[3] > add[11]
    # fill-in over the vocab: every copy of 3 masked
    logp, copies = trained.fill_in([deck], [3])
    r = ranks(logp, np.array([trained.index[3]]), [[trained.index[c] for c in rest.counts()]])
    assert r[0] == 1 and copies[0] == 3
    # an intruder of the other core is the most out of place card
    filler = deck.main[-1]  # one copy of a random filler card
    intruder = Deck(main=tuple(11 if c == filler else c for c in deck.main), extra=deck.extra)
    for kind in ("typicality", "support", "combined"):
        rem = trained.removal_scores(intruder, kind)
        assert set(rem) == set(intruder.counts())
        assert all(rem[11] > rem[c] for c in CORE["X"]), kind
    assert trained.removal_scores(intruder, "typicality") == trained.typicality_scores(intruder)
    # the core cards support each other; the intruder supports nothing: the core is the least removable
    sup = trained.support_scores(intruder)
    assert min(sup[c] for c in CORE["X"]) > sup[11]
    comb = trained.removal_scores(intruder)
    assert sorted(comb.values()) == [float(k) for k in range(1, len(comb) + 1)]  # positions from the end
    assert comb[11] == len(comb) and max(comb[c] for c in CORE["X"]) <= len(comb) // 3 + 1
    with pytest.raises(ValueError, match="unknown removal"):
        trained.removal_scores(intruder, "nope")


def test_interface_shapes_and_masking():
    model = DeckModel(VOCAB, np.zeros((len(VOCAB), 4), np.float32), DeckModelConfig(dim=16, layers=1, heads=2, ff=32))
    decks = [d for _, d in corpus(8)]
    idx, cnt = model.encode(decks)
    assert idx.shape == cnt.shape and idx.shape[0] == 8
    assert all(int(cnt[i].sum()) == len(d.main) + len(d.extra) for i, d in enumerate(decks))
    gen = torch.Generator().manual_seed(0)
    bi, bc, kind, target, copies = model.masked_batch(idx, cnt, generator=gen)
    width = idx.shape[1]
    assert width < bi.shape[1] <= 2 * width and target.shape == bi.shape
    assert ((target >= 0).sum(1) >= 1).all()  # at least one masked card a list
    at = target >= 0
    assert (kind[at] == 1).all() and ((copies[at] >= 1) & (copies[at] <= 3)).all()
    # per card: the copies shown + the copies masked = the list's copies
    for row in range(8):
        total = {}
        for j in range(bi.shape[1]):
            if kind[row, j] == 0:
                total[int(bi[row, j])] = total.get(int(bi[row, j]), 0) + int(bc[row, j])
            if target[row, j] >= 0:
                total[int(target[row, j])] = total.get(int(target[row, j]), 0) + int(copies[row, j])
        assert total == {int(i): int(n) for i, n in zip(idx[row], cnt[row], strict=True) if i >= 0}
    loss, stats = model.loss(bi, bc, kind, target, copies)
    loss.backward()
    assert np.isfinite(loss.item()) and set(stats) == {"card", "count", "top1"}
    deck = decks[0]
    assert set(model.removal_scores(deck)) == set(deck.counts())
    add = model.addition_scores(deck, [1, 2, 999])
    assert set(add) == {1, 2} and all(v <= 0 for v in add.values())  # log-probabilities; 999 is not in the vocab


def test_save_and_load_keep_the_scores(trained, tmp_path):
    path = tmp_path / "m.pt"
    trained.save(path, meta={"lists": 120})
    back = load_deck_model(path)
    deck = corpus(1)[0][1]
    assert back.meta == {"lists": 120} and back.environment == {"environment": "md-test"}
    assert back.passwords == trained.passwords and torch.equal(back.seen, trained.seen)
    a, b = trained.typicality_scores(deck), back.typicality_scores(deck)
    assert a.keys() == b.keys() and all(a[k] == pytest.approx(b[k], abs=1e-4) for k in a)
    with pytest.raises(ValueError, match="not a"):
        torch.save({"format": "x"}, tmp_path / "bad.pt")
        load_deck_model(tmp_path / "bad.pt")


def test_split_by_type_holds_out_whole_types():
    lists = [(f"T{i % 20}", Deck(main=(i,))) for i in range(200)]
    train, test = split_by_type(lists, 0.3)
    assert train and test and len(train) + len(test) == 200
    assert not {t for t, _ in train} & {t for t, _ in test}
    assert all(held_out(t, 0.3) for t, _ in test) and not held_out("T1", 0.0)
    tasks = fill_in_tasks(lists[:5] + [("X", Deck(main=(1, 2, 2)))], {1: 0, 2: 1})
    assert len(tasks) == 1 and tasks[0][1] in (1, 2)  # a list needs two distinct known cards


def test_ranks_exclude_present_cards_and_count_ties_against_the_target():
    scores = np.array([[0.9, 0.5, 0.5, 0.1], [0.9, 0.5, 0.5, 0.1]])
    assert ranks(scores, np.array([1, 1]), [[], [0]]).tolist() == [3, 2]
    assert topk(np.array([1, 3, 20]), (1, 5)) == {"top1": pytest.approx(1 / 3), "top5": pytest.approx(2 / 3),
                                                  "mrr": pytest.approx((1 + 1 / 3 + 1 / 20) / 3), "n": 3}  # fmt: skip


class Scorer:
    """A stand-in model: card 50 belongs most, 51 next; card 12 is the most removable, then 13 (``"typicality"``:
    card 14 first)."""

    def __init__(self):
        self.kinds = []

    def removal_scores(self, deck, kind="combined"):
        self.kinds.append(kind)
        top = {14: 6.0} if kind == "typicality" else {}
        return {c: {12: 5.0, 13: 4.0, **top}.get(c, 0.0) for c in deck.counts()}

    def addition_scores(self, deck, pool):
        return {c: {50: 0.0, 51: -1.0, 901: -2.0}.get(c, -30.0) for c in pool}


def test_learned_children_follow_the_scores_and_stay_legal():
    from tests.test_evolve import BASE, legal

    kids = learned_children(BASE, "Base", Scorer(), CardValueModel(), [50, 51, 52, 1, 901], legal=legal,
                            is_extra=lambda pw: pw >= 900, rng=np.random.default_rng(0), children=6,
                            max_bundle=2, temperature=0.1)  # fmt: skip
    assert kids and all(k.kind == "learned" and legal(k.deck) for k in kids)
    keys = {(tuple(sorted(k.deck.main)), k.deck.extra) for k in kids}
    assert len(keys) == len(kids)
    first = kids[0].edits[0]
    assert (first.into, first.out) == (50, 12)  # low temperature: the best addition for the worst card
    for k in kids:
        assert len({e.into for e in k.edits}) == len(k.edits)  # one copy of a new card
        assert all(e.into != 1 for e in k.edits)  # 1 is at 3 copies already
        assert all((e.section == "extra") == (e.into >= 900) for e in k.edits)
    assert learned_children(BASE, "Base", Scorer(), CardValueModel(), [50], legal=legal, is_extra=lambda pw: False,
                            rng=np.random.default_rng(0), children=0) == []  # fmt: skip
    kw = dict(legal=legal, is_extra=lambda pw: pw >= 900, children=4, max_bundle=1, temperature=0.1)
    # protected cards never go out; the removal ranking is selectable
    kept = learned_children(BASE, "Base", Scorer(), CardValueModel(), [50, 51], rng=np.random.default_rng(0),
                            protected={12}, **kw)  # fmt: skip
    assert kept and all(e.out != 12 for k in kept for e in k.edits) and kept[0].edits[0].out == 13
    scorer = Scorer()
    typ = learned_children(BASE, "Base", scorer, CardValueModel(), [50, 51], rng=np.random.default_rng(0),
                           removal="typicality", **kw)  # fmt: skip
    assert scorer.kinds == ["typicality"] and typ[0].edits[0].out == 14


def test_learned_children_in_the_evolution_step_record_their_generator(tmp_path):
    from tests.test_evolve import BASE, CONFIG, OPPONENTS, Env, make_lab

    lab = make_lab()
    config = RoundConfig(**{**CONFIG.__dict__, "learned": 2})
    with pytest.raises(ValueError, match="deck model"):
        Evolution(tmp_path / "none", Env()).run_round([Parent("corpus:base", BASE, "Base")], lab, config, OPPONENTS)
    lab.deck_model = Scorer()
    lab.card_pool = [50, 51]
    ev = Evolution(tmp_path / "s", Env())
    report = ev.run_round([Parent("corpus:base", BASE, "Base")], lab, config, OPPONENTS)
    assert lab.deck_model.kinds and set(lab.deck_model.kinds) == {"combined"}
    rows = [r for r in ev.lineage() if r["generator"] == "learned"]
    assert rows and all(r["kind"] == "learned" for r in rows)
    assert all({e["into"] for e in r["edits"]} <= {50, 51} for r in rows)
    assert report["generators"]["round"]["learned"]["children"] == len(rows)
    assert "learned " in (tmp_path / "s" / "rounds" / "0001" / "report.txt").read_text()


def test_history_lists_read_the_deck_dataset(tmp_path):
    from types import SimpleNamespace

    from ygorl.build.deck_model import history_lists
    from ygorl.data.deck_dataset import build_deck_dataset
    from ygorl.data.masterduelmeta import DeckRecord

    recs = [DeckRecord("X", "u1", "2026-01-02", 10.0, (1, 1, 2), (900,)),
            DeckRecord("Y", "u2", "2026-01-01", 10.0, (3,), ()),
            DeckRecord("Y", "u3", "2025-01-01", 10.0, (2, 1, 1), (900,))]  # fmt: skip
    build_deck_dataset(recs, lambda r: []).save(tmp_path)
    cards = {c: SimpleNamespace(is_extra_deck=c >= 900) for c in (1, 2, 3, 900)}
    lists = history_lists(tmp_path, cards)
    assert [(t, sorted(d.main), d.extra) for t, d in lists] == [("X", [1, 1, 2], (900,)), ("Y", [3], ())]


def test_learned_children_go_into_the_factorial_design_first(tmp_path):
    from tests.test_evolve import BASE, CONFIG, OPPONENTS, Env, make_lab

    lab = make_lab()
    lab.deck_model, lab.card_pool = Scorer(), [50, 51]
    config = RoundConfig(**{**CONFIG.__dict__, "evaluation": "factorial", "factorial_k": 4, "factorial_pairs": 20,
                            "learned": 2})  # fmt: skip
    ev = Evolution(tmp_path / "s", Env())
    ev.run_round([Parent("corpus:base", BASE, "Base")], lab, config, OPPONENTS)
    fac = ev._round_state(1)["results"][0]["factorial"]
    assert fac["kinds"][:2] == ["learned", "learned"] and len(fac["kinds"]) == 4
