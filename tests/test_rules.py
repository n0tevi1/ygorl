"""Association-rule candidates for deck evolution (ygorl.build.rules), on a hand-made corpus."""

import json

import numpy as np
import pytest

from ygorl.build.evolve import Evolution, Parent, RoundConfig
from ygorl.build.rules import GLOBAL, DeckRules, RuleConfig, rule_children
from ygorl.build.signals import CardValueModel
from ygorl.cards.ydk import Deck

# 10 lists: A ×4 run 1, 2, 3 (three copies of 3); B ×2 run 1, 5; D ×2 run 2, 6; C ×2 run 7. Global counts: 1 and 2 in
# 6 lists, 3 in 4, so 1 → 3 has confidence 4/6 and lift (4/6) / 0.4 = 5/3, and {1, 2} → 3 has confidence 1, lift 2.5.
CORPUS = ([("A", Deck(main=(1, 2, 3, 3, 3, 100 + i))) for i in range(4)]
          + [("B", Deck(main=(1, 5, 110 + i))) for i in range(2)] + [("D", Deck(main=(2, 6, 120 + i))) for i in range(2)]
          + [("C", Deck(main=(7, 130 + i))) for i in range(2)])  # fmt: skip


def mine(**cfg):
    return DeckRules(CORPUS, RuleConfig(**{"min_type_lists": 100, **cfg}))


def by_key(rules):
    return {(r.antecedent, r.consequent): r for r in rules}


def test_rule_statistics_on_a_hand_made_corpus():
    rules = by_key(mine(min_lift=1.5).rules([1, 2]))
    pair = rules[((1, 2), 3)]
    assert (pair.count, pair.support, pair.confidence, pair.lift) == (4, 0.4, 1.0, pytest.approx(2.5))
    single = rules[((1,), 3)]
    assert (single.count, single.confidence, single.lift) == (4, pytest.approx(4 / 6), pytest.approx(5 / 3))
    assert pair.stratum == single.stratum == GLOBAL and pair.lists == 10
    assert all(r.consequent not in (1, 2) for r in rules.values())  # never propose a card the deck has
    assert ((1, 2), 5) not in rules  # 1 and 2 are never together with 5: no pair rule without support
    # the thresholds bind: lift 2 drops the one-card rule, a count of 5 drops both
    assert set(by_key(mine().rules([1, 2]))) == {((1, 2), 3)}
    assert mine(min_count=5).rules([1, 2]) == []
    assert mine(min_confidence=1.01).rules([1, 2]) == []


def test_a_pair_rule_must_beat_its_one_card_rules():
    # 3 → 1 and 3 → 2 have confidence 1 already, so {3, x} → y adds nothing
    rules = mine(min_lift=1.0).rules([3, 1])
    assert ((1, 3), 2) not in by_key(rules) and ((3,), 2) in by_key(rules)


def test_type_frequencies_and_the_type_stratum():
    r = mine()
    assert r.frequencies("A") == {1: 1.0, 2: 1.0, 3: 1.0, **{100 + i: 0.25 for i in range(4)}}
    assert r.frequencies("nope") == {}
    assert r.typical_copies(3) == 3 and r.typical_copies(1) == 1 and r.typical_copies(999) == 1
    # a type rule: every A list runs 3 (lift 1 / 0.4); the stratum is still global (A has 4 < 100 lists)
    (t,) = [x for x in r.rules([1, 2], "A") if not x.antecedent]
    assert (t.consequent, t.stratum, t.confidence, t.lift, t.lists) == (3, "A", 1.0, pytest.approx(2.5), 4)
    assert r.stratum("A")[0] == GLOBAL
    # with enough lists the type's own lists are mined (and the lift is relative to them)
    strat = DeckRules(CORPUS, RuleConfig(min_type_lists=4, min_lift=1.0))
    assert strat.stratum("A")[0] == "A" and strat.stratum("B")[0] == GLOBAL
    inside = by_key(r for r in strat.rules([1], "A") if r.antecedent)
    assert inside[((1,), 3)].stratum == "A" and inside[((1,), 3)].lift == pytest.approx(1.0)


def test_package_completion():
    r = mine()
    assert [(x.consequent, x.confidence) for x in r.complete([1, 2])] == [(3, 1.0)]
    assert [x.consequent for x in r.complete([1, 2, 3])] == []
    # 5 runs in only 2 lists: below min_count, so the one-card rules decide (none pass here)
    assert r.complete([5]) == []
    # a package with an unseen card falls back to the rules of the cards it has
    assert [x.consequent for x in r.complete([1, 2, 99999])] == [3]


def test_rules_are_cached_per_environment(tmp_path):
    (tmp_path / "decks").mkdir()
    entries = []
    for k, (t, deck) in enumerate(CORPUS):
        (tmp_path / "decks" / f"d{k}.ydk").write_text(deck.to_ydk())
        entries.append({"type": t, "file": f"decks/d{k}.ydk"})
    (tmp_path / "deck_corpus.json").write_text(json.dumps({"decks": entries}))

    class Env:
        def __init__(self, fp):
            self.fp = fp

        def stamp(self):
            return {"environment": "md-test", "fingerprint": self.fp}

        def artifact_path(self, *parts):
            return tmp_path.joinpath(*parts)

    a = DeckRules.from_environment(Env("x"))
    assert DeckRules.from_environment(Env("x")) is a and len(a) == 10
    assert DeckRules.from_environment(Env("y")) is not a
    assert DeckRules.from_environment(Env("x"), RuleConfig(min_lift=3)) is not a


# a 40-card parent that runs 1 and 2 but not 3; 20 is lowest in the model's prior, 21 protected
PARENT = Deck(main=(1, 2, *range(10, 48)), extra=(900,))


def legal_40(deck):
    c = deck.counts()
    return len(deck.main) == 40 and max(c.values()) <= 3 and 20 in deck.main and 6 not in c  # 20 must stay, 6 banned


def test_rule_children_are_legal_protected_and_distinct():
    rules = DeckRules(CORPUS, RuleConfig(min_type_lists=100, min_lift=1.2, min_count=2, min_confidence=0.3))
    proposals = rules.additions(PARENT, "Z")
    assert {3, 5, 6} <= set(proposals) and proposals[3].rule.antecedent == (1, 2)
    model = CardValueModel(prior={20: -1.0, 11: -0.5})
    kids = rule_children(PARENT, "Z", rules, model, legal=legal_40, is_extra=lambda pw: pw >= 900,
                         rng=np.random.default_rng(0), children=4, max_bundle=3, protected={21})  # fmt: skip
    assert kids
    keys = {(tuple(sorted(k.deck.main)), tuple(sorted(k.deck.extra))) for k in kids}
    assert len(keys) == len(kids) and (tuple(sorted(PARENT.main)), (900,)) not in keys
    for k in kids:
        assert k.kind == "rules" and legal_40(k.deck) and 1 <= len(k.edits) <= 3
        for e in k.edits:
            assert e.into in proposals and e.into != 6  # the banned card is dropped, not forced in
            assert e.out not in (21, 20, 1, 2)  # protected, legality repair (20 drawn lowest), antecedents of 3
            assert e.section == "main"
    # 3 goes in with more than one copy (typically three) when the bundle has room; 11 (drawn low) goes out first
    assert any([e.into for e in k.edits].count(3) >= 2 for k in kids)
    assert all(k.edits[0].out == 11 for k in kids)
    # nothing to propose: no children
    empty = rule_children(PARENT, "C", DeckRules(CORPUS[-2:]), model, legal=legal_40, is_extra=lambda pw: False,
                          rng=np.random.default_rng(0))  # fmt: skip
    assert empty == []


def test_rule_children_in_the_evolution_step_record_their_generator(tmp_path):
    from tests.test_evolve import BASE, CONFIG, OPPONENTS, Env, make_lab

    # lists of BASE's type run 1 and 2 with the +12 pp card 50: the rules propose it
    corpus = [("Base", Deck(main=(1, 2, 50, 200 + i))) for i in range(4)] + [("X", Deck(main=(7, 300 + i)))
                                                                              for i in range(6)]  # fmt: skip
    lab = make_lab()
    config = RoundConfig(**{**CONFIG.__dict__, "rules": 2})
    with pytest.raises(ValueError, match="association rules"):
        Evolution(tmp_path / "none", Env()).run_round([Parent("corpus:base", BASE, "Base")], lab, config, OPPONENTS)
    lab.rules = DeckRules(corpus)
    ev = Evolution(tmp_path / "s", Env())
    report = ev.run_round([Parent("corpus:base", BASE, "Base")], lab, config, OPPONENTS)
    lineage = ev.lineage()
    rows = [r for r in lineage if r["generator"] == "rules"]
    assert rows and all(r["kind"] == "rules" for r in rows)
    assert all(any(e["into"] == 50 for e in r["edits"]) for r in rows)
    assert {r["generator"] for r in lineage} == {"mutation", "rules"}
    assert all(r["learned"][0]["source"] == "first_batch" for r in rows)  # rule children feed the signal library too
    g = report["generators"]["round"]["rules"]
    assert g["children"] == len(rows) and g["accepted_rate"] == g["accepted"] / g["children"]
    assert sum(x["children"] for x in report["generators"]["round"].values()) == report["children"]
    assert "rules " in (tmp_path / "s" / "rounds" / "0001" / "report.txt").read_text()
