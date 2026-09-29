"""Deck evolution step (ygorl.build.evolve) and its MAP-Elites archive (ygorl.build.archive), with stand-in games."""

import json

import numpy as np
import pytest

from ygorl.build.archive import DeckArchive, deck_descriptors
from ygorl.build.evolve import Evolution, Lab, Opponent, Parent, RoundConfig, deck_key, opponent_mix
from ygorl.build.signals import Calibration, CardValueModel, Observation
from ygorl.build.synergy_graph import Edge, SynergyGraph
from ygorl.cards.ydk import Deck
from ygorl.data.environment import EnvironmentConfigError
from ygorl.engine import constants as C

EXTRA = {900, 901}
BASE = Deck(main=tuple([1] * 3 + [2] * 3 + list(range(10, 44))), extra=(900,), name="base")
POOL = [50, 51, 52, 53, 54, 901]
EFFECT = {50: 0.12, 51: 0.06, 54: -0.08}  # per copy, on the win probability of every game


class Env:
    """Stand-in for an Environment: only its stamp matters here."""

    def __init__(self, version="md-test"):
        self.version = version

    def stamp(self):
        return {"environment": self.version, "fingerprint": f"fp-{self.version}"}

    def check_stamp(self, stamp):
        if stamp.get("environment") != self.version:
            raise EnvironmentConfigError(f"artifact belongs to environment {stamp.get('environment')!r}")


class Games:
    """Stand-in paired evaluator: game ``g`` of pair ``k`` is won when a uniform shared by every deck (common random
    numbers) is below 0.45 + the deck's card effects; going first adds 0.05."""

    def __init__(self, seed, fail_after=None):
        self.seed, self.games, self.fail_after, self.calls = seed, 0, fail_after, 0

    def play(self, jobs, per_game=False):
        assert per_game
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("machine reclaimed")
        out = []
        for deck, pairs in jobs:
            p = 0.45 + sum(EFFECT.get(c, 0.0) for c in deck.main)
            u = np.array([np.random.default_rng([self.seed % 2**32, k]).random(2) for k in pairs]).reshape(-1, 2)
            out.append((u < p + np.array([0.05, 0.0])).astype(float))
            self.games += 2 * len(pairs)
        return out


def legal(deck):
    c = deck.counts()
    return len(deck.main) == 40 and max(c.values()) <= 3


def make_lab(fail_after=None, played=None):
    def evaluator(seed, opponents):
        g = Games(seed, fail_after)
        if played is not None:
            played.append(g)
        return g

    return Lab(evaluator=evaluator, legal=legal, is_extra=lambda pw: pw in EXTRA, pool=lambda parent: POOL,
               descriptors=lambda d: {"hand_traps": float(d.main.count(52)), "combo_length": 1.0, "brick_rate": 0.3},
               checkpoint={"path": "ckpt.pt", "sha256": "abc", "update": 400})  # fmt: skip


CONFIG = RoundConfig(informed=4, explore=2, batch=10, max_pairs=120, look=40, cap=200, seed=7)
OPPONENTS = [Opponent("meta", Deck(main=(5,) * 40), 1.0)]
PARENTS = [Parent("corpus:base", BASE, "Base")]


def run(tmp, **kw):
    ev = Evolution(tmp, Env())
    return ev, ev.run_round(PARENTS, kw.pop("lab", None) or make_lab(), kw.pop("config", CONFIG), OPPONENTS, **kw)


def _strip_times(obj):
    if isinstance(obj, dict):
        return {k: _strip_times(v) for k, v in obj.items() if k not in ("time", "started", "finished")}
    if isinstance(obj, list):
        return [_strip_times(v) for v in obj]
    return obj


def test_a_round_is_reproducible_and_accepts_a_better_child(tmp_path):
    ev_a, a = run(tmp_path / "a")
    ev_b, b = run(tmp_path / "b")
    assert _strip_times(a) == _strip_times(b)
    assert _strip_times(ev_a.lineage()) == _strip_times(ev_b.lineage())
    assert (tmp_path / "a" / "manifest.json").read_text() == (tmp_path / "b" / "manifest.json").read_text()
    assert a["children"] == 6 and a["games"] > 0 and a["l0"] == "off" and a["l0_would_remove"] == 0
    assert a["accepted"] == 1  # a +12 pp card is found and confirmed
    manifest = json.loads((tmp_path / "a" / "manifest.json").read_text())
    assert manifest["environment"] == Env().stamp()
    (entry,) = manifest["decks"]
    assert entry["status"] == "probation" and entry["parent"] == "corpus:base" and entry["id"].startswith("evo-")
    assert (tmp_path / "a" / entry["file"]).is_file()
    accepted = next(r for r in ev_a.lineage() if r["accepted"])
    assert any(e["into"] == 50 for e in accepted["edits"])
    # every evaluated child fed the card-value model and the calibration table, and the state reloads
    again = Evolution(tmp_path / "a", Env())
    assert len(again.model.observations) == a["children"] and again.calibration.report()["model_gain"]["pairs"] == 6
    assert again.archive.to_dict() == ev_a.archive.to_dict() and len(again.archive) >= 1
    assert a["archive"]["admitted_this_round"] >= 1 and a["pool"]["decks"] == 1
    assert (tmp_path / "a" / "rounds" / "0001" / "report.txt").read_text().startswith("round 1")
    # the next round starts from the accepted deck's lineage and numbers itself 2
    assert again.pick_parents([*PARENTS, *again.pool()], 1)[0].id == entry["id"]  # never a parent yet


def test_lineage_records_are_complete(tmp_path):
    ev, report = run(tmp_path)
    lines = ev.lineage()
    assert len(lines) == report["children"]
    for r in lines:
        assert r["parent"]["id"] == "corpus:base" and r["edits"] and all({"out", "into", "section"} <= e.keys()
                                                                           for e in r["edits"])  # fmt: skip
        assert set(r["predicted"]) == {"mean", "sd"} and r["predicted"]["sd"] > 0
        assert r["search"]["pairs"] > 0 and len(r["search"]["ci"]) == 2 and r["search"]["diff"] is not None
        assert r["measured"]["pairs"] >= r["search"]["pairs"] and r["measured"]["stderr"] > 0
        assert r["games"] == 2 * r["measured"]["pairs"]
        assert r["checkpoint"] == {"path": "ckpt.pt", "sha256": "abc", "update": 400}
        assert r["environment"] == Env().stamp() and r["time"] and isinstance(r["accepted"], bool)
        assert r["archive"]["status"] in ("new", "improved", "rejected")
    chosen = [r for r in lines if r["validation"]]
    assert len(chosen) == 1 and {"pairs", "decision", "diff", "lower"} <= chosen[0]["validation"].keys()


def test_an_interrupted_round_resumes_without_replaying_its_games(tmp_path):
    _, whole = run(tmp_path / "whole")
    with pytest.raises(RuntimeError):
        run(tmp_path / "cut", lab=make_lab(fail_after=1))
    newer = make_lab()
    newer.checkpoint = {**newer.checkpoint, "sha256": "def"}
    with pytest.raises(ValueError, match="same checkpoint"):
        run(tmp_path / "cut", lab=newer)
    played = []
    ev, resumed = run(tmp_path / "cut", lab=make_lab(played=played))
    assert _strip_times(resumed) == _strip_times(whole)
    assert 0 < played[0].games < whole["games"]  # the first calls came from the game log
    assert ev.rounds() == [1]


def test_budget_stops_starting_parents(tmp_path):
    two = [PARENTS[0], Parent("corpus:other", BASE, "Base")]
    ev = Evolution(tmp_path, Env())
    report = ev.run_round(two, make_lab(), RoundConfig(**{**CONFIG.__dict__, "budget": 1}), OPPONENTS)
    assert report["parents"] == 1 and report["parents_planned"] == 2 and report["budget_stop"]


def test_environment_mismatch_is_an_error(tmp_path):
    run(tmp_path)
    with pytest.raises(EnvironmentConfigError, match="manifest"):
        Evolution(tmp_path, Env("other"))
    ev = Evolution(tmp_path / "fresh", Env())
    ev.check_checkpoint(Env().stamp())
    with pytest.raises(EnvironmentConfigError, match="checkpoint"):
        ev.check_checkpoint(Env("other").stamp(), "x.pt")
    with pytest.raises(EnvironmentConfigError, match="checkpoint"):
        ev.check_checkpoint(None, "default-rules.pt")  # trained under the default rules
    # a separate manifest from another environment is refused too
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"format": "ygorl-deck-pool", "version": 1, "environment": Env("x").stamp(),
                                 "decks": []}))  # fmt: skip
    with pytest.raises(EnvironmentConfigError):
        Evolution(tmp_path / "fresh2", Env(), manifest=other)


def test_archive_fills_empty_cells_and_replaces_a_worse_elite():
    a = DeckArchive({"x": (2, (0.0, 1.0)), "y": (2, (0.0, 1.0))})
    d1, d2, d3 = Deck(main=(1,)), Deck(main=(2,)), Deck(main=(3,))
    first = a.add("d1", d1, 0.5, {"x": 0.1, "y": 0.1})
    assert first.status == "new" and first.admitted and len(a) == 1 and a.coverage == 0.25
    worse = a.add("d2", d2, 0.4, {"x": 0.2, "y": 0.3})  # same cell, lower objective
    assert worse.status == "rejected" and not worse.admitted and worse.cell == first.cell
    tie = a.add("d2", d2, 0.5, {"x": 0.2, "y": 0.3})
    assert tie.status == "rejected"
    better = a.add("d3", d3, 0.6, {"x": 0.3, "y": 0.4})
    assert better.status == "improved" and better.replaced == "d1" and a.elite(first.cell)["id"] == "d3"
    other = a.add("d2", d2, 0.1, {"x": 0.9, "y": 5.0})  # beyond the range: the edge cell, a new niche
    assert other.status == "new" and other.cell != first.cell and len(a) == 2
    assert a.add("bad", d2, float("nan"), {"x": 0.1, "y": 0.1}).status == "invalid"
    back = DeckArchive.from_dict(json.loads(json.dumps(a.to_dict())))
    assert back.to_dict() == a.to_dict() and back.replaced == ["d1"]


def test_deck_descriptor_proxies_from_the_synergy_graph():
    deck = Deck(main=(1, 2, 3) + (9,) * 37)
    nodes = {p: {} for p in (1, 2, 3, 4, 9)}
    edges = [Edge(1, 2, "search", C.LOCATION_DECK, 3, "x"), Edge(2, 3, "special_summon", C.LOCATION_DECK, 3, "x"),
             Edge(3, 4, "search", C.LOCATION_DECK, 3, "x")]  # 4 is not in the deck  # fmt: skip
    d = deck_descriptors(deck, hand_traps={9}, graph=SynergyGraph(nodes, edges))
    assert d["hand_traps"] == 37 and d["combo_length"] == 2  # 1 -> 2 -> 3
    # starters 1 and 2: P(neither in 5 of 40) = C(38, 5) / C(40, 5)
    assert d["brick_rate"] == pytest.approx((35 * 34) / (40 * 39))
    assert deck_descriptors(deck)["brick_rate"] == 1.0


def test_opponents_mix_nash_weights_with_meta_shares():
    a, b, e = Deck(main=(1,)), Deck(main=(2,)), Deck(main=(3,))
    meta = [("A", a, 0.6), ("B", b, 0.2)]
    only = opponent_mix(meta)
    assert [(o.name, round(o.weight, 3)) for o in only] == [("A", 0.75), ("B", 0.25)]
    mixed = opponent_mix(meta, nash={"B": 1.0, "evo-1": 1.0, "A": 0.0}, decks={"evo-1": e}, nash_share=0.5)
    w = {o.name: o.weight for o in mixed}
    assert w == pytest.approx({"A": 0.375, "B": 0.125 + 0.25, "evo-1": 0.25})
    assert next(o for o in mixed if o.name == "evo-1").deck == e
    with pytest.raises(ValueError, match="neither"):
        opponent_mix(meta, nash={"ghost": 1.0})


def test_signal_library_state_round_trips():
    m = CardValueModel(prior={5: 0.01})
    m.add(Observation("T", (5, 6), (7, 8), 0.03, 0.01))
    back = CardValueModel.from_dict(json.loads(json.dumps(m.to_dict())))
    assert back.gain([5], [7], "T") == pytest.approx(m.gain([5], [7], "T"))
    cal = Calibration({"opening_effect": 0.0}, min_pairs=3)
    for x in range(5):
        cal.record("model_gain", x, x)
    assert Calibration.from_dict(json.loads(json.dumps(cal.to_dict()))).report() == cal.report()


def test_deck_key_follows_the_card_order():
    assert deck_key(BASE) != deck_key(Deck(main=tuple(reversed(BASE.main)), extra=BASE.extra))


def test_opening_hands_match_the_dealt_games():
    pytest.importorskip("torch")
    from ygorl.build.tuner import PairedEvaluator

    ev = PairedEvaluator(env_factory=None, policy=None, opponents=[Deck(main=(5,) * 40)], weights=[1.0], seed=3)
    hands = ev.opening_hands(BASE, range(4))
    specs = ev.specs(BASE, range(4))
    assert hands == [tuple(s.deck_a.main[-5:]) for s in specs[::2]]
    assert all(s.deck_a.main == t.deck_a.main for s, t in zip(specs[::2], specs[1::2], strict=True))


@pytest.mark.parametrize("mode", ["shadow", "on"])
def test_the_l0_screen_counts_in_shadow_and_removes_when_on(tmp_path, mode):
    lab = make_lab()
    lab.screen = lambda ev, parent, children: [-0.1 if 50 not in d.main else 0.0 for d in children]
    ev = Evolution(tmp_path, Env())
    report = ev.run_round(PARENTS, lab, RoundConfig(**{**CONFIG.__dict__, "l0": mode}), OPPONENTS)
    rows = ev.lineage()
    flagged = [r for r in rows if r["screened"]]
    assert report["l0_would_remove"] == len(flagged) > 0 and all(r["l0"] is not None for r in rows)
    if mode == "shadow":
        assert all(r["search"] for r in rows)  # counted, still played
    else:
        assert all(r["search"] is None and r["games"] == 0 for r in flagged)
        assert all(r["search"] for r in rows if not r["screened"])
    with pytest.raises(ValueError, match="screen"):
        Evolution(tmp_path / "x", Env()).run_round(PARENTS, make_lab(), RoundConfig(l0="on"), OPPONENTS)


def test_the_opening_value_screen_compares_children_with_the_parent_on_the_same_pairs():
    from ygorl.build.control import opening_value_screen

    class Specs:
        def specs(self, deck, pairs):
            return [(deck, k) for k in pairs for _ in range(2)]

    got = opening_value_screen(lambda specs: [0.2 * d.main.count(50) + 0.01 * k for d, k in specs], Specs(), BASE,
                               [Deck(main=(50,) * 2), BASE], hands=8)  # fmt: skip
    assert got == pytest.approx([0.2, 0.0])


def test_diagnosis_is_the_parents_baseline_and_feeds_only_the_calibration(tmp_path):
    class Diagnosed(Games):
        def opening_hands(self, deck, pairs):
            return [tuple(np.random.default_rng([k]).permutation(deck.main)[-5:].tolist()) for k in pairs]

    lab = make_lab()
    lab.evaluator = lambda seed, opponents: Diagnosed(seed)
    ev = Evolution(tmp_path, Env())
    report = ev.run_round(PARENTS, lab, RoundConfig(**{**CONFIG.__dict__, "diagnose_pairs": 30}), OPPONENTS)
    assert report["per_parent"][0]["parent_games"] >= 60
    cal = ev.calibration.report()
    assert cal["opening_effect"]["pairs"] == report["children"] and cal["opening_effect"]["weight"] == 0.0
    assert ev.model.prior == {}  # weight 0: the opening-hand effect does not move the prior
