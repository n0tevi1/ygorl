"""Deck evolution step (ygorl.build.evolve) and its MAP-Elites archive (ygorl.build.archive), with stand-in games."""

import json
import zlib

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

    def __init__(self, seed, fail_after=None, effects=None, copy=1.0):
        self.seed, self.games, self.fail_after, self.calls = seed, 0, fail_after, 0
        self.effects = EFFECT if effects is None else effects
        self.copy = (
            copy  # the chance a game follows the shared uniform; otherwise a deck-own one (less than perfect CRN)
        )

    def play(self, jobs, per_game=False):
        assert per_game
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("machine reclaimed")
        out = []
        for deck, pairs in jobs:
            p = 0.45 + sum(self.effects.get(c, 0.0) for c in deck.main)
            u = np.array([np.random.default_rng([self.seed % 2**32, k]).random(2) for k in pairs]).reshape(-1, 2)
            if self.copy < 1:
                own = zlib.crc32(deck_key(deck).encode())
                r = np.array([np.random.default_rng([self.seed % 2**32, k, own]).random(4) for k in pairs])
                u = np.where(r.reshape(-1, 4)[:, :2] < self.copy, u, r.reshape(-1, 4)[:, 2:])
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


# the bundled (non-cold-start) path; cold start has its own tests
CONFIG = RoundConfig(
    informed=4, explore=2, batch=10, max_pairs=120, look=40, cap=200, archive_min_pairs=10, seed=7, cold_candidates=0
)
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
    # one first-batch observation per child plus the validated child's fresh pairs; calibration from the first batch
    assert len(again.model.observations) == a["children"] + 1
    assert again.calibration.report()["model_gain"]["pairs"] == 6
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
        assert r["all_pairs"]["pairs"] >= r["search"]["pairs"] and r["all_pairs"]["stderr"] > 0
        assert r["games"] == 2 * r["all_pairs"]["pairs"] and "biased" in r["all_pairs"]["note"]
        first = r["learned"][0]
        assert first["source"] == "first_batch" and first["pairs"] == CONFIG.batch and first["stderr"] > 0
        assert r["checkpoint"] == {"path": "ckpt.pt", "sha256": "abc", "update": 400}
        assert r["environment"] == Env().stamp() and r["time"] and isinstance(r["accepted"], bool)
        assert r["archive"]["status"] in ("new", "improved", "rejected")
    chosen = [r for r in lines if r["validation"]]
    assert len(chosen) == 1 and {"pairs", "decision", "diff", "lower"} <= chosen[0]["validation"].keys()
    assert [x["source"] for x in chosen[0]["learned"]] == ["first_batch", "validation"]
    assert chosen[0]["learned"][1]["pairs"] == chosen[0]["validation"]["pairs"]


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


def test_the_signal_library_learns_only_unbiased_differences(tmp_path):
    """Close effects make the race's choice depend on noise: the chosen child's all-pairs difference is then biased
    upward, while what the card-value model receives (first batches, fresh validation pairs) is right on average."""
    effects = {50: 0.03, 51: 0.02, 52: 0.01, 53: -0.01, 54: -0.02}

    def truth(into, out):
        return sum(effects.get(c, 0.0) for c in into) - sum(effects.get(c, 0.0) for c in out)

    lab = make_lab()
    lab.evaluator = lambda seed, opponents: Games(seed, effects=effects, copy=0.8)
    config = RoundConfig(informed=4, explore=2, max_bundle=1, batch=10, max_pairs=60, look=20, cap=40,
                         archive_min_pairs=10, cold_candidates=0)  # fmt: skip
    learned, chosen_all, chosen_fresh = [], [], []
    for seed in range(40):
        ev = Evolution(tmp_path / str(seed), Env())
        ev.run_round(PARENTS, lab, RoundConfig(**{**config.__dict__, "seed": seed}), OPPONENTS)
        learned += [o.diff - truth(o.into, o.out) for o in ev.model.observations]
        for r in ev.lineage():
            if r["validation"]:
                t = truth([e["into"] for e in r["edits"]], [e["out"] for e in r["edits"]])
                chosen_all.append(r["all_pairs"]["diff"] - t)
                chosen_fresh.append(r["learned"][1]["diff"] - t)
    se = np.std(learned) / np.sqrt(len(learned))
    assert abs(np.mean(learned)) < 3 * se + 1e-3
    assert abs(np.mean(chosen_fresh)) < 3 * np.std(chosen_fresh) / np.sqrt(len(chosen_fresh)) + 1e-3
    assert np.mean(chosen_all) > np.mean(chosen_fresh) + 0.01  # the all-pairs number is selection-biased


def test_appends_cut_a_torn_last_line_first(tmp_path):
    from ygorl.build.evolve import GameLog, append_lines

    path = tmp_path / "log.jsonl"
    append_lines(path, ['{"a": 1}'])
    with open(path, "a") as f:
        f.write('{"a": 2, "cut by a cra')  # a crash in the middle of a write
    append_lines(path, ['{"a": 3}'])
    assert [json.loads(line) for line in path.read_text().splitlines()] == [{"a": 1}, {"a": 3}]
    only = tmp_path / "only.jsonl"
    only.write_text('{"frag')
    append_lines(only, ['{"b": 1}'])
    assert only.read_text() == '{"b": 1}\n'
    # a game log with a torn line reads what it can and appends cleanly
    games = tmp_path / "games.jsonl"
    log = GameLog(Games(1), games)
    log.play([(BASE, range(3))])
    with open(games, "a") as f:
        f.write('{"deck": "x", "pai')
    again = GameLog(Games(1), games)
    assert len(again.games_of(BASE)) == 3
    again.play([(BASE, range(5))])
    assert len(GameLog(Games(1), games).games_of(BASE)) == 5 and again.played == 4


def test_elites_scored_under_another_opponent_mix_are_stale():
    a = DeckArchive({"x": (2, (0.0, 1.0))})
    d1, d2, d3 = Deck(main=(1,)), Deck(main=(2,)), Deck(main=(3,))
    assert a.add("d1", d1, 0.8, {"x": 0.1}, mix="m1").status == "new"
    assert a.add("d2", d2, 0.5, {"x": 0.1}, mix="m1").status == "rejected"
    assert a.stale("m2") == 1 and a.stale("m1") == 0
    fresh = a.add("d2", d2, 0.5, {"x": 0.1}, mix="m2")  # lower objective, but the elite's score is from another mix
    assert fresh.status == "refreshed" and fresh.admitted and fresh.replaced == "d1" and a.stale("m2") == 0
    assert a.add("d3", d3, 0.4, {"x": 0.1}, mix="m2").status == "rejected"  # pyribs dropped the stale 0.8 too
    assert a.add("d3", d3, 0.6, {"x": 0.1}, mix="m2").status == "improved"
    back = DeckArchive.from_dict(json.loads(json.dumps(a.to_dict())))
    assert back.to_dict() == a.to_dict() and back.elite(fresh.cell)["mix"] == "m2"


def test_the_archive_objective_is_a_lower_bound_and_the_mix_is_recorded(tmp_path):
    ev, report = run(tmp_path)
    for e in ev.archive.elites():
        assert e["mix"] and e["objective"] < 1.0
    assert report["archive"]["stale"] == 0
    few = Evolution(tmp_path / "few", Env())
    few.run_round(PARENTS, make_lab(), RoundConfig(**{**CONFIG.__dict__, "archive_min_pairs": 10_000}), OPPONENTS)
    assert len(few.archive) == 0
    assert {v["status"] for v in few.archive_applied.values()} == {"too_few_pairs"}
    # a new opponent mix in the next round makes the old elites stale
    ev2 = Evolution(tmp_path, Env())
    other = [Opponent("meta2", Deck(main=(6,) * 40), 1.0)]
    report2 = ev2.run_round(PARENTS, make_lab(), CONFIG, other)
    assert report2["archive"]["stale"] < report["archive"]["elites"] + report2["archive"]["elites"]
    assert any(v["status"] == "refreshed" for k, v in ev2.archive_applied.items() if k.startswith("r0002"))


def test_manifest_edits_made_during_a_round_survive(tmp_path):
    lab = make_lab()
    manifest = tmp_path / "manifest.json"
    (tmp_path / "decks").mkdir(parents=True)
    (tmp_path / "decks" / "hand.ydk").write_text(BASE.to_ydk())

    def evaluator(seed, opponents):  # someone edits the manifest while the round runs
        data = {"format": "ygorl-deck-pool", "version": 1, "environment": Env().stamp(),
                "decks": [{"id": "hand", "file": "decks/hand.ydk", "status": "active", "weight": 2.0}]}  # fmt: skip
        manifest.write_text(json.dumps(data))
        return Games(seed)

    lab.evaluator = evaluator
    Evolution(tmp_path, Env()).run_round(PARENTS, lab, CONFIG, OPPONENTS)
    decks = json.loads(manifest.read_text())["decks"]
    assert [d["id"] for d in decks][0] == "hand" and decks[0]["weight"] == 2.0
    assert any(d["id"].startswith("evo-") for d in decks)


# ------------------------------------------------------------------ crossover children (#141)


def test_the_evolution_step_evaluates_crossover_children_and_records_both_parents(tmp_path):
    config = RoundConfig(**{**CONFIG.__dict__, "crossover": 2})
    ev, first = run(tmp_path, config=config)
    assert "crossover" not in first["generators"]["round"]  # the archive was empty when the round began
    assert len(ev.archive) >= 2
    again = Evolution(tmp_path, Env())
    report = again.run_round(PARENTS, make_lab(), config, OPPONENTS)
    assert report["generators"]["round"]["crossover"]["children"] == 2
    assert report["generators"]["cumulative"]["mutation"]["children"] == 12
    crossed = [r for r in again.lineage() if r["generator"] == "crossover"]
    assert len(crossed) == 2
    elites = {e["id"]: e for e in ev.archive.elites()}
    for r in crossed:
        assert r["round"] == 2 and r["kind"] == "crossover" and r["parent"]["id"] == "corpus:base"
        assert set(r["mate"]) == {"id", "cell", "key", "distance"} and r["mate"]["id"] in elites
        assert r["mate"]["cell"] == elites[r["mate"]["id"]]["cell"] and r["key"] != r["mate"]["key"]
        assert r["edits"] and r["search"]["pairs"] >= CONFIG.batch  # same evaluation path: Thompson's first batch
        assert r["learned"] and r["learned"][0]["source"] == "first_batch"
    assert all(r["generator"] == "mutation" and r["mate"] is None for r in again.lineage() if r["kind"] != "crossover")


def test_crossover_is_off_by_default(tmp_path):
    ev, _ = run(tmp_path)
    again = Evolution(tmp_path, Env())
    report = again.run_round(PARENTS, make_lab(), CONFIG, OPPONENTS)
    assert set(report["generators"]["cumulative"]) == {"mutation"}
    assert not any(r["kind"] == "crossover" for r in again.lineage())


def test_engine_members_are_protected_until_the_model_has_evidence(tmp_path):
    lab = make_lab()
    lab.engine = lambda deck: {1, 2, 10}
    ev, report = run(tmp_path / "a", lab=lab)
    (result,) = ev._round_state(1)["results"]
    assert {1, 2, 10} <= set(result["protected"]) and result["engine"] == [1, 2, 10]
    assert all(e["out"] not in (1, 2, 10) for r in ev.lineage() for e in r["edits"])
    # two observations of card 1 in the parent's type: 1 may go out now (after every non-engine card), 2 and 10 not
    ev = Evolution(tmp_path / "b", Env())
    for _ in range(2):
        ev.model.add(Observation("Base", (50,), (1,), 0.0, 0.05))
    ev.run_round(PARENTS, lab, CONFIG, OPPONENTS)
    (result,) = ev._round_state(1)["results"]
    assert 1 not in result["protected"] and {2, 10} <= set(result["protected"])


def test_a_cold_start_round_screens_many_single_swaps_then_bundles_once_evidence_exists(tmp_path):
    config = RoundConfig(**{**CONFIG.__dict__, "cold_candidates": 12, "cold_keep": 3, "cold_min_obs": 5})
    ev, report = run(tmp_path, config=config)
    (p,) = report["per_parent"]
    assert p["cold_start"] and "(cold start)" in (tmp_path / "rounds" / "0001" / "report.txt").read_text()
    lineage = ev.lineage()
    assert len(lineage) == report["children"] >= 10
    assert all(len(r["edits"]) == 1 for r in lineage)  # single swaps only
    assert all(r["screen"]["pairs"] == CONFIG.batch for r in lineage)
    assert sum(r["screen"]["kept"] for r in lineage) == 3 == sum(r["search"] is not None for r in lineage)
    assert all(r["search"] is None for r in lineage if not r["screen"]["kept"])
    # every screened child's first batch feeds the signal library (non-adaptive), the chosen child's validation too
    assert all(r["learned"][0]["source"] == "first_batch" for r in lineage)
    assert len(ev.model.observations) == len(lineage) + 1
    # the multiplicity counts every screened child
    assert ev._round_state(1)["results"][0]["candidates"] == len(lineage)
    assert ev._round_state(1)["results"][0]["stop_threshold"] == pytest.approx(1 - 0.05 / len(lineage))
    g = report["generators"]["round"]["mutation"]
    assert g["first_batch_children"] == len(lineage) and g["first_batch_mean"] is not None
    # the model now has evidence about the type: the next round bundles
    report2 = ev.run_round(PARENTS, make_lab(), config, OPPONENTS)
    assert not report2["per_parent"][0]["cold_start"]
    assert any(len(r["edits"]) > 1 for r in ev.lineage() if r["round"] == 2)


def test_warm_start_adds_each_observation_once(tmp_path):
    ev = Evolution(tmp_path, Env())
    items = [("warm:m2:x:0", Observation("Base", (65957473,), (1,), -0.05, 0.01)),
             ("warm:m2:x:1", Observation("Base", (65957473,), (2,), 0.01, 0.01))]  # fmt: skip
    assert ev.warm_start(items) == 2 and ev.warm_start(items) == 0
    again = Evolution(tmp_path, Env())
    assert len(again.model.observations) == 2 and again.warm_start(items) == 0
    assert again.model.evidence(1, "Base") == 1 and again.model.type_evidence("Base") == 2
