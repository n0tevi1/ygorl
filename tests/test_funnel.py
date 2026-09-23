"""ygorl.build.funnel and ygorl.build.first_turn: funnel stage 1 and its real-first-turn validation (T5.6).

Aggregation, the filter, the paired test and the first-turn players run without the solver binary
(a fake solver script stands in for it where a run is needed); the end-to-end test uses the real
binary when ``tools/build_combo_solver.sh`` has built it and is skipped otherwise.
"""

import math
from dataclasses import replace
from pathlib import Path

import pytest

from ygorl.agents import GreedyAgent
from ygorl.build.first_turn import (
    ExplorerAgent,
    explore,
    first_turn_duel,
    match_action,
    play_first_turn,
    real_config,
    replay_line,
)
from ygorl.build.funnel import (
    ASH_BLOSSOM,
    FunnelConfig,
    FunnelFilter,
    FunnelResult,
    HandOutcome,
    evaluate_deck,
    evaluate_genotype,
    fire_survives,
    mcnemar_exact,
    normalize_targets,
    opening_hands,
    paired_table,
)
from ygorl.cards.cdb import CardDB
from ygorl.cards.ydk import load_ydk
from ygorl.engine import constants as C
from ygorl.engine import messages as M
from ygorl.engine.actions import Action
from ygorl.engine.duel import Duel, DuelConfig, DuelSession
from ygorl.engine.replay import Replay, load_yrp
from ygorl.solver import Demonstration, SolverNotFound, convert_line, find_solver
from ygorl.solver.batch import PASSIVE_OPPONENT, sample_hand

DECKS = Path(__file__).parent / "decks"
PURRELY = load_ydk(DECKS / "purrely.ydk")
EPURRELY_HAPPINESS = 52645235


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


# ------------------------------------------------------------------ pure helpers


def test_normalize_targets():
    assert normalize_targets(["52645235"]) == (("52645235@atk",),)
    assert normalize_targets(["1225009", "5380979@szone:fd"]) == (("1225009@atk", "5380979@szone:fd"),)
    assert normalize_targets([["1"], ["2", "3@grave"]]) == (("1@atk",), ("2@atk", "3@grave"))
    for bad in ([], "52645235", [["1"], "2"], [[]], ["Ash Blossom"]):
        with pytest.raises(ValueError):
            normalize_targets(bad)


def test_mcnemar_exact_and_paired_table():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(3, 3) == 1.0
    assert mcnemar_exact(0, 5) == pytest.approx(2 / 32)
    assert mcnemar_exact(9, 1) == mcnemar_exact(1, 9) == pytest.approx(2 * 11 / 1024)
    t = paired_table([True, True, False, False, True], [True, False, True, False, True])
    assert (t["n"], t["both"], t["only_a"], t["only_b"], t["neither"]) == (5, 2, 1, 1, 1)
    assert t["rate_a"] == t["rate_b"] == pytest.approx(0.6) and t["agreement"] == pytest.approx(0.6) and t["p_value"] == 1.0
    with pytest.raises(ValueError):
        paired_table([True], [])


def test_fire_survives():
    assert fire_survives({"status": "no_window", "solver": {"windows": 0}}) is True
    assert fire_survives({"status": "solved", "solver": {"windows": 3, "converted": 3}}) is True
    assert fire_survives({"status": "solved", "solver": {"windows": 3, "converted": 1}}) is False  # the opponent picks the window
    assert fire_survives({"status": "unsolved", "solver": {"windows": 2, "converted": 0}}) is False
    assert fire_survives({"status": "error", "solver": {}}) is None


def _hand(i, status, actions=None, fire=None, solver_s=5.0):
    fires = {} if fire is None else {ASH_BLOSSOM: {"status": "x", "windows": 2, "converted": 2 if fire else 1, "survives": fire}}
    return HandOutcome(i, 100 + i, [1, 2, 3, 4, 5], status, 0 if status == "solved" else None, actions,
                       None if actions is None else 4 * actions, 1, fires, solver_s)  # fmt: skip


def _result(hands, fire=(ASH_BLOSSOM,), budget=120.0):
    cfg = {"fire": list(fire), "budget_s": budget}
    return FunnelResult({"name": "d", "main": [], "extra": []}, [["1@atk"]], cfg, hands, planned_hands=len(hands))


def test_result_rates_descriptors_and_json():
    hands = [_hand(0, "solved", 6, True), _hand(1, "solved", 10, False), _hand(2, "brick"), _hand(3, "brick"),
             _hand(4, "error", solver_s=0.5)]  # fmt: skip
    r = _result(hands)
    assert (r.valid_hands, r.solved, r.bricks, r.errors) == (4, 2, 2, 1)
    assert r.best_line_rate == 0.5 and r.brick_rate == 0.5
    lo, hi = r.brick_interval()
    assert lo < 0.5 < hi
    assert r.survival(ASH_BLOSSOM) == 0.25  # opens and survives, over the non-error hands
    assert r.survival(ASH_BLOSSOM, given_line=True) == 0.5
    assert r.hand_trap_survival == 0.25
    assert r.window_recovery(ASH_BLOSSOM) == pytest.approx(3 / 4)
    assert r.combo_length() == {"n": 2, "mean": 8.0, "median": 8.0, "min": 6, "max": 10}
    assert r.combo_length("combo_decisions")["mean"] == 32.0
    assert r.solver_s == 20.5 and r.within_budget
    assert r.descriptors() == {"brick_rate": 0.5, "combo_length": 8.0, "hand_trap_survival": 0.25}
    back = FunnelResult.from_json(r.to_json())
    assert back.hands == r.hands and back.summary() == r.summary()
    assert r.to_json()["summary"]["brick_rate_ci95"] == [round(lo, 4), round(hi, 4)]

    empty = _result([_hand(0, "brick"), _hand(1, "brick")], fire=(), budget=5.0)
    assert empty.hand_trap_survival is None and not empty.within_budget
    d = empty.descriptors()
    assert d["brick_rate"] == 1.0 and math.isnan(d["combo_length"]) and math.isnan(d["hand_trap_survival"])


def test_filter():
    gate = FunnelFilter(max_brick_rate=0.5, min_hand_trap_survival=0.3)
    ok = _result([_hand(0, "solved", 5, True), _hand(1, "solved", 5, True), _hand(2, "brick")])
    assert gate.passes(ok) and gate.reasons(ok) == []
    weak = _result([_hand(0, "solved", 5, False), _hand(1, "solved", 5, False), _hand(2, "brick")])
    assert gate.reasons(weak) == ["hand-trap survival 0.00 < 0.30"]
    bricky = _result([_hand(0, "solved", 5, True), _hand(1, "brick"), _hand(2, "brick")])
    assert any("brick rate 0.67" in s for s in gate.reasons(bricky))
    assert FunnelFilter().reasons(_result([_hand(0, "error")])) == ["no hand evaluated without error"]
    assert not gate.hopeless(6, 12) and gate.hopeless(7, 12)


# ------------------------------------------------------------------ evaluate_deck with a stand-in solver


QUIET_SOLVER = """#!/bin/sh
echo '@event {"type":"phase","phase":"search"}'
echo 'at best 0 of the 1 target cards'
exit 0
"""


def _fake_solver(tmp_path, text=QUIET_SOLVER):
    fake = tmp_path / "combosolver"
    fake.write_text(text)
    fake.chmod(0o755)
    return fake


def test_evaluate_deck_bricks_stop_early_and_are_paired_across_decks(tmp_path):
    cfg = FunnelConfig(hands=6, solve_ms=100, binary=_fake_solver(tmp_path))
    seen = []
    full = evaluate_deck(PURRELY, ["52645235"], cfg, scratch=tmp_path / "a", progress=seen.append)
    assert [h.status for h in full.hands] == ["brick"] * 6 and len(seen) == 6 and not full.stopped_early
    assert [h.hand for h in full.hands] == opening_hands(PURRELY, cfg)
    assert full.brick_rate == 1.0 and full.passed is None and full.hand_trap_survival == 0.0
    assert full.targets == [["52645235@atk"]] and full.deck["name"] == "purrely" and full.config["fire"] == [ASH_BLOSSOM]
    assert not (tmp_path / "a" / "hand0").exists()  # per-hand scratch is cleaned

    gated = evaluate_deck(PURRELY, ["52645235"], cfg, filter=FunnelFilter(max_brick_rate=0.5), scratch=tmp_path / "b")
    assert len(gated.hands) == 4 and gated.stopped_early and gated.passed is False  # 4 bricks > 0.5 x 6
    assert any("stopped early" in r for r in gated.reasons)

    # the same seeds shuffle a deck of the same size the same way: hands are paired across candidate decks
    renamed = replace(PURRELY, name="other")
    assert opening_hands(renamed, cfg) == opening_hands(PURRELY, cfg)
    assert FunnelConfig(seed=1).hand_seeds() != cfg.hand_seeds()


def test_evaluate_deck_errors_and_bad_targets(tmp_path):
    failing = _fake_solver(tmp_path, "#!/bin/sh\necho '!! decklist unreadable'\nexit 3\n")
    r = evaluate_deck(PURRELY, ["52645235"], FunnelConfig(hands=2, solve_ms=100, binary=failing), filter=FunnelFilter(),
                      scratch=tmp_path / "e")  # fmt: skip
    assert [h.status for h in r.hands] == ["error", "error"] and "exited with 3" in r.hands[0].error
    assert r.passed is False and r.reasons == ["no hand evaluated without error"]
    with pytest.raises(ValueError, match="99999999"):
        evaluate_deck(PURRELY, ["99999999"], FunnelConfig(hands=1, binary=failing), scratch=tmp_path / "f")


def test_max_rollouts_reaches_the_solver(tmp_path):
    log = tmp_path / "args.txt"
    recorder = _fake_solver(tmp_path, f'#!/bin/sh\necho "$@" >> {log}\necho "at best 0 of the 1 target cards"\nexit 0\n')
    r = evaluate_deck(PURRELY, ["52645235"], FunnelConfig(hands=1, solve_ms=100, max_rollouts=300, binary=recorder),
                      scratch=tmp_path / "m")  # fmt: skip
    assert r.config["max_rollouts"] == 300 and "--max-rollouts 300" in log.read_text()
    assert FunnelConfig().max_rollouts is None  # opt-in: the default budget stays wall time


def test_evaluate_deck_in_worker_processes(tmp_path):
    cfg = FunnelConfig(hands=3, solve_ms=100, workers=2, binary=_fake_solver(tmp_path))
    r = evaluate_deck(PURRELY, [["52645235"], ["52645235@def"]], cfg, scratch=tmp_path / "w")
    assert [h.index for h in r.hands] == [0, 1, 2] and all(h.status == "brick" for h in r.hands)
    assert r.targets == [["52645235@atk"], ["52645235@def"]]


def test_evaluate_genotype_decodes_through_the_space(tmp_path):
    class Space:  # the two methods of GenotypeSpace the funnel uses
        def decode(self, g, name=""):
            return replace(PURRELY, name=name)

        def genotype_to_json(self, g):
            return {"space": "fp", "packages": [0], "cards": {}}

    r = evaluate_genotype(Space(), object(), ["52645235"], FunnelConfig(hands=1, solve_ms=100, binary=_fake_solver(tmp_path)),
                          name="elite-1", scratch=tmp_path / "g")  # fmt: skip
    assert r.deck["name"] == "elite-1" and r.genotype["space"] == "fp" and len(r.hands) == 1


# ------------------------------------------------------------------ real first turns


def test_first_turn_duel_is_the_funnel_opening_without_pseudo_shuffle(db):
    seed = FunnelConfig().hand_seeds()[0]
    duel = first_turn_duel(PURRELY, seed, cards=db)
    assert duel.config.rule_flags & C.DUEL_PSEUDO_SHUFFLE == 0 and not duel.config.shuffle_decks
    assert real_config(DuelConfig(rule_flags=C.DUEL_MODE_MR5 | C.DUEL_PSEUDO_SHUFFLE)).rule_flags == C.DUEL_MODE_MR5
    loaded = duel.loaded_decks()[0][0]
    assert loaded[-5:] == sample_hand(PURRELY, seed)[0] and sorted(loaded) == sorted(PURRELY.main)
    session = DuelSession(duel)
    try:
        out = play_first_turn(session, GreedyAgent(0, cards=db), ["52645235"], cards=db)
    finally:
        session.close()
    assert out.board["turn"] == 2 and out.steps > 0 and out.reached == (out.missing == [])


def test_explore_is_seeded(db):
    seed = FunnelConfig().hand_seeds()[1]
    a = explore(PURRELY, seed, ["52645235"], 20, seed=3, cards=db)
    b = explore(PURRELY, seed, ["52645235"], 20, seed=3, cards=db)
    assert a["n"] == 20 and (a["reached"], a["first"], a["steps"]) == (b["reached"], b["first"], b["steps"])
    empty = explore(PURRELY, seed, [], 3, cards=db)  # no target: every first turn "reaches" it
    assert empty["reached"] == 3 and empty["first"] == 0


def test_explorer_prefers_proactive_moves():
    card = M.CardInfo(1, M.Location(0, C.LOCATION_HAND, 0))
    acts = [Action("summon", 0, card), Action("battle_phase"), Action("end_phase")]
    point = type("P", (), {"actions": acts})()
    picks = [ExplorerAgent(s, p_continue=1.0).act(point) for s in range(20)]
    assert set(picks) == {0}
    assert ExplorerAgent(0, p_continue=0.0).act(point) == 2


def test_match_action_ignores_order_in_deck_and_hand():
    def sel(code, loc, seq, i):
        return Action("select", i, M.CardInfo(code, M.Location(0, loc, seq)), value=i)

    ref = sel(7, C.LOCATION_DECK, 30, 4)
    assert match_action(ref, [sel(5, C.LOCATION_DECK, 30, 0), sel(7, C.LOCATION_DECK, 12, 1)]) == 1
    field = sel(7, C.LOCATION_MZONE, 2, 0)
    assert match_action(field, [sel(7, C.LOCATION_MZONE, 3, 0)]) is None  # on the field the zone matters
    assert match_action(field, [sel(7, C.LOCATION_MZONE, 3, 0), sel(7, C.LOCATION_MZONE, 2, 1)]) == 1
    assert match_action(Action("end_phase"), [Action("battle_phase"), Action("end_phase")]) == 1


def _pseudo_line(db, tmp_path, seed):
    """A 'solver line' made by Greedy in the solver's setting (pseudo-shuffle, passive opponent), as a Demonstration."""
    _, order = sample_hand(PURRELY, seed)
    deck = replace(PURRELY, main=tuple(order))
    cfg = DuelConfig(rule_flags=C.DUEL_MODE_MR5 | C.DUEL_PSEUDO_SHUFFLE, shuffle_decks=False, max_turns=1)
    duel = Duel(seed, None, deck, PASSIVE_OPPONENT, cards=db, config=cfg)
    result = duel.run(GreedyAgent(0, cards=db), GreedyAgent(1, cards=db))
    path = tmp_path / "line.yrpX"
    Replay.from_duel(duel, result).to_yrpx(path)
    yrp = load_yrp(path)
    line = convert_line(yrp, [], cards=db)
    mz = line.board["players"][0]["mzone"]
    targets = [str(mz[0]["code"])] if mz else []
    demo = Demonstration.start_from(yrp, deck=PURRELY, hand=order[-5:], hand_index=0, hand_seed=seed, variant="plain",
                                    targets=targets, environment=None)  # fmt: skip
    demo.lines.append(line)
    demo.status = "solved"
    return demo


def test_replay_line_follows_a_line_in_the_real_duel(db, tmp_path):
    seeds = FunnelConfig(hands=2).hand_seeds()
    demo = _pseudo_line(db, tmp_path, seeds[1])
    assert demo.start["core_seed"] == list(first_turn_duel(PURRELY, seeds[1], cards=db).core_seed)  # the funnel's opening
    out = replay_line(demo, 0, cards=db)
    assert out.error == "" and out.steps > 0 and out.board["turn"] == 2
    assert out.reached  # the target is a monster the line left on the board


def test_replay_line_reports_where_a_real_shuffle_breaks_the_line(db, tmp_path):
    """Hand 0: Purrely shuffles the deck, then looks at its top; with a real shuffle the line's next decision differs."""
    demo = _pseudo_line(db, tmp_path, FunnelConfig(hands=1).hand_seeds()[0])
    out = replay_line(demo, 0, cards=db)
    assert out.error.startswith("step ") and "in the real duel" in out.error
    assert out.board["turn"] == 2 and not out.reached


# ------------------------------------------------------------------ with the real solver binary


def test_end_to_end_funnel_on_a_real_deck(db, tmp_path):
    try:
        binary = find_solver()
    except SolverNotFound as exc:
        pytest.skip(f"combo solver binary not available ({exc}); build it with tools/build_combo_solver.sh")
    labrynth = load_ydk(DECKS / "labrynth.ydk")
    cfg = FunnelConfig(hands=2, solve_ms=4000, fire_ms=2000, binary=binary, keep_demos=True)
    r = evaluate_deck(labrynth, ["1225009", "5380979@szone:fd"], cfg, filter=FunnelFilter(max_brick_rate=1.0), scratch=tmp_path)
    assert len(r.hands) == 2 and r.errors == 0, [h.error for h in r.hands]
    assert r.passed is True and 0.0 <= r.brick_rate <= 1.0 and r.solver_s > 0
    for h in r.hands:
        if h.status == "solved":
            assert h.combo_actions > 0 and h.combo_decisions > 0 and ASH_BLOSSOM in h.fire
            assert h.fire[ASH_BLOSSOM]["survives"] in (True, False)
            demo = Demonstration.from_json(h.demos[0])
            assert replay_line(demo, 0, cards=db).board["turn"] == 2
