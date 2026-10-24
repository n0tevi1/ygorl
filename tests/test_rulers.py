"""Rulers R2 / R2b (tools/handtrap_ruler.py) and the first-seat line evaluation (tools/first_seat_line_eval.py).

``tests/data/handtrap_ruler_labels.jsonl`` holds the plain and fire records of seven held-out hands of
``out/claude-scale/handtrap/full_fire8.jsonl`` (the environment binding removed): blue-eyes 44 (two windows, none
recoverable), branded 43 / radiant-typhoon-zoodiac 49 / sky-striker 41 (window 1 a bait, a later window breaks),
dracotail 43 and elfnote 49 (every window recoverable) and orcust 45 (unverified fire lines: left out as an ambiguous
hold). The R2 reference below is ``out/claude-scale/handtrap_eval.py`` (2026-10-23) with its paths taken out.
"""

import json
import shutil
from collections import Counter
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import first_seat_line_eval as fsl  # noqa: E402
import handtrap_ruler as hr  # noqa: E402

HERE = Path(__file__).parent
LABELS = HERE / "data" / "handtrap_ruler_labels.jsonl"
LINES = HERE / "data" / "line_demos.jsonl"


@pytest.fixture(scope="module")
def setting():
    from ygorl.data import load_environment
    from ygorl.engine.duel import default_cards

    return load_environment("md-2026-09"), default_cards()


@pytest.fixture(scope="module")
def rows():
    return [json.loads(line) for line in LABELS.open()]


@pytest.fixture(scope="module")
def hands(rows, setting):
    return hr.load_hands(rows, *setting)


@pytest.fixture(scope="module")
def tiny_ckpts(tmp_path_factory):
    from ygorl.cards.cdb import CardVocab
    from ygorl.engine.duel import default_cards
    from ygorl.nets import NetConfig, PolicyNet
    from ygorl.nets.agent import save_checkpoint

    vocab = CardVocab.from_db(default_cards())
    out = []
    for seed in (0, 1):
        torch.manual_seed(seed)
        net = PolicyNet(NetConfig(vocab_size=len(vocab), d_model=16, n_heads=2, board_layers=1, history_layers=1))
        out.append(str(save_checkpoint(tmp_path_factory.mktemp("p") / "p.pt", net, vocab, event_length=16)))
    return out


# -- reference: handtrap_eval.py's selection, play and scoring ----------------------------------------------------
def reference_hands(rows, env, cards):
    from ygorl.solver import Demonstration

    plain = {(r["deck"]["name"], r["hand_index"]): Demonstration.from_json(r) for r in rows if r["variant"] == "plain"}
    fires = [
        Demonstration.from_json(r) for r in rows if r["variant"] == "fire" and (r["solver"].get("windows") or 0) > 0
    ]
    hands = []
    for fire in fires:
        base = plain[(fire.deck["name"], fire.hand_index)]
        p0 = [i for i, pl in zip(base.lines[0].actions, base.lines[0].players) if pl == 0]
        if fire.start is None:
            fire.start = json.loads(json.dumps(base.start))
            fire.start["decks"]["b"]["main"].append(hr.ASH)
        W = fire.solver["windows"]
        conv = fire.solver.get("converted") or 0
        rec = set() if conv == 0 else (set(range(1, W + 1)) if conv == W else hr.recovered_windows(fire, env, cards))
        if conv and conv != W and len(rec) < conv:
            continue
        hands.append((fire, p0, W, rec))
    return hands


def reference_play(fire, p0, agent, env, cards):
    from ygorl.engine.duel import DuelSession
    from ygorl.train.bc import passive_action

    duel = fire.replay(None).duel(env, cards=cards)
    session = DuelSession(duel)
    tracker = session.tracker
    if hook := getattr(agent, "on_duel_start", None):
        hook(duel)
    it = iter(p0)
    window = 0
    fired = None
    try:
        while not session.done and tracker.turn < 2:
            point = session.point
            if point is None:
                break
            if obs := getattr(agent, "observe", None):
                obs(point, session.core)
            if point.player == 0:
                idx = next(it, None)
                if idx is None:
                    break
            else:
                ai = hr.ash_index(point)
                if ai is not None:
                    window += 1
                    idx = agent.act(point)
                    a = point.actions[idx]
                    if a.card is not None and a.card.code == hr.ASH:
                        fired = window
                        if hook := getattr(agent, "on_decision", None):
                            hook(point, idx)
                        break
                else:
                    idx = passive_action(point)
            session.act(idx)
            if hook := getattr(agent, "on_decision", None):
                hook(point, idx)
    finally:
        session.close()
    return fired, window


def reference_r2(hands, factory, env, cards, seed=0):
    c = Counter()
    fired_at = []
    for n, (fire, p0, W, rec) in enumerate(hands):
        k, _ = reference_play(fire, p0, factory(seed * 100003 + n), env, cards)
        if k is None:
            c["missed" if len(rec) < W else "held"] += 1
        else:
            c["wasted" if k in rec else "broken"] += 1
            fired_at.append(k)
    n = len(hands)
    breakable = sum(len(r) < W for _, _, W, r in hands)
    return {"hands": n, **c, "score": (c["broken"] + c["held"]) / n, "break_rate": c["broken"] / max(1, breakable),
            "waste_rate": c["wasted"] / max(1, c["wasted"] + c["broken"]), "fire_rate": (c["wasted"] + c["broken"]) / n,
            "mean_fired_window": sum(fired_at) / max(1, len(fired_at))}  # fmt: skip


class FireAt:
    """Activates Ash at its ``k``-th window (0: never); P(Ash) 0.9 there, 0.1 elsewhere."""

    def __init__(self, k):
        self.k, self.seen, self.last_probs = k, 0, None

    def act(self, point):
        from ygorl.train.bc import passive_action

        self.seen += 1
        ai = hr.ash_index(point)
        self.last_probs = [0.0] * len(point.actions)
        self.last_probs[ai] = 0.9 if self.seen == self.k else 0.1
        return ai if self.seen == self.k else passive_action(point)


# -- R2 ---------------------------------------------------------------------------------------------------------------
def test_hand_selection_matches_the_reference(rows, setting, hands):
    ref = reference_hands(rows, *setting)
    assert [(h.fire.deck["name"], h.fire.hand_index, h.windows, h.recovered) for h in hands] == [
        (f.deck["name"], f.hand_index, W, rec) for f, _, W, rec in ref
    ]
    names = {(h.fire.deck["name"], h.fire.hand_index) for h in hands}
    assert ("orcust", 45) not in names and len(hands) == 6
    assert {(h.fire.deck["name"], h.fire.hand_index) for h in hands if h.bait_first} == {
        ("branded", 43), ("radiant-typhoon-zoodiac", 49), ("sky-striker", 41)}  # fmt: skip


@pytest.mark.parametrize("k", [0, 1, 2])
def test_r2_matches_handtrap_eval_on_scripted_agents(rows, setting, hands, k):
    env, cards = setting
    ref = reference_r2(reference_hands(rows, env, cards), lambda seed: FireAt(k), env, cards)
    fired = [hr.play(h, FireAt(k), env, cards)[0] for h in hands]
    assert hr.r2_summary(hands, fired) == ref


@pytest.mark.parametrize("sampled", [False, True])
def test_r2_matches_handtrap_eval_on_a_policy_with_a_fixed_seed(rows, setting, hands, tiny_ckpts, sampled):
    from ygorl.agents.registry import agent_factory

    env, cards = setting
    spec = f"policy:{tiny_ckpts[0]}" + ("" if sampled else "@greedy")
    ref = reference_r2(reference_hands(rows, env, cards), agent_factory(spec), env, cards, seed=7)
    got = hr.evaluate(hands, spec, env, cards, seed=7)
    got["r2"].pop("s")
    assert got["r2"] == ref
    assert got["r2b"]["bait_hands"] == 3 and 0 <= got["r2b"]["p_fire_bait"] <= 1


def test_r2_outcomes_on_synthetic_labels():
    hs = [
        hr.Hand(None, [], 2, set()),
        hr.Hand(None, [], 2, {1}),
        hr.Hand(None, [], 1, {1}),
        hr.Hand(None, [], 3, {1, 2}),
    ]
    r = hr.r2_summary(hs, [None, 1, None, 3])  # missed, wasted, held, broken
    assert (r["missed"], r["wasted"], r["held"], r["broken"]) == (1, 1, 1, 1)
    assert r["score"] == 0.5 and r["break_rate"] == 1 / 3 and r["waste_rate"] == 0.5 and r["mean_fired_window"] == 2


# -- R2b --------------------------------------------------------------------------------------------------------------
def test_r2b_reads_the_bait_and_the_first_breaking_window_of_synthetic_sequences():
    hs = [hr.Hand(None, [], 3, {1}), hr.Hand(None, [], 4, {1, 2}), hr.Hand(None, [], 2, {1})]
    assert all(h.bait_first for h in hs) and [h.first_break for h in hs] == [2, 3, 2]
    asked = [
        [(0.8, True), (0.9, True)],  # bait 0.8, break 0.9
        [(0.2, False), (0.4, False), (1.0, True)],  # bait 0.2 (and 0.4), break 1.0
        [(0.5, False)],  # the replay stopped before the breaking window: counts for the bait only
    ]
    r = hr.r2b_summary(hs, asked)
    assert r["bait_hands"] == 3 and r["break_windows"] == 2 and r["bait_windows"] == 4
    assert r["p_fire_bait"] == pytest.approx((0.8 + 0.2 + 0.5) / 3)
    assert r["p_fire_break"] == pytest.approx((0.9 + 1.0) / 2)
    assert r["p_fire_bait_all"] == pytest.approx((0.8 + 0.2 + 0.4 + 0.5) / 4)
    assert r["act_fire_bait"] == pytest.approx(1 / 3) and r["act_fire_break"] == 1.0


def test_r2b_subset_excludes_hands_without_a_bait_first():
    assert not hr.Hand(None, [], 2, set()).bait_first  # window 1 breaks
    assert not hr.Hand(None, [], 2, {1, 2}).bait_first  # nothing breaks
    assert not hr.Hand(None, [], 3, {2}).bait_first  # window 1 breaks, the bait comes later
    assert hr.r2b_summary([], [])["bait_hands"] == 0


def test_r2b_probe_passes_through_the_bait_and_stops_at_the_first_break(setting, hands):
    env, cards = setting
    for h in (h for h in hands if h.bait_first):
        asked = hr.play(h, FireAt(1), env, cards, probe=h.first_break)
        assert len(asked) == h.first_break  # the forced pass at window 1 let the line reach the breaking window
        assert asked[0] == (0.9, True) and all(p == 0.1 for p, _ in asked[1:])


def test_ruler_cli_json(tiny_ckpts, capsys):
    assert hr.main([str(LABELS), tiny_ckpts[0], "--json"]) == 0
    out = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(out) == 1 and out[0]["ckpt"] == tiny_ckpts[0] and out[0]["r2"]["hands"] == 6
    assert set(out[0]["r2b"]) >= {"p_fire_bait", "p_fire_break", "bait_hands"}


# -- first-seat line evaluation -----------------------------------------------------------------------------------------
def test_first_seat_line_eval_smoke_is_paired_and_deterministic(tiny_ckpts, tmp_path, capsys):
    decks = tmp_path / "decks"
    decks.mkdir()
    for name in ("snake_eye", "kashtira"):
        shutil.copy(HERE / "decks" / f"{name}.ydk", decks)
    args = [*tiny_ckpts, "--demos", str(LINES), "--decks", str(decks), "--games-per-line", "2", "--envs", "4",
            "--threads", "2", "--max-decisions", "600", "--json", "--line-recovery", str(LABELS)]  # fmt: skip
    reports = []
    for run in range(2):
        out = tmp_path / f"r{run}.json"
        assert fsl.main([*args, "--out", str(out)]) == 0
        report = json.loads(capsys.readouterr().out)
        for res in report["results"].values():
            res.pop("s")  # wall time
        reports.append(json.dumps(report))  # as text: NaN rates (no decided game) compare equal
        assert json.loads(out.read_text())["games"].keys() == set(tiny_ckpts)
    assert reports[0] == reports[1]  # same seed, same games, same draws
    a = json.loads(reports[0])
    assert a["lines"] == 3 and set(a["results"]) == set(tiny_ckpts)
    for res in a["results"].values():
        assert res["games"] == 6
        assert sum(res["fire_hist"].values()) == 6
        assert sum(res["line_status"].values()) == 6
        assert 0 <= res["line_completion"] <= 1
    assert a["paired"]["games"] == 6


def test_first_seat_helpers():
    from ygorl.train.lines import LineScript, LineStep

    steps = (LineStep(("a",), False), LineStep(("b",), True))
    script = LineScript(0, None, 0, (0, 0, 0, 0), 0, {}, steps)
    player = type("P", (), {"status": "playing", "played": 1, "pos": 1, "script": script, "skip_forced": True})()
    assert fsl.line_completed(player)  # only a forced step left, never asked
    player.skip_forced = False
    assert not fsl.line_completed(player)
    o = [fsl.GameOutcome(i, 0, "x", "d", w, r, 3, {"completed": c}, 2, f)
         for i, (w, r, c, f) in enumerate([(0, "win", True, []), (1, "win", False, [1]), (None, "max_turns", False, [2])])]  # fmt: skip
    s = fsl.summarize(o)
    assert (
        s["win_rate"] == 0.5 and s["score"] == pytest.approx(1.5 / 3) and s["line_completion"] == pytest.approx(1 / 3)
    )
    assert s["fire_hist"] == {"1": 1, "2": 1, "none": 1}
    assert s["p_fire_at"]["1"] == {"reached": 3, "p": pytest.approx(1 / 3)}
    assert s["p_fire_at"]["2"] == {"reached": 2, "p": 0.5}
    assert s["completion_if_fired"] == 0 and s["completion_if_held"] == 1
    flipped = [fsl.GameOutcome(x.index, 0, "x", "d", 1 - x.winner if x.winner is not None else None, x.reason, 3,
                               x.line, 2, x.fired) for x in o]  # fmt: skip
    p = fsl.paired(o, flipped)
    assert p["score_diff"] == 0 and p["b_better"] == 1 and p["a_better"] == 1
