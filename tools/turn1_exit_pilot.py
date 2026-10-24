"""P0d: how much first-seat win rate does choosing among a policy's own turn-1 lines buy? (first-seat plan §5.4 D, §6).

For ``--hands`` fixed first-seat hands (solved ``plain`` records of ``--demos``, hands ``--min-hand`` .. ``--max-hand``,
spread over the decks; the learner's deck is the record's in load order, so its opening hand is the record's whatever
the opponent):

1. **Candidates** (N = ``--candidates``): N - 1 turn-1 lines sampled from ``CKPT`` (temperature 1) against the first
   opponent sample, which passes on turn 1, recorded as encoded action keys; with ``--solver-candidate`` (default on)
   the record's solver blocking line is the N-th.
2. **Valuation** (M = ``--opponents`` scripted-line opponents drawn from ``--lines``, shared by all candidates: common
   random numbers): every (candidate, opponent) pair is a game from the opponent line's start (its deck in load order,
   its core seed and rules; the learner first). The learner replays the candidate on turn 1 by key
   (:class:`~ygorl.train.lines.LinePlayer`; a key with no match is a prefix mismatch, the pair is left out of the
   valuation and counted) and then plays ``CKPT`` with a stream seeded by (hand, opponent) only, so every candidate
   meets the same learner and opponent randomness; the opponent passes on turn 1, plays its line on turn 2 with the
   first-seat tolerances and is played by ``--snapshot`` (default CKPT) off the line. Games run to the end; J(i, m) is
   the learner's score (1 / 0.5 / 0).
3. **Cross-fit** (PR #167): the opponents are split in two halves; the candidate with the best mean J on one half is
   valued on the other against the mean of the policy candidates there (the policy's own value), and the two folds
   are averaged. Also reported: the in-sample (optimistic) margin and the solver line's margin.
4. **Realised**: the candidate with the best J over all M is played in fresh games, paired with the policy playing its
   own turn 1 (same seeds): ``--real-games`` games per hand against ``--real-opponent`` (G u576 on a meta deck
   shuffled by seed; it plays its turn 1 too) and as many against fresh scripted-line opponents (the snapshot plays
   their turn 1). The learner replays the chosen line while it matches and plays ``CKPT`` afterwards.

Margins are in win-rate points of the first seat.

  uv run --no-sync python tools/turn1_exit_pilot.py DEMOS.jsonl --ckpt CKPT --lines lines_train.jsonl \\
      --real-opponent G_u576.pt [--hands 100] [--candidates 16] [--opponents 8] [--real-games 8] --out OUT.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase0_common import STOP, Brain, Job, Net, generator, mean_se, passive_row, run_jobs, select_hands, turn  # noqa: E402,E501  # fmt: skip

from ygorl.cards.ydk import Deck, load_ydk  # noqa: E402
from ygorl.data import load_environment  # noqa: E402
from ygorl.data.environment import PlayerRules  # noqa: E402
from ygorl.engine.duel import DuelConfig, default_cards, shuffle_deck  # noqa: E402
from ygorl.env.encoded import EncodedVecEnv  # noqa: E402
from ygorl.env.pool import GameSpec  # noqa: E402
from ygorl.eval.arena import derive_seed  # noqa: E402
from ygorl.solver.demo import DemoError  # noqa: E402
from ygorl.train.lines import LineLibrary, LinePlayer, LineScript, LineStep, encoded_action_key, line_script  # noqa: E402

MAX_TURN1 = 300  # learner decisions on turn 1 before a candidate is cut (a loop)


def learner_deck(demo) -> Deck:
    s = demo.start
    return Deck(tuple(s["decks"]["a"]["main"]), tuple(s["decks"]["a"]["extra"]), (), demo.deck["name"])


def line_spec(deck: Deck, line: LineScript, seed: int, base: DuelConfig) -> GameSpec:
    """The learner (deck in load order) first against ``line``'s deck from its recorded start."""
    cfg = replace(base, rule_flags=line.rule_flags, player=PlayerRules(**line.player), shuffle_decks=False)
    return GameSpec(seed=seed, deck_a=deck, deck_b=line.deck, first=0, config=cfg, core_seed=line.core_seed)


def free_spec(deck: Deck, demo, opp: Deck, seed: int, base: DuelConfig) -> GameSpec:
    """The learner (deck in load order, the record's rules) first against ``opp`` shuffled by ``seed``."""
    s = demo.start
    cfg = replace(base, rule_flags=int(s["rule_flags"]), player=PlayerRules(**s["player"]), shuffle_decks=False)
    opp = replace(opp, main=tuple(shuffle_deck(opp.main, seed, 1)))
    return GameSpec(seed=seed, deck_a=deck, deck_b=opp, first=0, config=cfg)


class CandidateJob(Job):
    """Sample one turn-1 line for the learner (seat 0); the opponent passes; stops at turn 2."""

    def __init__(self, spec, gen) -> None:
        self.spec, self.gen = spec, gen
        self.keys: list[tuple] = []
        self.cut = False

    def decide(self, ev):
        if turn(ev.obs) >= 2:
            return STOP
        if ev.player != 0:
            return passive_row(ev.obs)
        if len(self.keys) >= MAX_TURN1:
            self.cut = True
            return STOP
        return Net("learner", self.gen)

    def chosen(self, ev, row: int) -> None:
        if ev.player == 0:
            self.keys.append(encoded_action_key(ev.obs, row))

    def finish(self, result):
        return {"keys": self.keys, "cut": self.cut, "ended": result is not None}


class GameJob(Job):
    """A full game, the learner first: turn 1 from ``plan`` (a LineScript replayed by key, None: the policy), then the
    policy. The opponent: ``line`` (first-seat LinePlayer) off which ``opponent`` plays; on turn 1 it passes when
    ``passive_turn1``, else ``opponent`` plays it."""

    def __init__(self, spec, plan, skip_forced: bool, learner_gen, opponent_gen, *, line=None, opponent="snapshot",
                 passive_turn1: bool = True) -> None:  # fmt: skip
        self.spec = spec
        self.plan = None if plan is None else LinePlayer(plan, skip_forced)
        self.line = None if line is None else LinePlayer(line, skip_forced, first_seat=True)
        self.learner_gen, self.opponent_gen = learner_gen, opponent_gen
        self.opponent, self.passive_turn1 = opponent, passive_turn1
        self.mismatch = False

    def decide(self, ev):
        obs = ev.obs
        if ev.player == 0:
            if self.plan is not None and self.plan.status == "playing" and turn(obs) == 1:
                row = self.plan.act(obs)
                if row is not None:
                    return row
                if self.plan.status == "broken":
                    self.mismatch = True
            return Net("learner", self.learner_gen)
        if self.line is not None:
            row = self.line.act(obs)
            if row is not None:
                return row
        if turn(obs) == 1 and self.passive_turn1:
            return passive_row(obs)
        return Net(self.opponent, self.opponent_gen)

    def finish(self, result):
        winner = None if result is None else result.get("winner")
        return {"score": 0.5 if winner is None else float(winner == 0), "mismatch": self.mismatch,
                "plan": None if self.plan is None else self.plan.status,
                "line": None if self.line is None else self.line.status,
                "reason": None if result is None else str(result.get("reason", "")),
                "turns": None if result is None else result.get("turns")}  # fmt: skip


def candidate_script(keys: list[tuple], deck: Deck, i: int) -> LineScript:
    return LineScript(i, deck, 0, (0, 0, 0, 0), 0, {}, tuple(LineStep(k, False) for k in keys))


def valuation(J: list[list[float | None]], policy: list[int], half: int) -> dict:
    """Cross-fitted, in-sample and solver margins of one hand's J matrix (candidates x opponents; None = left out)."""

    def mean(i, ms):
        xs = [J[i][m] for m in ms if J[i][m] is not None]
        return sum(xs) / len(xs) if xs else None

    def best(ms, pool):
        vals = [(mean(i, ms), i) for i in pool]
        vals = [(v, i) for v, i in vals if v is not None]
        return max(vals, key=lambda x: (x[0], -x[1]))[1] if vals else None

    def base(ms):
        xs = [mean(i, ms) for i in policy]
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    M = len(J[0])
    folds = (list(range(half)), list(range(half, M)))
    everyone = list(range(len(J)))
    out = {}
    for name, pool in (("all", everyone), ("policy_only", policy)):
        margins = []
        for sel, val in (folds, folds[::-1]):
            i = best(sel, pool)
            b = base(val)
            v = None if i is None else mean(i, val)
            if v is not None and b is not None:
                margins.append(v - b)
        out[f"crossfit_{name}"] = sum(margins) / len(margins) if margins else None
        allm = list(range(M))
        i = best(allm, pool)
        out[f"insample_{name}"] = None if i is None else mean(i, allm) - base(allm)
        out[f"chosen_{name}"] = i
    solver = [i for i in everyone if i not in policy]
    out["solver_margin"] = (
        None if not solver or mean(solver[0], range(M)) is None else mean(solver[0], range(M)) - base(range(M))
    )
    out["policy_value"] = base(range(M))
    return out


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("demos", nargs="+")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--lines", nargs="+", required=True)
    p.add_argument("--snapshot", default=None, help="plays scripted opponents off their line (default: CKPT)")
    p.add_argument("--real-opponent", default=None, help="policy of the realised test's free games (default: none)")
    p.add_argument("--hands", type=int, default=100)
    p.add_argument("--min-hand", type=int, default=40)
    p.add_argument("--max-hand", type=int, default=50)
    p.add_argument("--candidates", type=int, default=16)
    p.add_argument("--no-solver-candidate", action="store_true")
    p.add_argument("--opponents", type=int, default=8)
    p.add_argument("--real-games", type=int, default=8, help="paired games per hand and opponent type")
    p.add_argument("--env", default="md-2026-09")
    p.add_argument("--decks", nargs="*", default=None, help="decks of the free opponent (default: the meta decks)")
    p.add_argument("--max-lines", type=int, default=None)
    p.add_argument("--max-turns", type=int, default=None, help="game turn limit (default: the environment's)")
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--torch-threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    torch.set_num_threads(a.torch_threads)
    t0 = time.perf_counter()
    env = load_environment(a.env)
    cards = default_cards()
    learner = Brain(a.ckpt)
    brains = {"learner": learner, "snapshot": learner if a.snapshot in (None, a.ckpt) else Brain(a.snapshot)}
    if a.real_opponent:
        brains["real"] = Brain(a.real_opponent)
    for name, b in brains.items():
        if b.signature != learner.signature:
            raise SystemExit(f"{name}: vocabulary / event length differ from the learner's")
    library = LineLibrary.load(a.lines, learner.vocab, cards=cards, env=env)
    if a.max_lines is not None:
        library = LineLibrary(library.scripts[: a.max_lines], library.problems)
    paths = a.decks or sorted(str(x) for x in (Path(env.root) / "meta").glob("*.ydk"))
    free_decks = [load_ydk(x) for x in paths]
    base = DuelConfig.from_environment(env, **({"max_turns": a.max_turns} if a.max_turns else {}))
    cfg = learner.net_config
    venv = EncodedVecEnv(a.envs, a.threads, cards=cards, vocab=learner.vocab, event_length=learner.event_length,
                         skip_forced=True, selection_history=cfg.selection_history,
                         attack_history=cfg.attack_history)  # fmt: skip
    hands = select_hands(a.demos, a.hands, a.min_hand, a.max_hand)
    M, R = a.opponents, a.real_games
    per = []
    for h, demo in enumerate(hands):
        rng = np.random.default_rng(derive_seed(a.seed, 23, h))
        draw = rng.choice(len(library), size=M + R, replace=len(library) < M + R)  # without replacement when possible
        picks = [library.scripts[int(i)] for i in draw]
        solver = None
        if not a.no_solver_candidate:
            try:
                solver = line_script(demo, 0, learner.vocab, cards=cards, env=env)
            except (DemoError, ValueError):
                solver = None
        per.append({"demo": demo, "deck": learner_deck(demo), "opps": picks[:M], "fresh": picks[M:], "solver": solver,
                    "free": [free_decks[int(rng.integers(len(free_decks)))] for _ in range(R)]})  # fmt: skip
    # 1. candidates
    n_policy = a.candidates - (0 if a.no_solver_candidate else 1)
    jobs = [CandidateJob(line_spec(x["deck"], x["opps"][0], derive_seed(a.seed, h, 0, 0), base),
                         generator(a.seed, h, i, 101))
            for h, x in enumerate(per) for i in range(n_policy)]  # fmt: skip
    cand = run_jobs(venv, jobs, brains)
    t_cand = time.perf_counter() - t0
    for h, x in enumerate(per):
        rows = cand[h * n_policy : (h + 1) * n_policy]
        x["cands"] = [candidate_script(r["keys"], x["deck"], i) for i, r in enumerate(rows)]
        x["cut"] = sum(r["cut"] for r in rows)
        x["distinct"] = len({tuple(r["keys"]) for r in rows})
        x["policy"] = list(range(len(x["cands"])))
        if x["solver"] is not None:
            x["cands"].append(x["solver"])
    # 2. valuation
    jobs, index = [], []
    for h, x in enumerate(per):
        for i, plan in enumerate(x["cands"]):
            for m, line in enumerate(x["opps"]):
                seed = derive_seed(a.seed, h, m, 1)
                jobs.append(GameJob(line_spec(x["deck"], line, seed, base), plan, True, generator(seed, 1),
                                    generator(seed, 2), line=line))  # fmt: skip
                index.append((h, i, m))
    games = run_jobs(venv, jobs, brains)
    t_val = time.perf_counter() - t0 - t_cand
    J = {h: [[None] * M for _ in x["cands"]] for h, x in enumerate(per)}
    mismatch = {h: 0 for h in J}
    solver_mismatch = solver_pairs = 0
    for (h, i, m), g in zip(index, games):
        is_solver = i >= len(per[h]["policy"])
        solver_pairs += is_solver
        if g["mismatch"]:
            mismatch[h] += 1
            solver_mismatch += is_solver
        else:
            J[h][i][m] = g["score"]
    hand_rows = []
    for h, x in enumerate(per):
        v = valuation(J[h], x["policy"], M // 2)
        hand_rows.append({"deck": x["demo"].deck["name"], "hand_index": x["demo"].hand_index,
                          "candidates": len(x["cands"]), "distinct_policy_lines": x["distinct"], "cut": x["cut"],
                          "mismatch": mismatch[h], "pairs": len(x["cands"]) * M, "J": J[h], **v})  # fmt: skip
    # 4. realised
    real_jobs, real_index = [], []
    for h, x in enumerate(per):
        chosen = hand_rows[h]["chosen_all"]
        plan = None if chosen is None else x["cands"][chosen]
        for r in range(R):
            seed = derive_seed(a.seed, h, r, 2)
            kinds = [("line", line_spec(x["deck"], x["fresh"][r], seed, base), x["fresh"][r], "snapshot", False)]
            if "real" in brains:
                kinds.append(("free", free_spec(x["deck"], x["demo"], x["free"][r], seed, base), None, "real", False))
            for kind, spec, line, opp, passive in kinds:
                for arm, pl in (("policy", None), ("chosen", plan)):
                    real_jobs.append(GameJob(spec, pl, True, generator(seed, 1), generator(seed, 2), line=line,
                                             opponent=opp, passive_turn1=passive))  # fmt: skip
                    real_index.append((h, r, kind, arm))
    real = run_jobs(venv, real_jobs, brains) if R else []
    t_real = time.perf_counter() - t0 - t_cand - t_val
    paired: dict[str, dict[int, list]] = {}
    arms: dict[tuple[str, str], list[float]] = {}
    pending: dict[tuple, float] = {}
    plan_status: dict[str, int] = {}
    for (h, r, kind, arm), g in zip(real_index, real):
        arms.setdefault((kind, arm), []).append(g["score"])
        if arm == "chosen":
            plan_status[g["plan"] or "none"] = plan_status.get(g["plan"] or "none", 0) + 1
            paired.setdefault(kind, {}).setdefault(h, []).append(g["score"] - pending.pop((h, r, kind)))
        else:
            pending[(h, r, kind)] = g["score"]

    def pp(x):
        return None if x is None else 100 * x

    def stat(key):
        xs = [r[key] for r in hand_rows if r[key] is not None]
        s = mean_se(xs)
        return {"hands": s["n"], "pp": pp(s["mean"]), "se_pp": pp(s["se"])}

    pairs_total = sum(r["pairs"] for r in hand_rows)
    summary = {
        "hands": len(hand_rows),
        "crossfit_margin": stat("crossfit_all"), "crossfit_margin_policy_only": stat("crossfit_policy_only"),
        "insample_margin": stat("insample_all"), "insample_margin_policy_only": stat("insample_policy_only"),
        "solver_line_margin": stat("solver_margin"), "policy_value": stat("policy_value"),
        "chosen_is_solver": sum(r["chosen_all"] is not None and r["chosen_all"] >= len(per[h]["policy"])
                                for h, r in enumerate(hand_rows)),
        "mismatch_rate": sum(r["mismatch"] for r in hand_rows) / max(1, pairs_total),
        "solver_mismatch_rate": solver_mismatch / max(1, solver_pairs),
        "distinct_policy_lines": mean_se([r["distinct_policy_lines"] for r in hand_rows]),
        "realised": {},
    }  # fmt: skip
    for kind, by_hand in paired.items():
        per_hand = [sum(v) / len(v) for v in by_hand.values()]
        s = mean_se(per_hand)
        summary["realised"][kind] = {"hands": s["n"], "games_per_arm": len(arms[(kind, "policy")]),
                                     "margin_pp": pp(s["mean"]), "se_pp": pp(s["se"]),
                                     "win_rate_policy": sum(arms[(kind, "policy")]) / len(arms[(kind, "policy")]),
                                     "win_rate_chosen": sum(arms[(kind, "chosen")]) / len(arms[(kind, "chosen")])}  # fmt: skip
    summary["realised_plan_status"] = plan_status
    report = {"ckpt": a.ckpt, "snapshot": a.snapshot or a.ckpt, "real_opponent": a.real_opponent,
              "candidates": a.candidates, "opponents": M, "real_games": R, "seed": a.seed, "lines": len(library),
              "summary": summary, "rows": hand_rows,
              "seconds": {"candidates": round(t_cand, 1), "valuation": round(t_val, 1), "realised": round(t_real, 1),
                          "total": round(time.perf_counter() - t0, 1)}}  # fmt: skip
    text = json.dumps(report, indent=1)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n")
    else:
        print(json.dumps(report["summary"], indent=1))
    return report


if __name__ == "__main__":
    main()
