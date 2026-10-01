"""Does a better turn 1 win more games? (strength diagnosis, #83)

Paired (common random numbers) arena on random corpus deck pairings. Agent b is always the main policy; agent a is,
per arm:

- ``A``: the main policy for every decision (the baseline);
- any other arm ``NAME=SPEC``: a *composite* that answers the decisions of its own turn 1 (``point.turn == 1`` and
  it is the turn player, i.e. it went first) with ``SPEC`` -- e.g. a BC prior fine-tuned on blocking-board solver
  lines -- and every later decision with the main policy. An arm given with ``--own-arm`` instead plays the
  composite's first own turn with ``SPEC`` when it goes second too (game turn 2).

Both inner agents get every lockstep hook (``on_duel_start``, ``on_decision``, ``observe``), so their hosts follow the
whole game; only the active one's action is used. Every arm plays the same games: same deck pairing, shuffle seed,
agent seeds and first player (``Arena.game_specs``). Going second an ``--arm`` composite plays exactly as A, so
those games must match A exactly (``identical_games`` in the summary: a determinism check).

At the first decision of turn 2 the first player's (engine player 0) turn-1 end board is scored with
``ygorl.solver.blocking.board_interruptions`` (field, set, hand and graveyard interruptions from the card scripts).

Usage: tools/turn1_value.py MAIN.pt --arm B=policy:BC.pt@greedy [--arm ...] [--own-arm ...] [--pairings 600]
       [--workers 12] [--nice 19]
       [--decks 'out/corpus/train/*.ydk'] [--seed 0] [--out DIR]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import time
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path


class Turn1Composite:
    """``opening`` plays its own turn 1 (or its first own turn), ``main`` the rest; both follow every decision."""

    name = "turn1"

    def __init__(self, main, opening=None, *, first_own_turn: bool = False) -> None:
        self.main, self.opening, self.first_own_turn = main, opening, first_own_turn
        self.inner = [a for a in (main, opening) if a is not None]
        self.last_probs = None
        self.board = None  # engine player 0's board at the first decision of turn 2
        self.t1_steps = 0  # decisions of the first player during turn 1
        self.opening_steps = 0  # decisions answered by the opening policy

    def on_duel_start(self, duel) -> None:
        for a in self.inner:
            hook = getattr(a, "on_duel_start", None)
            if hook is not None:
                hook(duel)

    def observe(self, point, core) -> None:
        for a in self.inner:
            hook = getattr(a, "observe", None)
            if hook is not None:
                hook(point, core)
        if point.turn == 1 and point.player == 0:
            self.t1_steps += 1
        if self.board is None and point.turn >= 2:
            from ygorl.solver.targets import board_summary

            self.board = board_summary(core, point.turn, tuple(point.lp))

    def on_decision(self, point, index: int) -> None:
        for a in self.inner:
            hook = getattr(a, "on_decision", None)
            if hook is not None:
                hook(point, index)

    def _opening_turn(self, point) -> bool:
        if self.opening is None or point.turn_player != point.player:
            return False
        own_first = 1 if point.player == 0 else 2  # engine player 0 moves first
        return point.turn == (own_first if self.first_own_turn else 1)

    def act(self, point) -> int:
        agent = self.opening if self._opening_turn(point) else self.main
        self.opening_steps += agent is self.opening
        index = agent.act(point)
        self.last_probs = getattr(agent, "last_probs", None)
        return index


def _score(board, cards) -> dict:
    from ygorl.engine import constants as C
    from ygorl.solver.blocking import board_interruptions

    s = board_interruptions(board, cards, 0)
    side = board["players"][0]
    on_board = sum(w in ("field", "set") for _, w, _ in s.pieces)
    return {"interruptions": s.interruptions, "negates": s.negates, "board_interruptions": on_board,
            "monsters": len(side["mzone"]), "faceup_monsters": sum(bool(c["position"] & C.POS_FACEUP)
                                                                    for c in side["mzone"]),
            "spells_traps": len(side["szone"]), "hand": len(side["hand"])}  # fmt: skip


_CARDS = None


def job(args) -> dict:
    global _CARDS
    arm, opening_spec, first_own_turn, main_spec, pairing, path_a, path_b, seed, first = args
    import torch

    torch.set_num_threads(1)
    from ygorl.agents.registry import make_agent
    from ygorl.cards.ydk import load_ydk
    from ygorl.engine.duel import default_cards
    from ygorl.eval.arena import Arena

    if _CARDS is None:
        _CARDS = default_cards()
    deck_a, deck_b = load_ydk(path_a), load_ydk(path_b)
    spec = next(s for s in Arena(None, None).game_specs(deck_a, deck_b, 1, seed) if s.first == first)
    sa, sb = spec.agent_seeds
    out = {"arm": arm, "pairing": pairing, "deck_a": Path(path_a).stem, "deck_b": Path(path_b).stem, "seed": seed,
           "first": first}  # fmt: skip
    t0 = time.time()
    try:
        opening = make_agent(opening_spec, sa ^ 0x5EED) if opening_spec else None
        a = Turn1Composite(make_agent(main_spec, sa), opening, first_own_turn=first_own_turn)
        r = spec.duel().run(a, make_agent(main_spec, sb))
    except Exception as exc:  # noqa: BLE001 - recorded
        out.update(winner=None, reason="exception", error=f"{type(exc).__name__}: {exc}")
        return out
    out.update(winner=r.winner, reason=r.reason, turns=r.turns, decisions=r.decisions, lp=list(r.lp),
               error=r.error, t1_steps=a.t1_steps, opening_steps=a.opening_steps, seconds=time.time() - t0)  # fmt: skip
    out["turn1"] = _score(a.board, _CARDS) if a.board is not None else None
    return out


# ------------------------------------------------------------------ statistics


def score_a(r) -> float:
    return 1.0 if r["winner"] == 0 else 0.5 if r["winner"] is None else 0.0


def mean_ci(xs, z=1.959964) -> tuple[float, float, float]:
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    h = z * sd / math.sqrt(n)
    return m, m - h, m + h


def report(rows: list[dict], arms: list[str]) -> dict:
    by = {(r["arm"], r["pairing"], r["first"]): r for r in rows}
    keys = sorted({(r["pairing"], r["first"]) for r in rows})
    out = {"arms": {}, "paired": {}, "interruptions": {}}
    for arm in arms:
        rs = [by[(arm, *k)] for k in keys if (arm, *k) in by]
        ok = [r for r in rs if r["reason"] != "exception"]
        d = {"games": len(rs), "errors": len(rs) - len(ok)}
        for name, sel in (("all", lambda r: True), ("first", lambda r: r["first"] == 0),
                          ("second", lambda r: r["first"] == 1)):  # fmt: skip
            m, lo, hi = mean_ci([score_a(r) for r in ok if sel(r)])
            d[name] = {"n": sum(sel(r) for r in ok), "win_rate": m, "ci": [lo, hi]}
        fp = [score_a(r) if r["first"] == 0 else 1 - score_a(r) for r in ok]
        d["first_player_win_rate"] = sum(fp) / len(fp) if fp else None
        d["mean_turns"] = sum(r["turns"] for r in ok) / len(ok) if ok else None
        out["arms"][arm] = d
    for arm in arms[1:]:
        res = {}
        for name, firsts in (("first", (0,)), ("second", (1,)), ("all", (0, 1))):
            diffs, same = [], 0
            per_pairing = defaultdict(float)
            for p, f in keys:
                a, b = by.get((arms[0], p, f)), by.get((arm, p, f))
                if f not in firsts or a is None or b is None or "exception" in (a["reason"], b["reason"]):
                    continue
                dd = score_a(b) - score_a(a)
                diffs.append(dd)
                per_pairing[p] += dd
                same += a["decisions"] == b["decisions"] and a["winner"] == b["winner"] and a["turns"] == b["turns"]
            m, lo, hi = mean_ci(diffs)
            if name == "all":  # cluster by deck pairing (both games of a pairing share decks and seed)
                cl = list(per_pairing.values())
                k = len(diffs) / max(len(cl), 1)
                cm, clo, chi = mean_ci(cl)
                m, lo, hi = cm / k, clo / k, chi / k
            res[name] = {"n": len(diffs), "diff": m, "ci": [lo, hi], "identical_games": same,
                         "changed_outcome": sum(x != 0 for x in diffs)}  # fmt: skip
        out["paired"][arm] = res
    for arm in arms:
        table = defaultdict(lambda: [0, 0.0])
        table_b = defaultdict(lambda: [0, 0.0])
        for k in keys:
            r = by.get((arm, *k))
            if r is None or r["reason"] == "exception" or r.get("turn1") is None:
                continue
            fp = score_a(r) if r["first"] == 0 else 1 - score_a(r)
            t = table[min(r["turn1"]["interruptions"], 4)]
            t[0], t[1] = t[0] + 1, t[1] + fp
            t = table_b[min(r["turn1"]["board_interruptions"], 3)]
            t[0], t[1] = t[0] + 1, t[1] + fp
        out["interruptions"][arm] = {
            "all_places": {str(k): {"games": n, "first_player_win_rate": w / n} for k, (n, w) in sorted(table.items())},
            "field_and_set": {str(k): {"games": n, "first_player_win_rate": w / n}
                              for k, (n, w) in sorted(table_b.items())},
        }  # fmt: skip
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("main", help="main policy checkpoint")
    p.add_argument("--arm", action="append", default=[], help="NAME=SPEC: the opening policy of a composite arm")
    p.add_argument(
        "--own-arm",
        action="append",
        default=[],
        help="NAME=SPEC: like --arm, and the opening policy also plays turn 2 going second",
    )
    p.add_argument("--decks", default="out/corpus/train/*.ydk")
    p.add_argument("--pairings", type=int, default=600, help="random deck pairings (2 games each, both first players)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default=None, help="directory for games.jsonl and summary.json")
    p.add_argument("--nice", type=int, default=0, help="raise the niceness of this process and its workers")
    a = p.parse_args()
    if a.nice:
        os.nice(a.nice)

    from ygorl.eval.arena import derive_seed

    decks = sorted(glob.glob(a.decks))
    rng = random.Random(a.seed)
    pairings = [(i, *rng.sample(decks, 2), derive_seed(a.seed, i)) for i in range(a.pairings)]
    arms = [("A", None, False)] + [(*s.split("=", 1), False) for s in a.arm]
    arms += [(*s.split("=", 1), True) for s in a.own_arm]
    main_spec = f"policy:{a.main}"
    jobs = [(name, spec, own, main_spec, i, da, db, s, f)
            for i, da, db, s in pairings for f in (0, 1) for name, spec, own in arms]  # fmt: skip
    out_dir = Path(a.out) if a.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    with get_context("spawn").Pool(a.workers) as pool:
        sink = (out_dir / "games.jsonl").open("w") if out_dir else None
        for i, r in enumerate(pool.imap_unordered(job, jobs, chunksize=2)):
            rows.append(r)
            if sink:
                sink.write(json.dumps(r) + "\n")
                sink.flush()
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(jobs)} games ({time.time() - t0:.0f}s)", flush=True)
        if sink:
            sink.close()
    summary = report(rows, [n for n, _, _ in arms])
    summary["config"] = {"main": a.main, "arms": {n: {"opening": s, "first_own_turn": o} for n, s, o in arms},
                         "pairings": a.pairings, "seed": a.seed, "decks": a.decks}  # fmt: skip
    if out_dir:
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
