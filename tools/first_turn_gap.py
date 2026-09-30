"""Does the policy finish its combo on turn 1? (strength diagnosis, #83)

For each test deck with a solver target board (``tests/decks/solver_targets.json``) and ``--hands`` opening hands,
play player 0's first turn against a passive opponent and check whether the target card is on the board at the
start of turn 2 (``ygorl.build.first_turn``). Players:

- the policy checkpoint (sampled, and greedy with ``@greedy``), driven with its lockstep hooks;
- Greedy;
- a randomized explorer, best of ``--rollouts`` tries: the hand's target is reachable at all (a cheap reference;
  the combo solver finds more).

The gap between the explorer's reach and the policy's is the share of hands where a line exists but the policy
does not play it.

Usage: tools/first_turn_gap.py CHECKPOINT [--hands 12] [--rollouts 200] [--out gap.json]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path

DECKS = Path(__file__).resolve().parent.parent / "tests" / "decks"


def _policy_turn(session, agent, targets, cards):
    from ygorl.build.first_turn import MAX_TURN_STEPS, _finish, passive_index

    hook_start = getattr(agent, "on_duel_start", None)
    if hook_start is not None:
        hook_start(session.duel)
    steps = 0
    while not session.done and session.tracker.turn < 2 and steps < MAX_TURN_STEPS:
        point = session.point
        idx = agent.act(point) if point.player == 0 else passive_index(point)
        steps += point.player == 0
        hook = getattr(agent, "on_decision", None)
        if hook is not None:
            hook(point, idx)
        session.act(idx)
    return _finish(session, targets, cards, steps)


def job(args):
    deck_name, k, hand_seed, targets, ckpt, rollouts = args
    from ygorl.agents.greedy import GreedyAgent
    from ygorl.agents.registry import make_agent
    from ygorl.build.first_turn import explore, first_turn_duel
    from ygorl.cards.ydk import load_ydk
    from ygorl.engine.duel import DuelSession, default_cards
    from ygorl.solver.targets import parse_targets

    cards = default_cards()
    deck = load_ydk(DECKS / f"{deck_name}.ydk")
    tg = parse_targets(targets)
    out = {"deck": deck_name, "hand": k}
    for name, spec in (("policy", f"policy:{ckpt}"), ("policy_greedy", f"policy:{ckpt}@greedy")):
        try:
            session = DuelSession(first_turn_duel(deck, hand_seed, cards=cards))
            r = _policy_turn(session, make_agent(spec, hand_seed), tg, cards)
            out[name] = bool(r.reached)
            out[name + "_steps"] = r.steps
        except Exception as exc:  # noqa: BLE001 - recorded
            out[name] = None
            out[name + "_error"] = f"{type(exc).__name__}: {exc}"
    session = DuelSession(first_turn_duel(deck, hand_seed, cards=cards))
    from ygorl.build.first_turn import play_first_turn

    out["greedy"] = bool(play_first_turn(session, GreedyAgent(hand_seed, cards=cards), tg, cards=cards).reached)
    ex = explore(deck, hand_seed, targets, rollouts, seed=k, cards=cards)
    out["reachable"] = ex["reached"] > 0
    out["explorer_rate"] = ex["reached"] / max(rollouts, 1)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--hands", type=int, default=12)
    p.add_argument("--rollouts", type=int, default=200)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    from ygorl.eval.arena import derive_seed

    targets = json.loads((DECKS / "solver_targets.json").read_text())
    jobs = [(d, k, derive_seed(7, k), t["targets"], a.checkpoint, a.rollouts)
            for d, t in targets.items() if not d.startswith("_") for k in range(a.hands)]  # fmt: skip
    with get_context("spawn").Pool(a.workers) as pool:
        rows = pool.map(job, jobs, chunksize=1)
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1))
    by = defaultdict(list)
    for r in rows:
        by[r["deck"]].append(r)
    keys = ("reachable", "policy", "policy_greedy", "greedy")
    print(f"{'deck':20s} " + " ".join(f"{k:>14s}" for k in keys) + "   (hands reaching the target board)")
    tot = defaultdict(int)
    for d, rs in by.items():
        vals = [sum(bool(r.get(k)) for r in rs) for k in keys]
        for k, v in zip(keys, vals):
            tot[k] += v
        print(f"{d:20s} " + " ".join(f"{v:>11d}/{len(rs):<2d}" for v in vals))
    n = len(rows)
    print(f"{'total':20s} " + " ".join(f"{tot[k]:>11d}/{n:<2d}" for k in keys))
    reach = [r for r in rows if r["reachable"]]
    if reach:
        for k in ("policy", "policy_greedy", "greedy"):
            print(f"  of the {len(reach)} hands where the explorer reached the target: {k} reached "
                  f"{sum(bool(r.get(k)) for r in reach)} ({sum(bool(r.get(k)) for r in reach) / len(reach):.2f})")  # fmt: skip
    errs = [r for r in rows if r.get("policy_error")]
    if errs:
        print(f"  policy errors: {len(errs)}, e.g. {errs[0]['policy_error']}")


if __name__ == "__main__":
    main()
