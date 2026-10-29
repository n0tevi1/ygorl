"""#429/#439 hand-trap ruler: a policy plays the opponent's seat (player 1, holding Ash) against the solver's turn-1
blocking line of every labelled hand. Outcome per hand, from the solver labels (no rollout noise):
  broken   fired at a window the combo cannot recover from   (good)
  wasted   fired at a window the combo recovers from         (bad: Ash gone, board intact)
  missed   never fired although a breaking window existed     (bad)
  held     never fired and no breaking window existed         (good)
Score = (broken + held) / hands. Also reported: break rate over hands with a breaking window, waste rate over fires.

  uv run --no-sync python out/claude-scale/handtrap_eval.py LABELS.jsonl CKPT [CKPT ...] [--sampled] [--seed 0]
"""

import torch

torch.set_num_threads(2)
import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path("/home/ya0guang/Code/ygorl/.claude/worktrees/claude-rl-review")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, "/home/ya0guang/Code/ygorl/out/claude-scale")
from handtrap_dataset import ASH, ash_index, recovered_windows  # noqa: E402


def play(fire, p0, agent, env, cards):
    """Returns the window the agent fired at (1-based) or None, and the number of windows seen."""
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
                ai = ash_index(point)
                if ai is not None:
                    window += 1
                    idx = agent.act(point)
                    a = point.actions[idx]
                    if a.card is not None and a.card.code == ASH:
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("labels")
    p.add_argument("ckpts", nargs="+")
    p.add_argument("--sampled", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default=None)
    p.add_argument(
        "--min-hand", type=int, default=0, help="only hands with index >= N (held-out hands of the fine-tune)"
    )
    p.add_argument("--max-hand", type=int, default=10**9, help="only hands with index < N")
    a = p.parse_args()
    from ygorl.agents.registry import agent_factory
    from ygorl.data import load_environment
    from ygorl.engine.duel import default_cards
    from ygorl.solver import Demonstration

    env = load_environment("md-2026-09")
    cards = default_cards()
    rows = [json.loads(l) for l in open(a.labels)]
    plain = {(r["deck"]["name"], r["hand_index"]): Demonstration.from_json(r) for r in rows if r["variant"] == "plain"}
    fires = [
        Demonstration.from_json(r)
        for r in rows
        if r["variant"] == "fire"
        and (r["solver"].get("windows") or 0) > 0
        and a.min_hand <= r["hand_index"] < a.max_hand
    ]
    if a.limit:
        fires = fires[: a.limit]
    # labels per hand, computed once
    hands = []
    for fire in fires:
        base = plain[(fire.deck["name"], fire.hand_index)]
        p0 = [i for i, pl in zip(base.lines[0].actions, base.lines[0].players) if pl == 0]
        if fire.start is None:
            fire.start = json.loads(json.dumps(base.start))
            fire.start["decks"]["b"]["main"].append(ASH)
        W = fire.solver["windows"]
        conv = fire.solver.get("converted") or 0
        rec = set() if conv == 0 else (set(range(1, W + 1)) if conv == W else recovered_windows(fire, env, cards))
        if conv and conv != W and len(rec) < conv:
            continue  # ambiguous holds
        hands.append((fire, p0, W, rec))
    print(f"{len(hands)} hands; breaking window exists in {sum(len(r) < W for _, _, W, r in hands)}", flush=True)
    results = {}
    for ck in a.ckpts:
        spec = f"policy:{ck}" + ("" if a.sampled else "@greedy")
        factory = agent_factory(spec)
        t0 = time.time()
        c = Counter()
        fired_at = []
        for n, (fire, p0, W, rec) in enumerate(hands):
            agent = factory(a.seed * 100003 + n)
            k, seen = play(fire, p0, agent, env, cards)
            if k is None:
                c["missed" if len(rec) < W else "held"] += 1
            else:
                c["wasted" if k in rec else "broken"] += 1
                fired_at.append(k)
        n = len(hands)
        breakable = sum(len(r) < W for _, _, W, r in hands)
        res = {
            "hands": n,
            **c,
            "score": (c["broken"] + c["held"]) / n,
            "break_rate": c["broken"] / max(1, breakable),
            "waste_rate": c["wasted"] / max(1, c["wasted"] + c["broken"]),
            "fire_rate": (c["wasted"] + c["broken"]) / n,
            "mean_fired_window": sum(fired_at) / max(1, len(fired_at)),
            "s": round(time.time() - t0),
        }
        results[ck] = res
        name = ck.split("/ab1/")[-1] if "/ab1/" in ck else ck.split("/")[-1]
        print(
            f"{name}: " + " ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in res.items()),
            flush=True,
        )
    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=1))
    print("EVAL-DONE")


if __name__ == "__main__":
    main()
