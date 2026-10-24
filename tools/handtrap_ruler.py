"""Hand-trap ruler R2 / R2b (docs/training.md 「尺子」; plan docs/spikes/first-seat-interruption-plan-2026-10-24.md §6).

A policy plays the opponent's seat (engine player 1, Ash Blossom in hand) against the solver's turn-1 blocking line of
every labelled hand: player 0 follows the line, player 1 passes everywhere except at the Ash windows, where the policy
decides. Labels come from the solver's ``fire`` records (``--fire`` mode: the windows the combo recovers from), so
there is no rollout noise.

**R2** (moved here unchanged from ``out/claude-scale/handtrap_eval.py``), one outcome per hand:

- ``broken``  fired at a window the combo cannot recover from (good)
- ``wasted``  fired at a window the combo recovers from (bad: Ash gone, board intact)
- ``missed``  never fired although a breaking window existed (bad)
- ``held``    never fired and no breaking window existed (good)

``score`` = (broken + held) / hands; ``break_rate`` over hands with a breaking window; ``waste_rate`` over fires.

**R2b** (bait first): the hands whose first window is recoverable (a *bait*) and that have a breaking window later.
On them the policy is asked at every window up to the first breaking one while the opponent seat is made to pass
(forced passes), and the probability it puts on Ash is read: ``p_fire_bait`` = mean P(Ash) at window 1,
``p_fire_break`` = mean P(Ash) at the first breaking window. ``act_*`` are the same with the agent's own choice
(0 / 1: the greedy decision under ``@greedy``), ``p_fire_bait_all`` averages every bait window before the first
breaking one. The plan's acceptance reads ``p_fire_bait`` (≤ 0.4) and ``p_fire_break`` (≥ 0.9).

  uv run --no-sync python tools/handtrap_ruler.py LABELS.jsonl CKPT [CKPT ...] [--min-hand 40] [--sampled] [--json]

``LABELS.jsonl`` is a demonstration file with ``plain`` and ``fire`` records of the same hands (e.g.
``out/claude-scale/handtrap/full_fire8.jsonl``). Without ``--json`` it prints one ``name: key value ...`` line per
checkpoint (what ``handtrap_eval.py`` printed, R2b keys appended); with ``--json`` one JSON object per checkpoint
(``{"ckpt", "r2": {...}, "r2b": {...}}``), for loops such as ``colab/rulers_colab.sh``. ``--out`` writes all of them.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

ASH = 14558127  # Ash Blossom & Joyous Spring (train.line_labels.FIRE)


def ash_index(point, fire: int = ASH) -> int | None:
    """The candidate activating ``fire`` at ``point``, or None (not a window of the hand trap)."""
    for i, a in enumerate(point.actions):
        if a.card is not None and a.card.code == fire:
            return i
    return None


def recovered_windows(fire, env, cards) -> set[int]:
    """The window (1-based) at which each verified fire line activates Ash: the windows the combo recovers from."""
    from ygorl.engine.duel import DuelSession

    out = set()
    for ln in fire.lines:
        session = DuelSession(fire.replay(None).duel(env, cards=cards))
        window = 0
        try:
            for idx in ln.actions:
                point = session.point
                if point is None:
                    break
                if point.player == 1 and ash_index(point) is not None:
                    window += 1
                    a = point.actions[idx]
                    if a.card is not None and a.card.code == ASH:
                        out.add(window)
                        break
                session.act(idx)
        finally:
            session.close()
    return out


@dataclass
class Hand:
    """One labelled hand: the fire record (its start has Ash in the opponent's deck), the line's player-0 actions,
    the number of Ash windows and the recoverable ones."""

    fire: object  # ygorl.solver.Demonstration (variant "fire")
    p0: list[int]
    windows: int
    recovered: set[int] = field(default_factory=set)

    @property
    def breakable(self) -> bool:
        return len(self.recovered) < self.windows

    @property
    def first_break(self) -> int | None:
        return next((k for k in range(1, self.windows + 1) if k not in self.recovered), None)

    @property
    def bait_first(self) -> bool:
        """R2b's subset: window 1 is recoverable and a later window breaks."""
        return 1 in self.recovered and self.breakable


def load_hands(rows: Iterable[dict], env, cards, *, min_hand: int = 0, max_hand: int = 10**9,
               limit: int | None = None) -> list[Hand]:  # fmt: skip
    """The labelled hands of a demonstration file, in file order (``handtrap_eval.py``'s selection): fire records with
    at least one window in ``[min_hand, max_hand)``; hands whose recovered windows are not all identified (fewer
    distinct windows than the solver's ``converted`` count) are left out (ambiguous holds)."""
    from ygorl.solver import Demonstration

    rows = list(rows)
    plain = {(r["deck"]["name"], r["hand_index"]): Demonstration.from_json(r) for r in rows if r["variant"] == "plain"}
    fires = [Demonstration.from_json(r) for r in rows if r["variant"] == "fire" and (r["solver"].get("windows") or 0) > 0
             and min_hand <= r["hand_index"] < max_hand]  # fmt: skip
    if limit:
        fires = fires[:limit]
    hands = []
    for fire in fires:
        base = plain[(fire.deck["name"], fire.hand_index)]
        p0 = [i for i, pl in zip(base.lines[0].actions, base.lines[0].players, strict=True) if pl == 0]
        if fire.start is None:  # unsolved fire record: the plain start with Ash appended to the opponent's deck
            fire.start = json.loads(json.dumps(base.start))
            fire.start["decks"]["b"]["main"].append(ASH)
        W = fire.solver["windows"]
        conv = fire.solver.get("converted") or 0
        rec = set() if conv == 0 else (set(range(1, W + 1)) if conv == W else recovered_windows(fire, env, cards))
        if conv and conv != W and len(rec) < conv:
            continue
        hands.append(Hand(fire, p0, W, rec))
    return hands


def play(hand: Hand, agent, env, cards, *, probe: int | None = None):
    """Drive one hand from its fire start. Player 0 plays the line, player 1 passes except at Ash windows.

    ``probe=None`` (R2): the agent decides at every window; returns ``(window fired at or None, windows seen)``.
    ``probe=K`` (R2b): the agent is asked at windows 1..K but player 1 passes at each of them (forced passes);
    returns ``[(P(Ash), agent chose Ash), ...]`` for the windows reached.
    """
    from ygorl.engine.duel import DuelSession
    from ygorl.train.bc import passive_action

    session = DuelSession(hand.fire.replay(None).duel(env, cards=cards))
    tracker = session.tracker
    if hook := getattr(agent, "on_duel_start", None):
        hook(session.duel)
    it = iter(hand.p0)
    window, fired, asked = 0, None, []
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
                    chose = a.card is not None and a.card.code == ASH
                    if probe is not None:
                        probs = getattr(agent, "last_probs", None)
                        p = float(probs[ai]) if probs is not None else float(chose)
                        asked.append((p, bool(chose)))
                        if window >= probe:
                            break
                        idx = passive_action(point)
                    elif chose:
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
    return asked if probe is not None else (fired, window)


def r2_summary(hands: Sequence[Hand], fired: Sequence[int | None]) -> dict:
    """R2 from each hand's labels and the window the policy fired at (None: never)."""
    c: Counter = Counter()
    fired_at = []
    for h, k in zip(hands, fired, strict=True):
        if k is None:
            c["missed" if h.breakable else "held"] += 1
        else:
            c["wasted" if k in h.recovered else "broken"] += 1
            fired_at.append(k)
    n = len(hands)
    breakable = sum(h.breakable for h in hands)
    return {"hands": n, **c, "score": (c["broken"] + c["held"]) / max(1, n),
            "break_rate": c["broken"] / max(1, breakable), "waste_rate": c["wasted"] / max(1, c["wasted"] + c["broken"]),
            "fire_rate": (c["wasted"] + c["broken"]) / max(1, n),
            "mean_fired_window": sum(fired_at) / max(1, len(fired_at))}  # fmt: skip


def r2b_summary(hands: Sequence[Hand], asked: Sequence[Sequence[tuple[float, bool]]]) -> dict:
    """R2b from the bait-first hands and the forced-pass answers at their windows 1..first break (``play(probe=)``).
    A hand whose replay stops before its first breaking window counts for the bait windows it reached only."""
    bait, bait_all, brk, act_bait, act_brk = [], [], [], [], []
    for h, xs in zip(hands, asked, strict=True):
        b = h.first_break
        if not xs:
            continue
        bait.append(xs[0][0])
        act_bait.append(float(xs[0][1]))
        bait_all += [p for k, (p, _) in enumerate(xs, 1) if k in h.recovered and k < b]
        if len(xs) >= b:
            brk.append(xs[b - 1][0])
            act_brk.append(float(xs[b - 1][1]))

    def mean(v):
        return sum(v) / len(v) if v else float("nan")

    return {"bait_hands": len(hands), "p_fire_bait": mean(bait), "p_fire_break": mean(brk),
            "p_fire_bait_all": mean(bait_all), "act_fire_bait": mean(act_bait), "act_fire_break": mean(act_brk),
            "bait_windows": len(bait_all), "break_windows": len(brk)}  # fmt: skip


def evaluate(hands: Sequence[Hand], spec: str, env, cards, *, seed: int = 0) -> dict:
    """R2 and R2b of one agent spec (``policy:CKPT[@greedy]``). Agents are seeded per hand as handtrap_eval did
    (``seed * 100003 + n``); the R2b probes use fresh agents with the same seeds, so they do not shift R2's draws."""
    from ygorl.agents.registry import agent_factory

    factory = agent_factory(spec)
    t0 = time.time()
    fired = [play(h, factory(seed * 100003 + n), env, cards)[0] for n, h in enumerate(hands)]
    r2 = r2_summary(hands, fired)
    r2["s"] = round(time.time() - t0)
    subset = [(n, h) for n, h in enumerate(hands) if h.bait_first]
    asked = [play(h, factory(seed * 100003 + n), env, cards, probe=h.first_break) for n, h in subset]
    r2b = r2b_summary([h for _, h in subset], asked)
    return {"r2": r2, "r2b": r2b}


def _fmt(d: dict) -> str:
    return " ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in d.items())


def _name(ck: str) -> str:
    return ck.split("/ab1/")[-1] if "/ab1/" in ck else ck.split("/")[-1]


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("labels")
    p.add_argument("ckpts", nargs="+")
    p.add_argument("--sampled", action="store_true", help="sample from the policy instead of the argmax")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--min-hand", type=int, default=0, help="only hands with index >= N (40: the held-out hands)")
    p.add_argument("--max-hand", type=int, default=10**9, help="only hands with index < N")
    p.add_argument("--env", default="md-2026-09", help="environment version the labels were solved in")
    p.add_argument("--json", action="store_true", help="one JSON object per checkpoint on stdout")
    p.add_argument("--out", default=None, help="write {ckpt: {r2, r2b}} as JSON")
    a = p.parse_args(argv)

    import torch

    torch.set_num_threads(2)
    from ygorl.data import load_environment
    from ygorl.engine.duel import default_cards

    env = load_environment(a.env)
    cards = default_cards()
    with open(a.labels) as f:
        rows = [json.loads(line) for line in f]
    hands = load_hands(rows, env, cards, min_hand=a.min_hand, max_hand=a.max_hand, limit=a.limit)
    log = sys.stderr if a.json else sys.stdout
    print(f"{len(hands)} hands; breaking window exists in {sum(h.breakable for h in hands)}; "
          f"bait first in {sum(h.bait_first for h in hands)}", file=log, flush=True)  # fmt: skip
    results = {}
    for ck in a.ckpts:
        res = evaluate(hands, f"policy:{ck}" + ("" if a.sampled else "@greedy"), env, cards, seed=a.seed)
        results[ck] = res
        if a.json:
            print(json.dumps({"ckpt": ck, **res}), flush=True)
        else:
            print(f"{_name(ck)}: {_fmt(res['r2'])} | R2b {_fmt(res['r2b'])}", flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(results, f, indent=1)
    if not a.json:
        print("EVAL-DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
