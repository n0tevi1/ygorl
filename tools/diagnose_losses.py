"""Where does a policy lose? (strength diagnosis, #83)

Plays a policy checkpoint on random corpus deck pairings, both first players:

- ``--vs greedy`` (arena path, CPU workers): the policy against the one-ply Greedy baseline;
- ``--vs self`` (batched path): the policy against itself, to see how much the deck matchup alone decides.

Writes one JSON line per game (decks, who went first, winner, end reason, turns, decisions, LP) and prints loss
breakdowns: by first / second, by the turn the game ended, by end reason, by the policy's deck and the opponent's
deck (worst decks with enough games), by the final LP gap, and for ``--vs self`` the share of outcome variance the
deck pairing explains.

Usage: tools/diagnose_losses.py CHECKPOINT --vs greedy|self [--pairings N] [--out games.jsonl] [--workers W]
"""

from __future__ import annotations

import argparse
import functools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ygorl.agents.registry import make_agent
from ygorl.cards.ydk import load_ydk
from ygorl.engine.duel import DuelConfig
from ygorl.eval.arena import Arena, derive_seed


def pairings(decks, n, seed):
    rng = np.random.default_rng(seed)
    return [tuple(int(x) for x in rng.choice(len(decks), 2, replace=False)) for _ in range(n)]


def play_greedy(ckpt, decks, pairs, workers, seed):
    arena = Arena(functools.partial(_agent, f"policy:{ckpt}"), functools.partial(_agent, "greedy"),
                  config=DuelConfig(max_decisions=4000), workers=workers)  # fmt: skip
    specs = []
    for k, (a, b) in enumerate(pairs):
        specs.extend((k, a, b, s) for s in arena.game_specs(decks[a], decks[b], 1, derive_seed(seed, k)))
    records = arena.play([s for *_, s in specs])
    out = []
    for (k, a, b, _), r in zip(specs, records, strict=True):
        # agent a = policy on deck a; record.first: 0 = deck/agent a went first
        score = None if r.reason == "exception" else (0.5 if r.winner is None else float(r.winner == 0))
        out.append({"pairing": k, "deck": decks[a].name, "opp_deck": decks[b].name, "first": r.first == 0,
                    "score": score, "reason": r.reason, "win_reason": r.win_reason, "turns": r.turns,
                    "decisions": r.decisions, "lp": list(r.lp)})  # fmt: skip
    return out


def play_self(ckpt, decks, pairs, seed, device):
    import torch

    from ygorl.env.encoded import EncodedVecEnv
    from ygorl.eval.batched import paired_specs, play_policies
    from ygorl.train.checkpoint import load_actor

    pol = load_actor(ckpt)
    net = pol.net.to(device)
    config = DuelConfig(max_decisions=4000)
    specs, meta = [], []
    for k, (a, b) in enumerate(pairs):
        for s in paired_specs(decks[a], decks[b], 1, derive_seed(seed, k), config):
            specs.append(s)
            meta.append((k, a, b))
    env = EncodedVecEnv(min(256, len(specs)), 8, vocab=pol.vocab, event_length=pol.event_length, skip_forced=True)
    with torch.no_grad():
        records, _ = play_policies(env, specs, net, device=device)
    out = []
    for (k, a, b), sp, r in zip(meta, specs, records, strict=True):
        score = None if r.reason == "exception" else (0.5 if r.winner is None else float(r.winner == 0))
        out.append({"pairing": k, "deck": decks[a].name, "opp_deck": decks[b].name, "first": sp.first == 0,
                    "score": score, "reason": r.reason, "win_reason": r.win_reason, "turns": r.turns,
                    "decisions": r.decisions, "lp": list(r.lp)})  # fmt: skip
    return out


def _agent(spec, seed):
    return make_agent(spec, seed)


def report(games, vs):
    g = [x for x in games if x["score"] is not None]
    s = np.array([x["score"] for x in g])
    print(f"{len(g)} games (errors {len(games) - len(g)}), policy score {s.mean():.3f}")

    def split(name, key):
        by = defaultdict(list)
        for x in g:
            by[key(x)].append(x["score"])
        print(
            f"  by {name}: "
            + ", ".join(f"{k}: {np.mean(v):.3f} (n={len(v)})" for k, v in sorted(by.items(), key=lambda t: str(t[0])))
        )

    split("first", lambda x: "first" if x["first"] else "second")
    split("end turn", lambda x: "1-2" if x["turns"] <= 2 else "3-4" if x["turns"] <= 4 else "5-6" if x["turns"] <= 6
          else "7-10" if x["turns"] <= 10 else "11+")  # fmt: skip
    losses = [x for x in g if x["score"] == 0.0]
    wins = [x for x in g if x["score"] == 1.0]
    print(f"  losses {len(losses)}: end reason {dict(Counter(x['reason'] for x in losses))}, "
          f"win_reason {dict(Counter(x['win_reason'] for x in losses))}")  # fmt: skip
    for name, xs in (("loss", losses), ("win", wins)):
        if xs:
            t = np.array([x["turns"] for x in xs])
            print(f"  {name} turns: median {np.median(t):.0f}, quartiles {np.percentile(t, 25):.0f}-{np.percentile(t, 75):.0f}; "
                  f"first-player share {np.mean([x['first'] for x in xs]):.2f}")  # fmt: skip
    # LP at the end, from the policy's side (lp = (deck a, deck b))
    gap = np.array([x["lp"][0] - x["lp"][1] for x in losses]) if losses else np.array([])
    if len(gap):
        print(
            f"  loss LP gap (policy - opp): median {np.median(gap):.0f}; opp LP ≤ 2000 in {np.mean([x['lp'][1] <= 2000 for x in losses]):.2f} of losses"
        )
    for side in ("deck", "opp_deck"):
        by = defaultdict(list)
        for x in g:
            by[x[side]].append(x["score"])
        rows = sorted(((np.mean(v), len(v), k) for k, v in by.items() if len(v) >= 6))
        print(f"  worst {side}s (≥6 games): " + "; ".join(f"{k} {m:.2f}/{n}" for m, n, k in rows[:8]))
        spread = np.std([m for m, _, _ in rows]) if rows else float("nan")
        print(f"  {side} score spread (sd over decks with ≥6 games): {spread:.3f} over {len(rows)} decks")
    if vs == "self":
        # outcome variance explained by the pairing (both games of a pairing share decks, first player differs)
        by = defaultdict(list)
        for x in g:
            by[x["pairing"]].append(x["score"])
        pair_means = np.array([np.mean(v) for v in by.values() if len(v) == 2])
        within = np.mean([np.var(v) for v in by.values() if len(v) == 2])
        print(f"  pairing effect: sd of pairing means {pair_means.std():.3f}; total var {s.var():.3f}, within-pairing var {within:.3f} "
              f"-> deck pairing explains ~{1 - within / s.var():.2f} of outcome variance")  # fmt: skip


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--vs", choices=("greedy", "self"), required=True)
    p.add_argument("--decks", default="out/corpus/train")
    p.add_argument("--pairings", type=int, default=500)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    decks = [load_ydk(f) for f in sorted(Path(args.decks).glob("*.ydk"))]
    pairs = pairings(decks, args.pairings, args.seed)
    games = (play_greedy(args.checkpoint, decks, pairs, args.workers, args.seed) if args.vs == "greedy"
             else play_self(args.checkpoint, decks, pairs, args.seed, args.device))  # fmt: skip
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("".join(json.dumps(x) + "\n" for x in games))
    report(games, args.vs)


if __name__ == "__main__":
    main()
