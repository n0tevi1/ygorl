"""Tune a deck by one-card swaps judged by policy games against the environment's meta (ygorl.build.tuner).

  uv run python tools/tune_deck.py BASE.ydk --env md-2026-09 --checkpoint CKPT [--candidates 64] \\
      [--first-pairs 25] [--finalists 3] [--device cuda] [--envs 256] [--out out/tune/NAME]

The base deck's type is the type of its nearest list in the environment's deck corpus (card-count L1 distance);
candidate cards come from the other corpus lists of that type and the cards most common across the meta lists.
``--candidates`` random legal swaps go through successive halving (docs/tuning.md); the finalists and the base deck
then replay ``--validation-pairs`` fresh pairs, and only those decide significance (the search's own games are biased
by the selection). Writes ``report.json`` (every finalist's edit, search difference, fresh paired difference to the
base deck with its 95% interval, games, engine errors, seconds) and, when the best fresh interval is above 0,
``tuned.ydk``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", type=Path)
    ap.add_argument("--env", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--opponent-checkpoint", default=None, help="pilot of the meta decks (default: --checkpoint)")
    ap.add_argument("--candidates", type=int, default=64)
    ap.add_argument("--first-pairs", type=int, default=25)
    ap.add_argument("--finalists", type=int, default=3)
    ap.add_argument(
        "--validation-pairs",
        type=int,
        default=200,
        help="fresh pairs replayed by the finalists and the base deck; only they decide significance",
    )
    ap.add_argument("--meta-top", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import numpy as np
    import torch

    from ygorl.build.tuner import PairedEvaluator, neighbors, paired_difference, successive_halving, tech_pool, validate
    from ygorl.cards.ydk import load_ydk
    from ygorl.data.environment import load_environment
    from ygorl.engine.duel import default_cards
    from ygorl.env.encoded import EncodedVecEnv
    from ygorl.train.checkpoint import load_actor, vocab_passwords

    cards = default_cards()
    env = load_environment(args.env, cards=cards)
    base = load_ydk(args.base)
    if env.validate_deck(base, cards):
        raise SystemExit(f"{args.base}: not legal in {env.version}: {env.validate_deck(base, cards)}")
    corpus = json.loads(env.artifact_path("deck_corpus.json").read_text())
    lists = [(e["type"], load_ydk(env.artifacts_dir / e["file"])) for e in corpus["decks"]]
    base_c = base.counts()

    def dist(d):
        c = d.counts()
        return sum(abs(base_c[k] - c[k]) for k in base_c.keys() | c.keys())

    dtype = min(lists, key=lambda t: dist(t[1]))[0]
    same = [d for t, d in lists if t == dtype and dist(d) > 0]
    meta = [m.deck for m in env.meta_decks]
    pool = tech_pool(base, same, meta, meta_top=args.meta_top)
    rng = np.random.default_rng(args.seed)
    edits = neighbors(base, pool, lambda pw: cards[pw].is_extra_deck if pw in cards else False,
                      lambda d: not env.validate_deck(d, cards), limit=args.candidates, rng=rng)  # fmt: skip
    names = {pw: cards[pw].name for pw in pool + list(base.main) + list(base.extra) if pw in cards}
    print(
        f"{args.base.name}: type {dtype!r} ({len(same)} other lists), {len(pool)} candidate cards, "
        f"{len(edits)} legal swaps sampled",
        flush=True,
    )

    pol = load_actor(args.checkpoint)
    opp = load_actor(args.opponent_checkpoint) if args.opponent_checkpoint else pol
    if vocab_passwords(opp.vocab) != vocab_passwords(pol.vocab) or opp.event_length != pol.event_length:
        raise SystemExit("the two checkpoints need the same card vocab and event length")
    device = torch.device(args.device)
    net_a = pol.net.to(device)
    net_b = net_a if opp is pol else opp.net.to(device)
    evaluator = PairedEvaluator(
        env_factory=lambda n: EncodedVecEnv(n, args.threads, cards=cards, vocab=pol.vocab,
                                            event_length=pol.event_length, skip_forced=True),
        policy=net_a, opponent=net_b, opponents=meta, weights=[m.share for m in env.meta_decks], seed=args.seed,
        device=args.device, num_envs=args.envs)  # fmt: skip
    base_cand, finals = successive_halving(base, edits, evaluator, first_pairs=args.first_pairs,
                                           finalists=args.finalists, log=lambda m: print(m, flush=True))  # fmt: skip
    rows = []
    checked = validate(base, finals, evaluator, args.validation_pairs) if finals else []
    for c, fresh, base_fresh in checked:
        search = paired_difference(c.scores, base_cand.scores)
        mean, lo, hi = paired_difference(fresh, base_fresh)
        rows.append({"edit": c.edit.describe(names), "out": c.edit.out, "into": c.edit.into,
                     "section": c.edit.section, "search_pairs": len(c.scores), "search_diff": search[0],
                     "validation_pairs": len(fresh), "win_rate": float(np.nanmean(fresh)),
                     "base_win_rate": float(np.nanmean(base_fresh)), "diff": mean, "ci": [lo, hi],
                     "deck": c.deck})  # fmt: skip
    rows.sort(key=lambda r: r["diff"], reverse=True)
    for r in rows:
        print(
            f"  {r['edit']:60s} search {r['search_diff']:+.3f} | fresh {r['diff']:+.3f} ({r['ci'][0]:+.3f}, "
            f"{r['ci'][1]:+.3f}) over {r['validation_pairs']} pairs"
        )
    print(
        f"base win rate {np.nanmean(base_cand.scores):.3f} over {len(base_cand.scores)} search pairs; "
        f"{evaluator.games} games ({evaluator.errors} engine errors) in {evaluator.seconds:.0f}s"
    )
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        report = {"base": str(args.base), "env": env.stamp(), "checkpoint": args.checkpoint, "type": dtype,
                  "candidate_cards": len(pool), "candidates": len(edits),
                  "base_win_rate": float(np.nanmean(base_cand.scores)), "base_pairs": len(base_cand.scores),
                  "finalists": [{k: v for k, v in r.items() if k != "deck"} for r in rows], "games": evaluator.games,
                  "errors": evaluator.errors, "seconds": evaluator.seconds}  # fmt: skip
        (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
        if rows and rows[0]["ci"][0] > 0:
            best = rows[0]["deck"]
            (args.out / "tuned.ydk").write_text(best.to_ydk(), encoding="utf-8")
            print(f"significant improvement: wrote {args.out / 'tuned.ydk'}")
        else:
            print("no finalist is significantly better than the base deck")
    return 0


if __name__ == "__main__":
    sys.exit(main())
