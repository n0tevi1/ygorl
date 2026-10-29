"""Fixed-panel evaluation, identical to the terminal-critic study's evaluate_node: 64 pairings (seed 2026100904) of the
md-2026-09 meta decks x 4 games, against greedy / old-256x2 / initial-128x2 / historical-rl (256 games each; their
average is `mean`, the legacy score) plus two strong anchors E2016 / G576 (`mean_strong`). Every opponent entry and
the top-level `overall` report the candidate's win rate going first and second and the mean game length in turns.

  uv run --no-sync python out/claude-scale/panel.py CKPT [CKPT...] [--workers 8] [--opponents greedy,...]
Writes <ckpt dir>/panel/<ckpt stem>/<opponent>.jsonl + summary.json; prints score per opponent and the mean.
"""

import argparse, json, multiprocessing, os, sys, time
from dataclasses import asdict
from pathlib import Path

ROOT = Path("/home/ya0guang/Code/ygorl/.claude/worktrees/claude-rl-review")
os.chdir(ROOT)
R = Path("/home/ya0guang/Code/ygorl/out/research")
S = Path("/home/ya0guang/Code/ygorl/out/claude-scale/ab1")
OPP = {
    "greedy": "greedy",
    "old-256x2": f"policy:{R / 'bounded-capacity-2026-10-05/models/256x2/policy.pt'}",
    "initial-128x2": f"policy:{R / 'terminal-critic-policy-long-2026-10-06/seed-0/cold/run/checkpoints/update_00000000.pt'}",
    "historical-rl": f"policy:{R / 'terminal-critic-policy-long-2026-10-06/seed-0/warm/run/checkpoints/update_00000128.pt'}",
    # strong anchors (2026-10-22): the two best checkpoints of the scale-up arms; 0.50 head-to-head against each other
    # "E2016" (E-long-bf16 update_002016.pt) was lost in the 2026-10-27 disk cleanup (its chunk checkpoints were not
    # multiples of 64); mean_strong is G576 alone from then on — recompute old rows from their per-opponent results.
    "G576": f"policy:{S / 'G-sepcritic-init-bf16/checkpoints/update_000576.pt'}",
    # the turn-1 teacher (#442): the only opponent that builds an interruption board going first
    "teacher": "policy:/home/ya0guang/Code/ygorl/out/claude-scale/teacher/teacher_full.pt",
}
LEGACY = ("greedy", "old-256x2", "initial-128x2", "historical-rl")
STRONG = ("G576",)


def seat_stats(ok):
    """win rate going first / second and mean turns over healthy decided games (candidate is player 0)."""
    first = [r for r in ok if r.first == 0]
    second = [r for r in ok if r.first == 1]
    return {
        "first": sum(r.winner == 0 for r in first) / max(1, len(first)),
        "second": sum(r.winner == 0 for r in second) / max(1, len(second)),
        "mean_turns": sum(r.turns for r in ok) / max(1, len(ok)),
    }


EVAL_SEED = 2026100904


def play(job):
    import torch
    from ygorl.engine.duel import Duel
    from ygorl.eval.arena import GameRecord

    torch.set_num_threads(1)
    i, spec = job
    base = {"pair": spec.pair, "first": spec.first, "seed": spec.seed}
    try:
        duel = Duel(spec.seed, spec.env, spec.deck_a, spec.deck_b, config=spec.config, first=spec.first)
        r = duel.run(spec.agent_a(spec.agent_seeds[0]), spec.agent_b(spec.agent_seeds[1]))
        rec = GameRecord(
            **base,
            winner=r.winner,
            reason=r.reason,
            turns=r.turns,
            decisions=r.decisions,
            win_reason=r.win_reason,
            lp=r.lp,
            retries=r.retries,
            unknown_messages=r.unknown_messages,
            undecodable_messages=r.undecodable_messages,
            error=r.error,
            script_errors=len(r.script_errors),
        )
    except Exception as exc:  # noqa: BLE001
        rec = GameRecord(**base, winner=None, reason="exception", error=repr(exc))
    return i, asdict(rec)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", nargs="+")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--opponents", default=",".join(OPP))
    p.add_argument("--extra", action="append", default=[], help="name=checkpoint, added to the opponent table")
    p.add_argument(
        "--decks", default=None, help="directory of .ydk files to pilot instead of the environment meta decks"
    )
    p.add_argument("--tag", default="panel", help="output subdirectory name")
    a = p.parse_args()
    for item in a.extra:
        n, path = item.split("=", 1)
        OPP[n] = f"policy:{path}"
    from ygorl.agents.registry import agent_factory
    from ygorl.data import load_environment
    from ygorl.engine.duel import DuelConfig
    from ygorl.eval.agent_matrix import cell_specs, pairing_slots, sample_pairings
    from ygorl.eval.arena import GameRecord, wilson_interval

    env = load_environment("md-2026-09")
    cfg = DuelConfig.from_environment(env)
    if a.decks:
        from ygorl.cards.ydk import load_ydk

        decks = [load_ydk(str(f)) for f in sorted(Path(a.decks).glob("*.ydk"))]
    else:
        decks = [m.deck for m in sorted(env.meta_decks, key=lambda m: m.deck.name)]
    pairs = sample_pairings(len(decks), 64, EVAL_SEED)
    slots = pairing_slots(decks, pairs, EVAL_SEED, cfg)
    with multiprocessing.get_context("spawn").Pool(a.workers) as pool:
        for ck in a.ckpts:
            ck = Path(ck).resolve()
            out = ck.parent / a.tag / ck.stem
            out.mkdir(parents=True, exist_ok=True)
            summary = {}
            all_ok = []
            for name in a.opponents.split(","):
                raw = out / f"{name}.jsonl"
                if raw.exists():
                    rows = [json.loads(l) for l in raw.open()]
                else:
                    specs = cell_specs(agent_factory(f"policy:{ck}"), agent_factory(OPP[name]), slots, env, cfg)
                    t0 = time.time()
                    rows = [None] * len(specs)
                    for i, rec in pool.imap_unordered(play, list(enumerate(specs)), chunksize=1):
                        rows[i] = rec
                    with raw.open("w") as f:
                        for r in rows:
                            f.write(json.dumps(r) + "\n")
                    print(f"  {name}: {len(specs)} games in {time.time() - t0:.0f}s", flush=True)
                recs = [GameRecord(**r) for r in rows]
                ok = [r for r in recs if r.healthy and r.reason == "win" and r.winner in (0, 1)]
                wins = sum(r.winner == 0 for r in ok)
                lo, hi = wilson_interval(wins, len(ok))
                all_ok += ok
                summary[name] = {
                    "games": len(recs),
                    "healthy": len(ok),
                    "wins": wins,
                    "score": wins / max(1, len(ok)),
                    "ci": [lo, hi],
                    **seat_stats(ok),
                }
            groups = {"mean": LEGACY, "mean_strong": STRONG}
            for key, names in groups.items():
                got = [summary[n]["score"] for n in names if n in summary]
                if got:
                    summary[key] = sum(got) / len(got)
            summary["overall"] = seat_stats(all_ok)
            (out / "summary.json").write_text(json.dumps(summary, indent=1))
            line = " ".join(f"{k} {v['score']:.3f}" for k, v in summary.items() if isinstance(v, dict) and "score" in v)
            o = summary["overall"]
            print(
                f"{ck}: mean {summary.get('mean', float('nan')):.4f} strong {summary.get('mean_strong', float('nan')):.4f} "
                f"first {o['first']:.3f} second {o['second']:.3f} turns {o['mean_turns']:.1f} | {line}",
                flush=True,
            )


if __name__ == "__main__":
    main()
