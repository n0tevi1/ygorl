"""Fit the edit-value model (#151, ygorl.build.value_model; docs/tuning.md「改动价值模型」), or cross-validate it.

  .venv/bin/python tools/fit_value_model.py --env md-2026-09 --deck-model out/deckmodel/full.pt \\
      --target-checkpoint CKPT --paired SRC[@CKPT] ... [--games RUN ...] [--same-run DIR ...] \\
      [--out MODEL.pt] [--cv OUT.json] [--folds deck|random]

``--paired`` (repeatable) names a paired-evaluation source (ygorl.build.edit_labels.paired_labels: M1, M2, an
evolution state or lineage, a tuner comparison or re-validation, a factorial measurement); ``@CKPT`` gives the
policy that played it when the file does not record it (M2). ``--games RUN`` (repeatable) adds a training run's game
log (``--log-games``) as the auxiliary loss. ``--target-checkpoint`` is the policy the model predicts for: labels of
older updates of the same run (or of a ``--same-run`` directory: a continuation) are down-weighted by
``0.5 ** (age / --half-life)``, labels of another run count ``--foreign-age`` updates old.

``--out`` writes the fitted model (with the deck model's path and sha256); ``--cv`` writes held-out predictions
(whole parent decks left out by default) of the model, of the card-value model warm-started from the same folds and
of the masked model's own scores, and prints the Spearman table.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def parse_source(text: str) -> tuple[Path, str | None]:
    path, _, ckpt = text.partition("@")
    return Path(path), ckpt or None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", required=True)
    ap.add_argument("--deck-model", type=Path, required=True)
    ap.add_argument("--target-checkpoint", required=True, help="the policy the model predicts for")
    ap.add_argument("--target-update", type=int, default=None, help="its update (default: from the file name)")
    ap.add_argument("--paired", action="append", default=[], metavar="SRC[@CKPT]", help="paired evaluations")
    ap.add_argument("--games", type=Path, action="append", default=[], metavar="RUN", help="run with games.jsonl.gz")
    ap.add_argument("--same-run", action="append", default=[], metavar="DIR",
                    help="another run directory that continues the target's run")  # fmt: skip
    ap.add_argument("--half-life", type=float, default=200.0, help="label age (updates) that halves its weight")
    ap.add_argument("--foreign-age", type=float, default=400.0, help="age of a label of another run")
    ap.add_argument("--out", type=Path, default=None, help="fitted model (.pt)")
    ap.add_argument("--cv", type=Path, default=None, help="cross-validation report (JSON)")
    ap.add_argument("--folds", choices=("deck", "random"), default="deck")
    ap.add_argument("--members", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--aux-weight", type=float, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.out is None and args.cv is None:
        ap.error("give --out and / or --cv")

    import numpy as np

    from ygorl.build.deck_model import load_deck_model
    from ygorl.build.edit_labels import Clock, merge_games, paired_labels, read_game_log, unique
    from ygorl.build.evolve import check_environment
    from ygorl.build.value_model import (DeckModelFeatures, ValueModelConfig, cross_validate, file_sha256, fit,
                                         group_folds, summarize)  # fmt: skip
    from ygorl.data.environment import load_environment

    def say(msg):
        print(msg, flush=True)

    env = load_environment(args.env)
    deck_model = load_deck_model(args.deck_model)
    check_environment(env, deck_model.environment, f"deck model {args.deck_model}")
    clock = Clock.at(args.target_checkpoint, args.target_update, half_life=args.half_life,
                     foreign_age=args.foreign_age, same=tuple(args.same_run))  # fmt: skip
    labels = []
    for text in args.paired:
        path, ckpt = parse_source(text)
        got, skipped = paired_labels(path, env.artifacts_dir, checkpoint=ckpt)
        say(f"{path}: {len(got)} labels" + (f" ({skipped} skipped)" if skipped else ""))
        labels += got
    labels = unique(labels)
    games = None
    if args.games:
        games = merge_games(read_game_log(r) for r in args.games)
        say(f"games: {len(games.games)} decided games of {len(games.decks)} decks ({games.skipped} left out)")
    if not labels:
        raise SystemExit("no paired labels")
    over = {k: v for k, v in (("members", args.members), ("hidden", args.hidden), ("weight_decay", args.weight_decay),
                              ("aux_weight", args.aux_weight), ("steps", args.steps)) if v is not None}  # fmt: skip
    cfg = ValueModelConfig(**over, seed=args.seed)
    features = DeckModelFeatures(deck_model)
    groups = sorted({lab.group for lab in labels})
    say(f"{len(labels)} labels, {len(groups)} parent decks; target {clock.run} @ {clock.update}")
    if args.cv is not None:
        folds = group_folds(labels, args.folds, seed=args.seed)
        preds = cross_validate(features, labels, clock, cfg, folds=folds, games=games, log=say)
        table = summarize(labels, preds)
        say(f"{'method':<14} {'pooled':>8} {'within':>8} {'precise':>8} {'top gain':>9}")
        for name, row in table.items():
            say(f"{name:<14} {row['pooled']:+8.3f} {row['within']:+8.3f} {row['precise']:+8.3f} "
                f"{row['top_gain'] * 100:+8.2f}pp")  # fmt: skip
        args.cv.parent.mkdir(parents=True, exist_ok=True)
        args.cv.write_text(json.dumps({
            "config": cfg.__dict__, "clock": clock.to_dict(), "folds": args.folds, "summary": table,
            "labels": [{"id": lab.id, "source": lab.source, "type": lab.deck_type, "diff": lab.diff,
                        "stderr": lab.stderr, "checkpoint": lab.checkpoint.to_dict(), "fold": int(f),
                        **{k: (None if not math.isfinite(float(v[i])) else float(v[i])) for k, v in preds.items()}}
                       for i, (lab, f) in enumerate(zip(labels, folds, strict=True))]}, indent=1) + "\n")  # fmt: skip
    if args.out is not None:
        meta = {"deck_model": {"path": str(args.deck_model), "sha256": file_sha256(args.deck_model)},
                "environment": dict(env.stamp()), "labels": len(labels), "sources": args.paired,
                "games": [str(r) for r in args.games]}  # fmt: skip
        model, rep = fit(features, labels, clock, cfg, games=games, meta=meta, log=say)
        model.meta["fit"] = {"sigma0": rep.sigma0, "oob_spearman": None if np.isnan(rep.oob_spearman) else
                             rep.oob_spearman, "games": rep.games}  # fmt: skip
        args.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.out.with_name(f".{args.out.name}.tmp")
        model.save(tmp)
        tmp.replace(args.out)
        say(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
