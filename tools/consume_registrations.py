#!/usr/bin/env python3
"""Extend a fixed strength matrix with training registrations, once; safe to repeat independently of training."""

import argparse
import json
from pathlib import Path

from ygorl.commands import load_decks, load_env
from ygorl.train.registration import consume_registrations


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", type=Path, nargs="+")
    p.add_argument("--matrix", type=Path, required=True)
    p.add_argument("--env")
    p.add_argument("--decks", type=Path, nargs="+")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--device", help="only for a matrix created with batched evaluation")
    p.add_argument("--curve", type=Path)
    a = p.parse_args()
    env = load_env(a.env)
    if a.decks:
        pool = load_decks(a.decks, env)
        decks = {d.name: d for d in pool}
        if len(decks) != len(pool):
            p.error("duplicate deck names")
    elif env is not None:
        decks = {md.name: md.deck for md in env.meta_decks}
    else:
        p.error("pass --decks or --env with meta decks")
    curve = consume_registrations(
        a.runs, a.matrix, decks, env=env, workers=a.workers, device=a.device, curve_path=a.curve
    )
    print(json.dumps({"matrix": curve["matrix"], "agents": curve["agents"], "checkpoints": len(curve["records"])}))


if __name__ == "__main__":
    main()
