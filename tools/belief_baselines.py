"""Baseline numbers for the belief-calibration report (T3.5).

Usage: uv run python tools/belief_baselines.py [--n 20000] [--seed 0] [--bins 15]

Prints a Markdown table of `evaluate_beliefs` for the uniform / random / prior / oracle
predictors on a synthetic batch (see docs/belief-eval.md).
"""

from __future__ import annotations

import argparse

from ygorl.eval.beliefs import baseline_report, format_baselines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=20_000, help="synthetic samples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bins", type=int, default=15, help="ECE bins")
    args = ap.parse_args()
    print(format_baselines(baseline_report(args.n, seed=args.seed, n_bins=args.bins)))


if __name__ == "__main__":
    main()
