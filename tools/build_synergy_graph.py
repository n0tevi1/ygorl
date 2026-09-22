"""Build the script-mined synergy graph (T5.3), print its statistics and the proxy recall.

Usage: uv run python tools/build_synergy_graph.py [--out graph.json.gz] [--workers 2] [--max-fanout 100]

Without ``--out`` the graph goes to the on-disk cache used by
``ygorl.build.synergy_graph.load_or_build`` (``$YGORL_CACHE_DIR`` or ~/.cache/ygorl).
With ``--environment <version|dir>`` the graph is restricted to that
environment's card pool, stamped with it and written to its artifacts/
directory (unless ``--out`` is given).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

from ygorl.build.synergy_graph import DEFAULT_MAX_FANOUT, REACH_TYPES, build_graph, evaluate_recall, load_or_build

ROOT = Path(__file__).resolve().parents[1]


def proxy_packages() -> dict[str, list[int]]:
    spec = importlib.util.spec_from_file_location("make_proxy_packages", ROOT / "tools" / "make_proxy_packages.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_packages()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, help="write the graph here (.json or .json.gz)")
    parser.add_argument("--workers", type=int, default=1, help="script-analysis processes (keep <= 2 on shared machines)")
    parser.add_argument("--max-fanout", type=int, default=DEFAULT_MAX_FANOUT)
    parser.add_argument("--environment", help="restrict to an environment's card pool and stamp the graph")
    parser.add_argument("--threshold", type=float, default=0.8, help="package coverage needed to count as recovered")
    args = parser.parse_args()

    t0 = time.perf_counter()
    if args.out is None and args.environment is None:
        graph = load_or_build(max_fanout=args.max_fanout, workers=args.workers)
    else:
        graph = build_graph(max_fanout=args.max_fanout, workers=args.workers)
    if args.environment:
        from ygorl.data.environment import load_environment

        env = load_environment(args.environment)
        graph = graph.restrict(env.card_pool, stamp=env.stamp())
        out = args.out or env.artifact_path("synergy_graph.json.gz")
    else:
        out = args.out
    if out is not None:
        graph.save(out)
        print(f"wrote {out}")
    print(f"built/loaded in {time.perf_counter() - t0:.1f}s")
    print(json.dumps(graph.summary(), indent=1))

    packages = proxy_packages()
    for label, types in (("all edge types", None), ("search + special_summon", REACH_TYPES)):
        report = evaluate_recall(graph, packages, threshold=args.threshold, types=types)
        print(f"\nproxy recall ({label}, coverage >= {args.threshold}): {report.recall:.3f} "
              f"({len(report.recovered)}/{len(report.coverage)}), mean coverage {report.mean_coverage:.3f}")  # fmt: skip
        for name, cov in sorted(report.coverage.items()):
            print(f"  {name:18s} {cov:.2f}{'' if cov >= args.threshold else '  (missed)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
