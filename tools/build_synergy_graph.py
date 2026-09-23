"""Build the script-mined synergy graph (T5.3), print its statistics and the proxy recall.

Usage: uv run python tools/build_synergy_graph.py [--out graph.json.gz] [--workers 2] [--max-fanout 100]

The graph comes from (or is added to) the on-disk cache used by
``ygorl.build.synergy_graph.load_or_build`` (``$YGORL_CACHE_DIR`` or ~/.cache/ygorl);
``--out`` also writes it to a file.
With ``--environment <version|dir>`` the graph is restricted to that
environment's card pool, stamped with it and written to its artifacts/
directory (unless ``--out`` is given), and the recall on the environment's
meta engine packages (artifacts/meta_packages.json, tools/make_meta_packages.py)
is printed as well. ``--relations [TYPE,...]`` (needs ``--environment``) merges
the environment's Yugipedia relations as extra edge types
(``ygorl.build.relations``; default ``archetype_support``) and prints the recall
with and without them.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

from ygorl.build.relations import DEFAULT_RELATION_TYPES, RELATION_TYPES, add_relation_edges, load_relations
from ygorl.build.synergy_graph import DEFAULT_MAX_FANOUT, REACH_TYPES, evaluate_recall, load_or_build

ROOT = Path(__file__).resolve().parents[1]


def _tool(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def proxy_packages() -> dict[str, list[int]]:
    return _tool("make_proxy_packages").load_packages()


def print_recall(title: str, graph, packages: dict[str, list[int]], threshold: float, extra_types: tuple[str, ...] = ()) -> None:
    for label, types in (("all edge types", None), ("search + special_summon" + "".join(f" + {t}" for t in extra_types), REACH_TYPES + extra_types)):
        report = evaluate_recall(graph, packages, threshold=threshold, types=types)
        print(f"\n{title} ({label}, coverage >= {threshold}): {report.recall:.3f} "
              f"({len(report.recovered)}/{len(report.coverage)}), mean coverage {report.mean_coverage:.3f}")  # fmt: skip
        for name, cov in sorted(report.coverage.items()):
            print(f"  {name:24s} {cov:.2f}{'' if cov >= threshold else '  (missed)'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, help="write the graph here (.json or .json.gz)")
    parser.add_argument("--workers", type=int, default=1, help="script-analysis processes (keep <= 2 on shared machines)")
    parser.add_argument("--max-fanout", type=int, default=DEFAULT_MAX_FANOUT)
    parser.add_argument("--environment", help="restrict to an environment's card pool and stamp the graph")
    parser.add_argument("--threshold", type=float, default=0.8, help="package coverage needed to count as recovered")
    parser.add_argument("--relations", nargs="?", const=",".join(DEFAULT_RELATION_TYPES), metavar="TYPE[,TYPE]",
                        help=f"merge Yugipedia relations of the environment (types: {', '.join(RELATION_TYPES)})")  # fmt: skip
    args = parser.parse_args()
    if args.relations and not args.environment:
        parser.error("--relations needs --environment")

    t0 = time.perf_counter()
    graph = load_or_build(max_fanout=args.max_fanout, workers=args.workers)  # the disk cache is keyed on all inputs
    env = script_graph = None
    if args.environment:
        from ygorl.data.environment import load_environment

        env = load_environment(args.environment)
        graph = graph.restrict(env.card_pool, stamp=env.stamp())
        out = args.out or env.artifact_path("synergy_graph.json.gz")
        if args.relations:
            script_graph = graph.restrict(graph.nodes, stamp=env.stamp())  # copy without relation edges
            add_relation_edges(graph, load_relations(env), args.relations.split(","))
    else:
        out = args.out
    if out is not None:
        graph.save(out)
        print(f"wrote {out}")
    print(f"built/loaded in {time.perf_counter() - t0:.1f}s")
    print(json.dumps(graph.summary(), indent=1))

    base = script_graph if script_graph is not None else graph
    print_recall("proxy recall", base, proxy_packages(), args.threshold)
    if env is not None:
        meta_path = env.root / "artifacts" / "meta_packages.json"
        if not meta_path.is_file():
            print(f"\nno {meta_path}; run tools/make_meta_packages.py {env.version}")
            return 0
        meta = _tool("make_meta_packages").load_packages(meta_path)
        print_recall(f"meta recall, {env.version}, script edges", base, meta, args.threshold)
        if script_graph is not None:
            rel = tuple(args.relations.split(","))
            print_recall(f"meta recall, {env.version}, script + Yugipedia edges", graph, meta, args.threshold, rel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
