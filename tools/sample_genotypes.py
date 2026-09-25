"""Sample deck genotypes (T5.5), check every decoded deck for legality and print statistics.

Usage: uv run python tools/sample_genotypes.py [--n 10000] [--chain 10000] [--seed 2026]
                                               [--environment <version|dir>] [--top-packages 60]

Without ``--environment`` a test format is built in memory, like
tests/test_genotype.py: the card pool is the members of the ``--top-packages``
best engine packages plus the generic pool (tests/data/generic_pool.json), and a
banlist forbids / limits / semi-limits every 7th package member in turn plus a
few generic cards. The genotype space uses every enumerated package restricted
to the pool. ``--n`` random genotypes and ``--chain`` crossover + mutation
children (steady-state population of 100) are decoded and validated with
``Environment.validate_deck``; the script exits non-zero on any violation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from ygorl.build.genotype import GenotypeSpace
from ygorl.build.packages import enumerate_packages, setcodes_from_db
from ygorl.build.synergy_graph import load_or_build
from ygorl.cards.cdb import CardDB
from ygorl.cards.lflist import Banlist
from ygorl.data.environment import Environment, load_environment

ROOT = Path(__file__).resolve().parents[1]


def test_environment(pkgs, generic: dict[int, str], top: int) -> Environment:
    chosen = pkgs[:top]
    members = sorted({m for p in chosen for m in p.members})
    limits = {pw: i % 3 for i, pw in enumerate(members[::7])}
    limits.update({14558127: 1, 23434538: 0, 54693926: 1, 29301450: 0})  # Ash 1, Maxx "C" 0, DRNM 1, S:P 0
    return Environment(
        version="test-genotype", format="md", card_pool=frozenset(members) | frozenset(generic),
        banlist=Banlist("test", limits), rule_flags=0, meta_decks=(),
    )  # fmt: skip


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=10_000, help="random genotypes to sample")
    parser.add_argument("--chain", type=int, default=10_000, help="crossover + mutation children to generate")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--environment", help="use this environment's pool, banlist and deck rules")
    parser.add_argument(
        "--top-packages", type=int, default=60, help="test format: pool = members of the top-N packages"
    )
    parser.add_argument("--generic", type=Path, default=ROOT / "tests" / "data" / "generic_pool.json")
    parser.add_argument("--max-packages", type=int, default=3)
    args = parser.parse_args()

    db = CardDB.load()
    pkgs = enumerate_packages(load_or_build(), setcodes=setcodes_from_db(db))
    generic = {c["password"]: c["role"] for c in json.loads(args.generic.read_text())["cards"]}
    env = load_environment(args.environment) if args.environment else test_environment(pkgs, generic, args.top_packages)
    t0 = time.perf_counter()
    sp = GenotypeSpace.from_environment(env, db, pkgs, generic, max_packages=args.max_packages)
    print(f"environment {env.version}: pool {len(env.card_pool)} cards, {len(env.banlist.limits)} banlist entries")
    print(
        f"space: {len(sp)} cards ({sp.n_main} Main / {len(sp) - sp.n_main} Extra Deck), {len(sp.packages)} packages, "
        f"{int(sp.is_generic.sum())} generic; caps {dict(sorted(Counter(sp.cap.tolist()).items()))}; "
        f"main_range {sp.main_range}, extra_size {sp.extra_size}; built in {time.perf_counter() - t0:.2f}s"
    )

    rng = np.random.default_rng(args.seed)
    t0 = time.perf_counter()
    samples = sp.sample_many(args.n, rng)
    t_sample = time.perf_counter() - t0
    t0 = time.perf_counter()
    bad = [(n, vs) for n, g in enumerate(samples) if (vs := env.validate_deck(sp.decode(g), db))]
    t_check = time.perf_counter() - t0
    print(f"\nsampled {args.n} genotypes in {t_sample:.2f}s ({1e6 * t_sample / max(args.n, 1):.0f} us each); "
          f"validated in {t_check:.2f}s: {args.n - len(bad)} legal, {len(bad)} illegal")  # fmt: skip
    if samples:
        main = np.array([int(g.counts[: sp.n_main].sum()) for g in samples])
        extra = np.array([int(g.counts[sp.n_main :].sum()) for g in samples])
        n_pk = Counter(len(g.packages) for g in samples)
        generic_main = sp.is_generic & ~sp.is_extra
        share = np.array([g.counts[generic_main].sum() / g.counts[: sp.n_main].sum() for g in samples])
        roles = Counter()
        for g in samples:
            roles.update(sp.role_counts(g))
        print(f"  Main Deck size {dict(sorted(Counter(main.tolist()).items()))}")
        print(
            f"  Extra Deck size min {extra.min()} mean {extra.mean():.1f}; packages per genotype {dict(sorted(n_pk.items()))}"
        )
        print(f"  generic share of the Main Deck mean {share.mean():.2f}; mean copies per role "
              + ", ".join(f"{r} {n / len(samples):.1f}" for r, n in sorted(roles.items())))  # fmt: skip
        print(f"  distinct genotypes {len(set(samples))}, distinct decks {len({sp.decode(g) for g in samples})}")

    pop = samples[:100] if len(samples) >= 100 else sp.sample_many(100, rng)
    t0 = time.perf_counter()
    chain_bad = 0
    for _ in range(args.chain):
        i, j = rng.integers(len(pop), size=2)
        child = sp.mutate(sp.crossover(pop[i], pop[j], rng), rng, n_ops=int(rng.integers(1, 4)))
        chain_bad += bool(env.validate_deck(sp.decode(child), db))
        pop[int(rng.integers(len(pop)))] = child
    t_chain = time.perf_counter() - t0
    print(
        f"\n{args.chain} crossover + mutation children in {t_chain:.2f}s: {args.chain - chain_bad} legal, {chain_bad} illegal"
    )

    for n, vs in bad[:5]:
        print(f"sample {n}: " + "; ".join(v.message for v in vs))
    return 1 if bad or chain_bad else 0


if __name__ == "__main__":
    sys.exit(main())
