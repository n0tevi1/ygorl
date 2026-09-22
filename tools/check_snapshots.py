"""T2.8 acceptance: snapshot/restore at random points reproduces the uninterrupted game byte for byte.

For each game: record a random-vs-random game, then drive a snapshot-enabled core with the
recorded responses; at several random cut points take a snapshot, run ahead a random number
of responses, restore, and carry on. The concatenated message stream must equal the stream of
a plain core. Also measures restore time against replaying from the start to the same point.

Usage: uv run python tools/check_snapshots.py [--games 200] [--cuts 5] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

from ygorl import _core
from ygorl.agents import RandomAgent
from ygorl.cards.ydk import load_ydk
from ygorl.engine.duel import Duel, DuelConfig, default_cards

DECK_DIR = Path(__file__).resolve().parents[1] / "tests" / "decks"


def drive(core, responses, start, stop, awaiting):
    log, k = [], start
    while True:
        if awaiting:
            if k >= stop:
                return log, k, True
            core.set_response(responses[k])
            k += 1
        status = core.process()
        log.append(core.get_message())
        if status == _core.DUEL_STATUS_END:
            return log, k, False
        awaiting = status == _core.DUEL_STATUS_AWAITING


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--cuts", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cards = default_cards()
    decks = {p.stem: load_ydk(p) for p in sorted(DECK_DIR.glob("*.ydk"))}
    names = sorted(decks)
    rng = random.Random(args.seed)
    mismatches, escapes, restores = [], 0, 0
    restore_s, replay_s, sizes, used = [], [], [], []
    for g in range(args.games):
        seed = 900_000 + g
        a, b = decks[names[g % 10]], decks[names[(g * 3 + 7) % 10]]
        cfg = DuelConfig(max_turns=40)

        def make(snapshots, seed=seed, a=a, b=b, first=g % 2, cfg=cfg):
            return Duel(seed, None, a, b, cards=cards, first=first, config=cfg, snapshots=snapshots)

        result = make(False).run(RandomAgent(seed), RandomAgent(seed + 1))
        responses = result.responses
        n = len(responses)
        reference, _, _ = drive(make(False)._setup(), responses, 0, n, False)

        core = make(True)._setup()
        log, k, awaiting = [], 0, False
        for cut in sorted(rng.sample(range(n), min(args.cuts, n))):
            part, k, awaiting = drive(core, responses, k, cut, awaiting)
            log += part
            if not awaiting:
                break
            snap = core.snapshot()
            sizes.append(snap.nbytes)
            drive(core, responses, k, min(n, k + rng.randint(1, 40)), True)  # run ahead, then come back
            t0 = time.perf_counter()
            core.restore(snap)
            restore_s.append(time.perf_counter() - t0)
            restores += 1
            t0 = time.perf_counter()
            drive(make(False)._setup(), responses, 0, cut, False)
            replay_s.append(time.perf_counter() - t0)
        part, _, _ = drive(core, responses, k, n, awaiting)
        log += part
        used.append(core.arena_bytes())
        escapes += core.arena_escapes()
        if log != reference:
            mismatches.append(g)
        core.close()

    ratio = [r / p for r, p in zip(restore_s, replay_s) if p > 0]
    report = {
        "games": args.games,
        "restores": restores,
        "mismatches": len(mismatches),
        "arena_escapes": escapes,
        "snapshot_mib_median": round(statistics.median(sizes) / 2**20, 2),
        "snapshot_mib_max": round(max(sizes) / 2**20, 2),
        "arena_mib_max": round(max(used) / 2**20, 2),
        "restore_ms_median": round(statistics.median(restore_s) * 1000, 3),
        "replay_to_cut_ms_median": round(statistics.median(replay_s) * 1000, 2),
        "restore_over_replay_median": round(statistics.median(ratio), 4),
        "restore_over_replay_max": round(max(ratio), 4),
    }
    print(json.dumps(report, indent=2))
    if mismatches:
        print("mismatching games:", mismatches[:10], file=sys.stderr)
    return 1 if mismatches or escapes else 0


if __name__ == "__main__":
    raise SystemExit(main())
