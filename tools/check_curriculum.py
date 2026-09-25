"""T2.6 acceptance: every curriculum mode plays N games without errors.

Usage: uv run python tools/check_curriculum.py [--games 100] [--threads 2] [--envs 8] [--max-turns 200] [--compare]

For each mode (full / solo / handtrap) plays ``--games`` random-vs-random games
on the DuelPool as opening-balanced pairs (every seed once with the learner,
deck a, going first and once going second). Reports errors, retries,
unknown/undecodable messages, Lua script errors, end reasons, the learner's
win rate by seat, host-answered (auto) decisions and the opponent's chains in
the learner's turns (by origin: hand / other, and forced). ``--compare`` also
replays every game sequentially with ``Duel.run`` and from its response log
with ``Duel.replay`` and counts mismatches. Exits non-zero on any error,
retry, undecodable message, restriction breach or mismatch.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

from ygorl.agents import RandomAgent
from ygorl.cards.ydk import load_ydk
from ygorl.engine import constants as C
from ygorl.engine import messages as M
from ygorl.engine.curriculum import MODES
from ygorl.engine.duel import Duel, DuelConfig
from ygorl.env import GameSpec, paired_specs, run_games

DECK_DIR = Path(__file__).resolve().parents[1] / "tests" / "decks"


class Counting(RandomAgent):
    """Random agent that tallies the opponent's chain choices in the learner's turns."""

    def __init__(self, seed: int, is_learner: bool, tally: collections.Counter) -> None:
        super().__init__(seed)
        self.is_learner, self.tally = is_learner, tally

    def act(self, point) -> int:
        idx = super().act(point)
        if self.is_learner or point.turn_player == point.player:
            return idx
        a = point.actions[idx]
        self.tally["opponent_points"] += 1
        if a.kind in ("chain", "yes") and a.card is not None:
            forced = isinstance(point.decision, M.SelectChain) and point.decision.forced
            where = "hand" if a.card.loc.location == C.LOCATION_HAND else "other"
            self.tally[f"opponent_{a.kind}_{where}{'_forced' if forced else ''}"] += 1
        return idx


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100, help="games per mode (rounded up to even)")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--modes", nargs="*", default=list(MODES))
    parser.add_argument(
        "--compare", action="store_true", help="also check Duel.run and Duel.replay give the same games"
    )
    args = parser.parse_args()

    decks = {p.stem: load_ydk(p) for p in sorted(DECK_DIR.glob("*.ydk"))}
    names = sorted(decks)
    report, failed = {}, False
    for mode in args.modes:
        config = DuelConfig(curriculum=mode, max_turns=args.max_turns)
        base = [GameSpec(seed=700_000 + i, deck_a=decks[names[i % 10]], deck_b=decks[names[(i * 7 + 3) % 10]], config=config)
                for i in range((args.games + 1) // 2)]  # fmt: skip
        specs = paired_specs(base)
        tally: collections.Counter = collections.Counter()

        def agents(i, spec, tally=tally):
            return Counting(spec.seed, True, tally), Counting(spec.seed ^ 0xABC, False, tally)

        t0 = time.time()
        results = run_games(specs, agents, num_envs=args.envs, num_threads=args.threads)
        seconds = time.time() - t0

        wins = collections.Counter()
        for spec, r in zip(specs, results, strict=True):
            seat = "first" if spec.first == 0 else "second"
            wins[f"{seat}_games"] += 1
            wins[f"{seat}_learner_wins"] += r.winner == 0
        row = {
            "games": len(results),
            "errors": sum(r.reason == "error" for r in results),
            "retries": sum(r.retries for r in results),
            "unknown_messages": sum(r.unknown_messages for r in results),
            "undecodable_messages": sum(r.undecodable_messages for r in results),
            "games_with_script_errors": sum(bool(r.script_errors) for r in results),
            "reasons": dict(collections.Counter(r.reason for r in results)),
            "mean_turns": round(sum(r.turns for r in results) / len(results), 1),
            "agent_decisions": sum(r.decisions for r in results),
            "auto_decisions": sum(r.auto_decisions for r in results),
            "learner_win_rate_first": round(wins["first_learner_wins"] / max(1, wins["first_games"]), 3),
            "learner_win_rate_second": round(wins["second_learner_wins"] / max(1, wins["second_games"]), 3),
            **dict(sorted(tally.items())),
            "seconds": round(seconds, 1),
        }
        breach = 0
        if mode == "solo":
            breach = sum(
                v
                for k, v in tally.items()
                if k.startswith("opponent_") and not k.endswith("_forced") and k != "opponent_points"
            )
        elif mode == "handtrap":
            breach = sum(v for k, v in tally.items() if k.startswith("opponent_") and k.endswith("_other"))
        row["restriction_breaches"] = breach
        if args.compare:
            mismatches = 0
            for i, (spec, r) in enumerate(zip(specs, results, strict=True)):
                duel = Duel(spec.seed, None, spec.deck_a, spec.deck_b, first=spec.first, config=spec.config)
                seq = duel.run(*agents(i, spec, collections.Counter()))
                again = Duel(spec.seed, None, spec.deck_a, spec.deck_b, first=spec.first, config=spec.config).replay(
                    r.responses
                )
                same = (seq.responses == r.responses and seq.actions == r.actions and seq.auto_decisions == r.auto_decisions
                        and (seq.winner, seq.reason, seq.turns, seq.lp) == (r.winner, r.reason, r.turns, r.lp)
                        and (again.turns, again.lp) == (r.turns, r.lp) and (r.reason != "win" or again.winner == r.winner))  # fmt: skip
                mismatches += not same
            row["mismatches"] = mismatches
        failed |= bool(row["errors"] or row["retries"] or row["unknown_messages"] or row["undecodable_messages"]
                       or breach or row.get("mismatches"))  # fmt: skip
        report[mode] = row
        print(mode, json.dumps(row), file=sys.stderr, flush=True)
    print(
        json.dumps({"threads": args.threads, "envs": args.envs, "max_turns": args.max_turns, "modes": report}, indent=2)
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
