"""Where is the first-seat gap? Per-turn-segment look-ahead decomposition (first-seat plan §13).

The learner plays the same games (common random numbers, :mod:`ygorl.train.lookahead`) once as it is (``base``) and
once per condition with best-of-K look-ahead on one segment of its decisions: ``t1`` (turn 1), ``t2`` (turn 2),
``t3`` (turns 3 on), ``all``; ``opp`` instead gives the *opponent* look-ahead on its first own turn (turn 2 when the
learner moves first, turn 1 when it moves second) -- the reactive-opponent check. Each condition is reported against
``base`` game by game (paired difference of the learner's score and its standard error).

Opponent classes: ``anchors`` -- the strong anchors of the extended panel (``--opponent NAME=CKPT``, repeatable;
the panel's meta decks, pairings drawn like the panel's with ``--pairings``, the learner holding each deck slot);
``lines`` -- scripted solver lines (``--demos``, recovery lines ``--line-recovery``), the learner's checkpoint
playing the line seat once the line is over (as tools/first_seat_line_eval.py).

  uv run --no-sync python tools/seat_gap_decomposition.py CKPT --class anchors --opponent E2016=... --opponent G576=...
      [--seats 0 1] [--conditions base t1 t2 t3 all opp] [--pairings 100] [--k 8] [--envs 128] [--threads 16] --out DIR

Writes ``DIR/<class>_seat<s>_<condition>.json`` (per-game scores and look-ahead counts; an existing file is reused)
and ``DIR/<class>_summary.json``.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

CONDITIONS = ("base", "t1", "t2", "t3", "all", "opp")


def anchor_games(learner, opponents: dict, decks, pairings: int, seed: int, config, seat: int):
    """Learner first (``seat`` 0) or second: per pairing, per opponent, the learner holding each deck slot."""
    from ygorl.env.pool import GameSpec
    from ygorl.eval.agent_matrix import pairing_slots, sample_pairings
    from ygorl.eval.arena import derive_seed
    from ygorl.train.lookahead import GameSetup, SeatPlayer

    # the decks are shuffled per slot here (``config`` must shuffle), then dealt as they are (as the panel's cell_specs)
    slots = pairing_slots(decks, sample_pairings(len(decks), pairings, seed), seed, replace(config, shuffle_decks=True))
    cfg = replace(config, shuffle_decks=False)
    games = []
    for k, (s, pair, _) in enumerate(slots):
        for o, (name, module) in enumerate(opponents.items()):
            for m in (0, 1):
                spec = GameSpec(seed=s, deck_a=pair[m], deck_b=pair[1 - m], first=seat, config=cfg)
                players = [SeatPlayer(module), SeatPlayer(module)]
                players[seat] = SeatPlayer(learner)  # the learner holds deck a: engine seat == first
                games.append(GameSetup(spec, tuple(players), derive_seed(seed, 7, k, o, m),
                                       info={"pair": k, "opponent": name, "slot": m, "deck": pair[m].name,
                                             "opp_deck": pair[1 - m].name}))  # fmt: skip
    return games


def line_games(learner, scripts, decks, n: int, seed: int, config, seat: int):
    from ygorl.eval.arena import derive_seed
    from ygorl.train.lookahead import GameSetup, SeatPlayer

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(scripts))
    games = []
    for g in range(n):
        script = scripts[int(order[g % len(scripts)])]
        deck = decks[derive_seed(seed, 0, script.id, g) % len(decks)]
        spec = script.spec(deck, derive_seed(seed, 1, script.id, g), config, learner_seat=seat)
        games.append(GameSetup(spec, (SeatPlayer(learner), SeatPlayer(learner)), derive_seed(seed, 9, g),
                               line=script, line_seat=1 - seat,
                               info={"line": script.id, "line_name": script.name, "deck": deck.name}))  # fmt: skip
    return games


def with_condition(games, condition: str, seat: int, k: int, rollouts: int, mode: str = "sample", extra_turns: int = 0):
    """The same games with the condition's look-ahead (``base``: none)."""
    from ygorl.train.lookahead import LookaheadConfig, SeatPlayer

    if condition == "base":
        return games
    out = []
    for g in games:
        players = list(g.seats)
        if condition == "opp":
            if g.line is not None:
                raise ValueError("the opponent look-ahead condition needs policy opponents (class anchors)")
            turn = 2 if seat == 0 else 1  # the opponent's first own turn
            cfg = LookaheadConfig(
                k=k,
                rollouts=rollouts,
                first_turn=turn,
                last_turn=turn,
                own_turn=True,
                candidates=mode,
                extra_turns=extra_turns,
            )
            players[1 - seat] = SeatPlayer(players[1 - seat].module, cfg)
        else:
            cfg = LookaheadConfig.segment(condition, k=k, rollouts=rollouts, candidates=mode, extra_turns=extra_turns)
            players[seat] = SeatPlayer(players[seat].module, cfg)
        out.append(replace(g, seats=tuple(players)))
    return out


def run(env, games, seat: int, label: str, log) -> dict:
    from ygorl.train.lookahead import play

    t0 = time.time()
    last = [0.0]

    def progress(done, total):
        if time.time() - last[0] > 60:
            last[0] = time.time()
            print(f"  {label}: {done}/{total} games, {time.time() - t0:.0f}s", file=log, flush=True)

    outcomes, timing = play(env, games, progress=progress)
    stats = {"qualifying": 0, "branched": 0, "changed": 0, "rollout_steps": 0}
    for o in outcomes:
        for key in stats:
            stats[key] += getattr(o.stats, key)
    scores = [o.score(seat) for o in outcomes]
    decided = [o for o in outcomes if o.reason == "win" and o.winner in (0, 1)]
    return {"label": label, "games": len(outcomes), "seconds": round(time.time() - t0, 1), "timing": timing,
            "win_rate": sum(o.winner == seat for o in decided) / max(1, len(decided)),
            "score": float(np.mean(scores)), "stats": stats,
            "per_game": [{"score": s, "reason": o.reason, "turns": o.turns, "q": o.stats.qualifying,
                          "b": o.stats.branched, "c": o.stats.changed, "line": (o.line or {}).get("status"),
                          "by_turn": o.stats.as_dict()["by_turn"], **o.info} for s, o in zip(scores, outcomes)]}  # fmt: skip


def paired(base: dict, other: dict) -> dict:
    a = np.array([g["score"] for g in base["per_game"]])
    b = np.array([g["score"] for g in other["per_game"]])
    d = b - a
    return {"diff": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d))), "b_better": int((d > 0).sum()),
            "a_better": int((d < 0).sum()), "games": len(d)}  # fmt: skip


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpt")
    p.add_argument("--class", dest="kind", choices=("anchors", "lines"), required=True)
    p.add_argument("--opponent", action="append", default=[], help="NAME=CKPT (anchors)")
    p.add_argument("--demos", nargs="+", default=[], help="line demonstrations (lines)")
    p.add_argument("--line-recovery", action="append", default=[])
    p.add_argument("--library-cache", default=None, help="pickle of the loaded line library (written once, then read)")
    p.add_argument("--games", type=int, default=400, help="games per seat (lines)")
    p.add_argument("--pairings", type=int, default=100, help="pairings (anchors: 4 x pairings games per seat)")
    p.add_argument("--seats", type=int, nargs="+", default=[0, 1])
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--rollouts", type=int, default=1)
    p.add_argument("--extra-turns", type=int, default=0, help="horizon: the end of the decision's turn + N turns")
    p.add_argument(
        "--candidates",
        choices=("sample", "top"),
        default="sample",
        help="K policy draws, or the policy's draw plus the likeliest other actions",
    )
    p.add_argument("--seed", type=int, default=2026102701)
    p.add_argument("--env", default="md-2026-09")
    p.add_argument("--envs", type=int, default=128)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--torch-threads", type=int, default=8)
    p.add_argument("--max-decisions", type=int, default=4000)
    p.add_argument("--limit", type=int, default=None, help="only the first N games per seat (smoke runs)")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    import torch

    torch.set_num_threads(a.torch_threads)
    from ygorl.data import load_environment
    from ygorl.engine.duel import DuelConfig, default_cards
    from ygorl.env.encoded import EncodedVecEnv
    from ygorl.train.checkpoint import load_actor

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = sys.stdout
    environment = load_environment(a.env)
    cards = default_cards()
    ref = load_actor(a.ckpt)
    config = DuelConfig.from_environment(environment, max_decisions=a.max_decisions)
    decks = [m.deck for m in sorted(environment.meta_decks, key=lambda m: m.deck.name)]
    opponents = {}
    for item in a.opponent:
        name, path = item.split("=", 1)
        pol = load_actor(path)
        if why := ref.signature.mismatches(pol.signature):
            raise SystemExit(f"{name}: {'; '.join(why)}")
        opponents[name] = pol.net
    scripts = None
    if a.kind == "lines":
        sys.path.insert(0, str(Path(__file__).parent))
        from first_seat_line_eval import load_library

        t0 = time.time()
        cache = Path(a.library_cache) if a.library_cache else None
        if cache is not None and cache.exists():
            with cache.open("rb") as f:
                lib = pickle.load(f)
        else:
            lib = load_library(a.demos, ref.vocab, cards, environment, a.line_recovery)
            if cache is not None:
                with cache.open("wb") as f:
                    pickle.dump(lib, f)
        scripts = lib.scripts
        print(f"{len(scripts)} lines ({len(lib.problems)} left out, {lib.recovery_windows} recovery windows) "
              f"in {time.time() - t0:.0f}s", file=log, flush=True)  # fmt: skip
    elif not opponents:
        raise SystemExit("--class anchors needs --opponent NAME=CKPT")
    env = EncodedVecEnv(a.envs, a.threads, cards=cards, vocab=ref.vocab, privileged=True, event_length=ref.event_length,
                        skip_forced=True, selection_history=ref.net_config.selection_history,
                        attack_history=ref.net_config.attack_history, snapshots=True)  # fmt: skip
    summary = {"ckpt": a.ckpt, "class": a.kind, "k": a.k, "rollouts": a.rollouts, "candidates": a.candidates, "extra_turns": a.extra_turns,
               "seed": a.seed,
               "opponents": a.opponent, "demos": a.demos, "line_recovery": a.line_recovery, "seats": {}}  # fmt: skip
    for seat in a.seats:
        if a.kind == "anchors":
            games = anchor_games(ref.net, opponents, decks, a.pairings, a.seed, config, seat)
        else:
            games = line_games(ref.net, scripts, decks, a.games, a.seed, config, seat)
        games = games[: a.limit]
        results = {}
        for cond in a.conditions:
            if cond == "opp" and a.kind == "lines":
                continue
            path = out / f"{a.kind}_seat{seat}_{cond}.json"
            if path.exists():
                results[cond] = json.loads(path.read_text())
            else:
                r = run(env, with_condition(games, cond, seat, a.k, a.rollouts, a.candidates, a.extra_turns), seat, f"{a.kind} seat {seat} {cond}",
                        log)  # fmt: skip
                path.write_text(json.dumps(r))
                results[cond] = r
            r = results[cond]
            line = f"{a.kind} seat {seat} {cond}: score {r['score']:.3f} win {r['win_rate']:.3f} ({r['seconds']}s)"
            if cond != "base" and "base" in results:
                pr = paired(results["base"], r)
                line += f" | vs base {pr['diff'] * 100:+.1f} ± {pr['se'] * 100:.1f} pp"
            print(line + f" | {r['stats']}", file=log, flush=True)
        summary["seats"][str(seat)] = {
            cond: {"score": r["score"], "win_rate": r["win_rate"], "seconds": r["seconds"], "stats": r["stats"],
                   "timing": r["timing"], **({"paired": paired(results["base"], r)}
                                             if cond != "base" and "base" in results else {})}
            for cond, r in results.items()
        }  # fmt: skip
        (out / f"{a.kind}_summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
