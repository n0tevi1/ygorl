"""`ygorl replay PATH`: show a replay's metadata, re-simulate it and export it as ``.yrpX``."""

from __future__ import annotations

import argparse
from pathlib import Path

from ygorl.commands import SIDES, CommandError, add_env_option, describe_result, load_replay, replay_env

_COMPARED = ("winner", "reason", "win_reason", "turns", "lp")


def add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "replay",
        help="show, verify or export a recorded game",
        description="Print a replay's metadata. --verify re-runs the game from its response log and checks that it "
        "reaches the recorded end (exit code 1 if not); --export-yrpx writes an EDOPro replay (see docs/replays.md).",
    )
    p.add_argument("replay", type=Path, help="replay file (.json or .json.gz)")
    p.add_argument("--verify", action="store_true", help="re-simulate and compare with the recorded result")
    p.add_argument("--export-yrpx", type=Path, default=None, metavar="OUT", help="write an EDOPro replay (.yrpX)")
    p.add_argument("--yrpx-uncompressed", action="store_true",
                   help="write the .yrpX without LZMA compression (EDOPro reads both)")
    add_env_option(p, "environment of the replay (default: its recorded version under the environments root)")
    p.set_defaults(func=run)


def _result_text(r: dict, tail: str = "") -> str:
    winner, reason = describe_result(r.get("winner"), r.get("reason", "?"), r.get("win_reason"))
    lp = r.get("lp") or ("?", "?")
    tail = tail or f" decisions={r.get('decisions')}"
    return f"winner={winner} reason={reason} turns={r.get('turns')} lp={lp[0]}/{lp[1]}{tail}"


def run(args: argparse.Namespace) -> int:
    from ygorl.engine.replay import FORMAT, FORMAT_VERSION

    if args.yrpx_uncompressed and args.export_yrpx is None:
        raise CommandError("--yrpx-uncompressed needs --export-yrpx OUT")
    rep = load_replay(args.replay)
    lines = _metadata(args.replay, rep, FORMAT, FORMAT_VERSION)
    code = 0
    if args.verify or args.export_yrpx is not None:
        env = replay_env(rep, args.env)
        if env is not None:
            lines.append(f"checked    against environment {env.version} at {env.root}")
        try:
            if args.verify:
                result = rep.play(env)
                got = {"winner": result.winner, "reason": result.reason, "win_reason": result.win_reason,
                       "turns": result.turns, "lp": list(result.lp)}  # fmt: skip
                text = _result_text(got, f" responses={len(result.responses)}/{len(rep.responses)}")
                diff = [k for k in _COMPARED if rep.result and got[k] != rep.result.get(k)]
                if result.responses != rep.responses:
                    diff.append("responses")
                if diff:
                    lines.append(f"verify     MISMATCH ({', '.join(diff)}): replayed {text}")
                    code = 1
                elif not rep.result:
                    lines.append(f"verify     ok (no recorded result to compare): {text}")
                else:
                    lines.append(f"verify     ok: replayed to the recorded end, {text}")
            if args.export_yrpx is not None:
                args.export_yrpx.parent.mkdir(parents=True, exist_ok=True)
                rep.to_yrpx(args.export_yrpx, names=(rep.decks["a"].get("name") or "Player A",
                                                     rep.decks["b"].get("name") or "Player B"),
                           env=env, compress=not args.yrpx_uncompressed)  # fmt: skip
                lines.append(f"yrpX       {args.export_yrpx}")
        except (ValueError, OSError) as exc:
            print("\n".join(lines))
            raise CommandError(str(exc)) from None
    print("\n".join(lines))
    return code


def _metadata(path: Path, rep, fmt: str, version: int) -> list[str]:
    env = rep.environment
    ocg = rep.engine.get("ocgcore")
    p = rep.player
    lines = [
        f"replay     {path} ({fmt} v{version}" + (f", ocgcore {ocg[0]}.{ocg[1]})" if ocg else ")"),
        "environment " + (f"{env['version']} (fingerprint {env.get('fingerprint', '')[:12]})" if env else "none"),
        f"seed       {rep.seed} ({SIDES[rep.first]} moves first)",
        f"rules      flags 0x{rep.rule_flags:x}, lp {p['starting_lp']}, hand {p['starting_hand']}, "
        f"draw {p['draw_per_turn']}, max_turns {rep.max_turns}, max_decisions {rep.max_decisions}, "
        f"curriculum {rep.curriculum}" + (" (augmented start)" if rep.augmented_start else ""),
    ]
    for side in ("a", "b"):
        d = rep.decks[side]
        lines.append(f"deck_{side}     {d.get('name') or '(unnamed)'}: {len(d['main'])} main, {len(d['extra'])} extra, "
                     f"{len(d.get('side', ()))} side")  # fmt: skip
    lines.append(f"responses  {len(rep.responses)} ({len(rep.steps)} agent steps recorded)")
    lines.append(f"recorded   {_result_text(rep.result)}" if rep.result else "recorded   (no result)")
    return lines
