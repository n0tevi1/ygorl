"""`ygorl branch <replay> --at t --try all --policy <agent>`: compare candidate actions at a decision point."""

from __future__ import annotations

import argparse
from pathlib import Path

from ygorl.commands import SIDES as _SIDES
from ygorl.commands import CommandError, add_env_option, agents_help, load_replay, replay_env

_LOCATIONS = {0x1: "deck", 0x2: "hand", 0x4: "mzone", 0x8: "szone", 0x10: "grave", 0x20: "banished", 0x40: "extra",
              0x80: "overlay"}  # fmt: skip
_POSITIONS = {0x1: "faceup_attack", 0x2: "facedown_attack", 0x4: "faceup_defense", 0x8: "facedown_defense"}
_MAX_DESCRIPTION = 44


def add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "branch",
        help="fork a replay at a decision point and roll out candidate actions",
        description="Replay a recorded game up to agent step T, try candidate actions there and roll each out "
        "with a policy; prints one row per candidate (see docs/branching.md).",
    )
    p.add_argument("replay", type=Path, help="replay file (.json or .json.gz)")
    p.add_argument("--at", dest="t", type=int, required=True, metavar="T", help="decision point: agent step index, from 0")
    p.add_argument("--try", dest="candidates", type=_candidates, default=None, metavar="all|I,J,...",
                   help="candidate action indices at T (default: all legal actions)")  # fmt: skip
    p.add_argument("--policy", default="random", metavar="AGENT", help=f"rollout policy for both sides: {agents_help()}")
    p.add_argument("--seed", type=int, default=0, help="policy seed of the first rollout (default 0)")
    p.add_argument("--rollouts", type=int, default=1, metavar="N", help="rollouts per candidate (default 1)")
    add_env_option(p, "environment of the replay (default: its recorded version under the environments root)")
    p.set_defaults(func=run)


def _candidates(text: str) -> list[int] | None:
    if text == "all":
        return None
    try:
        return [int(x) for x in text.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected 'all' or comma-separated action indices, got {text!r}") from None


def run(args: argparse.Namespace) -> int:
    from ygorl.agents.registry import make_agent
    from ygorl.engine.branch import fork
    from ygorl.engine.duel import default_cards

    replay = load_replay(args.replay)
    env = replay_env(replay, args.env)
    try:
        make_agent(args.policy, args.seed)  # fail on a bad spec before replaying anything
        branch = fork(replay, args.t, env)
        outcomes = branch.try_all(lambda seed: make_agent(args.policy, seed), candidates=args.candidates,
                                  rollouts=args.rollouts, seed=args.seed)  # fmt: skip
    except (ValueError, OSError) as exc:
        raise CommandError(str(exc)) from None
    print(_report(args, replay, env, branch, outcomes, default_cards()))
    return 0


def describe_action(action, cards) -> str:
    """Short human-readable label of a legal action (card password, name and location when it has a card)."""
    parts = [action.kind]
    if action.card is not None and action.card.code:
        card = cards.get(action.card.code)
        parts.append(f"{action.card.code}")
        if card is not None and card.name:
            parts.append(card.name)
        loc = _LOCATIONS.get(action.card.loc.location)
        if loc and action.kind != "position":
            parts.append(f"@{loc}")
    if action.kind == "position":
        parts.append(_POSITIONS.get(action.value, str(action.value)))
    elif action.kind == "place":
        player, loc, seq = action.value >> 16, (action.value >> 8) & 0xFF, action.value & 0xFF
        parts.append(f"p{player} {_LOCATIONS.get(loc, loc)} {seq}")  # engine player
    elif action.kind == "declare":
        card = cards.get(action.value)
        parts.append(f"{action.value}" + (f" {card.name}" if card is not None and card.name else ""))
    elif action.kind in ("number", "race", "attribute", "rps"):
        parts.append(str(action.value))
    elif action.kind in ("option", "select", "unselect", "sort", "counter") and action.card is None:
        parts.append(f"#{action.index}")
    text = " ".join(parts)
    return text if len(text) <= _MAX_DESCRIPTION else text[: _MAX_DESCRIPTION - 3] + "..."


def _report(args, replay, env, branch, outcomes, cards) -> str:
    point, rec = branch.point, replay.result
    lines = [
        f"replay    {args.replay}: seed {replay.seed}, {_SIDES[replay.first]} moves first, "
        f"environment {env.version if env is not None else 'none'}",
        f"fork      t={branch.t}: turn {point.turn}, side {_SIDES[branch.side]} to act (engine player {point.player}), "
        f"{point.decision.name}, {len(point.actions)} legal action{'s' if len(point.actions) != 1 else ''}",
    ]
    recorded = f"action {branch.recorded_action}" if branch.recorded_action is not None else "no action recorded at t"
    if rec:
        lp = rec.get("lp") or ("?", "?")
        recorded += (f"; the game ended winner={_SIDES.get(rec.get('winner'), '?')} reason={rec.get('reason')} "
                     f"turns={rec.get('turns')} lp={lp[0]}/{lp[1]}")  # fmt: skip
    lines.append(f"recorded: {recorded}")
    per = "rollout" if args.rollouts == 1 else "rollouts"
    lines.append(f"rollouts  policy {args.policy}, seed {args.seed}, {args.rollouts} {per} per candidate; * = recorded action")
    if args.rollouts > 1:
        lines.append(f"          wins/draws/losses for side {_SIDES[branch.side]}; turns and lp are means")
    lines.append("")

    descs = [describe_action(o.action, cards) for o in outcomes]
    width = max([len("action"), *map(len, descs)])
    if args.rollouts == 1:
        head = f"  cand  {'action':<{width}}  winner  reason          turns   lp_a   lp_b"
    else:
        head = f"  cand  {'action':<{width}}  wins  draws  losses   win%   turns    lp_a    lp_b"
    lines.append(head)
    for o, desc in zip(outcomes, descs, strict=True):
        mark = "*" if o.recorded else " "
        cells = f"{mark}{o.index:>5}  {desc:<{width}}  "
        if args.rollouts == 1:
            r = o.results[0]
            cells += f"{_SIDES[r.winner]:<6}  {r.reason:<14}  {r.turns:>5}  {r.lp[0]:>5}  {r.lp[1]:>5}"
        else:
            n = len(o.results)
            turns = sum(r.turns for r in o.results) / n
            lp_a, lp_b = (sum(r.lp[i] for r in o.results) / n for i in (0, 1))
            cells += (f"{o.wins:>4}  {o.draws:>5}  {o.losses:>6}  {100 * o.wins / n:>5.1f}  {turns:>6.1f}  "
                      f"{lp_a:>6.0f}  {lp_b:>6.0f}")  # fmt: skip
        lines.append(cells)
    return "\n".join(lines)
