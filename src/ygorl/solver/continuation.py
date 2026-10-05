"""Search ordinary-shuffle openings from transferable demonstration prefixes.

The solver may backtrack a prefix. Only independently verified complete lines to
the requested final target are returned, never the intermediate stopping policy.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

from ygorl.build.first_turn import match_action
from ygorl.cards.ydk import Deck
from ygorl.data.environment import Environment
from ygorl.engine import constants as C
from ygorl.engine.duel import DuelSession, default_cards
from ygorl.engine.replay import Replay, load_yrp
from ygorl.solver.batch import _check_targets, _solver_meta
from ygorl.solver.combo_solver import SolveRequest, SolverError, Workdir, run_solver
from ygorl.solver.demo import DemoError, Demonstration, convert_line, iter_steps, verify_line
from ygorl.solver.targets import parse_targets


def start_identity(replay: Replay) -> dict:
    """Engine start identity in load order; reject host-shuffled/seated wrappers."""
    if replay.shuffle_decks or replay.first != 0:
        raise ValueError("continuation replays must specify load order and engine seats")
    return {
        "core_seed": replay.core_seed,
        "rule_flags": replay.rule_flags,
        "player": replay.player,
        "decks": {s: {k: list(replay.decks[s][k]) for k in ("main", "extra")} for s in "ab"},
    }


def require_same_start(expected: Replay, candidate: Replay) -> None:
    if start_identity(expected) != start_identity(candidate):
        raise ValueError("continuation start identity mismatch (seed, decks, rules or opponent)")


@dataclass
class OpeningPrefix:
    replay: Replay
    matched_steps: int
    divergence: str


def shuffled_prefix(demo: Demonstration, *, env: Environment | None = None, cards=None) -> OpeningPrefix:
    """Follow line 0 up to its solver boundary/first mismatch, keeping only complete core responses.

    Auto-closing actions are excluded. If divergence occurs mid multi-selection,
    export rolls back to the last complete core response, so no partial host-only
    selection state is smuggled into the solver root.
    """
    replay = demo.replay(0)
    replay.rule_flags &= ~C.DUEL_PSEUDO_SHUFFLE
    replay.responses = []
    session = DuelSession(replay.duel(env, cards=cards if cards is not None else default_cards()))
    matched, divergence = 0, ""
    steps = iter_steps(demo, 0, env=env, cards=cards)
    try:
        for n, (ref, index) in enumerate(steps):
            if n >= demo.lines[0].solver_steps:
                break
            point = session.point
            if point is None or session.tracker.turn >= 2:
                divergence = "ordinary duel stopped before the solver prefix ended"
                break
            if point.player != ref.player:
                divergence = f"step {n}: acting player changed"
                break
            real = match_action(ref.actions[index], point.actions)
            if real is None:
                action = ref.actions[index]
                divergence = f"step {n}: no {action.kind} of {action.card.code if action.card else '-'}"
                break
            session.act(real)
            matched += 1
        if session.tracker.result.reason == "error":
            raise DemoError(f"ordinary prefix engine error: {session.tracker.result.error}")
        if session.tracker.turn >= 2:
            raise DemoError("solver prefix already closed turn 1; cannot use it as an opening continuation")
        replay.responses = list(session.tracker.result.responses)
        return OpeningPrefix(replay, matched, divergence)
    finally:
        steps.close()
        session.close()


def continue_opening(base: Demonstration, targets: list[str], scratch: Path, *, env: Environment,
                     binary: Path, use_prefix: bool = True, solve_ms: int = 30_000,
                     finisher_ms: int = 20_000, seed: int = 0, timeout: float = 60,
                     lines: int = 1) -> Demonstration:  # fmt: skip
    """Compare from-start search to prefix-assisted search on exactly the same ordinary duel.

    Artifacts (inputs, command, native log, candidates) remain under ``scratch``.
    A fixed 12*(main+extra)+32 depth cap avoids a reference-length confound.
    Environment/legality/input failures raise before searching. Solver failures
    remain explicit error/unverified records, excluded from BC by its loader.
    """
    if base.environment != {"version": env.version, "fingerprint": env.fingerprint}:
        raise ValueError("continuation source environment identity mismatch")
    if not base.lines or base.status != "solved":
        raise ValueError("continuation requires a solved source line")
    if lines < 1:
        raise ValueError("lines must be positive")
    cards = default_cards()
    deck = Deck(tuple(base.deck["main"]), tuple(base.deck["extra"]), (), base.deck["name"])
    if violations := env.validate_deck(deck, cards):
        raise ValueError(f"illegal continuation source: {violations}")
    parsed = parse_targets(targets)
    _check_targets(deck, parsed, cards)
    verify_line(base, 0, env=env, cards=cards)
    prefix = shuffled_prefix(base, env=env, cards=cards)
    expected = replace(prefix.replay, responses=[])
    # The source record's start must actually contain the declared demonstrated
    # deck, not a legal label pasted onto a different replay.
    if sorted(expected.decks["a"]["main"]) != sorted(deck.main) or sorted(expected.decks["a"]["extra"]) != sorted(
        deck.extra
    ):
        raise ValueError("continuation source deck differs from its replay start")
    scratch = scratch.resolve()
    binary = binary.resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    reference, start, approach = (scratch / n for n in ("reference.yrpX", "start.yrpX", "prefix.yrpX"))
    base.replay(0).to_yrpx(reference, env=env, cards=cards)
    expected.to_yrpx(start, env=env, cards=cards)
    prefix.replay.to_yrpx(approach, env=env, cards=cards)
    # Re-read the exported headers, checking the exact representation the native
    # solver will see rather than assuming the writer preserved it.
    require_same_start(expected, Replay.from_yrp(start))
    require_same_start(expected, Replay.from_yrp(approach))
    request = SolveRequest(template=reference, start=start, targets=tuple(parsed),
                           approaches=(approach,) if use_prefix and prefix.replay.responses else (),
                           solve_ms=solve_ms, finisher_ms=finisher_ms, threads=1, seed=seed,
                           max_decisions=12 * (len(deck.main) + len(deck.extra)) + 32,
                           max_written=max(4, 2 * lines))  # fmt: skip
    meta = {"source_sha256": hashlib.sha256(json.dumps(base.to_json(), sort_keys=True).encode()).hexdigest(),
            "use_prefix": use_prefix, "prefix_responses": len(prefix.replay.responses),
            "matched_steps": prefix.matched_steps, "divergence": prefix.divergence,
            "ordinary_shuffle": True, "finisher_ms": finisher_ms, "max_decisions": request.max_decisions}  # fmt: skip
    demo = replace(base, targets=[t.to_arg() for t in parsed], start=start_identity(expected),
                   lines=[], rejected=[], solver={"continuation": meta}, status="pending", error="")  # fmt: skip
    try:
        run = run_solver(request, Workdir.create(scratch / "workdir"), scratch / "out", binary=binary, timeout=timeout)
        (scratch / "solver.log").write_text(run.stdout)
        (scratch / "command.json").write_text(json.dumps([str(binary), *run.args], indent=2) + "\n")
        demo.solver.update(_solver_meta(run, request))
        demo.solver["wall_s"] = round(run.elapsed_s, 2)
        seen = set()
        for solution in run.solutions:
            try:
                yrp = load_yrp(solution.path)
                require_same_start(expected, Replay.from_yrp(yrp))
                line = convert_line(yrp, parsed, cards=cards)
                if line.board["turn"] != 2:
                    raise DemoError("continuation must finish at the start of turn 2")
                if tuple(line.responses) in seen:
                    continue
                candidate = replace(demo, lines=[line])
                verify_line(candidate, 0, env=env, cards=cards)
                if candidate.replay(0).play(env, cards=cards).reason == "error":
                    raise DemoError("continuation replay ended with an engine error")
                line.source = solution.path.name
                line.score = {"burned": solution.burned, "actions": solution.actions, "alt": solution.alt}
                demo.lines.append(line)
                seen.add(tuple(line.responses))
                if len(demo.lines) >= lines:
                    break
            except (DemoError, ValueError) as exc:
                demo.rejected.append({"source": solution.path.name, "error": str(exc)})
        if run.returncode != 0 or run.timed_out:
            demo.status, demo.error, demo.lines = "error", f"solver failed/timed out: {run.problems()}", []
        else:
            demo.status = "solved" if demo.lines else ("unverified" if run.solutions else "unsolved")
    except (SolverError, OSError, ValueError) as exc:
        demo.status, demo.error, demo.lines = "error", str(exc), []
    return demo
