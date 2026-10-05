"""Solving opening hands in batch: hand sampling, the duel template, one hand end to end (T4a.1).

:func:`solve_hand` is the unit of work of ``tools/solve_openings.py``: sample (or take) an opening
hand, write the per-hand decklist and template, run the solver, and convert + verify its lines
into a :class:`~ygorl.solver.demo.Demonstration`. :func:`solve_fire` runs the ``--fire`` variant
on a solved hand: the opponent holds a hand trap and plays it at every legal window of the
demonstrated line, and the solver rebuilds the board from each window.
Everything a job needs is in :class:`HandJob` (picklable) so jobs can run in worker processes.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from ygorl import _core
from ygorl.cards.ydk import Deck, load_ydk
from ygorl.data.environment import Environment
from ygorl.engine.duel import DuelConfig, default_cards, shuffle_deck
from ygorl.engine.replay import Replay, YrpError, load_yrp
from ygorl.solver.combo_solver import (
    SolveRequest,
    SolverError,
    SolverRun,
    Workdir,
    find_solver,
    run_solver,
    solver_version,
)
from ygorl.solver.demo import DemoError, Demonstration, convert_line, read_jsonl, require_same_start, verify_line
from ygorl.solver.targets import board_summary_missing, parse_targets

# The template's opponent: 40 copies of a vanilla monster (Blue-Eyes White Dragon), so it never has an
# effect to play; the solver plays the opponent's windows as passes, and --fire adds the one card it plays.
PASSIVE_OPPONENT_CARD = 89631139
PASSIVE_OPPONENT = Deck((PASSIVE_OPPONENT_CARD,) * 40, (), (), "passive")
DEFAULT_FIRE_MS = 20_000
_BEST = re.compile(r"at best (\d+) of the (\d+) target cards")
_WINDOWS = re.compile(r"(\d+) window\(s\) where .* is playable")


def sample_hand(deck: Deck, seed: int, size: int = 5) -> tuple[list[int], list[int]]:
    """(opening hand, main deck order): the host shuffle of ``seed``; the hand is the top ``size`` cards.

    The core draws from the end of the loaded list, so the hand is ``order[-size:]``. The order is
    also the decklist handed to the solver, which keeps it (minus the hand) as the rest of the deck.
    """
    order = shuffle_deck(deck.main, seed, 0)
    return order[-size:], order


def make_template(path: str | Path, deck: Deck, seed: int, config: DuelConfig | None = None, cards=None,
                  scripts: _core.ScriptDirectory | None = None) -> Path:  # fmt: skip
    """Write the ``.yrpX`` the solver takes its duel parameters from (``--no-ref``).

    It supplies the rule flags and player rules of ``config``, the core seed words (from ``seed``)
    and the opponent (:data:`PASSIVE_OPPONENT`); the solver replaces our deck and hand. The template
    records no response.
    """
    config = config or DuelConfig()
    rep = Replay(seed=seed, first=0, rule_flags=config.rule_flags, player=asdict(config.player), shuffle_decks=False,
                 decks={"a": _deck_dict(deck), "b": _deck_dict(PASSIVE_OPPONENT)}, responses=[])  # fmt: skip
    rep.to_yrpx(path, names=(deck.name or "deck", PASSIVE_OPPONENT.name), **_engine_kwargs(cards, scripts))
    return Path(path)


def _engine_kwargs(cards, scripts) -> dict:
    return {k: v for k, v in (("cards", cards), ("scripts", scripts)) if v is not None}


def _deck_dict(deck: Deck) -> dict:
    return {"name": deck.name, "main": list(deck.main), "extra": list(deck.extra), "side": []}


@dataclass(frozen=True)
class HandJob:
    """One opening hand of one deck to solve (picklable)."""

    deck_path: Path
    hand_index: int
    hand_seed: int
    targets: tuple[str, ...]
    workdir: Path  # shared EDOPro-style work directory (Workdir.create)
    scratch: Path  # per-job files (decklist, template, solver output); removed unless keep_files
    solve_ms: int = 60_000
    threads: int | None = 1
    hand: tuple[int, ...] | None = None  # None: sample_hand(deck, hand_seed, hand size of the rules)
    lines: int = 1  # verified lines kept per hand
    fire: tuple[int, ...] = ()  # hand traps for the --fire variant, one record each
    fire_ms: int = DEFAULT_FIRE_MS
    binary: Path | None = None
    solver_seed: int | None = None
    max_rollouts: int | None = None  # solver --max-rollouts (count budget; see SolveRequest)
    env: str | None = None  # environment directory or version
    timeout_s: float | None = None
    keep_files: bool = False
    ordinary_shuffle: bool = False


def _config(env: Environment | None) -> DuelConfig:
    return DuelConfig.from_environment(env) if env is not None else DuelConfig()


def _load_env(spec: str | None) -> Environment | None:
    if spec is None:
        return None
    from ygorl.data import load_environment

    return load_environment(spec)


def _solver_meta(run: SolverRun, request: SolveRequest) -> dict:
    meta = {**solver_version(), "solve_ms": request.solve_ms, "threads": request.threads, "seed": run.seed,
            **({"max_rollouts": request.max_rollouts} if request.max_rollouts is not None else {}),
            "returncode": run.returncode, "elapsed_s": round(run.elapsed_s, 2), "candidates": run.candidates,
            "written": len(run.solutions), "timed_out": run.timed_out}  # fmt: skip
    if request.fire is not None:
        meta["fire_ms"] = request.fire_ms
        verdict = run.event("fireVerdict")
        found = _WINDOWS.findall(run.stdout)
        if verdict is not None:
            meta["windows"] = verdict.get("windows")
            meta["converted"] = verdict.get("converted")
        elif found:  # no verdict event when the card has no window at all
            meta["windows"], meta["converted"] = int(found[-1]), 0
    health = run.event("health")
    if health is not None:
        meta["reference_retries"] = health.get("msgRetry")
    best = _BEST.findall(run.stdout)
    if best:  # how close the search came: target cards placed, out of the target's
        meta["best_placed"], meta["target_cards"] = int(best[-1][0]), int(best[-1][1])
    if run.stdout and request.template.parent.is_dir():
        (request.template.parent / "solver.log").write_text(run.stdout, encoding="utf-8")
    return meta


def _collect(
    demo: Demonstration, run: SolverRun, keep: int, cards, scripts, expected_start: Replay | None = None
) -> None:
    """Convert and verify the solver's lines into ``demo`` (best first, distinct response lists)."""
    seen: set[tuple[bytes, ...]] = set()
    targets = demo.targets
    for sol in run.solutions:
        if len(demo.lines) >= keep:
            break
        try:
            yrp = load_yrp(sol.path)
            if expected_start is not None:
                require_same_start(expected_start, Replay.from_yrp(yrp))
            if demo.start is None:
                demo.set_start(yrp)
            line = convert_line(yrp, targets, cards=cards, scripts=scripts)
            if expected_start is not None:
                if line.board["turn"] != 2:
                    raise DemoError("ordinary opening must finish at the start of turn 2")
                replay = replace(expected_start, responses=list(line.responses))
                if replay.play(**_engine_kwargs(cards, scripts)).reason == "error":
                    raise DemoError("ordinary opening replay ended with an engine error")
        except (DemoError, YrpError, ValueError) as exc:
            demo.rejected.append({"source": sol.path.name, "error": str(exc)})
            continue
        key = tuple(line.responses)
        if key in seen:
            continue
        seen.add(key)
        line.score = {"burned": sol.burned, "actions": sol.actions, "alt": sol.alt}
        line.source = sol.path.name
        demo.lines.append(line)


def _validate_reference(reference: Demonstration, deck: Deck, targets, env: Environment | None,
                        cards=None, scripts=None) -> Replay:  # fmt: skip
    """Accept only complete, ordinary, same-environment/deck reference lines."""
    if env is None or reference.environment != {"version": env.version, "fingerprint": env.fingerprint}:
        raise ValueError("reference environment identity mismatch")
    if (
        reference.variant != "plain"
        or reference.fire is not None
        or reference.status != "solved"
        or not reference.lines
    ):
        raise ValueError("reference requires a solved plain line")
    cards = cards if cards is not None else default_cards()
    if violations := env.validate_deck(deck, cards):
        raise ValueError(f"illegal reference deck: {violations}")
    replay = reference.replay(0)
    cfg = _config(env)
    if replay.rule_flags != cfg.rule_flags or replay.player != asdict(cfg.player):
        raise ValueError("reference rules differ from its environment")
    for zone in ("main", "extra"):
        if sorted(reference.deck[zone]) != sorted(getattr(deck, zone)) or sorted(replay.decks["a"][zone]) != sorted(
            getattr(deck, zone)
        ):
            raise ValueError("reference deck differs from the requested deck or replay start")
        if replay.decks["b"][zone] != list(getattr(PASSIVE_OPPONENT, zone)):
            raise ValueError("reference requires the passive opponent")
    if replay.first != 0 or replay.shuffle_decks:
        raise ValueError("reference requires a recorded first-player start")
    verify_line(reference, 0, env=env, cards=cards, scripts=scripts)
    if reference.lines[0].board["turn"] != 2 or board_summary_missing(reference.lines[0].board, targets, cards):
        raise ValueError("reference must finish turn 1 with the requested final targets")
    if replay.play(env, **_engine_kwargs(cards, scripts)).reason == "error":
        raise ValueError("reference replay ended with an engine error")
    return replay


def solve_hand(job: HandJob, *, cards=None, scripts: _core.ScriptDirectory | None = None,
               reference: Demonstration | None = None, reference_guidance: bool = False) -> Demonstration:  # fmt: skip
    """Solve one opening hand and return its verified record (never raises for solver-side failures)."""
    env = _load_env(job.env)
    config = _config(env)
    deck = load_ydk(job.deck_path)
    if job.hand is not None:
        hand = list(job.hand)
        order = _order_with_hand(deck, hand)
    else:
        hand, order = sample_hand(deck, job.hand_seed, config.player.starting_hand)
    targets = parse_targets(job.targets)
    demo = Demonstration.new(deck, hand, hand_index=job.hand_index, hand_seed=job.hand_seed, variant="plain",
                             targets=targets, environment=env)  # fmt: skip
    scratch = Path(job.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    try:
        _check_targets(deck, targets, cards)
        if reference_guidance and reference is None:
            raise ValueError("reference guidance requires a complete reference")
        if reference is not None and not job.ordinary_shuffle:
            raise ValueError("reference comparison requires ordinary-shuffle generation")
        if job.ordinary_shuffle and job.fire:
            raise ValueError("ordinary-shuffle generation currently supports plain openings only")
        deck_file = scratch / f"{deck.name}.ydk"
        deck_file.write_text(Deck(tuple(order), deck.extra, (), deck.name).to_ydk(), encoding="utf-8")
        template_deck = replace(deck, main=tuple(order)) if job.ordinary_shuffle else deck
        template = make_template(scratch / "template.yrpX", template_deck, job.hand_seed, config, cards, scripts)
        expected_start = None
        if job.ordinary_shuffle:
            demo.set_start(load_yrp(template))  # preserve the true start even when search finds no line
            expected_start = Replay.from_yrp(template)
        search_reference, reference_meta = template, {}
        if reference is not None:
            replay = _validate_reference(reference, deck, targets, env, cards, scripts)
            search_reference = scratch / "reference.yrpX"
            replay.to_yrpx(search_reference, env=env, **_engine_kwargs(cards, scripts))
            reference_meta = {
                "source_sha256": hashlib.sha256(json.dumps(reference.to_json(), sort_keys=True).encode()).hexdigest(),
                "guided": reference_guidance,
                "max_decisions": 12 * (len(deck.main) + len(deck.extra)) + 32,
            }
        request = SolveRequest(template=search_reference, deck=None if job.ordinary_shuffle else deck_file,
                               hand=() if job.ordinary_shuffle else tuple(hand), targets=tuple(targets),
                               start=template if job.ordinary_shuffle else None,
                               no_reference=job.ordinary_shuffle and reference is None, reference_guidance=reference_guidance,
                               max_decisions=reference_meta.get("max_decisions"),
                               solve_ms=job.solve_ms, threads=job.threads, seed=job.solver_seed,
                               max_written=max(4, 2 * job.lines), max_rollouts=job.max_rollouts)  # fmt: skip
        run = run_solver(
            request, Workdir.create(job.workdir), scratch / "out", binary=job.binary, timeout=job.timeout_s
        )
        demo.solver = _solver_meta(run, request)
        if reference_meta:
            demo.solver["reference"] = reference_meta
            (scratch / "command.json").write_text(
                json.dumps([str(job.binary or find_solver()), *run.args], indent=2) + "\n"
            )
        if (run.returncode != 0 and (not run.solutions or job.ordinary_shuffle)) or (
            job.ordinary_shuffle and run.timed_out
        ):
            demo.status, demo.error = "error", f"solver exited with {run.returncode}: {run.problems()}"
        else:
            _collect(demo, run, job.lines, cards, scripts, expected_start)
            demo.status = "solved" if demo.lines else ("unverified" if run.solutions else "unsolved")
            if demo.status == "unverified":
                demo.error = f"none of the {len(run.solutions)} solver lines survived the fresh replay"
    except (SolverError, OSError, ValueError) as exc:
        demo.status, demo.error = "error", str(exc)
    finally:
        demo.solver["wall_s"] = round(time.monotonic() - start, 2)
        if not job.keep_files:
            shutil.rmtree(scratch / "out", ignore_errors=True)
    return demo


def solve_fire(job: HandJob, base: Demonstration, fire: int, *, cards=None,
               scripts: _core.ScriptDirectory | None = None) -> Demonstration:  # fmt: skip
    """The ``--fire`` variant of a solved hand: the opponent plays ``fire`` at every legal window of its best line."""
    demo = replace(
        base, variant="fire", fire=fire, status="pending", error="", start=None, lines=[], rejected=[], solver={}
    )
    if not base.lines:
        demo.status, demo.error = "skipped", "the plain opening has no verified line to test"
        return demo
    scratch = Path(job.scratch) / f"fire_{fire}"
    scratch.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    try:
        # The base line already runs to turn 2 (passive closing), which the solver's --fire reference needs: it
        # captures the target board at the end of turn 1, and misreads a line that stops right after its last summon.
        reference = scratch / "line.yrpX"
        rep = base.replay(0)
        rep.environment = None  # the solver re-runs the file itself; the environment was checked when solving
        rep.to_yrpx(reference, names=(base.deck["name"], PASSIVE_OPPONENT.name), **_engine_kwargs(cards, scripts))
        request = SolveRequest(template=reference, fire=fire, fire_ms=job.fire_ms, solve_ms=job.solve_ms,
                               threads=job.threads, seed=job.solver_seed, max_written=max(4, 2 * job.lines),
                               max_rollouts=job.max_rollouts)  # fmt: skip
        timeout = job.timeout_s if job.timeout_s is not None else job.solve_ms / 1000 + 120 + job.fire_ms / 1000 * 12
        run = run_solver(request, Workdir.create(job.workdir), scratch / "out", binary=job.binary, timeout=timeout)
        demo.solver = _solver_meta(run, request)
        if run.returncode != 0 and not run.solutions:
            demo.status, demo.error = "error", f"solver exited with {run.returncode}: {run.problems()}"
        elif not demo.solver.get("windows"):
            demo.status = "no_window" if "windows" in demo.solver else "error"
            demo.error = "" if demo.status == "no_window" else f"no --fire verdict: {run.problems()}"
        else:
            _collect(demo, run, job.lines, cards, scripts)
            demo.status = "solved" if demo.lines else ("unverified" if run.solutions else "unsolved")
            if demo.status == "unverified":
                demo.error = f"none of the {len(run.solutions)} solver lines survived the fresh replay"
    except (SolverError, OSError, ValueError) as exc:
        demo.status, demo.error = "error", str(exc)
    finally:
        demo.solver["wall_s"] = round(time.monotonic() - start, 2)
        if not job.keep_files:
            shutil.rmtree(scratch, ignore_errors=True)
    return demo


def _check_targets(deck: Deck, targets, cards) -> None:
    """Every target card must be in the deck (main or extra), as often as the target asks."""
    from collections import Counter

    from ygorl.engine.duel import default_cards

    db = cards if cards is not None else default_cards()
    have = Counter(db.canonical(c) for c in (*deck.main, *deck.extra))
    want = Counter(db.canonical(t.code) for t in targets)
    short = [f"{code} (x{n}, deck has {have[code]})" for code, n in sorted(want.items()) if have[code] < n]
    if short:
        raise ValueError(f"target cards not in deck {deck.name}: {', '.join(short)}")


def _order_with_hand(deck: Deck, hand: list[int]) -> list[int]:
    """A deck order with ``hand`` on top (end of the list); raises ValueError if the deck lacks a card."""
    rest = list(deck.main)
    for code in hand:
        if code not in rest:
            raise ValueError(f"opening hand card {code} is not (often enough) in the main deck of {deck.name}")
        rest.remove(code)
    return rest + list(hand)


def run_job(job: HandJob, base: Demonstration | None = None) -> list[dict]:
    """Worker entry point: the plain record and one ``--fire`` record per hand trap, as JSON dicts."""
    base = base if base is not None else solve_hand(job)
    out = [base.to_json()]
    for fire in job.fire:
        out.append(solve_fire(job, base, fire).to_json())
    if not job.keep_files:
        shutil.rmtree(job.scratch, ignore_errors=True)
    return out


def resume_job(item: tuple[HandJob, Demonstration | None]) -> list[dict]:
    """Pool entry point: missing fire records must descend from the persisted plain line."""
    return run_job(*item)


# ------------------------------------------------------------------ planning, resuming, summarising a batch


def hand_seed(seed: int, deck_name: str, index: int) -> int:
    """The shuffle seed of hand ``index`` of a deck: stable across runs, machines and Python versions."""
    digest = hashlib.sha256(f"{seed}:{deck_name}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def load_targets(path: str | Path) -> dict[str, dict]:
    """Per-deck solver settings: ``{deck name: {"targets": [password[@zone[:fd]], ...], "fire": [password, ...]}}``.

    A plain list is accepted as the targets. Keys starting with ``_`` are comments. Targets are validated.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for name, spec in raw.items():
        if name.startswith("_"):
            continue
        spec = {"targets": spec} if isinstance(spec, list) else dict(spec)
        try:
            parse_targets(spec.get("targets", []))
        except ValueError as exc:
            raise ValueError(f"{path}: deck {name}: {exc}") from None
        if not spec.get("targets"):
            raise ValueError(f"{path}: deck {name} has no targets")
        spec["fire"] = [int(c) for c in spec.get("fire", [])]
        out[name] = spec
    return out


def repair_jsonl(path: str | Path) -> int:
    """Cut a partial last line left by an interrupted run (so appends start on a fresh line); bytes removed."""
    path = Path(path)
    if not path.exists():
        return 0
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return 0
    keep = data.rfind(b"\n") + 1
    with open(path, "r+b") as f:
        f.truncate(keep)
    return len(data) - keep


def done_keys(path: str | Path, environment: dict | None) -> set[tuple]:
    """Keys (:attr:`Demonstration.key`) already in a demonstration file; refuses a file of another environment."""
    path = Path(path)
    keys: set[tuple] = set()
    if not path.exists():
        return keys
    for demo in read_jsonl(path):
        if demo.environment != environment:
            raise ValueError(
                f"{path} holds demonstrations of environment {demo.environment}, not {environment}; use another --out"
            )
        if demo.key in keys:
            raise ValueError(f"{path} holds duplicate demonstration key {demo.key}")
        keys.add(demo.key)
    return keys


def job_done(job_deck: str, job: HandJob, keys: set[tuple]) -> bool:
    return (job_deck, job.hand_index, "plain", None) in keys and all(
        (job_deck, job.hand_index, "fire", f) in keys for f in job.fire
    )


def summarize(records: list[dict]) -> dict:
    """Counts per (deck, variant): hands, statuses, verified lines, rejected lines, solver wall time."""
    table: dict[str, dict] = {}
    for r in records:
        variant = r["variant"] if r.get("fire") is None else f"fire:{r['fire']}"
        row = table.setdefault(f"{r['deck']['name']} {variant}", {"hands": 0, "lines": 0, "rejected": 0, "wall_s": 0.0})
        row["hands"] += 1
        row[r["status"]] = row.get(r["status"], 0) + 1
        row["lines"] += len(r.get("lines", []))
        row["rejected"] += len(r.get("rejected", []))
        row["wall_s"] = round(row["wall_s"] + r.get("solver", {}).get("wall_s", 0.0), 2)
        if r.get("fire") is not None:
            row["windows"] = row.get("windows", 0) + (r.get("solver", {}).get("windows") or 0)
            row["converted"] = row.get("converted", 0) + (r.get("solver", {}).get("converted") or 0)
    return table


__all__ = ["DEFAULT_FIRE_MS", "PASSIVE_OPPONENT", "HandJob", "done_keys", "hand_seed", "job_done", "load_targets",
           "make_template", "repair_jsonl", "run_job", "sample_hand", "solve_fire", "solve_hand", "summarize"]  # fmt: skip
