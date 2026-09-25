"""Subprocess wrapper around the ygo-combo-solver binary (T4a.1; see docs/solver.md).

The solver (AGPL-3.0, github.com/96jonesa/ygo-combo-solver) is built from source against our
pinned, patched core by ``tools/build_combo_solver.sh``; nothing of it is vendored here. This
module finds the binary, lays out the EDOPro-style work directory it reads cards and scripts
from, assembles its command line, runs it under a wall-clock timeout and parses what it prints
(``@event`` JSON lines from ``--json``) and writes (``solution_NN_bB_aA[_alt].yrp``).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from ygorl import paths
from ygorl.solver.targets import TargetCard

SOLVER_REPO = "https://github.com/96jonesa/ygo-combo-solver"
SOLVER_COMMIT = "e0c7221802a23657d5a9a0b020025e0ceca04675"  # keep in sync with tools/build_combo_solver.sh
BINARY_NAME = "combosolver"
SCRIPT_ROOT_VIEW = "script-root"
_SOLUTION = re.compile(r"^solution_(\d+)_b(\d+)_a(\d+)(_alt)?\.yrp$")


class SolverError(RuntimeError):
    """The solver could not be run, or failed."""


class SolverNotFound(SolverError):
    """No solver binary was found."""


def default_binary() -> Path:
    """Where ``tools/build_combo_solver.sh`` puts the binary (``build/combo-solver/bin/``)."""
    return paths.third_party().parent / "build" / "combo-solver" / "bin" / BINARY_NAME


def find_solver(path: str | Path | None = None) -> Path:
    """The solver binary: ``path``, else ``$YGORL_COMBO_SOLVER``, else the build directory, else ``PATH``."""
    hint = "build it with tools/build_combo_solver.sh or set YGORL_COMBO_SOLVER"
    if path is not None:
        candidates = [Path(path)]
    elif os.environ.get("YGORL_COMBO_SOLVER"):
        candidates = [Path(os.environ["YGORL_COMBO_SOLVER"])]
    else:
        candidates = []
        try:
            candidates.append(default_binary())
        except FileNotFoundError:
            pass
        found = shutil.which(BINARY_NAME)
        if found:
            candidates.append(Path(found))
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return c
    where = ", ".join(str(c) for c in candidates) or "nowhere"
    raise SolverNotFound(f"combo solver binary not found (looked at {where}); {hint}")


@dataclass(frozen=True)
class Workdir:
    """The EDOPro-style directory the solver reads: ``cards.cdb`` plus script directories (``--scriptdir``).

    The scripts are ours (``ygorl.paths.script_directories()``) in our priority order. The solver
    adds every immediate subdirectory of a ``--scriptdir`` behind it, so the CardScripts root is
    passed as a view holding only its top-level ``*.lua`` files; the subdirectories follow in our order.
    """

    path: Path
    scriptdirs: tuple[Path, ...]

    @classmethod
    def create(cls, path: str | Path) -> Workdir:
        path = Path(path)
        view = path / SCRIPT_ROOT_VIEW
        view.mkdir(parents=True, exist_ok=True)
        _link(path / "cards.cdb", paths.cards_cdb())
        dirs = paths.script_directories()
        root = dirs[0]
        for f in sorted(root.glob("*.lua")):
            _link(view / f.name, f)
        return cls(path, (view, *(d for d in dirs[1:] if d.is_dir())))


def _link(link: Path, target: Path) -> None:
    target = target.resolve()
    if link.is_symlink() and link.resolve() == target:
        return
    if link.exists() or link.is_symlink():
        link.unlink()
    try:
        link.symlink_to(target)
    except FileExistsError:  # a concurrent worker made the same link
        pass


@dataclass(frozen=True)
class SolveRequest:
    """One solver run.

    Two modes: an opening (``deck`` + ``hand`` + ``targets``; ``template`` only supplies the rule
    flags, life points, core seed and the opponent, ``--no-ref``), or a ``--fire`` test of the line
    recorded in ``template`` (the opponent plays ``fire`` at every legal window and the solver
    rebuilds the board; ``--fire-bake`` writes the card into the replays' headers).
    """

    template: Path
    deck: Path | None = None
    hand: tuple[int, ...] = ()
    targets: tuple[TargetCard, ...] = ()
    solve_ms: int = 60_000
    fire: int | None = None
    fire_ms: int | None = None
    threads: int | None = None
    seed: int | None = None
    max_written: int | None = None
    max_decisions: int | None = None
    max_rollouts: int | None = None  # --max-rollouts: rollouts per worker and search phase (wall time still bounds)
    extra_args: tuple[str, ...] = ()

    def args(self, workdir: Workdir, outdir: Path) -> list[str]:
        out = [str(self.template), "--workdir", str(workdir.path)]
        for d in workdir.scriptdirs:
            out += ["--scriptdir", str(d)]
        if self.fire is not None:
            out += ["--fire", str(self.fire), "--fire-bake"]
            if self.fire_ms is not None:
                out += ["--fire-ms", str(self.fire_ms)]
        else:
            if self.deck is None or not self.hand:
                raise ValueError("an opening needs a deck and an opening hand")
            if not self.targets:
                raise ValueError("an opening needs at least one target card")
            out += ["--no-ref", "--deck", str(self.deck), "--hand", "|".join(str(c) for c in self.hand)]
            for t in self.targets:
                out += ["--target", t.to_arg()]
        out += ["--solve-ms", str(self.solve_ms)]
        for flag, value in (("--threads", self.threads), ("--seed", self.seed), ("--max-written", self.max_written),
                            ("--max-decisions", self.max_decisions), ("--max-rollouts", self.max_rollouts)):  # fmt: skip
            if value is not None:
                out += [flag, str(value)]
        out += ["--json", *self.extra_args, "--outdir", str(outdir)]
        return out


@dataclass(frozen=True)
class SolutionFile:
    path: Path
    index: int  # rank in the solver's sorted output (0 = best)
    burned: int  # cards sent to the graveyard or banished
    actions: int  # summons and activations
    alt: bool  # reaches an alternative goal (--fire-spare), not the full board


def parse_solution_name(name: str, directory: Path | None = None) -> SolutionFile | None:
    m = _SOLUTION.match(name)
    if m is None:
        return None
    return SolutionFile(Path(directory or ".") / name, int(m[1]), int(m[2]), int(m[3]), bool(m[4]))


def parse_events(stdout: str) -> list[dict]:
    """The ``@event {json}`` lines of a ``--json`` run, in order (malformed lines are skipped)."""
    events = []
    for line in stdout.splitlines():
        if not line.startswith("@event "):
            continue
        try:
            ev = json.loads(line[7:])
        except json.JSONDecodeError:
            continue
        if isinstance(ev, dict):
            events.append(ev)
    return events


@dataclass
class SolverRun:
    args: list[str]
    returncode: int
    elapsed_s: float
    stdout: str
    events: list[dict] = field(default_factory=list)
    solutions: list[SolutionFile] = field(default_factory=list)  # best first
    timed_out: bool = False

    def event(self, kind: str) -> dict | None:
        """The last event of type ``kind``."""
        return next((e for e in reversed(self.events) if e.get("type") == kind), None)

    @property
    def seed(self) -> int | None:
        ev = self.event("seed")
        return ev.get("seed") if ev else None

    @property
    def candidates(self) -> int:
        ev = self.event("written")
        return int(ev.get("candidates", 0)) if ev else 0

    def problems(self, limit: int = 6) -> str:
        """The report's ``!!`` lines (the solver's error and warning channel), for error messages."""
        lines = [
            ln.strip()
            for ln in self.stdout.splitlines()
            if ln.lstrip().startswith("!!") and "scripts not found" not in ln
        ]
        return "; ".join(lines[:limit]) or (self.stdout.strip().splitlines() or ["no output"])[-1]


def run_solver(request: SolveRequest, workdir: Workdir, outdir: str | Path, *, binary: str | Path | None = None,
               timeout: float | None = None) -> SolverRun:  # fmt: skip
    """Run the solver once; raise :class:`SolverError` only when it cannot be started.

    ``timeout`` (seconds, default: the search budget plus two minutes) bounds the wall time; on
    expiry the solver is interrupted (it then writes what it found, like on Ctrl-C) and killed
    if it does not stop within 20 s.
    """
    exe = find_solver(binary)
    outdir = Path(outdir)
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)
    args = request.args(workdir, outdir)
    if timeout is None:
        timeout = request.solve_ms / 1000 + 120 + (request.fire_ms or 0) / 1000 * 8
    start = time.monotonic()
    try:
        proc = subprocess.Popen([str(exe), *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                errors="replace", cwd=outdir.parent)  # fmt: skip
    except OSError as exc:
        raise SolverError(f"cannot start {exe}: {exc}") from None
    timed_out = False
    try:
        stdout, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.send_signal(signal.SIGINT)
        try:
            stdout, _ = proc.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, _ = proc.communicate()
    elapsed = time.monotonic() - start
    solutions = sorted((s for f in outdir.iterdir() if (s := parse_solution_name(f.name, outdir)) is not None),
                       key=lambda s: s.index)  # fmt: skip
    return SolverRun(args, proc.returncode, elapsed, stdout, parse_events(stdout), solutions, timed_out)


def solver_version() -> dict:
    return {"repo": SOLVER_REPO, "commit": SOLVER_COMMIT}


__all__ = ["SOLVER_COMMIT", "SolutionFile", "SolveRequest", "SolverError", "SolverNotFound", "SolverRun", "Workdir",
           "find_solver", "parse_events", "parse_solution_name", "run_solver", "solver_version"]  # fmt: skip
