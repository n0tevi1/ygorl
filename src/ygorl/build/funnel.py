"""Funnel stage 1: opening-hand analysis with the combo solver, as a cheap filter and QD descriptors (T5.6).

A candidate deck (or a :class:`~ygorl.build.genotype.Genotype` decoded by its space) gets ``hands``
opening hands with fixed seeds (hand ``i`` is the host shuffle of ``hand_seed(seed, "funnel", i)``,
the same permutation for every deck of the same size: common random numbers across candidates).
Every hand is handed to the T4a.1 solver (:func:`ygorl.solver.solve_hand`) with the deck's target
board; a hand whose search finds no verified line within ``solve_ms`` is a brick (卡手). Every solved
hand is then re-run with each ``--fire`` hand trap (:func:`ygorl.solver.solve_fire`): the opponent plays
it at every legal window of the line and the solver must rebuild the board.

:class:`FunnelResult` aggregates the hands: best-line rate, brick rate (with a Wilson interval),
hand-trap survival (抗手坑率), combo length of the best lines, and the evaluation time against the
stage-1 budget. :class:`FunnelFilter` is the pass/fail gate (with early stopping once a deck can no
longer pass) and :meth:`FunnelResult.descriptors` the numbers a QD archive (T5.8) indexes on.
Definitions and measurements are in docs/funnel.md.
"""

from __future__ import annotations

import math
import re
import shutil
import statistics
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from ygorl.cards.ydk import Deck
from ygorl.eval.arena import wilson_interval
from ygorl.solver.batch import HandJob, _check_targets, hand_seed, sample_hand, solve_fire, solve_hand
from ygorl.solver.combo_solver import Workdir
from ygorl.solver.targets import parse_targets

ASH_BLOSSOM = 14558127  # Ash Blossom & Joyous Spring, the default --fire hand trap
DEFAULT_BUDGET_S = 120.0  # solver process-seconds per deck (docs/funnel.md, "budget")
FUNNEL_FORMAT = "ygorl-funnel"
FUNNEL_FORMAT_VERSION = 1
HAND_SEED_KEY = "funnel"  # hand_seed(seed, HAND_SEED_KEY, i): independent of the deck, so hands are paired across decks
DESCRIPTORS = ("brick_rate", "combo_length", "hand_trap_survival")


@dataclass(frozen=True)
class FunnelConfig:
    """How a deck is evaluated. The defaults fit :data:`DEFAULT_BUDGET_S` (see docs/funnel.md)."""

    hands: int = 12
    seed: int = 0  # base seed of the hand shuffles
    solve_ms: int = 10_000  # solver search budget per hand (and per target alternative); 5 s over-calls long combos
    fire: tuple[int, ...] = (ASH_BLOSSOM,)  # hand traps of the --fire variant; () skips it
    fire_ms: int = 3_000  # solver budget per --fire window
    workers: int = 1  # parallel solver processes for one deck (1: in this process)
    threads: int = 1  # solver threads per process
    solver_seed: int | None = 1  # fixed rollout seed (the search is still bounded by wall time)
    # Opt-in count budget (solver --max-rollouts): rollouts per solver phase and worker. solve_ms still bounds every
    # phase, so with threads=1 a run is reproducible only when the count binds first (solve_ms set well above it).
    max_rollouts: int | None = None
    env: str | None = None  # environment directory or version: rules of the duel; stamps the result
    binary: Path | None = None
    budget_s: float = DEFAULT_BUDGET_S  # stage-1 budget in solver process-seconds per deck
    keep_demos: bool = False  # keep the solver records (Demonstration JSON) in each HandOutcome

    def __post_init__(self) -> None:
        if self.hands < 1:
            raise ValueError("hands must be >= 1")
        if self.workers < 1:
            raise ValueError("workers must be >= 1")

    def hand_seeds(self) -> list[int]:
        return [hand_seed(self.seed, HAND_SEED_KEY, i) for i in range(self.hands)]


@dataclass
class HandOutcome:
    """One opening hand: did a line exist, how long was it, did it survive each hand trap."""

    index: int
    hand_seed: int
    hand: list[int]
    status: str  # "solved" (a verified line exists), "brick" (none within the budget) or "error"
    target: int | None = None  # index of the target alternative the line reaches
    combo_actions: int | None = None  # best line: summons and activations (solver score)
    combo_decisions: int | None = None  # best line: our action steps that came from the solver
    burned: int | None = None  # best line: cards sent to the grave or banished (solver score)
    fire: dict[int, dict] = field(default_factory=dict)  # hand trap -> status, windows, converted, survives, wall_s
    solver_s: float = 0.0  # solver wall time spent on this hand (plain + fire), process-seconds
    error: str = ""
    demos: list[dict] = field(default_factory=list)  # Demonstration.to_json() of every solver run (keep_demos)

    def survives(self, fire: int) -> bool | None:
        """Opened and survived ``fire``: True / False; None when the hand is an error or was not tested."""
        if self.status == "error":
            return None
        if self.status == "brick":
            return False
        rec = self.fire.get(fire)
        return None if rec is None else rec["survives"]

    def to_json(self) -> dict:
        d = asdict(self)
        d["fire"] = {str(k): v for k, v in self.fire.items()}
        return d

    @classmethod
    def from_json(cls, d: Mapping) -> HandOutcome:
        return cls(**{**d, "fire": {int(k): v for k, v in d.get("fire", {}).items()}})


def fire_survives(record: Mapping) -> bool | None:
    """Whether a ``--fire`` record (Demonstration JSON) shows the line surviving the hand trap.

    ``no_window``: the hand trap never had a legal window on the line (survives). ``solved``: a
    verified line rebuilt the board, and the solver recovered from *every* window (the opponent
    picks its timing). ``unsolved``: some window was not recovered. Anything else is unknown (None).
    """
    status = record["status"]
    if status == "no_window":
        return True
    solver = record.get("solver", {})
    if status == "solved":
        return (solver.get("converted") or 0) >= (solver.get("windows") or 0)
    if status == "unsolved":
        return False
    return None


# ------------------------------------------------------------------ one hand (a worker job)


@dataclass(frozen=True)
class _HandTask:
    deck_path: Path
    index: int
    hand_seed: int
    targets: tuple[tuple[str, ...], ...]
    config: FunnelConfig
    workdir: Path
    scratch: Path


def _evaluate_hand(task: _HandTask) -> dict:
    """Solve one hand against each target alternative until one is reached, then run the --fire variants."""
    cfg = task.config
    records: list[dict] = []
    out = HandOutcome(task.index, task.hand_seed, [], "brick")
    errors: list[str] = []
    solved = None
    for alt, targets in enumerate(task.targets):
        job = HandJob(deck_path=task.deck_path, hand_index=task.index, hand_seed=task.hand_seed, targets=targets,
                      workdir=task.workdir, scratch=task.scratch / f"t{alt}", solve_ms=cfg.solve_ms, threads=cfg.threads,
                      fire=cfg.fire, fire_ms=cfg.fire_ms, binary=cfg.binary, solver_seed=cfg.solver_seed,
                      max_rollouts=cfg.max_rollouts, env=cfg.env)  # fmt: skip
        demo = solve_hand(job)
        records.append(demo.to_json())
        out.hand = list(demo.hand)
        out.solver_s += demo.solver.get("wall_s", 0.0)
        if demo.status == "solved":
            solved, out.target = (job, demo), alt
            break
        if demo.status != "unsolved":  # error / unverified: a failure of the tooling, not a brick
            errors.append(f"target {alt}: {demo.status}: {demo.error}")
    if solved is None:
        out.status, out.error = ("error", "; ".join(errors)) if errors else ("brick", "")
    else:
        job, demo = solved
        best = demo.lines[0]
        out.status = "solved"
        out.combo_actions = best.score.get("actions")
        out.burned = best.score.get("burned")
        out.combo_decisions = best.solver_steps
        for fire in cfg.fire:
            rec = solve_fire(job, demo, fire).to_json()
            records.append(rec)
            solver = rec.get("solver", {})
            out.solver_s += solver.get("wall_s", 0.0)
            out.fire[fire] = {"status": rec["status"], "windows": solver.get("windows"), "converted": solver.get("converted"),
                              "survives": fire_survives(rec), "wall_s": solver.get("wall_s", 0.0)}  # fmt: skip
        shutil.rmtree(job.scratch, ignore_errors=True)
    out.solver_s = round(out.solver_s, 2)
    if cfg.keep_demos:
        out.demos = records
    shutil.rmtree(task.scratch, ignore_errors=True)
    return out.to_json()


# ------------------------------------------------------------------ the filter


@dataclass(frozen=True)
class FunnelFilter:
    """The stage-1 gate: a deck passes when its brick rate and hand-trap survival clear the bars.

    ``max_brick_rate`` bounds the share of bricked hands; ``min_hand_trap_survival`` is the least
    share of hands that open *and* survive every ``--fire`` hand trap (0 disables the bar); a deck
    whose hands all end in ``error`` fails. Early stopping: once the bricks alone exceed
    ``max_brick_rate`` of the planned hands the deck cannot pass, and the remaining hands are skipped.
    """

    max_brick_rate: float = 0.5
    min_hand_trap_survival: float = 0.0

    def hopeless(self, bricks: int, planned: int) -> bool:
        return bricks > self.max_brick_rate * planned + 1e-9

    def reasons(self, result: FunnelResult) -> list[str]:
        """Why ``result`` fails the gate (empty: it passes)."""
        out = []
        if result.valid_hands == 0:
            return ["no hand evaluated without error"]
        if result.stopped_early:
            out.append(f"stopped early: {result.bricks} bricks of {result.planned_hands} planned hands")
        if result.brick_rate > self.max_brick_rate + 1e-9:
            out.append(f"brick rate {result.brick_rate:.2f} > {self.max_brick_rate:.2f}")
        survival = result.hand_trap_survival
        if self.min_hand_trap_survival > 0 and (survival is None or survival < self.min_hand_trap_survival - 1e-9):
            shown = "n/a" if survival is None else f"{survival:.2f}"
            out.append(f"hand-trap survival {shown} < {self.min_hand_trap_survival:.2f}")
        return out

    def passes(self, result: FunnelResult) -> bool:
        return not self.reasons(result)


# ------------------------------------------------------------------ the result


def _mean(values: Sequence[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


@dataclass
class FunnelResult:
    """Stage-1 evaluation of one deck: per-hand outcomes and their aggregates."""

    deck: dict  # {"name", "main", "extra"}
    targets: list[list[str]]  # target alternatives, password@zone[:fd]
    config: dict
    hands: list[HandOutcome]
    planned_hands: int
    stopped_early: bool = False
    wall_s: float = 0.0  # wall clock of the whole evaluation
    environment: dict | None = None  # {"version", "fingerprint"}
    genotype: dict | None = None  # GenotypeSpace.genotype_to_json() when evaluated from a genotype
    passed: bool | None = None  # FunnelFilter verdict, when a filter was given
    reasons: list[str] = field(default_factory=list)

    # -- counts -----------------------------------------------------------
    @property
    def valid_hands(self) -> int:
        return sum(h.status != "error" for h in self.hands)

    @property
    def solved(self) -> int:
        return sum(h.status == "solved" for h in self.hands)

    @property
    def bricks(self) -> int:
        return sum(h.status == "brick" for h in self.hands)

    @property
    def errors(self) -> int:
        return sum(h.status == "error" for h in self.hands)

    # -- rates ------------------------------------------------------------
    @property
    def best_line_rate(self) -> float:
        """Share of (non-error) hands with a verified line to the target board."""
        return self.solved / self.valid_hands if self.valid_hands else 0.0

    @property
    def brick_rate(self) -> float:
        """Share of (non-error) hands without a line within the budget (卡手率)."""
        return 1.0 - self.best_line_rate if self.valid_hands else 1.0

    def brick_interval(self, confidence: float = 0.95) -> tuple[float, float]:
        return wilson_interval(self.bricks, self.valid_hands, confidence)

    def survival(self, fire: int, *, given_line: bool = False) -> float | None:
        """Share of hands that open and survive ``fire`` (``given_line``: among the solved hands only)."""
        flags = [h.survives(fire) for h in self.hands if not given_line or h.status == "solved"]
        flags = [f for f in flags if f is not None]
        return sum(flags) / len(flags) if flags else None

    @property
    def hand_trap_survival(self) -> float | None:
        """Share of hands that open and survive every ``--fire`` hand trap (抗手坑率); None without --fire."""
        fires = self.config.get("fire") or []
        if not fires:
            return None
        flags = []
        for h in self.hands:
            per = [h.survives(f) for f in fires]
            if h.status == "error" or any(p is None for p in per):
                continue
            flags.append(all(per))
        return sum(flags) / len(flags) if flags else None

    def window_recovery(self, fire: int) -> float | None:
        """Share of the hand trap's legal windows (over all solved hands) from which the board was rebuilt."""
        windows = converted = 0
        for h in self.hands:
            rec = h.fire.get(fire)
            if rec and rec.get("windows"):
                windows += rec["windows"]
                converted += rec.get("converted") or 0
        return converted / windows if windows else None

    def combo_length(self, key: str = "combo_actions") -> dict:
        """Best-line length over the solved hands: mean, median, min, max (``combo_actions`` or ``combo_decisions``)."""
        values = [getattr(h, key) for h in self.hands if h.status == "solved" and getattr(h, key) is not None]
        if not values:
            return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
        return {"n": len(values), "mean": _mean(values), "median": statistics.median(values), "min": min(values),
                "max": max(values)}  # fmt: skip

    # -- time -------------------------------------------------------------
    @property
    def solver_s(self) -> float:
        """Solver wall time summed over hands: process-seconds, independent of the worker count."""
        return round(sum(h.solver_s for h in self.hands), 2)

    @property
    def within_budget(self) -> bool:
        return self.solver_s <= self.config.get("budget_s", DEFAULT_BUDGET_S)

    # -- QD ---------------------------------------------------------------
    def descriptors(self) -> dict[str, float]:
        """Numbers for a QD archive (T5.8): brick rate, mean combo length (actions), hand-trap survival.

        ``combo_length`` is NaN when no hand was solved and ``hand_trap_survival`` NaN without --fire;
        such decks normally fail the filter before they reach an archive.
        """
        combo = self.combo_length()["mean"]
        survival = self.hand_trap_survival
        return {"brick_rate": self.brick_rate, "combo_length": math.nan if combo is None else float(combo),
                "hand_trap_survival": math.nan if survival is None else survival}  # fmt: skip

    def summary(self) -> dict:
        lo, hi = self.brick_interval()
        fires = self.config.get("fire") or []
        return {
            "deck": self.deck["name"], "hands": len(self.hands), "planned_hands": self.planned_hands,
            "valid_hands": self.valid_hands, "solved": self.solved, "bricks": self.bricks, "errors": self.errors,
            "best_line_rate": round(self.best_line_rate, 4), "brick_rate": round(self.brick_rate, 4),
            "brick_rate_ci95": [round(lo, 4), round(hi, 4)], "hand_trap_survival": self.hand_trap_survival,
            "survival_given_line": {str(f): self.survival(f, given_line=True) for f in fires},
            "window_recovery": {str(f): self.window_recovery(f) for f in fires},
            "combo_actions": self.combo_length(), "combo_decisions": self.combo_length("combo_decisions"),
            "solver_s": self.solver_s, "wall_s": self.wall_s, "budget_s": self.config.get("budget_s"),
            "within_budget": self.within_budget, "stopped_early": self.stopped_early, "passed": self.passed,
            "reasons": self.reasons,
        }  # fmt: skip

    # -- storage ----------------------------------------------------------
    def to_json(self) -> dict:
        d = asdict(self)
        d["hands"] = [h.to_json() for h in self.hands]
        return {"format": FUNNEL_FORMAT, "format_version": FUNNEL_FORMAT_VERSION, **d, "summary": self.summary()}

    @classmethod
    def from_json(cls, data: Mapping) -> FunnelResult:
        if data.get("format") != FUNNEL_FORMAT or data.get("format_version") != FUNNEL_FORMAT_VERSION:
            raise ValueError(f"not a {FUNNEL_FORMAT} v{FUNNEL_FORMAT_VERSION} record")
        fields_ = {k: v for k, v in data.items() if k not in ("format", "format_version", "summary")}
        fields_["hands"] = [HandOutcome.from_json(h) for h in fields_["hands"]]
        return cls(**fields_)


# ------------------------------------------------------------------ evaluation


def normalize_targets(targets: Sequence[str] | Sequence[Sequence[str]]) -> tuple[tuple[str, ...], ...]:
    """One target board (a list of ``password[@zone[:fd]]``) or several alternatives (a list of such lists)."""
    if not targets:
        raise ValueError("no target board")
    if isinstance(targets, str):
        raise ValueError("targets must be a list of target cards, not a string")
    kinds = {isinstance(t, str) for t in targets}
    if len(kinds) > 1:
        raise ValueError("targets mix cards and alternatives: give one list of cards or a list of lists")
    alts = [tuple(targets)] if kinds == {True} else [tuple(t) for t in targets]  # type: ignore[arg-type]
    out = []
    for alt in alts:
        if not alt:
            raise ValueError("empty target alternative")
        out.append(tuple(t.to_arg() for t in parse_targets(alt)))
    return tuple(out)


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "deck"


def _env_stamp(spec: str | None) -> dict | None:
    if spec is None:
        return None
    from ygorl.data import load_environment

    env = load_environment(spec)
    return {"version": env.version, "fingerprint": env.fingerprint}


def evaluate_deck(deck: Deck, targets: Sequence[str] | Sequence[Sequence[str]], config: FunnelConfig | None = None, *,
                  filter: FunnelFilter | None = None, scratch: str | Path | None = None,
                  progress: Callable[[HandOutcome], None] | None = None) -> FunnelResult:  # fmt: skip
    """Run funnel stage 1 on ``deck``: solve its ``config.hands`` opening hands and aggregate.

    ``targets`` is one target board or a list of alternatives (a hand is solved if it reaches any;
    they are tried in order). With ``filter`` the verdict is filled in and the evaluation stops as
    soon as the deck cannot pass. ``progress`` is called with every finished hand.
    """
    config = config or FunnelConfig()
    alts = normalize_targets(targets)
    name = deck.name or "deck"
    deck = replace(deck, name=_safe_name(name))
    for alt in alts:
        _check_targets(deck, parse_targets(alt), None)
    own_scratch = scratch is None
    root = Path(tempfile.mkdtemp(prefix="ygorl-funnel-")) if own_scratch else Path(scratch)
    root.mkdir(parents=True, exist_ok=True)
    deck_path = root / f"{deck.name}.ydk"
    deck_path.write_text(deck.to_ydk(), encoding="utf-8")
    workdir = Workdir.create(root / "workdir").path
    tasks = [
        _HandTask(deck_path, i, s, alts, config, workdir, root / f"hand{i}") for i, s in enumerate(config.hand_seeds())
    ]
    outcomes: list[HandOutcome] = []
    stopped = False
    start = time.monotonic()

    def done(rec: dict) -> bool:
        h = HandOutcome.from_json(rec)
        outcomes.append(h)
        if progress is not None:
            progress(h)
        return filter is not None and filter.hopeless(sum(o.status == "brick" for o in outcomes), config.hands)

    try:
        if config.workers == 1:
            for task in tasks:
                if done(_evaluate_hand(task)):
                    stopped = len(outcomes) < len(tasks)
                    break
        else:
            import multiprocessing

            with ProcessPoolExecutor(config.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                pending = {pool.submit(_evaluate_hand, t) for t in tasks}
                while pending:
                    finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                    if any([done(f.result()) for f in finished]) and pending:
                        stopped = True
                        for f in pending:
                            f.cancel()
                        running = [f for f in pending if not f.cancelled()]
                        for f in running:  # already started: keep what they found
                            done(f.result())
                        pending = set()
    finally:
        if own_scratch:
            shutil.rmtree(root, ignore_errors=True)
    outcomes.sort(key=lambda h: h.index)
    cfg = asdict(config)
    cfg["binary"] = str(config.binary) if config.binary is not None else None
    cfg["fire"] = list(config.fire)
    result = FunnelResult(deck={"name": name, "main": list(deck.main), "extra": list(deck.extra)},
                          targets=[list(a) for a in alts], config=cfg, hands=outcomes, planned_hands=config.hands,
                          stopped_early=stopped, wall_s=round(time.monotonic() - start, 2),
                          environment=_env_stamp(config.env))  # fmt: skip
    if filter is not None:
        result.reasons = filter.reasons(result)
        result.passed = not result.reasons
    return result


def evaluate_genotype(space: Any, genotype: Any, targets: Sequence[str] | Sequence[Sequence[str]],
                      config: FunnelConfig | None = None, *, name: str = "genotype", **kwargs) -> FunnelResult:  # fmt: skip
    """:func:`evaluate_deck` on ``space.decode(genotype)`` (a T5.5 :class:`~ygorl.build.genotype.GenotypeSpace`)."""
    result = evaluate_deck(space.decode(genotype, name=name), targets, config, **kwargs)
    result.genotype = space.genotype_to_json(genotype)
    return result


def opening_hands(deck: Deck, config: FunnelConfig | None = None, hand_size: int = 5) -> list[list[int]]:
    """The opening hands :func:`evaluate_deck` will solve (the host shuffle of each fixed seed)."""
    config = config or FunnelConfig()
    return [sample_hand(deck, s, hand_size)[0] for s in config.hand_seeds()]


# ------------------------------------------------------------------ paired comparison


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pair counts ``b`` and ``c`` (binomial, p = 1/2)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_table(a: Sequence[bool], b: Sequence[bool]) -> dict:
    """2x2 table of paired binary outcomes (e.g. brick by the funnel vs brick in a real first turn) and McNemar.

    ``both`` / ``neither``: concordant; ``only_a`` / ``only_b``: discordant. ``rate_a`` / ``rate_b`` are
    the two marginal rates; ``p_value`` the exact McNemar test of equal marginals.
    """
    if len(a) != len(b):
        raise ValueError("paired outcomes must have the same length")
    both = sum(x and y for x, y in zip(a, b, strict=True))
    only_a = sum(x and not y for x, y in zip(a, b, strict=True))
    only_b = sum(y and not x for x, y in zip(a, b, strict=True))
    n = len(a)
    return {"n": n, "both": both, "only_a": only_a, "only_b": only_b, "neither": n - both - only_a - only_b,
            "rate_a": (both + only_a) / n if n else None, "rate_b": (both + only_b) / n if n else None,
            "agreement": (n - only_a - only_b) / n if n else None, "p_value": mcnemar_exact(only_a, only_b)}  # fmt: skip


__all__ = ["ASH_BLOSSOM", "DEFAULT_BUDGET_S", "DESCRIPTORS", "FunnelConfig", "FunnelFilter", "FunnelResult", "HandOutcome",
           "evaluate_deck", "evaluate_genotype", "fire_survives", "mcnemar_exact", "normalize_targets", "opening_hands",
           "paired_table"]  # fmt: skip
