"""ygorl.solver: opening lines from ygo-combo-solver, turned into verified demonstrations (T4a.1).

See docs/solver.md. The solver binary is built by ``tools/build_combo_solver.sh``; the batch
driver is ``tools/solve_openings.py``.
"""

from ygorl.solver.batch import DEFAULT_FIRE_MS, PASSIVE_OPPONENT, HandJob, make_template, run_job, sample_hand, solve_fire, solve_hand
from ygorl.solver.combo_solver import (
    SOLVER_COMMIT,
    SolutionFile,
    SolveRequest,
    SolverError,
    SolverNotFound,
    SolverRun,
    Workdir,
    find_solver,
    parse_events,
    parse_solution_name,
    run_solver,
)
from ygorl.solver.demo import (
    DEMO_FORMAT,
    DemoError,
    DemoLine,
    Demonstration,
    canonical_response,
    convert_line,
    iter_steps,
    read_jsonl,
    verify_line,
)
from ygorl.solver.targets import TargetCard, board_summary, board_summary_missing, parse_targets

__all__ = [
    "DEFAULT_FIRE_MS", "DEMO_FORMAT", "PASSIVE_OPPONENT", "SOLVER_COMMIT", "DemoError", "DemoLine", "Demonstration",
    "HandJob", "SolutionFile", "SolveRequest", "SolverError", "SolverNotFound", "SolverRun", "TargetCard", "Workdir",
    "board_summary", "board_summary_missing", "canonical_response", "convert_line", "find_solver", "iter_steps",
    "make_template", "parse_events", "parse_solution_name", "parse_targets", "read_jsonl", "run_job", "run_solver",
    "sample_hand", "solve_fire", "solve_hand", "verify_line",
]  # fmt: skip
