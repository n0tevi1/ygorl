"""Evaluation: paired-seed arena, matchup matrices and meta-game solutions.

``ygorl.eval.matchup`` (matrix, Nash, alpha-rank) is imported on its own so that
using the arena does not load nashpy and scipy.
"""

from ygorl.eval.arena import Arena, ArenaReport, GameRecord, derive_seed, merge, wilson_interval

__all__ = ["Arena", "ArenaReport", "GameRecord", "derive_seed", "merge", "wilson_interval"]
