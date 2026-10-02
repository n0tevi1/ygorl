"""Fixed-opponent checkpoint comparisons, clustered by deck pairing / shuffle seed."""

from collections.abc import Mapping

import numpy as np


def compare_panel(scores: Mapping[str, np.ndarray], control: str, *, seed: int = 0, resamples: int = 10000) -> dict:
    """Arrays are [pairing, opponent, four games], in identical order for every candidate.

    NaN marks an error, never a draw. Remove the entire cluster from every row if any of its games failed.
    Resample clusters together across all opponents and candidates; opponents are a fixed, equally weighted panel.
    Intervals describe game sampling for these checkpoints, not variation between training runs.
    """
    if control not in scores or len(scores) < 2:
        raise ValueError("provide a control and at least one candidate")
    if resamples < 2:
        raise ValueError("at least two bootstrap resamples required")
    arrays = {k: np.asarray(v, dtype=float) for k, v in scores.items()}
    shape = arrays[control].shape
    if len(shape) != 3 or shape[1] < 1 or shape[2] != 4:
        raise ValueError("scores must have shape [pairings, opponents, 4]")
    valid = np.ones(shape[0], dtype=bool)
    for values in arrays.values():
        if values.shape != shape:
            raise ValueError("all candidates must share pairing / opponent / game axes")
        if np.any(np.isinf(values)) or np.any((values < 0) | (values > 1)):
            raise ValueError("scores must be in [0, 1], or NaN for errors")
        valid &= np.isfinite(values).all(axis=(1, 2))
    n = int(valid.sum())
    if n < 2:
        raise ValueError("at least two complete common pairings required")
    indices = np.random.default_rng(seed).integers(n, size=(resamples, n))
    paired = {k: v[valid].mean(axis=(1, 2)) for k, v in arrays.items()}
    results = {}
    for name, values in paired.items():
        diff = values - paired[control]
        ci = np.quantile(diff[indices].mean(axis=1), [0.025, 0.975])
        results[name] = {
            "score": float(values.mean()),
            "by_opponent": arrays[name][valid].mean(axis=(0, 2)).tolist(),
            "difference_vs_control": float(diff.mean()),
            "ci95": ci.tolist(),
            "error_games": int(np.isnan(arrays[name]).sum()),
        }
    return {
        "control": control,
        "pairings_total": shape[0],
        "pairings_used": n,
        "pairings_excluded": shape[0] - n,
        "games_per_candidate_used": n * shape[1] * 4,
        "bootstrap_seed": seed,
        "resamples": resamples,
        "interval_scope": "pointwise; fixed checkpoints and opponents; complete common pairing clusters",
        "candidates": results,
    }
