"""Fixed corpus mixture weights for cached critic diagnostics."""

import math

import torch


def mixture_weights(game_ids, corpus_by_game, masses):
    """Give each corpus its registered mass and each eligible game equal mass.

    Inputs enumerate eligible games only; zero-idle outcomes remain in the
    caller's audit. More sampled turns must not give a game extra weight.
    """
    if game_ids.ndim != 1 or game_ids.dtype != torch.long or not len(game_ids):
        raise ValueError("nonempty int64 sample-to-game IDs required")
    if set(corpus_by_game) != set(masses):
        raise ValueError("registered corpus masses must match eligible corpora")
    if not all(math.isfinite(v) and v > 0 for v in masses.values()) or not math.isclose(sum(masses.values()), 1.0):
        raise ValueError("positive finite corpus masses must sum to one")
    if bool(((game_ids < 0) | (game_ids >= len(corpus_by_game))).any()):
        raise ValueError("game ID outside corpus metadata")
    counts = torch.bincount(game_ids, minlength=len(corpus_by_game))
    if bool((counts == 0).any()):
        raise ValueError("corpus metadata must enumerate eligible games only")
    games_per_corpus = {key: corpus_by_game.count(key) for key in masses}
    per_game = torch.tensor(
        [masses[key] / games_per_corpus[key] for key in corpus_by_game],
        dtype=torch.float64,
        device=game_ids.device,
    )
    return per_game[game_ids] / counts[game_ids]
