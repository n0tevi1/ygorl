"""Conservative card-exposure accounting for encoded policy datasets.

Count every identity-bearing column, including masked rows. Hold out whole
source hands/games, never individual decisions from an otherwise seen game.
This proves interaction-data exclusion, not absence from language pretraining.
"""

from collections import defaultdict

import numpy as np


def card_ids_per_row(obs):
    """Cards, declared/effect cards, and both event-card fields (vocab indices)."""
    return np.concatenate(
        [
            obs["cards"][..., 0],
            obs["actions"][..., 2],
            obs["actions"][..., 3],
            obs["events"][..., 2],
            obs["events"][..., 3],
        ],
        axis=1,
    )


def source_group(meta):
    if "deck" in meta and "hand_index" in meta:
        return ("opening", meta["deck"], int(meta["hand_index"]))
    if "game" in meta and "deck_a" in meta and "deck_b" in meta:
        return ("game", meta["deck_a"], meta["deck_b"], int(meta["game"]), int(meta["first"]))
    raise ValueError("missing source hand/game identity")


def reserved_action_rows(obs, reserved):
    """An offered legal action names or points to a reserved card (no label used)."""
    actions = obs["actions"]
    refs = actions[..., 1]
    card_ids = np.take_along_axis(obs["cards"][..., 0], (refs - 1).clip(0, obs["cards"].shape[1] - 1), axis=1)
    card_ids = np.where(refs > 0, card_ids, 0)
    hit = (
        np.isin(card_ids, list(reserved))
        | np.isin(actions[..., 2], list(reserved))
        | np.isin(actions[..., 3], list(reserved))
    )
    return (hit & obs["action_mask"].astype(bool)).any(axis=1)


def exclude_card_groups(obs, meta, reserved):
    """Return kept rows and excluded source groups; all variants share a hand."""
    ids = card_ids_per_row(obs)
    if len(ids) != len(meta):
        raise ValueError("observation/metadata row count differs")
    touched = np.isin(ids, list(reserved)).any(axis=1)
    groups = [source_group(m) for m in meta]
    excluded = {g for g, hit in zip(groups, touched, strict=True) if hit}
    keep = np.array([g not in excluded for g in groups], dtype=bool)
    return keep, excluded


def grouped_metrics(nll, correct, meta):
    """Retain one mean per independent hand/game for later paired intervals."""
    groups = defaultdict(list)
    for i, m in enumerate(meta):
        groups[source_group(m)].append(i)
    return [
        dict(
            group=list(k),
            rows=len(v),
            nll=float(np.asarray(nll)[v].mean()),
            accuracy=float(np.asarray(correct)[v].mean()),
        )
        for k, v in sorted(groups.items())
    ]
