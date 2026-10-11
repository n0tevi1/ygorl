"""Known-opponent feature for a private-critic diagnostic, never actor inference."""

import torch


def opponent_features(resources, game_ids, opponents, *, conditioned):
    """Append a game-aligned indicator; reject unknown policies instead of guessing.

    Unconditioned controls return resources without reading opponent metadata.
    The two registered policies are a closed diagnostic vocabulary, not an
    embedding or a generalization mechanism for unseen opponents.
    """
    if not conditioned:
        return resources
    labels = {"old-256x2": -1.0, "mirror-u528": 1.0}
    unknown = set(opponents) - labels.keys()
    if unknown:
        raise ValueError(f"unknown diagnostic opponent: {sorted(unknown)}")
    if game_ids.ndim != 1 or len(game_ids) != len(resources) or game_ids.dtype != torch.long:
        raise ValueError("one int64 game ID is required per resource row")
    if bool(((game_ids < 0) | (game_ids >= len(opponents))).any()):
        raise ValueError("game ID outside opponent metadata")
    values = resources.new_tensor([labels[name] for name in opponents])
    return torch.cat((resources, values[game_ids.to(resources.device), None]), dim=-1)
