"""Known-opponent resource interactions for a frozen critic diagnostic."""

import torch

from critic_opponent_probe import opponent_features


def interaction_features(resources, game_ids, opponents):
    """Allow each registered opponent its own linear resource slopes.

    This reuses the closed-vocabulary alignment checks. It adds capacity, not
    unseen-opponent inference, and leaves fitting/normalization to training.
    """
    shared = opponent_features(resources, game_ids, opponents, conditioned=True)
    return torch.cat((shared, resources * shared[:, -1:]), dim=-1)
