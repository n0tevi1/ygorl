"""Research-only full declaration scoring; no change to PPO's fixed-width storage.

Uses only the engine's public legal action list. Row order remains engine order,
including unknown-vocabulary cards. Normalize once over the returned full logits.
"""

import numpy as np
import torch

from ygorl.env.encoding import A_ACTION, ACTION_KINDS


def declaration_rows(actions, vocab):
    """Encode an AnnounceCard menu without the legacy 128-row truncation."""
    if not actions or any(a.kind != "declare" or a.card is not None or a.description or a.index != -1 for a in actions):
        raise ValueError("expected a nonempty plain AnnounceCard menu")
    rows = np.zeros((len(actions), A_ACTION), dtype=np.int32)
    rows[:, 0] = ACTION_KINDS.index("declare") + 1
    rows[:, 2] = [vocab.index(a.value) for a in actions]
    return rows


@torch.no_grad()
def chunked_logits(net, features, rows, *, chunk_size=128, belief=None):
    """Score [B,A,10] candidate rows against one previously computed public context.

    Inference only: callers retain the original context/global action count. Full
    logits need one global softmax; separate chunk softmaxes would change policy.
    This utility does not alter encoded environments, collectors or PPO replay.
    """
    if net.training:
        raise ValueError("chunked probe requires eval mode")
    if (
        chunk_size < 1
        or rows.ndim != 3
        or rows.shape[0] != features.context.shape[0]
        or rows.shape[1] == 0
        or rows.shape[2] != A_ACTION
    ):
        raise ValueError("expected nonempty [B,A,10] rows and positive chunk size")
    logits = []
    for part in rows.split(chunk_size, dim=1):
        candidates = net.action_encoder(part, features.cards)
        mask = torch.ones(part.shape[:2], dtype=torch.bool, device=part.device)
        logits.append(net.head(features.context, candidates, mask, belief))
    return torch.cat(logits, dim=1)
