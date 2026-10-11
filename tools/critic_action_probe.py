"""Research-only all-action view of a resource critic, not a PPO checkpoint adapter.

Inputs must come from the frozen reference actor used to fit the critic. A changed
behavior actor's embeddings are not interchangeable with that coordinate system.
"""

from __future__ import annotations

import torch

from critic_opponent_probe import opponent_features


def all_action_values(net, batch, *, ridge=None, opponents=None):
    """Return Q[B,A], V[B]; preserve zero Q on illegal/padded action rows.

    ``candidates`` and ``context`` are frozen reference features. Detach them so
    a critic loss cannot accidentally update the actor. A ridge correction is
    broadcast to legal Q rows only; it is clipped after adding raw head outputs.
    Known opponent labels are a closed diagnostic vocabulary, not an inference
    mechanism for new opponents. Without a ridge, reproduce the raw base head.
    """
    context = batch["context"].detach()
    candidates = batch["candidates"].detach()
    mask = batch["action_mask"]
    if candidates.ndim != 3 or mask.shape != candidates.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("one boolean legality mask per candidate is required")
    if context.shape[0] != candidates.shape[0]:
        raise ValueError("context and candidate batch sizes differ")
    resources = batch["resources"]
    c = net.model.critic
    public = resources if net.resources else torch.zeros_like(resources)
    x = torch.cat((context, net.model.privileged(batch["privileged"])), -1)
    hidden = c.trunk[0][0](x) + net.public(public)
    hidden = c.trunk[1](c.trunk[0][2](c.trunk[0][1](hidden)))
    q, v = c.q_head(hidden, candidates, mask), c.v_head(hidden)
    if ridge is not None:
        if opponents is None or len(opponents) != len(context):
            raise ValueError("one registered opponent per observation is required")
        game_ids = torch.arange(len(context), device=resources.device)
        features = opponent_features(resources, game_ids, opponents, conditioned=True).double()
        state = {k: ridge[k].to(device=features.device, dtype=features.dtype) for k in ["mean", "scale", "beta"]}
        design = torch.cat(((features - state["mean"]) / state["scale"], features.new_ones((len(features), 1))), -1)
        offset = design @ state["beta"]
        q = (q.double() + offset[:, 0, None]).clamp(-1, 1)
        v = (v.double() + offset[:, 1]).clamp(-1, 1)
    return q.masked_fill(~mask, 0), v
