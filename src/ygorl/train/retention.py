"""Opt-in native-deal coverage and frozen-policy distillation; never PPO replay."""

from collections import Counter
import math

import numpy as np
import torch

from ygorl.engine import constants as C
from ygorl.nets.batch import policy_logits
from ygorl.train.selfplay import DeckPool


MECHANISMS = {
    "ritual": C.TYPE_RITUAL,
    "fusion": C.TYPE_FUSION,
    "synchro": C.TYPE_SYNCHRO,
    "xyz": C.TYPE_XYZ,
    "link": C.TYPE_LINK,
    "pendulum": C.TYPE_PENDULUM,
    "quickplay": C.TYPE_QUICKPLAY,
    "trap": C.TYPE_TRAP,
}


class MechanismDeckPool(DeckPool):
    """50% uniform pairs + 50% uniform nonempty mechanism then uniform matching pair.

    All original pairs retain positive support. Deck membership is an opportunity
    proxy, not proof of encountering or learning a mechanism. No actions change.
    """

    def __init__(self, decks, cards, pairings="cross", share=0.5):
        super().__init__(decks, pairings)
        if not 0 <= share <= 1:
            raise ValueError("mechanism share must be in [0, 1]")
        self.share = share
        tags = [
            {k for k, flag in MECHANISMS.items() if any(cards[p].type & flag for p in (*d.main, *d.extra))}
            for d in decks
        ]
        self.groups = {k: [n for n, (i, j) in enumerate(self.pairs) if k in tags[i] | tags[j]] for k in MECHANISMS}
        self.groups = {k: v for k, v in self.groups.items() if v}
        if not self.groups:
            raise ValueError("no registered mechanisms in deck pool")
        self.probabilities = np.full(len(self.pairs), (1 - share) / len(self.pairs))
        for indices in self.groups.values():
            self.probabilities[indices] += share / len(self.groups) / len(indices)
        self.draws = Counter()

    def sample(self, rng):
        if self.share == 0:  # exact baseline RNG path
            return super().sample(rng)
        n = int(rng.choice(len(self.pairs), p=self.probabilities))
        i, j = self.pairs[n]
        self.draws[f"{i}:{j}"] += 1
        return self.decks[i], self.decks[j]


def prompt_groups(obs):
    """Group BEFORE action choice by prompt and phase, never reward or chosen action."""
    g = obs["globals"]
    # globals: phase at 4, prompt message at 18 (docs/encoding.md).
    return g[:, 18].long() * 65536 + g[:, 4].long()


class FrozenReplay:
    """Owned, detached observations and teacher probabilities; independent CPU RNG.

    Samples groups uniformly, then rows uniformly within a group. The fixed bank
    must be environment/schema-bound by its owner, as a rollout's tensors alone
    cannot prove provenance. Its RNG state can be stored alongside that binding.
    """

    def __init__(self, obs, probabilities, groups, *, seed=0):
        n = len(probabilities)
        mask = obs["action_mask"].detach().cpu().bool()
        p = probabilities.detach().cpu().float()
        groups = groups.detach().cpu()
        if not n or p.ndim != 2 or p.shape != mask.shape or groups.shape != (n,):
            raise ValueError("replay shapes differ or bank is empty")
        if any(len(v) != n for v in obs.values()):
            raise ValueError("replay observation row count differs")
        if not torch.isfinite(p).all() or (p < 0).any() or (p[~mask] != 0).any():
            raise ValueError("invalid or illegal teacher probabilities")
        if not mask.any(-1).all() or not torch.allclose(p.sum(-1), torch.ones(n), atol=1e-5):
            raise ValueError("empty mask or unnormalized teacher probabilities")
        self.obs = {k: v.detach().cpu().clone() for k, v in obs.items()}
        self.probabilities = p.clone()
        self.groups = groups.clone()
        self.indices = [torch.where(groups == g)[0] for g in torch.unique(groups)]
        self.generator = torch.Generator().manual_seed(seed)
        self.sampled = 0

    def sample(self, count):
        if count < 1:
            raise ValueError("positive replay count required")
        group = torch.randint(len(self.indices), (count,), generator=self.generator)
        indices = torch.empty(count, dtype=torch.long)
        for g in group.unique().tolist():
            positions = torch.where(group == g)[0]
            rows = self.indices[g]
            indices[positions] = rows[torch.randint(len(rows), (len(positions),), generator=self.generator)]
        self.sampled += count
        return indices

    def loss(self, model, count):
        idx = self.sample(count)
        dev = next(model.parameters()).device
        obs = {k: v[idx].to(dev) for k, v in self.obs.items()}
        logits = policy_logits(model, obs).float()
        mask = obs["action_mask"].bool()
        p = self.probabilities[idx].to(dev)
        # Remask explicitly: teacher/student must share the same legal support.
        logp = torch.log_softmax(logits.masked_fill(~mask, -torch.inf), -1)
        delta = p.clamp_min(torch.finfo(p.dtype).tiny).log() - logp.masked_fill(~mask, 0)
        return (p * delta).sum(-1).mean()

    def state_dict(self):
        return {"rng": self.generator.get_state(), "sampled": self.sampled}

    def load_state_dict(self, state):
        if not math.isfinite(state["sampled"]) or state["sampled"] < 0:
            raise ValueError("invalid replay sample count")
        self.generator.set_state(state["rng"])
        self.sampled = int(state["sampled"])
