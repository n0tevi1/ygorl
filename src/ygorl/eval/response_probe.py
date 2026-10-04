"""Observation-only response features and a small frozen ridge reranker for diagnostic pilots."""

from dataclasses import dataclass

import numpy as np
import torch

from ygorl.nets.batch import collate


@torch.no_grad()
def public_features(actor, observations, device):
    """Uses exactly the policy observation; no privileged tensors, engine, or critic input."""
    batch = collate(observations, device)
    f = actor.features(batch)
    # State-action interactions in the existing public actor representation, plus the action representation.
    return torch.cat((f.actions * f.context[:, None, :], f.actions), dim=-1).float().cpu().numpy()


@dataclass
class ResponseRidge:
    scale: np.ndarray
    weight: np.ndarray

    @classmethod
    def fit(cls, features, targets, *, regularization=10.0):
        x, y = np.asarray(features, dtype=float), np.asarray(targets, dtype=float)
        if len(x) < 2 or x.ndim != 2 or y.shape != (len(x),) or regularization <= 0:
            raise ValueError("need at least two feature/target pairs and positive regularization")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("training features/targets must be finite")
        scale = np.maximum(np.sqrt(np.mean(x * x, axis=0)), 1e-6)
        z = x / scale
        weight = np.linalg.solve(z.T @ z + regularization * np.eye(z.shape[1]), z.T @ y)
        return cls(scale, weight)

    def scores(self, features):
        return np.asarray(features) / self.scale @ self.weight


def common_valid(scores):
    """A failed continuation is missing for every candidate, never a draw."""
    x = np.asarray(scores, dtype=float)
    if x.ndim != 2 or np.isinf(x).any() or ((x < 0) | (x > 1)).any():
        raise ValueError("scores must be [actions, seeds] in [0,1], or NaN for errors")
    return np.isfinite(x).all(axis=0)


def bootstrap_clusters(values, clusters, *, seed=0, resamples=10000):
    """Mean over equally sized four-game clusters (zeros included for games with no eligible response)."""
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    unique = np.unique(clusters)
    if values.shape != clusters.shape or len(unique) < 2 or not np.isfinite(values).all():
        raise ValueError("need finite values and at least two clusters")
    counts = np.array([(clusters == c).sum() for c in unique])
    if (counts != counts[0]).any():
        raise ValueError("clusters must have equal sizes")
    means = np.array([values[clusters == c].mean() for c in unique])
    ids = np.random.default_rng(seed).integers(len(unique), size=(resamples, len(unique)))
    ci = np.quantile(means[ids].mean(axis=1), [0.025, 0.975])
    return {"difference": float(means.mean()), "ci95": ci.tolist(), "clusters": len(unique)}


def response_window(obs):
    """Return legal action indices and the pass action for the deployed response window, or None."""
    from ygorl.engine import constants as C
    from ygorl.env.encoding import ACTION_KINDS

    g = obs["globals"]
    if int(g[3]) > 4 or int(g[2]) != 0 or int(g[18]) != C.MSG_SELECT_CHAIN:
        return None
    legal = np.flatnonzero(obs["action_mask"])
    passes = [int(k) for k in legal if obs["actions"][k, 0] == ACTION_KINDS.index("pass") + 1]
    if not passes or not 1 < len(legal) <= 8:
        return None
    return legal, passes[0]


def simple_response(probabilities, legal, passed, *, pass_bias=None):
    """A deterministic logit offset on pass, or always activate the actor's preferred non-pass action."""
    legal = np.asarray(legal, dtype=int)
    if passed not in legal or len(legal) < 2:
        raise ValueError("need pass and another legal action")
    if pass_bias is None:
        choices = legal[legal != passed]
        return int(choices[np.argmax(np.asarray(probabilities)[choices])])
    scores = np.log(np.maximum(np.asarray(probabilities)[legal], 1e-30))
    scores[legal == passed] += pass_bias
    return int(legal[np.argmax(scores)])


def response_contrasts(scores, *, seed=0, resamples=10000):
    """Five prespecified paired contrasts, common complete clusters, Bonferroni familywise 95% intervals."""
    names = [(n, "control") for n in ("reranked", "pass_bias", "always_activate")]
    names += [("reranked", n) for n in ("pass_bias", "always_activate")]
    values = {k: np.asarray(v, dtype=float) for k, v in scores.items()}
    shape = values["control"].shape
    if any(v.shape != shape or np.isinf(v).any() or ((v < 0) | (v > 1)).any() for v in values.values()):
        raise ValueError("incompatible score panels")
    valid = np.logical_and.reduce([np.isfinite(v).all(axis=(1, 2)) for v in values.values()])
    n = int(valid.sum())
    if n < 2:
        raise ValueError("need two complete pairing clusters")
    ids = np.random.default_rng(seed).integers(n, size=(resamples, n))
    result = {}
    for a, b in names:
        diff = (values[a][valid] - values[b][valid]).mean(axis=(1, 2))
        ci = np.quantile(diff[ids].mean(axis=1), [0.005, 0.995])
        result[f"{a}-{b}"] = {"difference": float(diff.mean()), "ci99": ci.tolist(), "clusters": n}
    return {"family": "5 prespecified contrasts; Bonferroni 95% familywise", "contrasts": result}
