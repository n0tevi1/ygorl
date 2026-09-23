"""PPO update for self-play (T4b.4, design I1 / I2 / I8): clipped surrogate, entropy bonus, KL to a slow
reference (and optionally a BC prior), Q / V critic losses, VRPO or GAE advantages.

Per update (docs/training.md §8):

1. Targets once per rollout with :func:`ygorl.train.advantages.estimate` (VRPO by default, GAE switch),
   limit-cut games bootstrapped from the critic (``Rollout.truncated``).
2. The reference policy (EMA of the learner, design I8's "slow reference", MMD / R-NaD style) and the
   optional BC prior score every row once, without gradient.
3. ``epochs`` passes over shuffled minibatches of ``minibatch_size`` rows:
   ``loss = objective + kl_ref_coef KL(π‖π_ref) + kl_prior_coef KL(π‖π_prior) - entropy_coef H(π)
   + q_coef L_Q + v_coef L_V``, gradients clipped to ``max_grad_norm``. With ``kl_prior_turns > 0`` the prior
   KL is kept on the turn player's decisions up to that turn only (the states the BC data covers), the other rows
   count as 0.
4. ``reference <- (1 - reference_ema) reference + reference_ema θ``.

The policy objective is pluggable (:func:`register_objective`): ``"ppo_clip"`` (default) is the clipped
surrogate, ``"pg"`` the plain on-policy policy gradient. An objective sees the rollout once
(:meth:`PolicyObjective.prepare`, e.g. to regroup rows by start state for a MaxRL-style success objective,
docs/eng-plan.md "M4 待议的消融") and each minibatch (:meth:`PolicyObjective.loss`); the collector, the
league and the critic losses do not change.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields

import torch
from torch import Tensor, nn

from ygorl.nets.policy import trim_padding
from ygorl.train.advantages import ESTIMATORS, NORMALIZE_MODES, VRPO_MODES, Estimate, estimate
from ygorl.train.critic import q_loss, v_loss
from ygorl.train.rollout import Rollout, _logits_of


@dataclass(frozen=True)
class PPOConfig:
    objective: str = "ppo_clip"  # policy objective (register_objective); "ppo_clip" or "pg"
    estimator: str = "vrpo"  # "vrpo" (design I2) or "gae" (control)
    vrpo_mode: str = "return"
    gamma: float = 1.0
    lam: float = 0.95
    clip: float = 0.2  # PPO ratio clip
    entropy_coef: float = 0.05  # design I1: 0.05-0.2
    kl_ref_coef: float = 0.05  # KL(π‖π_ref) to the EMA reference (design I8)
    reference_ema: float = 0.02  # reference <- (1 - τ) reference + τ θ after every update
    kl_prior_coef: float = 0.0  # KL(π‖π_prior) to a BC prior checkpoint, when one is given (design I4)
    # > 0: the prior KL only on the turn player's own decisions up to this turn (globals is_my_turn / turn), the
    # states the turn-1 solver demonstrations cover (docs/bc.md「补救实验」); other rows count as 0; 0 = every row
    kl_prior_turns: int = 0
    # stop an update's remaining minibatches once one exceeds 1.5 x target_kl (approx_kl); None = every epoch
    target_kl: float | None = None
    q_coef: float = 0.5
    v_coef: float = 0.5
    lr: float = 1e-3
    adam_eps: float = 1e-5
    max_grad_norm: float = 0.5
    epochs: int = 4  # 4 x 256-row minibatches of 2,048 rows: approx_kl ~5e-3 per update (docs/benchmarks.md)
    minibatch_size: int = 256
    adv_norm: str = "standard"  # normalize_advantages mode, over the whole rollout

    def __post_init__(self) -> None:
        if self.estimator not in ESTIMATORS:
            raise ValueError(f"estimator must be one of {ESTIMATORS}, got {self.estimator!r}")
        if self.vrpo_mode not in VRPO_MODES:
            raise ValueError(f"vrpo_mode must be one of {VRPO_MODES}, got {self.vrpo_mode!r}")
        if self.adv_norm not in NORMALIZE_MODES:
            raise ValueError(f"adv_norm must be one of {NORMALIZE_MODES}, got {self.adv_norm!r}")
        if self.objective not in _OBJECTIVES:
            raise ValueError(f"unknown objective {self.objective!r}; registered: {', '.join(sorted(_OBJECTIVES))}")
        if not 0 <= self.reference_ema <= 1:
            raise ValueError("reference_ema must be in [0, 1]")
        if self.epochs < 1 or self.minibatch_size < 1:
            raise ValueError("epochs and minibatch_size must be at least 1")
        if self.kl_prior_turns < 0:
            raise ValueError("kl_prior_turns must be >= 0 (0 = every row)")
        if self.target_kl is not None and not self.target_kl > 0:
            raise ValueError("target_kl must be positive (or None)")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> PPOConfig:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


# ------------------------------------------------------------------------------------------ objectives


@dataclass
class ObjectiveInputs:
    """One minibatch, flattened rows ``[n]`` (``[n, A]`` per candidate)."""

    log_probs: Tensor  # [n, A] current policy (with gradient)
    actions: Tensor  # [n]
    old_log_probs: Tensor  # [n] behaviour policy at acting time
    advantages: Tensor  # [n] from PolicyObjective.prepare
    returns: Tensor  # [n] Q-head targets (Expected-SARSA / λ-returns)
    action_mask: Tensor  # [n, A]
    index: Tensor  # [n] row index into the flattened rollout (t * B + b), for objectives that group rows


class PolicyObjective:
    """Policy-loss plug-in. Subclass, then ``register_objective(name, factory)``; ``factory(cfg)`` builds it."""

    def prepare(self, rollout: Rollout, est: Estimate) -> Tensor:
        """Per-row weights ``[T, B]`` for the policy gradient; default: the estimator's advantages."""
        return est.advantages

    def loss(self, x: ObjectiveInputs) -> tuple[Tensor, dict[str, Tensor]]:
        raise NotImplementedError


class ClippedSurrogate(PolicyObjective):
    """PPO: ``-mean min(ρ A, clip(ρ, 1 ± ε) A)`` with ``ρ = π(a|s) / π_old(a|s)``."""

    def __init__(self, clip: float = 0.2) -> None:
        self.clip = clip

    def loss(self, x: ObjectiveInputs) -> tuple[Tensor, dict[str, Tensor]]:
        new = x.log_probs.gather(1, x.actions.unsqueeze(1)).squeeze(1)
        log_ratio = new - x.old_log_probs
        ratio = log_ratio.exp()
        surrogate = torch.minimum(ratio * x.advantages, ratio.clamp(1 - self.clip, 1 + self.clip) * x.advantages)
        with torch.no_grad():
            stats = {"approx_kl": ((ratio - 1) - log_ratio).mean(), "clip_frac": ((ratio - 1).abs() > self.clip).float().mean()}
        return -surrogate.mean(), stats


class PolicyGradient(PolicyObjective):
    """Plain on-policy policy gradient ``-mean A log π(a|s)`` (no ratio, no clipping); an ablation baseline."""

    def loss(self, x: ObjectiveInputs) -> tuple[Tensor, dict[str, Tensor]]:
        new = x.log_probs.gather(1, x.actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            log_ratio = new - x.old_log_probs
            stats = {"approx_kl": ((log_ratio.exp() - 1) - log_ratio).mean(), "clip_frac": torch.zeros(())}
        return -(x.advantages * new).mean(), stats


_OBJECTIVES: dict[str, Callable[[PPOConfig], PolicyObjective]] = {}


def register_objective(name: str, factory: Callable[[PPOConfig], PolicyObjective] | None) -> None:
    """Register a policy objective under ``name`` (``factory(cfg) -> PolicyObjective``); None removes it."""
    if factory is None:
        _OBJECTIVES.pop(name, None)
    elif name in _OBJECTIVES:
        raise ValueError(f"objective {name!r} is already registered")
    else:
        _OBJECTIVES[name] = factory


def available_objectives() -> list[str]:
    return sorted(_OBJECTIVES)


register_objective("ppo_clip", lambda cfg: ClippedSurrogate(cfg.clip))
register_objective("pg", lambda cfg: PolicyGradient())


# ------------------------------------------------------------------------------------------ learner


def _index(x, idx: Tensor):
    if x is None:
        return None
    if isinstance(x, Tensor):
        return x[idx]
    return {k: v[idx] for k, v in x.items()}


def _masked_mean_rows(x: Tensor, mask: Tensor) -> Tensor:
    return torch.where(mask, x, torch.zeros((), dtype=x.dtype, device=x.device)).sum(-1)


def _advantage_by_kind(obs, actions: Tensor, adv: Tensor) -> dict[str, float]:
    """Diagnostics: mean advantage (``adv/<kind>``) and row count (``rows/<kind>``) per kind of the chosen action.

    Needs the encoded action table (``obs["actions"]`` column 0 is the kind + 1, docs/encoding.md); {} otherwise.
    """
    if not isinstance(obs, dict) or "actions" not in obs:
        return {}
    from ygorl.env.encoding import ACTION_KINDS

    rows = torch.arange(actions.shape[0])
    kind = obs["actions"][rows, actions, 0].long() - 1
    out: dict[str, float] = {}
    for k in torch.unique(kind).tolist():
        if 0 <= k < len(ACTION_KINDS):
            sel = kind == k
            out[f"adv/{ACTION_KINDS[k]}"] = float(adv[sel].mean())
            out[f"rows/{ACTION_KINDS[k]}"] = int(sel.sum())
    return out


class PPOLearner:
    """Owns the model's optimizer, the EMA reference policy and the optional BC prior."""

    def __init__(self, model: nn.Module, cfg: PPOConfig, prior: nn.Module | None = None) -> None:
        self.model = model
        self.cfg = cfg
        self.objective = _OBJECTIVES[cfg.objective](cfg)
        self.reference = self._frozen(model)
        self.prior = None if prior is None else self._frozen(prior)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=cfg.adam_eps)
        self.updates = 0

    @staticmethod
    def _frozen(module: nn.Module) -> nn.Module:
        m = copy.deepcopy(module)
        m.eval()
        m.requires_grad_(False)
        return m

    # -- KL / entropy on masked logits ------------------------------------------------------------
    @staticmethod
    def kl(logits: Tensor, ref_logits: Tensor, mask: Tensor, rows: Tensor | None = None) -> Tensor:
        """Mean over rows of ``KL(π‖π_ref) = Σ_a π (log π - log π_ref)`` over legal candidates.

        ``rows`` (bool ``[n]``): the other rows count as 0, so a selected row weighs as much as without ``rows``.
        """
        logp, ref = torch.log_softmax(logits, -1), torch.log_softmax(ref_logits, -1)
        per_row = _masked_mean_rows(logp.exp() * (logp - ref), mask)
        if rows is not None:
            per_row = torch.where(rows, per_row, torch.zeros((), dtype=per_row.dtype))
        return per_row.mean()

    @staticmethod
    def prior_rows(obs, turns: int) -> Tensor | None:
        """Rows the prior KL applies to: every row (None) for ``turns == 0``, else the turn player's own decisions
        up to turn ``turns`` (``globals`` column 2 ``is_my_turn`` and column 3 ``turn``, docs/encoding.md)."""
        if turns <= 0:
            return None
        if not isinstance(obs, dict) or "globals" not in obs:
            raise ValueError("kl_prior_turns needs observations with globals (is_my_turn, turn)")
        g = obs["globals"]
        return (g[:, 2] == 1) & (g[:, 3] <= turns)

    @staticmethod
    def entropy(logits: Tensor, mask: Tensor) -> Tensor:
        logp = torch.log_softmax(logits, -1)
        return -_masked_mean_rows(logp.exp() * logp, mask).mean()

    @torch.no_grad()
    def _score(self, module: nn.Module, obs, n: int) -> Tensor:
        chunk = max(1, self.cfg.minibatch_size)
        out = [_logits_of(module, _index(obs, torch.arange(i, min(i + chunk, n)))).float() for i in range(0, n, chunk)]
        return torch.cat(out)

    # -- update --------------------------------------------------------------------------------------
    def targets(self, ro: Rollout) -> Estimate:
        cfg = self.cfg
        return estimate(cfg.estimator, rewards=ro.rewards, dones=ro.dones, players=ro.players, actions=ro.actions,
                        action_mask=ro.action_mask, probs=ro.probs, q=ro.q, values=ro.values, gamma=cfg.gamma,
                        lam=cfg.lam, bootstrap_player=ro.bootstrap_player, bootstrap_value=ro.bootstrap_value,
                        bootstrap_q=ro.bootstrap_q, bootstrap_probs=ro.bootstrap_probs,
                        bootstrap_mask=ro.bootstrap_mask, truncated=ro.truncated, vrpo_mode=cfg.vrpo_mode,
                        normalize=cfg.adv_norm)  # fmt: skip

    def update(self, ro: Rollout) -> dict[str, float]:
        cfg, model = self.cfg, self.model
        est = self.targets(ro)
        adv = self.objective.prepare(ro, est).reshape(-1).float()
        q_targets = est.q_targets.reshape(-1).float()
        v_targets = est.v_targets.reshape(-1).float()
        actions = ro.actions.reshape(-1)
        old_logp = ro.log_probs.reshape(-1)
        # Cut the rollout's padding once (nets.policy.trim_padding): every minibatch then gathers from the smaller
        # tensors, and logits / masks only span the rollout's longest action list (same values, see tests).
        obs = trim_padding(ro.obs)
        mask = ro.action_mask.reshape(actions.shape[0], -1)[:, : obs["action_mask"].shape[1]]
        n = actions.shape[0]
        ref_logits = self._score(self.reference, obs, n) if cfg.kl_ref_coef > 0 else None
        prior_logits = self._score(self.prior, obs, n) if self.prior is not None and cfg.kl_prior_coef > 0 else None
        prior_rows = self.prior_rows(obs, cfg.kl_prior_turns) if prior_logits is not None else None

        model.train()
        sums: dict[str, float] = defaultdict(float)
        steps = 0
        stopped = False
        for _ in range(cfg.epochs):
            if stopped:
                break
            perm = torch.randperm(n)
            for start in range(0, n, cfg.minibatch_size):
                idx = perm[start : start + cfg.minibatch_size]
                out = model(_index(obs, idx), _index(ro.privileged, idx))
                logits = out.logits.float()
                m = mask[idx]
                logp = torch.log_softmax(logits, -1)
                policy_loss, extra = self.objective.loss(ObjectiveInputs(logp, actions[idx], old_logp[idx], adv[idx],
                                                                         q_targets[idx], m, idx))  # fmt: skip
                ent = self.entropy(logits, m)
                loss = policy_loss - cfg.entropy_coef * ent
                kl_ref = self.kl(logits, ref_logits[idx], m) if ref_logits is not None else torch.zeros(())
                kl_prior = (self.kl(logits, prior_logits[idx], m, None if prior_rows is None else prior_rows[idx])
                            if prior_logits is not None else torch.zeros(()))  # fmt: skip
                loss = loss + cfg.kl_ref_coef * kl_ref + cfg.kl_prior_coef * kl_prior
                ql = q_loss(out.q.float(), actions[idx], q_targets[idx])
                vl = v_loss(out.v.float(), v_targets[idx])
                loss = loss + cfg.q_coef * ql + cfg.v_coef * vl
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                for key, val in (("loss", loss), ("policy_loss", policy_loss), ("entropy", ent), ("kl_ref", kl_ref),
                                 ("kl_prior", kl_prior), ("q_loss", ql), ("v_loss", vl), ("grad_norm", grad_norm),
                                 *extra.items()):  # fmt: skip
                    sums[key] += float(val.detach()) if isinstance(val, Tensor) else float(val)
                steps += 1
                kl_now = extra.get("approx_kl")
                if cfg.target_kl is not None and kl_now is not None and float(kl_now) > 1.5 * cfg.target_kl:
                    stopped = True  # the policy moved enough for this batch of data
                    break
        self.updates += 1
        self.update_reference()
        stats = {k: v / steps for k, v in sums.items()}
        taken_q = ro.q.gather(-1, ro.actions.unsqueeze(-1)).squeeze(-1).reshape(-1)
        stats["q_explained_var"] = _explained_variance(taken_q, q_targets)
        stats["adv_mean"], stats["adv_std"] = float(adv.mean()), float(adv.std(correction=0))
        stats["rows"] = n
        stats["minibatches"], stats["early_stop"] = steps, int(stopped)
        stats.update(_advantage_by_kind(obs, actions, adv))
        if prior_rows is not None:
            stats["kl_prior_rows"] = int(prior_rows.sum())
        return stats

    @torch.no_grad()
    def update_reference(self) -> None:
        tau = self.cfg.reference_ema
        ref = self.reference.state_dict()
        for name, value in self.model.state_dict().items():
            if value.dtype.is_floating_point:
                ref[name].mul_(1 - tau).add_(value.detach(), alpha=tau)
            else:
                ref[name].copy_(value)

    # -- persistence --------------------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {"model": self.model.state_dict(), "reference": self.reference.state_dict(),
                "optimizer": self.optimizer.state_dict(), "updates": self.updates}  # fmt: skip

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state["model"])
        self.reference.load_state_dict(state["reference"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.updates = int(state["updates"])


def _explained_variance(pred: Tensor, target: Tensor) -> float:
    var = float(target.var(correction=0))
    return float("nan") if var == 0 else 1.0 - float((target - pred.to(target)).var(correction=0)) / var


__all__ = ["ClippedSurrogate", "ObjectiveInputs", "PPOConfig", "PPOLearner", "PolicyGradient", "PolicyObjective",
           "available_objectives", "register_objective"]  # fmt: skip
