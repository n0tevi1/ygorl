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
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields

import torch
from torch import Tensor, nn

from ygorl.nets.batch import policy_logits
from ygorl.nets.policy import trim_padding
from ygorl.train.advantages import (
    ESTIMATORS,
    NORMALIZE_MODES,
    VRPO_MODES,
    Estimate,
    estimate,
    normalize_advantages,
    terminal_returns,
)
from ygorl.train.aux import aux_targets, phi_shaping, turn_boundaries
from ygorl.train.critic import aux_loss, load_allowing_new_heads, phi_loss, q_loss, v_loss
from ygorl.train.rollout import Rollout


BRANCH_MODES = ("off", "replace", "add")
BRANCH_CE_WEIGHTS = ("delta", "uniform")


@torch.no_grad()
def branch_advantages(ro: Rollout) -> tuple[Tensor, Tensor, Tensor]:
    """The window counterfactual advantage of each branched row (first-seat plan §5.1 A, a COMA-style baseline):
    ``(adv, used, in_set)`` -- ``adv`` [T, B] is ``V_b(a_taken) - Σ_{a in set} π̃(a) V_b(a)`` with π̃ the behaviour
    probabilities renormalized on the branch set (0 off ``used``), ``used`` [T, B] the valid branched rows whose taken
    action is in the set, ``in_set`` [T, B, A] the set's candidates of the valid rows."""
    value, valid = ro.branch_value, ro.branch_valid
    in_set = torch.isfinite(value) & valid.unsqueeze(-1)
    used = in_set.gather(-1, ro.actions.unsqueeze(-1)).squeeze(-1)
    v = torch.where(in_set, value, torch.zeros((), dtype=value.dtype))
    pi = torch.where(in_set, ro.probs.to(value), torch.zeros((), dtype=value.dtype))
    baseline = (pi * v).sum(-1) / pi.sum(-1).clamp(min=1e-12)
    taken = v.gather(-1, ro.actions.unsqueeze(-1)).squeeze(-1)
    return torch.where(used, taken - baseline, torch.zeros((), dtype=value.dtype)), used, in_set


@torch.no_grad()
def branch_ce_targets(ro: Rollout, margin: float, weight: str = "delta") -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """The supervised targets of the window counterfactuals (first-seat plan §11.8 items 2-3, ``branch_ce_coef``):
    ``(target, weight, gap, tie)``. On each valid branched row with two or more candidates, ``gap`` [T, B] is the best
    candidate's branch value (the mean over the rollouts) minus the second best; the row is a tie when
    ``gap <= margin`` (value units) and gets no target. ``target`` [T, B] is the best candidate (-1: none),
    ``weight`` [T, B] is ``gap`` over its mean on the rollout's targeted rows (``"delta"``) or 1 (``"uniform"``), 0
    off the targets; ``tie`` [T, B] marks the tied rows."""
    value, valid = ro.branch_value, ro.branch_valid
    in_set = torch.isfinite(value) & valid.unsqueeze(-1)
    two = in_set.sum(-1) >= 2
    zero = torch.zeros((), dtype=value.dtype)
    v = torch.where(in_set, value, torch.full((), -float("inf"), dtype=value.dtype))
    if v.shape[-1] < 2:
        return torch.full_like(valid, -1, dtype=torch.long), zero.expand(valid.shape), zero.expand(valid.shape), two
    top = v.topk(2, dim=-1)
    gap = torch.where(two, top.values[..., 0] - top.values[..., 1], zero)
    decided = two & (gap > margin)
    target = torch.where(decided, top.indices[..., 0], torch.full((), -1, dtype=torch.long))
    if weight == "uniform":
        w = decided.to(value.dtype)
    else:
        mean = gap[decided].mean() if decided.any() else torch.ones((), dtype=value.dtype)
        w = torch.where(decided, gap / mean, zero)
    return target, w, gap, two & ~decided


@dataclass(frozen=True)
class PPOConfig:
    objective: str = "ppo_clip"  # policy objective (register_objective); "ppo_clip" or "pg"
    estimator: str = "vrpo"  # "vrpo" (design I2) or "gae" (control)
    vrpo_mode: str = "return"
    gamma: float = 1.0
    lam: float = 0.5  # design I2: lower-variance advantages; 0.3 / the critic mode are worse (docs/benchmarks.md)
    critic_lam: float | None = None  # Q / V target λ (None: lam); 1.0 still retains VRPO control variates/bootstrap
    critic_target: str = "lambda"  # experimental "terminal": actual z for Q/V, same policy advantages
    clip: float = 0.2  # PPO ratio clip
    entropy_coef: float = 0.05  # design I1: 0.05-0.2
    kl_ref_coef: float = 0.05  # KL(π‖π_ref) to the EMA reference (design I8)
    reference_ema: float = 0.02  # reference <- (1 - τ) reference + τ θ after every update
    kl_prior_coef: float = 0.0  # KL(π‖π_prior) to a BC prior checkpoint, when one is given (design I4)
    # > 0: the prior KL only on the turn player's own decisions up to this turn (globals is_my_turn / turn), the
    # states the turn-1 solver demonstrations cover (docs/bc.md「补救实验」); other rows count as 0; 0 = every row
    kl_prior_turns: int = 0
    # Reject this and remaining minibatches before Adam when measured approx_kl exceeds 1.5 x target_kl.
    # An accepted step can still overshoot: this is an early-stop heuristic, not a hard KL bound. None = off.
    # From a BC warm start the fixed 32 steps overshoot (docs/benchmarks.md 「BC 热启动 + PPO」); from scratch it
    # rarely triggers (approx_kl ~0.006-0.009 per update)
    target_kl: float | None = 0.01
    q_coef: float = 0.5
    v_coef: float = 0.5
    lr: float = 1e-3
    adam_eps: float = 1e-5
    max_grad_norm: float = 0.5
    epochs: int = 2  # over the 16,384-row batch in 2,048-row minibatches (target_kl may stop earlier); design I1: 2 = 4 in strength
    minibatch_size: int = 2048  # 8 minibatches per epoch of the 16,384-row default batch (design I1)
    adv_norm: str = "standard"  # normalize_advantages mode, over the whole rollout
    # Advantage filtering (MirrorForce's "kept rows"): only rows whose |A| is at or above this quantile of the
    # rollout's |A| enter the update (policy and critic losses alike); the rest are forced-like decisions whose
    # gradient is noise. 0 = every row. The KL guard then measures approx_kl on the kept rows only.
    adv_keep: float = 0.0
    # "rows": only the kept rows enter the update at all (cheapest); "policy": every row is forwarded, the critic,
    # entropy and KL terms see all rows, only the policy loss is restricted to the kept rows (their advantages are
    # kept, the others zeroed, the surrogate rescaled to the kept count)
    adv_keep_mode: str = "rows"
    # Policy-gradient weight multiplier on rows decided during the opponent's turn (globals is_my_turn == 0: chain
    # windows, responses, selections inside the opponent's effects). 1 = unchanged. Diagnostic for whether those
    # decisions lack gradient rather than signal (docs/spikes/rl-methodology-review-2026-10-20.md §7.2)
    response_weight: float = 1.0
    # multiply the policy-gradient weight of rows decided by the seat that moved first (engine player 0; in self-play
    # both seats are the learner): a curriculum on the harder seat (#442: the first seat wins 8-12 pp less); 1 = off
    first_seat_weight: float = 1.0
    aux_coef: float = 0.0  # weight of the critic's auxiliary-head loss (#440); needs a model built with aux heads
    # solver hand-trap labels on scripted-line games (train.line_labels, TrainConfig.line_labels): at a labelled chain
    # window with an Ash candidate, line_bonus_coef * (u_k - max later u) is added to the advantage of the taken
    # action (the sign flipped when it held) before the advantages are normalized; 0 = off (rollouts unchanged)
    line_bonus_coef: float = 0.0
    # the same labels as a supervised term (LineLabels.ce_targets): at each labelled window with an Ash candidate the
    # target is the activation when u_k - max later u > 0, the passive answer otherwise; line_ce_coef * the mean over
    # the minibatch's labelled rows of -log π(target), each weighted by |u_k - max later u| (line_ce_weighted; off: 1)
    # is added to the policy loss. 0 = off (updates unchanged); combinable with line_bonus_coef
    line_ce_coef: float = 0.0
    line_ce_weighted: bool = True
    # survival potential Φ (docs/training.md §8.2; needs a model built with critic_phi): phi_coef weighs the BCE of the
    # critic's phi head against train.aux.turn_boundaries targets at the deciding seat's turn boundaries; with
    # phi_shape_coef > 0 each boundary row's reward gets phi_shape_coef * (γ^k Φ(next boundary) − Φ(this one)), Φ = 0
    # after a finished game (Ng et al. 1999 potential-based shaping), before the estimator runs. 0 = off
    phi_coef: float = 0.0
    phi_shape_coef: float = 0.0
    phi_shape_first_seat_only: bool = False  # shape only the rows of engine player 0 (the seat that moved first)
    # window counterfactual advantages (TrainConfig.branch_windows, Rollout.branch_value; docs/training.md §8.2
    # 「窗口反事实优势」): on a branched row whose taken action is in the branch set, Â = V_b(a_taken) - Σ π̃(a) V_b(a)
    # (π̃ the behaviour policy renormalized on the set); "replace" puts Â in place of the row's VRPO advantage,
    # "add" adds branch_coef * Â to it (both before normalization, where line_bonus goes); a row whose taken action is
    # outside the set keeps VRPO. branch_q_coef = w > 0: the Q targets of the set's actions become
    # (1 - w) * (λ target of the taken action, acting-time Q of the others) + w * V_b(a). "off" / 0 = unchanged
    branch_mode: str = "off"
    branch_coef: float = 1.0
    branch_q_coef: float = 0.0
    # the window counterfactuals as a supervised term instead (first-seat plan §11.8 items 2-3; docs/training.md §8.2
    # 「窗口反事实监督项」): on each valid branched row the target is the branch-set candidate of highest branch value,
    # given only when it beats the second best by more than branch_ce_margin board points (times the collector's
    # board scale, PPOLearner.branch_board_scale; ties get none); branch_ce_coef * the mean over the minibatch's
    # targeted rows of -log π(target), weighted by gap / its mean over the rollout's targets ("delta") or 1
    # ("uniform"), is added to the policy loss (as line_ce, with which it adds up). Works with branch_mode "off".
    # 0 = off (updates unchanged)
    branch_ce_coef: float = 0.0
    branch_ce_margin: float = 0.5
    branch_ce_weight: str = "delta"

    def __post_init__(self) -> None:
        if self.critic_target not in ("lambda", "terminal"):
            raise ValueError("critic_target must be lambda or terminal")
        if self.critic_target == "terminal" and (self.gamma != 1 or self.critic_lam is not None):
            raise ValueError("terminal targets require gamma=1 and no critic_lam override")
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
        if not 0 <= self.adv_keep < 1:
            raise ValueError("adv_keep must be a quantile in [0, 1)")
        if self.adv_keep_mode not in ("rows", "policy"):
            raise ValueError("adv_keep_mode must be rows or policy")
        if not self.response_weight > 0:
            raise ValueError("response_weight must be positive")
        if not self.first_seat_weight > 0:
            raise ValueError("first_seat_weight must be positive")
        if self.aux_coef < 0:
            raise ValueError("aux_coef must be nonnegative")
        if self.line_bonus_coef < 0:
            raise ValueError("line_bonus_coef must be nonnegative")
        if self.line_ce_coef < 0:
            raise ValueError("line_ce_coef must be nonnegative")
        if self.phi_coef < 0 or self.phi_shape_coef < 0:
            raise ValueError("phi_coef and phi_shape_coef must be nonnegative")
        if self.branch_mode not in BRANCH_MODES:
            raise ValueError(f"branch_mode must be one of {BRANCH_MODES}, got {self.branch_mode!r}")
        if self.branch_coef < 0 or not 0 <= self.branch_q_coef <= 1:
            raise ValueError("branch_coef must be nonnegative and branch_q_coef in [0, 1]")
        if self.branch_ce_coef < 0 or self.branch_ce_margin < 0:
            raise ValueError("branch_ce_coef and branch_ce_margin must be nonnegative")
        if self.branch_ce_weight not in BRANCH_CE_WEIGHTS:
            raise ValueError(f"branch_ce_weight must be one of {BRANCH_CE_WEIGHTS}, got {self.branch_ce_weight!r}")

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
            stats = {
                "approx_kl": ((ratio - 1) - log_ratio).mean(),
                "clip_frac": ((ratio - 1).abs() > self.clip).float().mean(),
            }
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

    rows = torch.arange(actions.shape[0], device=actions.device)
    kind = obs["actions"][rows, actions, 0].long() - 1
    out: dict[str, float] = {}
    for k in torch.unique(kind).tolist():
        if 0 <= k < len(ACTION_KINDS):
            sel = kind == k
            out[f"adv/{ACTION_KINDS[k]}"] = float(adv[sel].mean())
            out[f"rows/{ACTION_KINDS[k]}"] = int(sel.sum())
    return out


class _Laps:
    """Update timing (``PPOLearner.timing``): ``lap(name)`` waits for the device, then adds the time since the last
    lap to ``name``. Waiting only moves when the clock is read, never what is computed. Off: every lap is a no-op."""

    def __init__(self, device: torch.device | None) -> None:
        self.on = device is not None
        self.sync = device is not None and device.type == "cuda"
        self.times: dict[str, float] = defaultdict(float)
        self.t0 = self.last = time.perf_counter() if self.on else 0.0

    def lap(self, name: str) -> None:
        if not self.on:
            return
        if self.sync:
            torch.cuda.synchronize()
        now = time.perf_counter()
        self.times[name] += now - self.last
        self.last = now

    def stats(self) -> dict[str, float]:
        """``time/<section>`` seconds plus ``time/total``; ``time/other`` is whatever no lap covered."""
        if not self.on:
            return {}
        total = self.last - self.t0
        out = {f"time/{k}": v for k, v in self.times.items()}
        out["time/total"] = total
        out["time/other"] = total - sum(self.times.values())
        return out


class PPOLearner:
    """Owns the model's optimizer, the EMA reference policy, the optional BC prior and the optional solver-window
    labels (``line_labels``, a :class:`ygorl.train.line_labels.LineLabels`; used with ``cfg.line_bonus_coef > 0`` and /
    or ``cfg.line_ce_coef > 0``). ``branch_board_scale`` is the collector's branch value per board point
    (``BranchConfig.board_scale``): ``cfg.branch_ce_margin`` is in board points."""

    def __init__(self, model: nn.Module, cfg: PPOConfig, prior: nn.Module | None = None, line_labels=None,
                 branch_board_scale: float = 0.1) -> None:  # fmt: skip
        self.model = model
        self.cfg = cfg
        self.objective = _OBJECTIVES[cfg.objective](cfg)
        self.reference = self._frozen(model)
        self.prior = None if prior is None else self._frozen(prior)
        self.line_labels = line_labels
        self._line_stats: dict[str, float] = {}
        self._line_ce: tuple[Tensor, Tensor] | None = None  # [T, B] target candidate (-1: none) and weight
        self._phi_stats: dict[str, float] = {}
        self._boundaries = None  # train.aux.TurnBoundaries of the rollout being updated on (phi_coef / phi_shape_coef)
        self._branch_stats: dict[str, float] = {}
        self._branch_q: Tensor | None = None  # [T, B, A] Q targets of the branch sets' untaken actions (NaN: none)
        self._branch_ce: tuple[Tensor, Tensor] | None = None  # [T, B] branch_ce_coef target (-1: none) and weight
        self.branch_board_scale = branch_board_scale
        self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=cfg.adam_eps)
        self.updates = 0
        self.new_heads: list[str] = []  # critic parameters a loaded checkpoint lacked (started fresh)
        self.timing = False  # add time/<section> of each update to its stats (tools/bench_train.py)

    def _policy_params(self) -> list[nn.Parameter]:
        """Parameters the policy's logits depend on: the actor (with the trunk the critic may share)."""
        actor = getattr(self.model, "actor", None)
        if actor is None:
            raise ValueError("critic warm-up needs a model with an .actor (the policy's parameters)")
        return list(actor.parameters())

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
            per_row = torch.where(rows, per_row, torch.zeros((), dtype=per_row.dtype, device=per_row.device))
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
    def line_ce(logp: Tensor, target: Tensor, weight: Tensor) -> Tensor:
        """Mean over the rows with a target (``target >= 0``) of ``weight * -log π(target)``; 0 without one."""
        has = target >= 0
        nll = -logp.gather(1, target.clamp(min=0).unsqueeze(1)).squeeze(1)
        per_row = torch.where(has, weight.to(nll) * nll, torch.zeros((), dtype=nll.dtype, device=nll.device))
        return per_row.sum() / has.sum().clamp(min=1)

    @staticmethod
    def entropy(logits: Tensor, mask: Tensor) -> Tensor:
        logp = torch.log_softmax(logits, -1)
        return -_masked_mean_rows(logp.exp() * logp, mask).mean()

    @torch.no_grad()
    def _score(self, module: nn.Module, obs, n: int) -> Tensor:
        chunk = max(1, self.cfg.minibatch_size)
        out = [policy_logits(module, _index(obs, slice(i, min(i + chunk, n)))).float() for i in range(0, n, chunk)]
        return torch.cat(out)

    # -- update --------------------------------------------------------------------------------------
    def targets(self, ro: Rollout) -> Estimate:
        """The estimator's targets; with line labels and ``line_bonus_coef > 0`` the per-row label bonus
        (:meth:`ygorl.train.line_labels.LineLabels.bonus`) is added to the raw advantages, which are then
        normalized as usual (the estimator normalizes itself when there is no bonus: the same operations). With
        ``line_ce_coef > 0`` the supervised targets of the labelled rows (:meth:`~ygorl.train.line_labels.LineLabels.
        ce_targets`) are kept for :meth:`update`."""
        cfg = self.cfg
        self._line_stats = {}
        self._line_ce = None
        self._phi_stats = {}
        self._boundaries = None
        self._branch_stats = {}
        self._branch_q = None
        self._branch_ce = None
        rewards = ro.rewards
        if cfg.phi_coef > 0 or cfg.phi_shape_coef > 0:
            self._boundaries = turn_boundaries(ro.obs, ro.players, ro.dones, ro.rewards, ro.truncated)
        if cfg.phi_shape_coef > 0:
            rewards = rewards + self.phi_shaping(ro)
        bonus = None
        if self.line_labels is not None and cfg.line_bonus_coef > 0:
            bonus, self._line_stats = self.line_labels.bonus(ro)
        if self.line_labels is not None and cfg.line_ce_coef > 0:
            target, weight, ce_stats = self.line_labels.ce_targets(ro)
            self._line_ce = (target, weight if cfg.line_ce_weighted else (target >= 0).float())
            self._line_stats.update(ce_stats)
        branch = None
        if ro.branch_value is not None and (
            cfg.branch_mode != "off" or cfg.branch_q_coef > 0 or cfg.branch_ce_coef > 0
        ):
            branch = branch_advantages(ro)
            ce = branch_ce_targets(ro, cfg.branch_ce_margin * self.branch_board_scale, cfg.branch_ce_weight)
            self._branch_stats = self._branch_summary(ro, *branch, ce)
            if cfg.branch_ce_coef > 0:
                self._branch_ce = (ce[0], ce[1])
        shift = bonus is not None or (branch is not None and cfg.branch_mode != "off")
        est = estimate(cfg.estimator, rewards=rewards, dones=ro.dones, players=ro.players, actions=ro.actions,
                        action_mask=ro.action_mask, probs=ro.probs, q=ro.q, values=ro.values, gamma=cfg.gamma,
                        lam=cfg.lam, bootstrap_player=ro.bootstrap_player, bootstrap_value=ro.bootstrap_value,
                        bootstrap_q=ro.bootstrap_q, bootstrap_probs=ro.bootstrap_probs,
                        bootstrap_mask=ro.bootstrap_mask, truncated=ro.truncated, vrpo_mode=cfg.vrpo_mode,
                        normalize="none" if shift else cfg.adv_norm, target_lam=cfg.critic_lam)  # fmt: skip
        if shift:
            raw = est.advantages
            if branch is not None and cfg.branch_mode != "off":
                adv, used, _ = branch
                if cfg.branch_mode == "replace":
                    raw = torch.where(used, adv.to(raw), raw)
                else:
                    raw = raw + cfg.branch_coef * adv.to(raw)
            if bonus is not None:
                raw = raw + cfg.line_bonus_coef * bonus.to(raw)
            est.advantages = normalize_advantages(raw, mode=cfg.adv_norm)
        if cfg.critic_target == "terminal":
            if not ro.complete_games:
                raise ValueError("terminal targets require complete-game collection")
            z = terminal_returns(ro.rewards, ro.dones, ro.players, ro.truncated)
            est.q_targets = est.v_targets = z
        if branch is not None and cfg.branch_q_coef > 0 and est.q_targets is not None:
            est.q_targets, self._branch_q = self.branch_q_targets(ro, est.q_targets, branch[1], branch[2])
        return est

    def branch_q_targets(self, ro: Rollout, q_targets: Tensor, used: Tensor, in_set: Tensor) -> tuple[Tensor, Tensor]:
        """``branch_q_coef`` w: the taken action's Q target on the rows ``used`` becomes ``(1 - w) G + w V_b(a)``; the
        set's other actions get ``(1 - w) Q_act(a) + w V_b(a)`` (Q at acting time, ``Rollout.q``), returned as
        ``[T, B, A]`` (NaN off the set and at the taken action) for the extra Q loss of :meth:`update`."""
        w = self.cfg.branch_q_coef
        value = ro.branch_value.to(q_targets)
        taken = value.gather(-1, ro.actions.unsqueeze(-1)).squeeze(-1)
        q_targets = torch.where(used, (1 - w) * q_targets + w * taken, q_targets)
        others = in_set.clone()
        others.scatter_(-1, ro.actions.unsqueeze(-1), False)
        extra = torch.where(
            others, (1 - w) * ro.q.to(value) + w * value, torch.full((), float("nan"), dtype=value.dtype)
        )
        return q_targets, extra

    @staticmethod
    def _branch_summary(ro: Rollout, adv: Tensor, used: Tensor, in_set: Tensor, ce) -> dict[str, float]:
        """``branch_rows`` (branched rows whose taken action is in the set: their advantage is the branch one),
        ``branch_valid_rows``, ``branch_fallback_rows`` (taken action outside the set: VRPO), ``branch_adv_mean`` (Â
        over the used rows, before normalization), ``branch_fire_share`` (of the valid rows with a pass in the set,
        those whose best candidate is an activation: the counterfactual says fire; exact ties count as not); from
        :func:`branch_ce_targets` (``ce``, with the margin): ``branch_ce_rows`` (rows with a target), and of the
        valid rows with a pass in the set ``branch_tie_share`` (within the margin) and ``branch_hold_share`` (the
        target is the pass)."""
        valid = ro.branch_valid
        target, tie = ce[0], ce[3]
        out = {"branch_rows": int(used.sum()), "branch_valid_rows": int(valid.sum()),
               "branch_fallback_rows": int((valid & ~used).sum()),
               "branch_adv_mean": float(adv[used].mean()) if used.any() else 0.0,
               "branch_ce_rows": int((target >= 0).sum())}  # fmt: skip
        has_pass = valid & (ro.branch_pass >= 0) if ro.branch_pass is not None else torch.zeros_like(valid)
        if has_pass.any():
            value = torch.where(in_set, ro.branch_value, torch.full((), -float("inf"), dtype=ro.branch_value.dtype))
            hold = value.gather(-1, ro.branch_pass.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            fire = value.scatter(-1, ro.branch_pass.clamp(min=0).unsqueeze(-1), -float("inf")).amax(-1)
            out["branch_fire_share"] = float((fire > hold)[has_pass].float().mean())
            out["branch_tie_share"] = float(tie[has_pass].float().mean())
            out["branch_hold_share"] = float((target == ro.branch_pass)[has_pass].float().mean())
        else:
            out["branch_fire_share"] = out["branch_tie_share"] = out["branch_hold_share"] = 0.0
        return out

    @torch.no_grad()
    def phi_shaping(self, ro: Rollout) -> Tensor:
        """``phi_shape_coef *`` the potential-based shaping rewards ``[T, B]`` of :func:`ygorl.train.aux.phi_shaping`,
        with Φ the sigmoid of the current phi head at the rollout's turn boundaries (one forward pass over them)."""
        cfg, tb = self.cfg, self._boundaries
        T, B = ro.shape
        phi = torch.zeros(T * B)
        idx = tb.boundary.reshape(-1).nonzero().squeeze(1)
        if idx.numel():
            dev = next(self.model.parameters()).device
            training = self.model.training
            self.model.eval()
            obs = trim_padding(ro.obs)
            outs = []
            for i in range(0, idx.numel(), max(1, cfg.minibatch_size)):
                rows = idx[i : i + cfg.minibatch_size].to(dev)
                out = self.model(_index(obs, rows), _index(ro.privileged, rows))
                if out.phi is None:
                    raise ValueError("phi_shape_coef > 0 needs a model with the survival-potential head (critic_phi)")
                outs.append(torch.sigmoid(out.phi.float()).cpu())
            self.model.train(training)
            phi[idx] = torch.cat(outs)
        rows = (ro.players == 0).cpu() if cfg.phi_shape_first_seat_only else None
        shaping = cfg.phi_shape_coef * phi_shaping(phi.view(T, B), tb, cfg.gamma, rows)
        shaped = tb.boundary if rows is None else tb.boundary & rows
        self._phi_stats = {"phi_shape_rows": int(shaped.sum()), "phi_shape_mean": float(shaping[shaped].mean())
                           if shaped.any() else 0.0, "phi_mean": float(phi[idx].mean()) if idx.numel() else 0.0}  # fmt: skip
        return shaping.to(ro.rewards)

    def update(self, ro: Rollout, policy: bool = True) -> dict[str, float]:
        """One PPO update on ``ro``. ``policy=False`` (critic warm-up): only the Q / V losses, and only parameters the
        policy does not use are stepped, so the policy (actor and shared trunk) stays exactly as it was."""
        cfg, model = self.cfg, self.model
        dev = next(model.parameters()).device
        laps = _Laps(dev if self.timing else None)
        est = self.targets(ro)  # on the CPU with the rollout's per-row data; the batch tensors go to the model
        laps.lap("targets")
        adv = self.objective.prepare(ro, est).reshape(-1).float().to(dev)
        if cfg.response_weight != 1.0:
            if not isinstance(ro.obs, dict) or "globals" not in ro.obs:
                raise ValueError("response_weight needs observations with globals (is_my_turn)")
            opp_turn = (ro.obs["globals"][:, 2] == 0).to(dev)
            adv = torch.where(opp_turn, adv * cfg.response_weight, adv)
        if cfg.first_seat_weight != 1.0:
            first_seat = (ro.players.reshape(-1) == 0).to(dev)
            adv = torch.where(first_seat, adv * cfg.first_seat_weight, adv)
        q_targets = est.q_targets.reshape(-1).float().to(dev)
        v_targets = est.v_targets.reshape(-1).float().to(dev)
        actions = ro.actions.reshape(-1).to(dev)
        old_logp = ro.log_probs.reshape(-1).to(dev)
        # Cut the rollout's padding once (nets.policy.trim_padding): every minibatch then gathers from the smaller
        # tensors, and logits / masks only span the rollout's longest action list (same values, see tests).
        obs = trim_padding(ro.obs)
        mask = ro.action_mask.reshape(actions.shape[0], -1)[:, : obs["action_mask"].shape[1]].to(dev)
        n = actions.shape[0]
        bq = None
        if self._branch_q is not None:
            bq = self._branch_q.reshape(n, -1)[:, : mask.shape[1]].to(dev)
            bq_sum = torch.zeros((), dtype=torch.float64, device=dev)
        ce_t = ce_w = None
        if self._line_ce is not None and policy:
            ce_t, ce_w = (x.reshape(-1).to(dev) for x in self._line_ce)
            ce_sum = ce_batches = torch.zeros((), dtype=torch.float64, device=dev)
        bce_t = bce_w = None
        if self._branch_ce is not None and policy:
            bce_t, bce_w = (x.reshape(-1).to(dev) for x in self._branch_ce)
            bce_sum = bce_batches = torch.zeros((), dtype=torch.float64, device=dev)
        aux_t = aux_v = None
        if cfg.aux_coef > 0:
            at, av = aux_targets(ro.obs, ro.players, ro.dones)
            aux_t, aux_v = at.reshape(n, -1).to(dev), av.reshape(n).to(dev)
        phi_t = phi_v = None
        if cfg.phi_coef > 0:
            phi_t, phi_v = self._boundaries.survive.reshape(n).to(dev), self._boundaries.valid.reshape(n).to(dev)
        # advantage filtering: the rows the minibatches draw from (all of them with adv_keep = 0 or in a critic warm-up)
        if policy and cfg.adv_keep > 0:
            keep = adv.abs() >= torch.quantile(adv.abs(), cfg.adv_keep)
            if cfg.adv_keep_mode == "policy":  # all rows forwarded; the policy loss only sees the kept ones
                adv = torch.where(keep, adv * (n / max(1, int(keep.sum()))), torch.zeros_like(adv))
                kept = torch.arange(n, device=dev)
            else:
                kept = keep.nonzero().squeeze(1)
        else:
            kept = torch.arange(n, device=dev)
        n_kept = kept.shape[0]
        laps.lap("setup")
        ref_logits = self._score(self.reference, obs, n) if cfg.kl_ref_coef > 0 else None
        prior_logits = self._score(self.prior, obs, n) if self.prior is not None and cfg.kl_prior_coef > 0 else None
        prior_rows = self.prior_rows(obs, cfg.kl_prior_turns) if prior_logits is not None else None
        laps.lap("reference")

        model.train()
        sums: dict[str, float] = defaultdict(float)
        steps, checked = 0, 0
        nonfinite_loss, nonfinite_grad = 0, 0
        grad_sum, stop_kl = 0.0, 0.0
        stopped = False
        for _ in range(cfg.epochs):
            if stopped:
                break
            perm = kept[torch.randperm(n_kept, device=dev)]
            laps.lap("setup")
            for start in range(0, n_kept, cfg.minibatch_size):
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
                if ce_t is not None:
                    ce = self.line_ce(logp, ce_t[idx], ce_w[idx])
                    loss = loss + cfg.line_ce_coef * ce
                if bce_t is not None:  # the window counterfactuals' targets (branch_ce_coef), averaged as line_ce
                    bce = self.line_ce(logp, bce_t[idx], bce_w[idx])
                    loss = loss + cfg.branch_ce_coef * bce
                ql = q_loss(out.q.float(), actions[idx], q_targets[idx])
                if bq is not None:  # the branch sets' other actions (branch_q_coef): each entry weighs like a row
                    t = bq[idx]
                    has = torch.isfinite(t)
                    sq = torch.where(has, (out.q.float() - torch.nan_to_num(t)) ** 2, torch.zeros((), device=dev))
                    bql = 0.5 * sq.sum() / idx.shape[0]
                    ql = ql + bql
                vl = v_loss(out.v.float(), v_targets[idx])
                al = torch.zeros((), device=dev)
                if aux_t is not None:
                    if out.aux is None:
                        raise ValueError("aux_coef > 0 needs a model with auxiliary critic heads (critic_aux)")
                    al = aux_loss(out.aux.float(), actions[idx], aux_t[idx], aux_v[idx])
                loss = loss + cfg.q_coef * ql + cfg.v_coef * vl + cfg.aux_coef * al
                if not policy:
                    loss = cfg.q_coef * ql + cfg.v_coef * vl + cfg.aux_coef * al
                if phi_t is not None:
                    if out.phi is None:
                        raise ValueError("phi_coef > 0 needs a model with the survival-potential head (critic_phi)")
                    pl = phi_loss(out.phi.float(), phi_t[idx], phi_v[idx])
                    loss = loss + cfg.phi_coef * pl
                laps.lap("forward")
                if not bool(torch.isfinite(loss)):
                    # A NaN / inf loss would poison every parameter through clip_grad_norm_ (the total norm scales
                    # all gradients): skip the minibatch entirely, neither stepped nor averaged into the stats.
                    nonfinite_loss += 1
                    continue
                kl_now = extra.get("approx_kl")
                rejected = (
                    policy and cfg.target_kl is not None and kl_now is not None and float(kl_now) > 1.5 * cfg.target_kl
                )
                laps.lap("kl_guard")
                if not rejected:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if not policy:
                        for p in self._policy_params():
                            p.grad = None  # Adam skips these parameters: the policy does not move
                    laps.lap("backward")
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    if bool(torch.isfinite(grad_norm)):
                        self.optimizer.step()
                        grad_sum = grad_sum + grad_norm.double()  # read once after the loop, not per minibatch
                        steps += 1
                    else:  # finite loss, overflowing gradient: the step is skipped, the forward stats still count
                        self.optimizer.zero_grad(set_to_none=True)
                        nonfinite_grad += 1
                    laps.lap("optimizer")
                # Forward diagnostics include a rejected batch; grad_norm counts only applied steps.
                for key, val in (("loss", loss), ("policy_loss", policy_loss), ("entropy", ent), ("kl_ref", kl_ref),
                                 ("kl_prior", kl_prior), ("q_loss", ql), ("v_loss", vl), ("aux_loss", al),
                                 *extra.items(), *((("phi_loss", pl),) if phi_t is not None else ())):  # fmt: skip
                    # device sums in float64: the same values as summing the floats one by one, without a wait
                    sums[key] += val.detach().double() if isinstance(val, Tensor) else float(val)
                if bq is not None:
                    bq_sum = bq_sum + bql.detach().double()
                if ce_t is not None:  # the mean over the minibatches that hold a labelled row
                    ce_sum = ce_sum + ce.detach().double()
                    ce_batches = ce_batches + (ce_t[idx] >= 0).any().double()
                if bce_t is not None:
                    bce_sum = bce_sum + bce.detach().double()
                    bce_batches = bce_batches + (bce_t[idx] >= 0).any().double()
                checked += 1
                laps.lap("bookkeeping")
                if rejected:
                    stopped, stop_kl = True, float(kl_now)
                    break
        if steps:
            self._check_finite_parameters()
        self.updates += 1
        self.update_reference()
        laps.lap("ema")
        stats = {k: float(v) / checked for k, v in sums.items()}
        for key in ("loss", "policy_loss", "entropy", "kl_ref", "kl_prior", "q_loss", "v_loss", "aux_loss"):
            stats.setdefault(key, float("nan"))  # every minibatch was non-finite
        stats["nonfinite_minibatches"], stats["nonfinite_grad_minibatches"] = nonfinite_loss, nonfinite_grad
        stats["grad_norm"] = float(grad_sum) / steps if steps else 0.0
        taken_q = ro.q.gather(-1, ro.actions.unsqueeze(-1)).squeeze(-1).reshape(-1)
        stats["q_explained_var"] = _explained_variance(taken_q, q_targets)
        if ro.complete_games:
            z = terminal_returns(ro.rewards, ro.dones, ro.players, ro.truncated).reshape(-1)
            # Pre-update predictions on new, complete on-policy games, same z metric in both arms.
            stats["q_terminal_ev"] = _explained_variance(taken_q, z)
            stats["v_terminal_ev"] = _explained_variance(ro.values.reshape(-1), z)
            stats["q_terminal_mse"] = float((taken_q - z).square().mean())
            stats["v_terminal_mse"] = float((ro.values.reshape(-1) - z).square().mean())
        stats["adv_mean"], stats["adv_std"] = float(adv.mean()), float(adv.std(correction=0))
        stats["rows"], stats["kept_rows"] = n, n_kept
        if aux_v is not None:
            stats["aux_valid_rows"] = int(aux_v.sum())
        if phi_v is not None:
            stats["phi_rows"] = int(phi_v.sum())
            stats["phi_rate"] = float(phi_t[phi_v].mean()) if bool(phi_v.any()) else float("nan")
        if self._boundaries is not None:
            stats["phi_boundaries"] = int(self._boundaries.boundary.sum())
        stats.update(self._phi_stats)
        stats["minibatches"], stats["early_stop"] = steps, int(stopped)
        stats["evaluated_minibatches"], stats["stop_approx_kl"] = checked, stop_kl
        stats["critic_warmup"] = int(not policy)
        stats.update(_advantage_by_kind(obs, actions, adv))
        if prior_rows is not None:
            stats["kl_prior_rows"] = int(prior_rows.sum())
        stats.update(self._line_stats)
        stats.update(self._branch_stats)
        if bq is not None:
            stats["branch_q_loss"] = float(bq_sum) / max(1, checked)
        if ce_t is not None:
            stats["line_ce_loss"] = float(ce_sum / ce_batches.clamp(min=1))
        if bce_t is not None:
            stats["branch_ce_loss"] = float(bce_sum / bce_batches.clamp(min=1))
        stats.update(laps.stats())
        return stats

    @torch.no_grad()
    def _check_finite_parameters(self) -> None:
        """Raise naming the first non-finite parameter (one device sync), rather than let NaN weights fail later in a
        rollout's sampling."""
        named = list(self.model.named_parameters())
        finite = torch.stack([torch.isfinite(p).all() for _, p in named])
        if not bool(finite.all()):
            name = named[int((~finite).nonzero()[0])][0]
            raise RuntimeError(f"PPO update {self.updates + 1}: parameter {name!r} became non-finite after an "
                               "optimizer step (finite loss and gradient norm); stopping before the next rollout")  # fmt: skip

    @torch.no_grad()
    def update_reference(self) -> None:
        tau = self.cfg.reference_ema
        # keep_vars preserves Parameter/buffer identity across module aliases. A detached state_dict
        # has distinct Tensor wrappers for the same parameter, so it cannot be deduplicated by id.
        ref = self.reference.state_dict(keep_vars=True)
        seen: set[int] = set()
        for name, value in self.model.state_dict().items():
            if id(ref[name]) in seen:
                continue
            seen.add(id(ref[name]))
            if value.dtype.is_floating_point:
                ref[name].mul_(1 - tau).add_(value.detach(), alpha=tau)
            else:
                ref[name].copy_(value)

    # -- persistence --------------------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {"model": self.model.state_dict(), "reference": self.reference.state_dict(),
                "optimizer": self.optimizer.state_dict(), "updates": self.updates}  # fmt: skip

    def load_state_dict(self, state: dict) -> None:
        """Restore the learner; a checkpoint saved before a new critic head (critic.NEW_HEAD_PREFIXES) existed loads
        with that head fresh and its optimizer state empty (the head's parameters come last in the model)."""
        fresh = load_allowing_new_heads(self.model, state["model"])
        load_allowing_new_heads(self.reference, state["reference"])
        opt = state["optimizer"]
        if fresh:
            names = [k for k, _ in self.model.named_parameters()]
            n_new = sum(1 for k in names if k in fresh)
            if sorted(names[len(names) - n_new :]) != sorted(k for k in fresh if k in names):
                raise ValueError(f"new heads {fresh} are not the model's last parameters; cannot extend the optimizer")
            group = opt["param_groups"][0]
            if len(opt["param_groups"]) != 1 or len(group["params"]) != len(names) - n_new:
                raise ValueError("the checkpoint's optimizer does not match the model without its new heads")
            opt = {**opt, "param_groups": [{**group, "params": list(range(len(names)))}]}
        self.optimizer.load_state_dict(opt)
        self.updates = int(state["updates"])
        self.new_heads = fresh


def _explained_variance(pred: Tensor, target: Tensor) -> float:
    var = float(target.var(correction=0))
    return float("nan") if var == 0 else 1.0 - float((target - pred.to(target)).var(correction=0)) / var


__all__ = ["BRANCH_CE_WEIGHTS", "BRANCH_MODES", "ClippedSurrogate", "ObjectiveInputs", "PPOConfig", "PPOLearner",
           "PolicyGradient", "PolicyObjective", "available_objectives", "branch_advantages", "branch_ce_targets",
           "register_objective"]  # fmt: skip
