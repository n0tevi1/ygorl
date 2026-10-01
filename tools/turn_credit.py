"""Where does PPO's learning signal go, by game turn? (strength diagnosis, #83)

Resumes a PPO checkpoint into a :class:`~ygorl.train.trainer.Trainer` (same config, optionally fewer envs / steps and
another device), collects real rollouts and computes the learner's targets exactly as ``PPOLearner.update`` does
(``learner.targets``: the configured estimator, λ and advantage normalization). Without updating, it reports per
bucket of game turn (1 = the first player's turn 1, 2, 3-4, 5+) x own decision (turn player) / response:

- share of rows; mean |A| (normalized, as the loss sees it), mean A, std A;
- signal share: Var(Q(s,a) - V̄(s)) / Var(A_raw), the part of the advantage the critic attributes to the action
  vs the sampled λ-correction; and the critic's action gap sqrt(Σ π (Q - V̄)²) in normalized units;
- policy entropy, max probability, legal-action count of the behaviour policy.

On the first player's turn-1 own rows (and, for scale, an equal sample of turn 5+ own rows) it then takes the
actor's gradient of each loss term as the learner weighs it (policy gradient at ratio 1: -A log π(a); -c_H H;
c_KL KL(π‖π_ref)) and reports gradient norms, cosines, split-half consistency (games split by column parity), and the
first-order effect of a step along each term on the rows' mean log max-probability (> 0: the term sharpens).

Usage: tools/turn_credit.py CKPT OUT_DIR [--envs 32] [--steps 64] [--warmup 6] [--rollouts 2] [--device cpu]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ygorl.env.encoding import ACTION_KINDS
from ygorl.nets.policy import trim_padding
from ygorl.train.advantages import estimate, expected_values
from ygorl.train.ppo import PPOLearner, _index
from ygorl.train.trainer import TrainConfig, Trainer

BUCKETS = ("t1", "t2", "t3-4", "t5+")


def bucket(turn: int) -> str:
    return "t1" if turn <= 1 else "t2" if turn == 2 else "t3-4" if turn <= 4 else "t5+"


def row_table(learner: PPOLearner, ro) -> dict[str, np.ndarray]:
    cfg = learner.cfg
    est = learner.targets(ro)
    raw = estimate(cfg.estimator, rewards=ro.rewards, dones=ro.dones, players=ro.players, actions=ro.actions,
                   action_mask=ro.action_mask, probs=ro.probs, q=ro.q, values=ro.values, gamma=cfg.gamma, lam=cfg.lam,
                   bootstrap_player=ro.bootstrap_player, bootstrap_value=ro.bootstrap_value,
                   bootstrap_q=ro.bootstrap_q, bootstrap_probs=ro.bootstrap_probs, bootstrap_mask=ro.bootstrap_mask,
                   truncated=ro.truncated, vrpo_mode=cfg.vrpo_mode, normalize="none").advantages  # fmt: skip
    vbar = expected_values(ro.q, ro.probs, ro.action_mask)
    taken_q = ro.q.gather(-1, ro.actions.unsqueeze(-1)).squeeze(-1)
    critic = taken_q - vbar
    p = torch.where(ro.action_mask, ro.probs, torch.zeros(()))
    p = p / p.sum(-1, keepdim=True).clamp_min(1e-12)
    gap = (p * torch.where(ro.action_mask, ro.q - vbar.unsqueeze(-1), torch.zeros(())) ** 2).sum(-1).sqrt()
    logp = torch.where(p > 0, p.log(), torch.zeros(()))
    ent = -(p * logp).sum(-1)
    obs = trim_padding(ro.obs)
    g = obs["globals"].cpu()
    n = ro.actions.numel()
    acts = ro.actions.reshape(-1)
    kind = obs["actions"][torch.arange(n), acts, 0].long().cpu() - 1
    T, B = ro.actions.shape
    return {"turn": g[:, 3].numpy(), "mine": g[:, 2].numpy(), "dtype": g[:, 18].numpy(), "kind": kind.numpy(),
            "adv": est.advantages.reshape(-1).numpy(), "raw": raw.reshape(-1).numpy(),
            "critic": critic.reshape(-1).numpy(), "gap": gap.reshape(-1).numpy(), "ent": ent.reshape(-1).numpy(),
            "pmax": p.max(-1).values.reshape(-1).numpy(), "nlegal": ro.action_mask.sum(-1).reshape(-1).numpy(),
            "col": np.tile(np.arange(B), T), "std_raw": np.full(n, float(raw.std()))}  # fmt: skip


def summarize(rows: dict[str, np.ndarray]) -> list[dict]:
    out = []
    n = len(rows["adv"])
    for b in BUCKETS:
        for mine in (1, 0):
            sel = (np.vectorize(bucket)(rows["turn"]) == b) & (rows["mine"] == mine)
            if not sel.any():
                continue
            a, raw, cr = rows["adv"][sel], rows["raw"][sel], rows["critic"][sel]
            out.append({"bucket": b, "who": "own" if mine else "resp", "share": sel.sum() / n, "rows": int(sel.sum()),
                        "mean|A|": float(np.abs(a).mean()), "meanA": float(a.mean()), "stdA": float(a.std()),
                        "signal_share": float(cr.var() / max(raw.var(), 1e-12)),
                        "corr(A,critic)": float(np.corrcoef(raw, cr)[0, 1]) if sel.sum() > 2 else float("nan"),
                        "gap/std": float((rows["gap"][sel] / rows["std_raw"][sel]).mean()),
                        "entropy": float(rows["ent"][sel].mean()), "pmax": float(rows["pmax"][sel].mean()),
                        "pmax>0.9": float((rows["pmax"][sel] > 0.9).mean()),
                        "nlegal": float(rows["nlegal"][sel].mean())})  # fmt: skip
    return out


def kind_breakdown(rows, sel) -> dict:
    out = {}
    for k in np.unique(rows["kind"][sel]):
        s = sel & (rows["kind"] == k)
        out[ACTION_KINDS[k] if 0 <= k < len(ACTION_KINDS) else str(k)] = {
            "rows": int(s.sum()), "meanA": round(float(rows["adv"][s].mean()), 3),
            "pmax": round(float(rows["pmax"][s].mean()), 3), "entropy": round(float(rows["ent"][s].mean()), 3)}  # fmt: skip
    return out


def term_grads(learner: PPOLearner, ro, adv: torch.Tensor, idx: np.ndarray, chunk: int = 256) -> dict:
    """Actor gradients of the loss terms summed over rows ``idx`` (flattened rollout rows), each as the learner
    weighs a row (the 1 / minibatch factor dropped: it is common to all terms)."""
    cfg, model = learner.cfg, learner.model
    params = [p for p in learner._policy_params() if p.requires_grad]
    obs = trim_padding(ro.obs)
    mask = ro.action_mask.reshape(ro.actions.numel(), -1)[:, : obs["action_mask"].shape[1]]
    acts = ro.actions.reshape(-1)
    names = ("pg", "ent", "kl", "sharp")
    grads = {k: [torch.zeros_like(p) for p in params] for k in names}
    model.eval()  # no dropout: the gradient of the expected loss
    for s in range(0, len(idx), chunk):
        i = torch.as_tensor(idx[s : s + chunk])
        sub = _index(obs, i)
        m = mask[i]
        logits = model.policy_logits(sub).float()
        logp = torch.log_softmax(logits, -1)
        with torch.no_grad():
            ref = torch.log_softmax(learner.reference.policy_logits(sub).float(), -1)
        pz = torch.where(m, logp.exp(), torch.zeros(()))
        lz = torch.where(m, logp, torch.zeros(()))
        terms = {
            "pg": -(adv[i] * logp.gather(1, acts[i].unsqueeze(1)).squeeze(1)).sum(),
            "ent": cfg.entropy_coef * (pz * lz).sum(),  # -c H
            "kl": cfg.kl_ref_coef * (pz * (lz - torch.where(m, ref, torch.zeros(())))).sum(),
            "sharp": torch.where(m, logp, torch.full((), -1e9)).max(-1).values.sum(),  # Σ log max-prob
        }
        for k in names:
            gs = torch.autograd.grad(terms[k], params, retain_graph=k != names[-1], allow_unused=True)
            for acc, gg in zip(grads[k], gs, strict=True):
                if gg is not None:
                    acc += gg
    return {k: torch.cat([g.reshape(-1) for g in v]) for k, v in grads.items()}


def grad_report(learner, ro, adv, rows, sel, rng, max_rows: int) -> dict:
    idx = np.flatnonzero(sel)
    if len(idx) > max_rows:
        idx = np.sort(rng.choice(idx, max_rows, replace=False))
    g = term_grads(learner, ro, adv, idx)
    halves = [idx[rows["col"][idx] % 2 == h] for h in (0, 1)]
    gh = [term_grads(learner, ro, adv, h) for h in halves]
    sharp = g["sharp"] / len(idx)

    def cos(a, b):
        return float(a @ b / (a.norm() * b.norm() + 1e-20))

    out = {"rows": len(idx)}
    for k in ("pg", "ent", "kl"):
        out[k] = {"norm": float(g[k].norm()), "cos_with_pg": cos(g[k], g["pg"]),
                  "split_half_cos": cos(gh[0][k], gh[1][k]),
                  # SGD step -g: first-order change of the rows' mean log max-prob, per unit step
                  "d_mean_logpmax": float(-(g[k] @ sharp))}  # fmt: skip
    out["pg_ent_norm_ratio"] = out["pg"]["norm"] / max(out["ent"]["norm"], 1e-20)
    tot = g["pg"] + g["ent"] + g["kl"]
    out["total"] = {"norm": float(tot.norm()), "d_mean_logpmax": float(-(tot @ sharp))}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("out", type=Path)
    p.add_argument("--envs", type=int, default=32)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--rollouts", type=int, default=2)
    p.add_argument(
        "--warmup",
        type=int,
        default=6,
        help="rollouts collected and dropped first: a resumed run starts every "
        "game fresh, so the first segments over-sample early turns",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--root", default=".", help="directory the config's relative deck paths resolve against")
    p.add_argument("--grad-rows", type=int, default=1500)
    p.add_argument("--lam", type=float, default=None, help="override the advantage λ (what-if on the same rollouts)")
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    from ygorl.train.checkpoint import load_checkpoint

    cfg = TrainConfig.from_dict(load_checkpoint(a.checkpoint)["config"])
    decks = [str(Path(a.root) / d) for d in cfg.decks]
    over = {} if a.lam is None else {"ppo": dataclasses.replace(cfg.ppo, lam=a.lam)}
    trainer = Trainer.resume(a.checkpoint, a.out / "run", num_envs=a.envs, steps=a.steps, device=a.device,
                             decks=decks, eval_every=0, overlap_collect=False, bf16=False, env_threads=4,
                             torch_threads=a.threads, collect_threads=a.threads, **over)  # fmt: skip
    learner = trainer.learner
    print(f"ppo: {learner.cfg}", flush=True)
    rng = np.random.default_rng(0)
    all_rows, grads = defaultdict(list), []
    for _ in range(a.warmup):
        with torch.no_grad():
            trainer._collect()
    for r in range(a.rollouts):
        with torch.no_grad():
            ro = trainer._collect()
        rows = row_table(learner, ro)
        for k, v in rows.items():
            all_rows[k].append(v)
        adv = torch.as_tensor(rows["adv"]).float()
        t1 = (rows["turn"] <= 1) & (rows["mine"] == 1)
        late = (rows["turn"] >= 5) & (rows["mine"] == 1)
        rep = {
            "rollout": r,
            "games": len(ro.games),
            "t1_own": grad_report(learner, ro, adv, rows, t1, rng, a.grad_rows),
        }
        rep["t5+_own"] = grad_report(learner, ro, adv, rows, late, rng, rep["t1_own"]["rows"])
        grads.append(rep)
        print(json.dumps(rep, indent=1), flush=True)
    rows = {k: np.concatenate(v) for k, v in all_rows.items()}
    table = summarize(rows)
    t1 = (rows["turn"] <= 1) & (rows["mine"] == 1)
    report = {"table": table, "t1_own_kinds": kind_breakdown(rows, t1), "grads": grads}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "turn_credit.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    np.savez_compressed(a.out / "turn_credit_rows.npz", **rows)
    cols = list(table[0])
    print("\n" + " ".join(f"{c:>13s}" for c in cols))
    for t in table:
        print(" ".join(f"{t[c]:>13.3f}" if isinstance(t[c], float) else f"{t[c]!s:>13s}" for c in cols))
    print("\nturn-1 own rows by chosen kind:", json.dumps(report["t1_own_kinds"]))


if __name__ == "__main__":
    main()
