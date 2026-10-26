"""Why the window counterfactual advantage (design A) does not move the policy: magnitude, noise and the main
objective's pressure, measured on CPU rollouts of a checkpoint (first-seat plan §11).

  uv run --no-sync python tools/branch_adv_diagnosis.py CKPT --line-recovery FIRE.jsonl \\
      [--envs 128 --steps 128 --rounds 2 --rollouts 4] [--no-crn] [--updates] [--compare CKPT ...] --out OUT.json

Resumes ``CKPT`` (its decks, lines, league pool, critic) on the CPU and collects ``--rounds`` rollouts with every
qualifying window branched (``fraction = 1``, K = 2, no recovery requirement, board horizon, scale 0.1) and
``--rollouts`` M rollouts per candidate, keeping every rollout's value (``--no-crn``: the candidates of a rollout use
different seeds instead of common random numbers). Per window it keeps the line's state (recovery line for the
activation or not, broken), the policy's fire probability, the critic's Q of the two candidates, the raw VRPO
advantage of the row and of the game's later rows, and the game's result. The report has:

- ``magnitude``: Â (the COMA-style ``branch_advantages``) emulating the A1n arm (M = 1, half the windows) against the
  rollout's raw VRPO advantages, before and after the whole-rollout standardization, and the first-order push of each
  row group on the fire logit (``Σ A (1[a = fire] - π_fire)``, normalized advantages);
- ``noise``: sign agreement of single rollouts with each other and with the mean of the others, tie rates, the share
  of the variance of the window deltas that is rollout noise, and P(Â says fire) by line state;
- ``pressure``: the raw VRPO advantage of window rows that fired / held, of the later rows of games where the
  learner fired / did not, their results, and how often the critic's Q prefers firing;
- ``updates`` (``--updates``): one PPO update of copies of the learner per variant (control, A1n, a high-dose replace,
  Â scaled separately, a supervised term on the branch rows) and the change of the fire probability on the window
  rows, by the sign of the counterfactual;
- ``compare`` (``--compare``): the fire probability of other checkpoints on the same window observations (does
  training move the policy on its own distribution).
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import statistics
import tempfile
import time
from dataclasses import replace
from pathlib import Path

PASS, FIRE = 0, 1  # candidate positions in a K = 2 branch set


def _stats(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    a = [abs(x) for x in xs]
    return {"n": len(xs), "mean": statistics.fmean(xs), "abs_mean": statistics.fmean(a), "abs_median": statistics.median(a),
            "std": statistics.pstdev(xs), "zero": sum(x == 0 for x in xs) / len(xs), "pos": sum(x > 0 for x in xs) / len(xs),
            "neg": sum(x < 0 for x in xs) / len(xs)}  # fmt: skip


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def instrument(collector, crn: bool):
    """Record each branched window's per-rollout values on its row (``row.diag``) and the rows of each rollout
    (``captured``); without ``crn`` every candidate of a rollout gets its own seed."""
    from ygorl.train.rollout import CHAIN_KIND, _mix

    hooks, cfg = collector.branch_hooks, collector.branch
    captured: list[list] = []
    boards: dict[int, float] = {}
    orig_window, orig_value, orig_next = collector._branch_window, collector._branch_value, collector._branch_next
    orig_fields, orig_board = collector._branch_fields, hooks.board

    def window(slot, ev, a, probs, counted, effect, occurrence):
        st = orig_window(slot, ev, a, probs, counted, effect, occurrence)
        if st is not None:
            rec = hooks.candidates(slot, ev, probs, cfg.k, True, effect, occurrence) is not None
            kind = int(ev.obs["actions"][a][0])
            taken = "pass" if a == st.pass_row else "fire" if a in st.candidates else "other"
            st.diag = {"first": bool(slot.line.first_seat), "status": slot.line.status, "has_recovery": rec,
                       "taken": taken, "taken_chain": kind == CHAIN_KIND, "pass_row": st.pass_row,
                       "fire_row": st.candidates[FIRE], "values": [[] for _ in st.candidates],
                       "boards": [[] for _ in st.candidates]}  # fmt: skip
            st.base_seed = st.seed
        return st

    def board(ev):
        boards[ev.env_id] = value = orig_board(ev)
        return value

    def value(env_id, slot, v):
        st = slot.branch
        st.diag["values"][st.k].append(v)
        st.diag["boards"][st.k].append(boards.pop(env_id, None))
        return orig_value(env_id, slot, v)

    def nxt(env_id, slot):
        st = slot.branch
        if st.m < cfg.rollouts:
            if not crn:
                st.seed = _mix(st.base_seed, st.k + 1)
        else:
            st.row.diag = st.diag
        return orig_next(env_id, slot)

    def fields(rows, T, B):
        captured.append([(i, r.diag) for i, r in enumerate(rows) if hasattr(r, "diag")])
        return orig_fields(rows, T, B)

    collector._branch_window, collector._branch_value, collector._branch_next = window, value, nxt
    collector._branch_fields, hooks.board = fields, board
    return captured


def window_probs(model, ro, rows):
    """The policy's probabilities on the rollout rows ``rows`` (flat indices), [len(rows), A]."""
    import torch

    from ygorl.nets.batch import policy_logits
    from ygorl.nets.policy import trim_padding
    from ygorl.train.ppo import _index

    if not rows:
        return torch.zeros(0, 0)
    was = model.training
    model.eval()
    with torch.no_grad():
        obs = trim_padding(ro.obs)
        logits = policy_logits(model, _index(obs, torch.tensor(rows))).float()
    model.train(was)
    return torch.softmax(logits, -1)


def analyse_round(learner, ro, windows, rnd: int) -> list[dict]:
    """One record per branched window with every rollout valid: the rollout values, Â, the raw VRPO advantage of the
    row and of its game's later rows, the game's result."""
    T, B = ro.players.shape
    raw = vrpo_raw(learner, ro).reshape(-1)
    probs, q = ro.probs.reshape(T * B, -1), ro.q.reshape(T * B, -1)
    dones, rewards = ro.dones.reshape(-1), ro.rewards.reshape(-1)
    out = []
    for i, d in windows:
        vals = d["values"]
        if any(v is None for c in vals for v in c) or len(vals) != 2:
            continue
        t, b = divmod(i, B)
        # the game's later rows in this column, and its result (None: still running at the rollout's end)
        later, result = [], None
        for tt in range(t, T):
            j = tt * B + b
            if tt > t:
                later.append(float(raw[j]))
            if bool(dones[j]):
                result = float(rewards[j])
                break
        f, p = d["fire_row"], d["pass_row"]
        out.append({"round": rnd, "row": i, "t": t, "b": b, "first": d["first"], "status": d["status"],
                    "has_recovery": d["has_recovery"], "taken": d["taken"], "taken_chain": d["taken_chain"],
                    "pi_fire": float(probs[i, f]), "pi_pass": float(probs[i, p]),
                    "pi_chain": float(probs[i].sum() - probs[i, p]), "q_fire": float(q[i, f]), "q_pass": float(q[i, p]),
                    "v_fire": vals[FIRE], "v_pass": vals[PASS], "board_fire": d["boards"][FIRE],
                    "board_pass": d["boards"][PASS], "vrpo_raw": float(raw[i]), "later_raw": later, "result": result,
                    "fire_row": f, "pass_row": p})  # fmt: skip
    return out


def vrpo_raw(learner, ro):
    """The rollout's VRPO advantages before normalization (no branch term)."""
    saved = learner.cfg
    learner.cfg = replace(saved, branch_mode="off", branch_q_coef=0.0, adv_norm="none")
    try:
        return learner.targets(ro).advantages
    finally:
        learner.cfg = saved


def coma(v_fire: float, v_pass: float, pi_fire: float, pi_pass: float, taken: str) -> float:
    """``branch_advantages`` of one K = 2 row: V(taken) - Σ π̃ V."""
    s = pi_fire + pi_pass
    base = (pi_fire * v_fire + pi_pass * v_pass) / s
    return (v_fire if taken == "fire" else v_pass) - base


def rescale(values: list[float | None], boards: list[float | None], scale: float) -> list[float]:
    """Rollout values at another board scale (terminal values unchanged)."""
    return [v if bd is None else -scale * bd for v, bd in zip(values, boards)]


def branch_rollout(ro, records: list[dict], m: int | None, scale: float, keep: set[int] | None):
    """``ro`` with branch fields rebuilt from ``records``: the mean of the first ``m`` rollouts (all: None) at board
    scale ``scale``, only the rows in ``keep`` (all: None)."""
    import torch

    T, B, A = ro.probs.shape
    value = torch.full((T * B, A), float("nan"))
    valid = torch.zeros(T * B, dtype=torch.bool)
    passes = torch.full((T * B,), -1, dtype=torch.long)
    for r in records:
        if keep is not None and r["row"] not in keep:
            continue
        for row, vs, bd in (
            (r["fire_row"], r["v_fire"], r["board_fire"]),
            (r["pass_row"], r["v_pass"], r["board_pass"]),
        ):
            xs = rescale(vs, bd, scale)[:m]
            value[r["row"], row] = sum(xs) / len(xs)
        valid[r["row"]], passes[r["row"]] = True, r["pass_row"]
    return replace(ro, branch_value=value.reshape(T, B, A), branch_valid=valid.reshape(T, B),
                   branch_pass=passes.reshape(T, B))  # fmt: skip


def magnitude(learner, ro, records: list[dict], fraction_seed: int) -> dict:
    """A1n emulation on one rollout: M = 1, half the windows, board scale 0.1, ``replace``."""
    import torch

    from ygorl.train.ppo import branch_advantages

    rng = random.Random(fraction_seed)
    keep = {r["row"] for r in records if rng.random() < 0.5}
    a1n = branch_rollout(ro, records, 1, 0.1, keep)
    adv, used, _ = branch_advantages(a1n)
    raw = vrpo_raw(learner, ro)
    saved = learner.cfg
    learner.cfg = replace(saved, branch_mode="replace", branch_q_coef=0.0)
    try:
        norm = learner.targets(a1n).advantages
    finally:
        learner.cfg = saved
    learner.cfg = replace(saved, branch_mode="off", branch_q_coef=0.0)
    try:
        norm_ctrl = learner.targets(ro).advantages
    finally:
        learner.cfg = saved
    T, B = ro.players.shape
    flat = lambda x: x.reshape(-1)  # noqa: E731
    used_f, adv_f, raw_f, norm_f, ctrl_f = flat(used), flat(adv), flat(raw), flat(norm), flat(norm_ctrl)
    rows = {r["row"]: r for r in records}
    g = ro.obs["globals"].reshape(T * B, -1) if isinstance(ro.obs, dict) else None
    from ygorl.engine.constants import MSG_SELECT_CHAIN

    chain = (g[:, 18] == MSG_SELECT_CHAIN) if g is not None else torch.zeros(T * B, dtype=torch.bool)
    window = torch.zeros(T * B, dtype=torch.bool)
    window[list(rows)] = True
    out = {
        "rows": T * B, "windows": len(records), "branch_rows": int(used_f.sum()),
        "raw_branch": _stats([float(x) for x in adv_f[used_f]]),
        "raw_branch_taken_pass_share": float(sum(rows[i]["taken"] == "pass" for i in used_f.nonzero().flatten().tolist())
                                             / max(1, int(used_f.sum()))),
        "raw_all": _stats([float(x) for x in raw_f]),
        "raw_chain_rows": _stats([float(x) for x in raw_f[chain]]),
        "raw_window_rows_unbranched": _stats([float(x) for x in raw_f[window & ~used_f]]),
        "raw_nonwindow_rows": _stats([float(x) for x in raw_f[~window]]),
        "norm_branch": _stats([float(x) for x in norm_f[used_f]]),
        "norm_all": _stats([float(x) for x in norm_f]),
        "norm_window_rows_unbranched": _stats([float(x) for x in norm_f[window & ~used_f]]),
        "norm_abs_share_branch": float(norm_f[used_f].abs().sum() / norm_f.abs().sum()),
        "norm_abs_share_window_unbranched": float(norm_f[window & ~used_f].abs().sum() / norm_f.abs().sum()),
        "rollout_raw_std": float(raw_f.std(correction=0)),
    }  # fmt: skip
    # Pass-taken rows carry most of |Â|: Â(fire) = (1 - π̃_f) Δ, Â(pass) = -π̃_f Δ
    by_taken = {}
    for taken in ("fire", "pass"):
        idx = [i for i in used_f.nonzero().flatten().tolist() if rows[i]["taken"] == taken]
        by_taken[taken] = {"rows": len(idx), "abs_raw_sum": float(adv_f[idx].abs().sum()) if idx else 0.0,
                           "abs_norm_sum": float(norm_f[idx].abs().sum()) if idx else 0.0}  # fmt: skip
    out["by_taken"] = by_taken

    # first-order push on the fire logit: Σ A (1[a = fire] - π_fire), with the advantages the update sees
    def push(advs, idx) -> float:
        total = 0.0
        for i in idx:
            r = rows[i]
            total += float(advs[i]) * ((r["taken"] == "fire") - r["pi_fire"])
        return total

    branched = used_f.nonzero().flatten().tolist()
    unbranched = [i for i in rows if not bool(used_f[i])]
    out["fire_logit_push"] = {"a1n_branch_rows": push(norm_f, branched), "a1n_unbranched_window_rows": push(norm_f, unbranched),
                              "ctrl_all_window_rows": push(ctrl_f, list(rows)),
                              "ctrl_same_branch_rows": push(ctrl_f, branched)}  # fmt: skip
    # the same rows split by the counterfactual (all M): hold better / fire better / tie
    split = {}
    for name, test in (("hold_better", lambda d: d < 0), ("fire_better", lambda d: d > 0), ("tie", lambda d: d == 0)):
        idx = [i for i in branched if test(_delta(rows[i], None))]
        split[name] = {"rows": len(idx), "a1n_push": push(norm_f, idx), "ctrl_push": push(ctrl_f, idx)}
    out["fire_logit_push_by_cf"] = split
    return out


def _delta(r: dict, m: int | None, scale: float = 0.1) -> float:
    f = rescale(r["v_fire"], r["board_fire"], scale)[:m]
    p = rescale(r["v_pass"], r["board_pass"], scale)[:m]
    return sum(f) / len(f) - sum(p) / len(p)


def noise(records: list[dict]) -> dict:
    """Sign agreement of single rollouts and the share of rollout noise in the window deltas."""
    M = min(len(r["v_fire"]) for r in records) if records else 0
    out: dict = {"windows": len(records), "M": M}
    if M >= 2:
        d = [[r["v_fire"][m] - r["v_pass"][m] for m in range(M)] for r in records]
        pairs = [(x[0], x[1]) for x in d]
        decided = [(a, b) for a, b in pairs if a != 0 and b != 0]
        out["m1_pair_tie"] = sum(a == 0 for a, _ in pairs) / len(pairs)
        out["m1_pair_sign_agree_all"] = sum(_sign(a) == _sign(b) for a, b in pairs) / len(pairs)
        out["m1_pair_sign_agree_decided"] = sum(_sign(a) == _sign(b) for a, b in decided) / max(1, len(decided))
        rest = [(x[0], sum(x[1:]) / (M - 1)) for x in d]
        out["m1_vs_rest_agree_all"] = sum(_sign(a) == _sign(b) for a, b in rest) / len(rest)
        dec = [(a, b) for a, b in rest if a != 0 and b != 0]
        out["m1_vs_rest_agree_decided"] = sum(_sign(a) == _sign(b) for a, b in dec) / max(1, len(dec))
        if M >= 4:
            half = [(sum(x[: M // 2]), sum(x[M // 2 :])) for x in d]
            dech = [(a, b) for a, b in half if a != 0 and b != 0]
            out["m_half_vs_half_agree_decided"] = sum(_sign(a) == _sign(b) for a, b in dech) / max(1, len(dech))
            out["m_half_tie"] = sum(a == 0 for a, _ in half) / len(half)
        within = statistics.fmean(statistics.pvariance(x) * M / (M - 1) for x in d)
        means = [statistics.fmean(x) for x in d]
        total = statistics.pvariance([v for x in d for v in x])
        out["delta_var_total_single"] = total
        out["delta_var_within"] = within
        out["noise_share_single"] = within / total if total else None
        out["noise_share_mean_of_M"] = (
            (within / M) / statistics.pvariance(means) if statistics.pvariance(means) else None
        )
    groups = {
        "recovery_line": lambda r: r["status"] == "playing" and r["has_recovery"],
        "playing_no_recovery": lambda r: r["status"] == "playing" and not r["has_recovery"],
        "line_recovering": lambda r: r["status"] == "recovering",
        "line_broken": lambda r: r["status"] == "broken",
        "line_other": lambda r: r["status"] not in ("playing", "recovering", "broken"),
    }
    by = {}
    for name, test in groups.items():
        rs = [r for r in records if test(r)]
        for seat in ("first", "second", "all"):
            ss = [r for r in rs if seat == "all" or r["first"] == (seat == "first")]
            if not ss:
                continue
            for m in (1, None):
                ds = [_delta(r, m) for r in ss]
                key = f"{name}/{seat}/{'m1' if m == 1 else 'mall'}"
                by[key] = {"n": len(ds), "says_fire": sum(x > 0 for x in ds) / len(ds),
                           "says_hold": sum(x < 0 for x in ds) / len(ds), "tie": sum(x == 0 for x in ds) / len(ds),
                           "mean_delta": statistics.fmean(ds), "pi_fire": statistics.fmean(r["pi_fire"] for r in ss),
                           "fired": statistics.fmean(r["taken"] == "fire" for r in ss)}  # fmt: skip
    out["by_line_state"] = by
    return out


def pressure(records: list[dict]) -> dict:
    """The main objective at the windows: raw VRPO advantages of fired / held rows, of the later rows of games where
    the learner fired / did not, their results, and the critic's preference."""
    out: dict = {}
    for taken in ("fire", "pass"):
        rs = [r for r in records if r["taken"] == taken]
        if not rs:
            continue
        out[f"window_rows_{taken}"] = {**_stats([r["vrpo_raw"] for r in rs]),
                                       "cf_hold_better": sum(_delta(r, None) < 0 for r in rs) / len(rs)}  # fmt: skip
        hold = [r for r in rs if _delta(r, None) < 0]
        out[f"window_rows_{taken}_where_cf_hold_better"] = _stats([r["vrpo_raw"] for r in hold])
    games: dict[tuple, list] = {}
    # a game is identified by (round, column, its end row): windows of one game share the column and the first done
    for r in records:
        games.setdefault((r["round"], r["b"], r["t"] + len(r["later_raw"])), []).append(r)
    for fired in (True, False):
        gs = [g for g in games.values() if any(x["taken_chain"] for x in g) == fired]
        later = [v for g in gs for v in min(g, key=lambda x: x["t"])["later_raw"]]
        results = [g[0]["result"] for g in gs if g[0]["result"] is not None]
        out[f"games_{'fired' if fired else 'held'}"] = {
            "games": len(gs), "finished": len(results),
            "win_rate": sum(x > 0 for x in results) / len(results) if results else None,
            "later_rows": _stats(later), "later_sum_per_game": (sum(later) / len(gs)) if gs else None,
        }  # fmt: skip
    q = [r["q_fire"] - r["q_pass"] for r in records]
    out["critic_q_prefers_fire"] = sum(x > 0 for x in q) / max(1, len(q))
    agree = [(_sign(r["q_fire"] - r["q_pass"]), _sign(_delta(r, None))) for r in records if _delta(r, None) != 0]
    out["critic_q_agrees_with_cf_decided"] = sum(a == b for a, b in agree) / max(1, len(agree))
    out["q_fire_minus_pass"] = _stats(q)
    out["fire_prob"] = _stats([r["pi_fire"] / (r["pi_fire"] + r["pi_pass"]) for r in records])
    return out


def updates(learner, ro, records: list[dict], seed: int, ce_coef: float) -> dict:
    """One PPO update of a copy of ``learner`` per variant; the change of π̃(fire) on the window rows."""
    import torch

    from ygorl.train.ppo import branch_advantages

    rows = [r["row"] for r in records]
    before = window_probs(learner.model, ro, rows)
    rng = random.Random(seed)
    half = {r["row"] for r in records if rng.random() < 0.5}
    M = min(len(r["v_fire"]) for r in records)

    def run(name, cfg_changes, ro_v, adv_fn=None, ce=None):
        L = copy.deepcopy(learner)
        L.cfg = replace(L.cfg, **cfg_changes)
        if adv_fn is not None or ce is not None:
            orig = L.targets

            def targets(r):
                est = orig(r)
                if adv_fn is not None:
                    est.advantages = adv_fn(est.advantages)
                if ce is not None:
                    L._line_ce = ce
                return est

            L.targets = targets
        torch.manual_seed(seed)
        t0 = time.perf_counter()
        stats = L.update(ro_v)
        after = window_probs(L.model, ro, rows)
        res = {
            "seconds": round(time.perf_counter() - t0, 1),
            **{k: stats.get(k) for k in ("approx_kl", "clip_frac", "early_stop", "minibatches", "policy_loss")},
        }
        for group, test in (("hold_better", lambda d: d < 0), ("fire_better", lambda d: d > 0), ("tie", lambda d: d == 0),
                            ("all", lambda d: True)):  # fmt: skip
            idx = [j for j, r in enumerate(records) if test(_delta(r, None))]
            if not idx:
                continue
            f = torch.tensor([records[j]["fire_row"] for j in idx])
            p = torch.tensor([records[j]["pass_row"] for j in idx])
            ii = torch.tensor(idx)
            pf0 = before[ii, f] / (before[ii, f] + before[ii, p])
            pf1 = after[ii, f] / (after[ii, f] + after[ii, p])
            lg0 = torch.log(before[ii, f]) - torch.log(before[ii, p])
            lg1 = torch.log(after[ii, f]) - torch.log(after[ii, p])
            res[group] = {"n": len(idx), "fire_prob_before": float(pf0.mean()), "fire_prob_change": float((pf1 - pf0).mean()),
                          "fire_minus_pass_logit_change": float((lg1 - lg0).mean())}  # fmt: skip
        print(json.dumps({"update": name, **res}), flush=True)
        return name, res

    T, B = ro.players.shape
    out = {}
    out.update([run("ctrl", {"branch_mode": "off", "branch_q_coef": 0.0}, ro)])
    out.update([run("a1n_replace_m1_half_scale0.1", {"branch_mode": "replace", "branch_q_coef": 0.0},
                    branch_rollout(ro, records, 1, 0.1, half))])  # fmt: skip
    out.update([run(f"a2_replace_m{min(2, M)}_all_scale0.5", {"branch_mode": "replace", "branch_q_coef": 0.0},
                    branch_rollout(ro, records, min(2, M), 0.5, None))])  # fmt: skip
    # Â scaled on its own: the rollout standardized without the branch rows' VRPO, then the branch rows get Â / std(Â)
    full = branch_rollout(ro, records, None, 0.1, None)
    adv_b, used, _ = branch_advantages(full)
    sd = float(adv_b[used].std(correction=0)) or 1.0

    def separate(norm):
        return torch.where(used, adv_b / sd, norm)

    out.update([run(f"separate_scale_m{M}_all", {"branch_mode": "off", "branch_q_coef": 0.0}, ro, separate)])
    # set-level advantage without the COMA shrink: both branch candidates' sign on the taken row, ± |Δ| / std
    delta = torch.zeros(T * B)
    for r in records:
        d = _delta(r, None)
        delta[r["row"]] = d if r["taken"] == "fire" else -d if r["taken"] == "pass" else 0.0
    delta = delta.reshape(T, B)
    sdd = float(delta[used].std(correction=0)) or 1.0
    out.update([run(f"delta_unshrunk_m{M}_all", {"branch_mode": "off", "branch_q_coef": 0.0}, ro,
                    lambda norm: torch.where(used, delta / sdd, norm))])  # fmt: skip
    # a supervised term: target the better candidate where the rollouts decide, weight |Δ| / mean |Δ|
    target = torch.full((T * B,), -1, dtype=torch.long)
    weight = torch.zeros(T * B)
    decided = [r for r in records if _delta(r, None) != 0]
    mean_abs = statistics.fmean(abs(_delta(r, None)) for r in decided) if decided else 1.0
    for r in decided:
        d = _delta(r, None)
        target[r["row"]] = r["fire_row"] if d > 0 else r["pass_row"]
        weight[r["row"]] = abs(d) / mean_abs
    ce = (target.reshape(T, B), weight.reshape(T, B))
    out.update([run(f"ce_coef{ce_coef}_m{M}_all", {"branch_mode": "off", "branch_q_coef": 0.0, "line_ce_coef": ce_coef},
                    ro, None, ce)])  # fmt: skip
    # the same with the VRPO advantage of the window rows masked (only the counterfactual speaks there)
    window = torch.zeros(T * B, dtype=torch.bool)
    window[rows] = True
    window = window.reshape(T, B)
    out.update([run(f"ce_coef{ce_coef}_mask_window_vrpo", {"branch_mode": "off", "branch_q_coef": 0.0,
                                                           "line_ce_coef": ce_coef}, ro,
                    lambda norm: torch.where(window, torch.zeros_like(norm), norm), ce)])  # fmt: skip
    return out


def compare(model, paths: list[str], ro, records: list[dict]) -> dict:
    """π̃(fire) of other checkpoints' actors on the same window observations."""
    import torch

    from ygorl.train.checkpoint import load_checkpoint

    rows = [r["row"] for r in records]
    out = {}
    for path in paths:
        m = copy.deepcopy(model)
        m.load_state_dict(load_checkpoint(path)["learner"]["model"])
        pr = window_probs(m, ro, rows)
        res = {}
        for group, test in (("hold_better", lambda d: d < 0), ("fire_better", lambda d: d > 0), ("tie", lambda d: d == 0),
                            ("all", lambda d: True)):  # fmt: skip
            idx = [j for j, r in enumerate(records) if test(_delta(r, None))]
            if not idx:
                continue
            ii = torch.tensor(idx)
            f = torch.tensor([records[j]["fire_row"] for j in idx])
            p = torch.tensor([records[j]["pass_row"] for j in idx])
            pf = pr[ii, f] / (pr[ii, f] + pr[ii, p])
            chain = 1 - pr[ii, p]
            res[group] = {"n": len(idx), "fire_prob": float(pf.mean()), "chain_prob": float(chain.mean()),
                          "greedy_fire": float((pr[ii, f] > pr[ii, p]).float().mean())}  # fmt: skip
        out[path] = res
    return out


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("ckpt", help="the checkpoint to resume (its decks, lines, league pool and critic)")
    p.add_argument("--line-recovery", action="append", default=[], help="--fire records of the lines (repeatable)")
    p.add_argument("--envs", type=int, default=128)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--rollouts", type=int, default=4, help="M rollouts per candidate (each one is kept)")
    p.add_argument("--no-crn", action="store_true", help="the candidates of a rollout use different seeds")
    p.add_argument("--env-threads", type=int, default=2)
    p.add_argument("--torch-threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=None, help="collector seed (default: the checkpoint's)")
    p.add_argument("--updates", action="store_true", help="one PPO update per variant on the first rollout")
    p.add_argument("--ce-coef", type=float, default=0.2)
    p.add_argument("--compare", action="append", default=[], help="checkpoints whose actors score the same windows")
    p.add_argument("--records", action="store_true", help="keep every window's record in the report")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    import torch

    from ygorl.train.trainer import Trainer

    t_start = time.perf_counter()
    torch.set_num_threads(a.torch_threads)
    over = dict(device="cpu", bf16=False, num_envs=a.envs, steps=a.steps, env_threads=a.env_threads,
                collect_threads=a.torch_threads, torch_threads=a.torch_threads, eval_every=0, checkpoint_every=0,
                snapshot_every=0, log_games=False, trace_games=False, branch_windows=True, branch_fraction=1.0,
                branch_require_recovery=False, branch_k=2, branch_rollouts=a.rollouts, branch_horizon="board",
                branch_board_scale=0.1)  # fmt: skip
    if a.line_recovery:
        over["line_recovery"] = tuple(a.line_recovery)
    if a.seed is not None:
        over["seed"] = a.seed
    with tempfile.TemporaryDirectory() as tmp:
        trainer = Trainer.resume(a.ckpt, tmp, log=print, **over)
    learner, collector = trainer.learner, trainer.collector
    captured = instrument(collector, crn=not a.no_crn)
    records, rollouts, first_ro, collect_s = [], [], None, []
    for rnd in range(a.rounds):
        t0 = time.perf_counter()
        ro = collector.collect()
        collect_s.append(round(time.perf_counter() - t0, 1))
        recs = analyse_round(learner, ro, captured[-1], rnd)
        records.extend(recs)
        rollouts.append({"round": rnd, "seconds": collect_s[-1], "windows": ro.branch_windows,
                         "valid_windows": len(recs), "branch_decisions": ro.branch_decisions,
                         "line_games": sum(g.line is not None for g in ro.games), "games": len(ro.games),
                         "magnitude": magnitude(learner, ro, recs, 1000 + rnd)})  # fmt: skip
        print(json.dumps({k: v for k, v in rollouts[-1].items() if k != "magnitude"}), flush=True)
        if first_ro is None:
            first_ro, first_recs = ro, recs
    report = {"ckpt": a.ckpt, "crn": not a.no_crn, "rollouts_per_candidate": a.rollouts, "envs": a.envs,
              "steps": a.steps, "rounds": rollouts, "noise": noise(records), "pressure": pressure(records)}  # fmt: skip
    if a.records:
        report["records"] = records

    def write() -> None:  # after each stage: the updates are slow on the CPU
        report["seconds"] = round(time.perf_counter() - t_start, 1)
        Path(a.out).write_text(json.dumps(report, indent=1))

    write()
    if a.compare and first_ro is not None:
        report["compare"] = compare(trainer.model, [a.ckpt, *a.compare], first_ro, first_recs)
        write()
    if a.updates and first_ro is not None and first_recs:
        report["updates"] = updates(learner, first_ro, first_recs, 7, a.ce_coef)
        write()
    return report


if __name__ == "__main__":
    main()
