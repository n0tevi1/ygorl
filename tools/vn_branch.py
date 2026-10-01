"""Action gap of the oracle value network by engine branching (docs/oracle-value.md).

For a sample of validation positions of ``tools/vn_collect.py`` shards: rebuild the position by replaying its game's
seed and action prefix in the batched C++ env (deterministic; checked against the PPO critic's stored V), then play
every candidate action (the legal rows, at most ``--max-candidates`` by policy probability) and read the successor
position. Per candidate ``a``:

- ``Q_VN(s, a)``: the value network on the successor, from the acting player's side (negated when the opponent acts
  next; the outcome ``z`` when the action ends the game);
- ``Q_PPO(s, a)``: the PPO critic's Q head at ``s``; ``V_PPO(s'_a)``: its V head on the successor (sign-corrected).

The action gap of each is the policy-weighted standard deviation over the candidates; it is compared with the
outcome noise ``std(z - V(s))`` of the same estimator on the validation set (the scale of a Monte Carlo advantage).

Usage: vn_branch.py DATA_DIR RUN_DIR [RUN_DIR ...] --out OUT.json [--states 300]
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

from ygorl.cards.ydk import load_ydk
from ygorl.engine.duel import DuelConfig, default_cards
from ygorl.env import GameSpec
from ygorl.env.encoded import EncodedVecEnv
from ygorl.nets import collate
from ygorl.nets.actor_critic import collate_privileged
from ygorl.nets.oracle_value import load_value_net
from ygorl.train.checkpoint import load_actor_critic, torch_device

BUCKETS = {"1-2": (1, 2), "3-4": (3, 4), "5-6": (5, 6), "7+": (7, 999)}


def run_jobs(env: EncodedVecEnv, jobs: list[tuple[GameSpec, list[int]]]) -> list[tuple]:
    """Play each job's action prefix; -> per job ``("event", obs, privileged, player)`` at the decision after the
    prefix, or ``("end", result)`` when the game ends first."""
    out: list = [None] * len(jobs)
    todo = list(range(len(jobs)))[::-1]
    slots: dict[int, list[int]] = {}

    def launch(e: int) -> None:
        if todo:
            j = todo.pop()
            env.reset(e, jobs[j][0])
            slots[e] = [j, 0]
        else:
            slots.pop(e, None)

    for e in range(env.num_envs):
        launch(e)
    while slots:
        for ev in env.recv(1):
            j, pos = slots[ev.env_id]
            prefix = jobs[j][1]
            if ev.result is not None:
                out[j] = ("end", ev.result)
                launch(ev.env_id)
            elif pos == len(prefix):
                out[j] = ("event", ev.obs, ev.privileged, ev.player)
                launch(ev.env_id)
            else:
                slots[ev.env_id][1] += 1
                env.step(ev.env_id, prefix[pos])
    return out


@torch.no_grad()
def evaluate(model, events: list, device, size: int = 256):
    """Model outputs on ``("event", obs, priv, player)`` items, in chunks."""
    outs = []
    for s in range(0, len(events), size):
        chunk = events[s : s + size]
        obs = collate([e[1] for e in chunk], device)
        priv = collate_privileged([e[2] for e in chunk], device)
        outs.append(model(obs, priv))
    return outs


def pi_std(q: np.ndarray, p: np.ndarray) -> float:
    m = float((p * q).sum())
    return float(np.sqrt((p * (q - m) ** 2).sum()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("runs", type=Path, nargs="+", help="vn_train.py output directories")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--checkpoint", default="out/why/bb_lam05/checkpoints/update_000400.pt")
    ap.add_argument("--corpus", type=Path, default=Path("out/corpus/train"))
    ap.add_argument("--states", type=int, default=300)
    ap.add_argument("--max-candidates", type=int, default=16)
    ap.add_argument("--val-mod", type=int, default=10)
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    device = torch_device(args.device)
    rng = np.random.default_rng(0)

    # sample validation positions with a real choice, with their games' specs and action lists
    deck_names = json.loads((args.data / "decks.json").read_text())
    decks = {}
    rows = []
    for f in sorted(glob.glob(str(args.data / "shard_*.npz")))[: args.shards]:
        d = np.load(f)
        offs = np.concatenate([[0], np.cumsum(d["game_n_actions"])])
        games = {}
        for k, m in enumerate(d["game_meta"]):
            g = json.loads(str(m))
            games[g["game"]] = (g, d["game_actions"][offs[k] : offs[k + 1]].astype(int).tolist())
        ok = np.flatnonzero((d["game"] % args.val_mod == 0) & (d["n_legal"] >= 2))
        for i in ok:
            rows.append({"game": games[int(d["game"][i])], "step": int(d["step"][i]), "player": int(d["player"][i]),
                         "turn": int(d["turn"][i]), "z": float(d["z"][i]), "v": float(d["v"][i])})  # fmt: skip
    rows = [rows[i] for i in rng.choice(len(rows), min(args.states, len(rows)), replace=False)]

    def spec(g: dict) -> GameSpec:
        for k in (g["deck_a"], g["deck_b"]):
            if k not in decks:
                decks[k] = load_ydk(args.corpus / deck_names[k])
        return GameSpec(seed=g["seed"], deck_a=decks[g["deck_a"]], deck_b=decks[g["deck_b"]], first=g["first"],
                        config=DuelConfig(max_decisions=4000))  # fmt: skip

    ac = load_actor_critic(args.checkpoint)
    model = ac.model.to(device)
    env = EncodedVecEnv(64, 8, cards=default_cards(), vocab=ac.vocab, event_length=ac.event_length, privileged=True,
                        skip_forced=True)  # fmt: skip
    # phase 1: the positions themselves
    base = run_jobs(env, [(spec(r["game"][0]), r["game"][1][: r["step"]]) for r in rows])
    keep = [i for i, b in enumerate(base) if b[0] == "event" and b[3] == rows[i]["player"]]
    rows, base = [rows[i] for i in keep], [base[i] for i in keep]
    outs = evaluate(model, base, device)
    v_ppo = torch.cat([o.v for o in outs]).float().cpu().numpy()
    q_ppo = torch.cat([o.q for o in outs]).float().cpu().numpy()
    probs = torch.cat([torch.softmax(o.logits.float(), -1) for o in outs]).cpu().numpy()
    mismatch = np.abs(v_ppo - np.array([r["v"] for r in rows]))
    print(f"{len(rows)} positions rebuilt; max |V - stored V| = {mismatch.max():.2e}", flush=True)
    good = mismatch < 1e-3
    # phase 2: every candidate action
    jobs, owner = [], []
    for i, r in enumerate(rows):
        if not good[i]:
            continue
        legal = np.flatnonzero(base[i][1]["action_mask"])
        cand = legal[np.argsort(-probs[i][legal])][: args.max_candidates]
        r["cand"] = cand.tolist()
        for a in cand:
            jobs.append((spec(r["game"][0]), r["game"][1][: r["step"]] + [int(a)]))
            owner.append((i, int(a)))
    succ = run_jobs(env, jobs)
    events = [s for s in succ if s[0] == "event"]
    sign = {}
    for (i, a), s in zip(owner, succ, strict=True):
        if s[0] == "end":
            w = s[1].get("winner")
            sign[(i, a)] = ("z", 0.0 if w is None else (1.0 if w == rows[i]["player"] else -1.0))
        else:
            sign[(i, a)] = ("v", 1.0 if s[3] == rows[i]["player"] else -1.0)
    print(f"{len(jobs)} successors ({len(jobs) - len(events)} terminal)", flush=True)
    estimators = {"ppo_v_succ": np.concatenate([o.v.float().cpu().numpy() for o in evaluate(model, events, device)])}
    vn_state = {}
    for run in args.runs:
        net, _, _ = load_value_net(run / "value_net.pt")
        net.to(device)

        def fwd(obs, priv, net=net):
            return net(obs, None if net.privileged is None else priv)

        estimators[run.name] = torch.cat(evaluate(fwd, events, device)).float().cpu().numpy()
        vn_state[run.name] = torch.cat(evaluate(fwd, base, device)).float().cpu().numpy()
    # per position gaps
    succ_index = {}
    k = 0
    for key, s in zip(owner, succ, strict=True):
        if s[0] == "event":
            succ_index[key] = k
            k += 1
    names = list(estimators)
    per = {n: [] for n in ["ppo_q", *names]}  # policy-weighted std of Q over the candidates
    uni = {n: [] for n in per}  # unweighted std
    td = {n: [] for n in names}
    turns = []
    for i, r in enumerate(rows):
        if "cand" not in r:
            continue
        cand = r["cand"]
        p = probs[i][cand] / probs[i][cand].sum()
        per["ppo_q"].append(pi_std(q_ppo[i][cand], p))
        uni["ppo_q"].append(float(np.std(q_ppo[i][cand])))
        for n in names:
            q = []
            for a in cand:
                kind, x = sign[(i, a)]
                q.append(x if kind == "z" else x * estimators[n][succ_index[(i, a)]])
            q = np.array(q)
            per[n].append(pi_std(q, p))
            uni[n].append(float(np.std(q)))
            v_s = v_ppo[i] if n == "ppo_v_succ" else vn_state[n][i]
            td[n].append(float((p * q).sum()) - v_s)  # E_pi[Q(s, a)] - V(s): consistency of the estimator
        turns.append(r["turn"])
    turns = np.array(turns)
    # outcome noise std(z - V(s)) of each estimator on the full validation set (from the training runs)
    noise = {}
    for run in args.runs:
        pr = np.load(run / "val_predictions.npz")
        noise[run.name] = float(np.std(pr["z"] - pr["vn"]))
        noise["ppo"] = float(np.std(pr["z"] - pr["v"]))
    res = {"positions": int(len(turns)), "successors": len(jobs), "noise_std": noise, "gap": {}}
    for n, gaps in per.items():
        g = np.array(gaps)
        ref = noise["ppo"] if n.startswith("ppo") else noise[n]
        u = np.array(uni[n])
        entry = {"mean_gap": float(g.mean()), "median_gap": float(np.median(g)), "gap_over_noise": float(g.mean() / ref),
                 "mean_uniform_gap": float(u.mean()), "uniform_gap_over_noise": float(u.mean() / ref)}  # fmt: skip
        entry["by_turn"] = {b: float(g[(turns >= lo) & (turns <= hi)].mean()) for b, (lo, hi) in BUCKETS.items()
                            if ((turns >= lo) & (turns <= hi)).any()}  # fmt: skip
        if n in td:
            entry["mean_abs_Epi_Q_minus_V"] = float(np.abs(td[n]).mean())
        res["gap"][n] = entry
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
