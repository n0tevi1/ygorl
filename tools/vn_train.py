"""Train the offline oracle value network on ``tools/vn_collect.py`` shards (docs/oracle-value.md).

Supervised MSE of ``tanh`` output on the game outcome ``z`` of the acting player; game-level split (game id
``% --val-mod == 0`` is validation). Reports the explained variance ``1 - Var(z - V) / Var(z)`` of the network and
of the PPO critic's ``V`` (stored by the collector) on the same validation positions, overall and by turn, every
``--eval-every`` steps; keeps the best step's weights and validation predictions.

Usage: vn_train.py DATA_DIR [DATA_DIR ...] --out OUT_DIR --inputs oracle|privileged|public [--d-model 64] [--epochs 4]
       [--init-critic | --init-actor] [--draws order|identity] [--keep 0.3,1] [--eval-every STEPS]
"""

from __future__ import annotations

import argparse
import glob
import json
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch

from ygorl.env.privileged import PRIVILEGED_KEYS
from ygorl.nets.config import NetConfig
from ygorl.nets.oracle_value import OracleValueNet, save_value_net
from ygorl.train.checkpoint import load_actor_critic, load_checkpoint, load_policy, torch_device, vocab_from_text

WIDE = {"cards": 9, "events": 12}  # the one int32 column of each table (type bitmask / event payload)
BUCKETS = {"1-2": (1, 2), "3-4": (3, 4), "5-6": (5, 6), "7+": (7, 999)}
SCALARS = ("z", "v", "q_taken", "q_bar", "q_std", "turn", "player", "game", "step", "n_legal")


class Positions:
    """Every stored position of the shards in RAM: ragged tables as int16 (+ the wide int32 column)."""

    def __init__(self, files: list[str], keep: float = 1.0, seed: int = 0) -> None:
        """``keep``: load each stored position with this probability (fewer correlated positions per game)."""
        parts: dict[str, list] = {}
        rng = np.random.default_rng(seed)
        for f in files:
            d = dict(np.load(f))
            if keep < 1:
                rows = rng.random(len(d["z"])) < keep
                for key, count in (("cards", "n_cards"), ("actions", "n_actions"), ("events", "n_events")):
                    d[key] = d[key][np.repeat(rows, d[count])]
                d["action_mask"] = d["action_mask"][np.repeat(rows, d["n_actions"])]
                for k in (
                    "n_cards",
                    "n_actions",
                    "n_events",
                    "globals",
                    *SCALARS,
                    *("priv_" + k for k in PRIVILEGED_KEYS),
                ):
                    d[k] = d[k][rows]
            for key in ("cards", "events"):
                a = d[key]
                parts.setdefault(key + "_wide", []).append(a[:, WIDE[key]].astype(np.int32))
                lo = np.clip(a, 0, 32767).astype(np.int16)
                lo[:, WIDE[key]] = 0
                parts.setdefault(key, []).append(lo)
            parts.setdefault("actions", []).append(np.clip(d["actions"], 0, 32767).astype(np.int16))
            for k in ("action_mask", "n_cards", "n_actions", "n_events", "globals", *SCALARS):
                parts.setdefault(k, []).append(d[k])
            for k in PRIVILEGED_KEYS:
                parts.setdefault("priv_" + k, []).append(d["priv_" + k])
        self.a = {k: np.concatenate(v) for k, v in parts.items()}
        if files:
            self._index()

    @classmethod
    def merge(cls, parts: list[Positions]) -> Positions:
        out = cls([])
        out.a = {k: np.concatenate([p.a[k] for p in parts]) for k in parts[0].a}
        out._index()
        return out

    def _index(self) -> None:
        self.n = len(self.a["z"])
        self.offset = {}
        for key, count in (("cards", "n_cards"), ("actions", "n_actions"), ("events", "n_events")):
            self.offset[key] = np.concatenate([[0], np.cumsum(self.a[count], dtype=np.int64)[:-1]])

    def _ragged(self, key: str, count: str, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lens = self.a[count][idx]
        m = max(1, int(lens.max()))
        ar = np.arange(m)
        valid = ar[None, :] < lens[:, None]
        src = self.offset[key][idx][:, None] + ar[None, :]
        table = self.a[key]
        out = np.zeros((len(idx), m, table.shape[1]), np.int64)
        out[valid] = table[src[valid]]
        if key in WIDE:
            out[..., WIDE[key]][valid] = self.a[key + "_wide"][src[valid]]  # a view: writes into out
        return out, valid

    def batch(self, idx: np.ndarray) -> tuple[dict, dict, np.ndarray]:
        cards, _ = self._ragged("cards", "n_cards", idx)
        actions, amask_valid = self._ragged("actions", "n_actions", idx)
        events, emask = self._ragged("events", "n_events", idx)
        amask = np.zeros(amask_valid.shape, bool)
        src = self.offset["actions"][idx][:, None] + np.arange(amask.shape[1])[None, :]
        amask[amask_valid] = self.a["action_mask"][src[amask_valid]] != 0
        obs = {"cards": cards, "globals": self.a["globals"][idx].astype(np.int64), "actions": actions,
               "action_mask": amask, "events": events, "event_mask": emask}  # fmt: skip
        priv = {k: self.a["priv_" + k][idx].astype(np.int64) for k in PRIVILEGED_KEYS}
        return obs, priv, self.a["z"][idx]


def to_device(obs, priv, z, device):
    o = {k: torch.from_numpy(v).to(device, non_blocking=True) for k, v in obs.items()}
    p = {k: torch.from_numpy(v).to(device, non_blocking=True) for k, v in priv.items()}
    return o, p, torch.from_numpy(z).to(device)


def batches(data: Positions, idx: np.ndarray, size: int, prefetch: int = 6):
    """Background-thread batch assembly (numpy) in ``idx`` order."""
    q: queue.Queue = queue.Queue(prefetch)

    def work():
        for s in range(0, len(idx), size):
            q.put(data.batch(idx[s : s + size]))
        q.put(None)

    threading.Thread(target=work, daemon=True).start()
    while (item := q.get()) is not None:
        yield item


def explained_variance(z: np.ndarray, v: np.ndarray) -> float:
    return float(1 - np.var(z - v) / np.var(z)) if len(z) > 1 else float("nan")


def ev_table(z, v, turn) -> dict:
    out = {"all": {"n": len(z), "ev": explained_variance(z, v)}}
    for name, (lo, hi) in BUCKETS.items():
        m = (turn >= lo) & (turn <= hi)
        out[name] = {"n": int(m.sum()), "ev": explained_variance(z[m], v[m])}
    return out


@torch.no_grad()
def predict(net, data: Positions, idx: np.ndarray, device, size: int = 1024) -> np.ndarray:
    net.eval()
    out = []
    for obs, priv, z in batches(data, idx, size):
        o, p, _ = to_device(obs, priv, z, device)
        out.append(net(o, None if net.privileged is None else p).float().cpu().numpy())
    net.train()
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path, nargs="+", help="shard directories")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--inputs", choices=("oracle", "privileged", "public"), default="oracle")
    ap.add_argument("--checkpoint", default="out/why/bb_lam05/checkpoints/update_000400.pt",
                    help="the collecting PPO checkpoint: vocab, event length, base net options")  # fmt: skip
    ap.add_argument("--init-actor", action="store_true", help="warm-start the trunk from the checkpoint's actor")
    ap.add_argument(
        "--init-critic",
        action="store_true",
        help="start from the checkpoint's critic (trunk, privileged encoder, critic MLP + V head)",
    )
    ap.add_argument(
        "--draws",
        choices=("order", "identity"),
        default="order",
        help="oracle inputs: deck-order encoder (own card table) or the trunk's card identity encoder",
    )
    ap.add_argument("--trunk-lr-mult", type=float, default=1.0, help="trunk learning rate / head learning rate")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--layers", type=int, default=1, help="board and history Transformer layers")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.0, help="NetConfig.dropout of the trunk")
    ap.add_argument("--id-dropout", type=float, default=0.0, help="NetConfig.id_dropout (card ID embedding)")
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--eval-every", type=int, default=0, help="validate every N steps (0 = every epoch)")
    ap.add_argument("--val-mod", type=int, default=10)
    ap.add_argument("--max-train-games", type=int, default=0, help="0 = all")
    ap.add_argument("--shards", type=int, default=0, help="read only the first N shards of each directory (0 = all)")
    ap.add_argument("--keep", default="", help="per data directory: fraction of stored positions to load, e.g. 0.3,1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = torch_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    keep = [float(x) for x in args.keep.split(",")] if args.keep else [1.0] * len(args.data)
    parts = []
    for d, k in zip(args.data, keep, strict=True):
        found = sorted(glob.glob(str(d / "shard_*.npz")))
        parts.append(Positions(found[: args.shards] if args.shards else found, k, args.seed))
    data = parts[0] if len(parts) == 1 else Positions.merge(parts)
    files = [f for d in args.data for f in glob.glob(str(d / "shard_*.npz"))]
    games = data.a["game"]
    val = games % args.val_mod == 0
    train_idx = np.flatnonzero(~val)
    if args.max_train_games:
        keep = np.unique(games[train_idx])[: args.max_train_games]
        train_idx = train_idx[np.isin(games[train_idx], keep)]
    val_idx = np.flatnonzero(val)
    print(f"{data.n} positions ({len(np.unique(games))} games) from {len(files)} shards in {time.time() - t0:.0f}s; "
          f"train {len(train_idx)} ({len(np.unique(games[train_idx]))} games), val {len(val_idx)} "
          f"({len(np.unique(games[val_idx]))} games)", flush=True)  # fmt: skip

    state = load_checkpoint(args.checkpoint)
    vocab = vocab_from_text(state["vocab"])
    event_length = int(state["config"]["event_length"])
    base = NetConfig.from_dict(state["net_config"]).to_dict()
    if not (args.init_actor or args.init_critic):
        base |= {"d_model": args.d_model, "board_layers": args.layers, "history_layers": args.layers}
    cfg = NetConfig.from_dict(base | {"dropout": args.dropout, "id_dropout": args.id_dropout})
    if args.init_critic:
        net = OracleValueNet.from_actor_critic(load_actor_critic(args.checkpoint).model, args.inputs, args.draws)
        net.trunk.cfg = net.cfg = cfg  # the training-time dropout options
        for m in net.trunk.modules():
            if hasattr(m, "id_dropout"):
                m.id_dropout = args.id_dropout
    else:
        net = OracleValueNet(cfg, inputs=args.inputs, hidden=args.hidden, draws=args.draws)
    if args.init_actor and not args.init_critic:
        net.trunk.load_state_dict(load_policy(args.checkpoint).net.state_dict())
    net.to(device).train()
    n_params = sum(p.numel() for p in net.parameters())
    print(f"inputs={args.inputs} d_model={cfg.d_model} layers={cfg.board_layers} params={n_params:,}", flush=True)

    trunk = list(net.trunk.parameters())
    rest = [p for n, p in net.named_parameters() if not n.startswith("trunk.")]
    lrs = [args.lr * args.trunk_lr_mult, args.lr]
    opt = torch.optim.AdamW([{"params": trunk, "lr": lrs[0]}, {"params": rest, "lr": lrs[1]}],
                            weight_decay=args.weight_decay)  # fmt: skip
    per_epoch = (len(train_idx) + args.batch - 1) // args.batch
    steps = int(args.epochs * per_epoch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lrs, total_steps=steps, pct_start=0.05)
    eval_every = args.eval_every or per_epoch
    zv, turn_v = data.a["z"][val_idx], data.a["turn"][val_idx]
    critic = ev_table(zv, data.a["v"][val_idx], turn_v)
    print("ppo critic", json.dumps({k: round(x["ev"], 4) for k, x in critic.items()}), flush=True)
    var_train = float(np.var(data.a["z"][train_idx]))
    rng = np.random.default_rng(args.seed)
    history, best = [], None
    if args.init_critic:
        start = ev_table(zv, predict(net, data, val_idx, device), turn_v)
        print("step 0", json.dumps({k: round(x["ev"], 4) for k, x in start.items()}), flush=True)
    step, t1, total, count = 0, time.time(), 0.0, 0
    while step < steps:
        for obs, priv, z in batches(data, rng.permutation(train_idx), args.batch):
            o, p, zt = to_device(obs, priv, z, device)
            v = net(o, None if net.privileged is None else p)
            loss = ((v - zt) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            total += float(loss.detach()) * len(z)
            count += len(z)
            if step % eval_every and step < steps:
                continue
            pred = predict(net, data, val_idx, device)
            table = ev_table(zv, pred, turn_v)
            train_ev = 1 - (total / count) / var_train
            history.append({"step": step, "epoch": step / per_epoch, "train_ev": train_ev, "val": table,
                            "seconds": time.time() - t1})  # fmt: skip
            print(f"step {step} (epoch {step / per_epoch:.2f}): train_ev {train_ev:.4f} val",
                  json.dumps({k: round(x["ev"], 4) for k, x in table.items()}), f"{time.time() - t1:.0f}s",
                  flush=True)  # fmt: skip
            total, count = 0.0, 0
            if best is None or table["all"]["ev"] > best[0]:
                best = (table["all"]["ev"], pred)
                save_value_net(net.cpu(), vocab, event_length, args.out / "value_net.pt", step=step)
                net.to(device)
            if step >= steps:
                break
    np.savez_compressed(args.out / "val_predictions.npz", vn=best[1],
                        **{k: data.a[k][val_idx] for k in SCALARS})  # fmt: skip
    result = {"args": {k: str(v) for k, v in vars(args).items()}, "params": n_params, "train_positions":
              len(train_idx), "val_positions": len(val_idx), "ppo_critic": critic, "history": history,
              "best": ev_table(zv, best[1], turn_v)}  # fmt: skip
    (args.out / "result.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
