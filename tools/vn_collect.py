"""Collect self-play positions for the offline oracle value network (docs/oracle-value.md).

Self-play of one PPO checkpoint (both seats, sampled actions) on random corpus pairings in the batched C++ env with
``privileged=True``. Every decision is answered; a random subset (``--keep``) is stored with the actor observation,
the privileged tensors (opponent ground truth + the next draws of both players), the acting player, turn, game id,
decision index and the PPO critic's outputs (V, Q of the taken action, policy-weighted Q mean / spread). When the
game ends each stored row gets the outcome ``z`` in {-1, 0, 1} from its acting player's side. Every game's spec and
full action list are stored too, so a position can be rebuilt by replaying its prefix (``tools/vn_branch.py``).

Shards (``shard_XXXX.npz``, compressed): observations are ragged — only the valid prefix rows of ``cards`` /
``actions`` / ``events`` are kept, with per-row counts (``n_cards`` / ``n_actions`` / ``n_events``).

Usage: vn_collect.py CHECKPOINT OUT_DIR --games N [--keep 0.35] [--slots 256] [--threads 4] [--torch-threads 8]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ygorl.cards.ydk import load_ydk
from ygorl.engine.duel import DuelConfig, default_cards
from ygorl.env import GameSpec
from ygorl.env.driver import drive
from ygorl.env.encoded import EncodedVecEnv
from ygorl.env.privileged import PRIVILEGED_KEYS
from ygorl.eval.arena import derive_seed
from ygorl.nets import collate
from ygorl.nets.actor_critic import collate_privileged
from ygorl.train.checkpoint import load_actor_critic

RAGGED = (("cards", "n_cards"), ("actions", "n_actions"), ("events", "n_events"))


def game_specs(corpus: Path, n: int, seed: int) -> tuple[list[GameSpec], list[tuple[int, int, int]], list[str]]:
    paths = sorted(corpus.glob("*.ydk"))
    decks = [load_ydk(p) for p in paths]
    rng = np.random.default_rng(seed)
    specs, meta = [], []
    for i in range(n):
        a, b = (int(x) for x in rng.choice(len(decks), 2, replace=False))
        first = int(rng.integers(2))
        specs.append(GameSpec(seed=derive_seed(seed, i), deck_a=decks[a], deck_b=decks[b], first=first,
                              config=DuelConfig(max_decisions=4000)))  # fmt: skip
        meta.append((a, b, first))
    return specs, meta, [p.name for p in paths]


def valid_prefix(mask: np.ndarray) -> int:
    nz = np.flatnonzero(mask)
    return int(nz[-1]) + 1 if len(nz) else 0


class ShardWriter:
    def __init__(self, out: Path, games_per_shard: int, start: int = 0) -> None:
        self.out, self.per, self.index = out, games_per_shard, start
        self.rows: list[dict] = []
        self.games: list[dict] = []

    def add_game(self, rows: list[dict], game: dict) -> None:
        self.rows.extend(rows)
        self.games.append(game)
        if len(self.games) >= self.per:
            self.flush()

    def flush(self) -> None:
        if not self.games:
            return
        r = self.rows
        data: dict[str, np.ndarray] = {}
        for key, count in RAGGED:
            data[count] = np.array([x[key].shape[0] for x in r], np.int32)
            data[key] = np.concatenate([x[key] for x in r]) if r else np.zeros((0, 1), np.int32)
        data["action_mask"] = np.concatenate([x["action_mask"] for x in r]).astype(np.uint8)
        data["globals"] = np.stack([x["globals"] for x in r]).astype(np.int32)
        for k in PRIVILEGED_KEYS:
            data["priv_" + k] = np.stack([x["priv"][k] for x in r]).astype(np.int16)
        for k in ("player", "turn", "game", "step", "action", "n_legal"):
            data[k] = np.array([x[k] for x in r], np.int32)
        for k in ("z", "v", "q_taken", "q_bar", "q_std", "logp"):
            data[k] = np.array([x[k] for x in r], np.float32)
        steps = [g.pop("actions") for g in self.games]
        data["game_actions"] = np.concatenate(steps).astype(np.int16) if steps else np.zeros(0, np.int16)
        data["game_n_actions"] = np.array([len(s) for s in steps], np.int32)
        data["game_meta"] = np.array([json.dumps(g) for g in self.games])
        np.savez_compressed(self.out / f"shard_{self.index:04d}.npz", **data)
        self.index += 1
        self.rows, self.games = [], []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("out", type=Path)
    ap.add_argument("--games", type=int, required=True)
    ap.add_argument("--first-game", type=int, default=0, help="game ids start here (several collectors)")
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--keep", type=float, default=0.35, help="probability of storing a decision")
    ap.add_argument("--slots", type=int, default=256)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--torch-threads", type=int, default=8)
    ap.add_argument("--min-batch", type=int, default=64)
    ap.add_argument("--games-per-shard", type=int, default=250)
    ap.add_argument("--corpus", type=Path, default=Path("out/corpus/train"))
    args = ap.parse_args()
    torch.set_num_threads(args.torch_threads)
    args.out.mkdir(parents=True, exist_ok=True)

    ac = load_actor_critic(args.checkpoint)
    model, vocab = ac.model, ac.vocab
    all_specs, meta, deck_names = game_specs(args.corpus, args.first_game + args.games, args.seed)
    specs = all_specs[args.first_game :]
    meta = meta[args.first_game :]
    (args.out / "decks.json").write_text(json.dumps(deck_names))
    env = EncodedVecEnv(args.slots, args.threads, cards=default_cards(), vocab=vocab, event_length=ac.event_length,
                        privileged=True, skip_forced=True)  # fmt: skip
    writer = ShardWriter(args.out, args.games_per_shard, start=args.first_game // args.games_per_shard)
    gen = torch.Generator().manual_seed(args.seed + args.first_game)
    keep_rng = np.random.default_rng(args.seed + 7 + args.first_game)
    t0, done, positions = time.time(), [0], [0]

    def start(i, spec):
        return {"rows": [], "actions": []}

    def decide(ready):
        obs = [ev.obs for _, ev in ready]
        with torch.no_grad():
            o = model(collate(obs), collate_privileged([ev.privileged for _, ev in ready]))
            logp = torch.log_softmax(o.logits.float(), -1)
            probs = logp.exp()
            acts = torch.multinomial(probs, 1, generator=gen).squeeze(1)
            q = o.q.float()
            q_bar = (probs * q).sum(-1)
            q_std = (probs * (q - q_bar.unsqueeze(1)) ** 2).sum(-1).sqrt()
            q_taken = q.gather(1, acts.unsqueeze(1)).squeeze(1)
            lp = logp.gather(1, acts.unsqueeze(1)).squeeze(1)
        acts_l = acts.tolist()
        for j, (game, ev) in enumerate(ready):
            st = game.state
            if keep_rng.random() < args.keep:
                ob = ev.obs
                n_c, n_a, n_e = valid_prefix(ob["cards"][:, 0]), valid_prefix(ob["action_mask"]), 0
                if "event_mask" in ob:
                    n_e = valid_prefix(ob["event_mask"])
                st["rows"].append({
                    "cards": ob["cards"][:n_c].copy(), "actions": ob["actions"][:n_a].copy(),
                    "action_mask": ob["action_mask"][:n_a].copy(),
                    "events": ob["events"][:n_e].copy() if n_e else np.zeros((0, ob["events"].shape[1]), np.int32),
                    "globals": ob["globals"].copy(), "priv": {k: ev.privileged[k].copy() for k in PRIVILEGED_KEYS},
                    "player": ev.player, "turn": int(ob["globals"][3]), "game": args.first_game + game.index,
                    "step": len(st["actions"]), "action": acts_l[j], "n_legal": int(ob["action_mask"].sum()),
                    "v": float(o.v[j]), "q_taken": float(q_taken[j]), "q_bar": float(q_bar[j]),
                    "q_std": float(q_std[j]), "logp": float(lp[j]),
                })  # fmt: skip
            st["actions"].append(acts_l[j])
        return acts_l

    def on_result(game, result):
        st = game.state
        w = result.get("winner")
        reason = str(result.get("reason", ""))
        rows = [] if reason == "error" else st["rows"]
        for r in rows:
            r["z"] = 0.0 if w is None else (1.0 if w == r["player"] else -1.0)
        a, b, first = meta[game.index]
        writer.add_game(rows, {"game": args.first_game + game.index, "seed": game.spec.seed, "deck_a": a,
                               "deck_b": b, "first": first, "winner": w, "reason": reason,
                               "actions": st["actions"]})  # fmt: skip
        done[0] += 1
        positions[0] += len(rows)
        if done[0] % 100 == 0:
            dt = time.time() - t0
            print(f"{done[0]} games {positions[0]} positions {dt:.0f}s ({done[0] / dt:.2f} games/s)", flush=True)

    decisions = drive(env, specs, decide, on_result, min_batch=args.min_batch, start=start)
    writer.flush()
    dt = time.time() - t0
    print(f"done: {done[0]} games, {positions[0]} positions, {decisions} decisions in {dt:.0f}s", flush=True)


if __name__ == "__main__":
    main()
