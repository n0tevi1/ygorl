"""Belief-head experiment (T4c.1): random self-play data, brief CPU training, T3.5 evaluation.

Usage:
    uv run python tools/train_beliefs.py [--games 1500] [--seconds 300] [--out DIR] [--threads 2]

1. Meta table: the first 8 test decks (tests/decks, sorted by name) are the meta types with
   Zipf shares; the last 2 are rogue decks ("other", share 0.15). 30% of meta decks get 1-3
   random main-deck swaps (off-meta noise; the label stays the meta type). Candidates = meta
   union + tests/data/generic_pool.json; roles = hand_trap / ash / maxx_c / nibiru / extender.
2. Collect: ``EncodedVecEnv(privileged=True)``, both seats pick uniformly random legal actions,
   games capped at ``--max-turns``. A decision is kept with probability 0.1, or always when the
   chosen action is an activation (the only points with a ``responded`` label). Every 5th game is test.
3. Train, on the training games: (a) heads on the HDT features only; (b) heads on a small
   ``PolicyNet`` trunk (board + event history, action-level ``responded``), belief losses only.
   AdamW, ``--seconds`` each; the parameters with the best validation loss are kept.
4. Report ``evaluate_beliefs`` on the test games for: uniform, frequency prior (fit on train),
   the HDT filter (``hdt_prior``; ``responded`` = train base rate), (a) and (b).

Writes ``report.json`` and ``report.md`` to ``--out`` (default: a temporary directory) and prints
the Markdown table. Numbers: docs/belief-heads.md.
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from ygorl import paths
from ygorl.cards.cdb import CardDB, CardVocab
from ygorl.cards.ydk import Deck, load_ydk
from ygorl.engine.duel import DuelConfig
from ygorl.env import GameSpec
from ygorl.env.belief_prior import (
    Evidence,
    EvidenceTracker,
    MetaTable,
    default_roles,
    hdt_prior,
    responded_labels,
    role_targets,
)
from ygorl.env.driver import Game, drive
from ygorl.env.encoded import EncodedEvent, EncodedVecEnv
from ygorl.env.encoding import ACTION_KINDS
from ygorl.env.events import E_EVENT
from ygorl.env.privileged import belief_targets
from ygorl.eval.beliefs import BeliefBatch, Head, evaluate_beliefs, prior_predictor, uniform_predictor
from ygorl.eval.calibration import roc_auc
from ygorl.nets import NetConfig, PolicyNet, to_tensors
from ygorl.nets.belief import (
    HEADS,
    PRIOR_KEYS,
    BeliefConfig,
    BeliefHeads,
    BeliefPolicy,
    belief_losses,
    evaluation_batch,
    loss_weights,
    prior_tensors,
)

ROOT = Path(__file__).resolve().parents[1]
ACTIVATE = ACTION_KINDS.index("activate") + 1
CARD_ROWS = 96  # card-table rows kept per sample (the tail is the viewer's own deck composition)
EVENT_WINDOW = 48  # event tokens per sample fed to the trunk
STREAM = 1024  # event tokens per observation while collecting (the full stream, for the responded labels)
TARGET_KEYS = ("deck_type", "remaining_copies", "hand", "hand_mask", "hand_roles", "hand_roles_mask", "set_cards",
               "set_cards_mask", "responded", "responded_action")  # fmt: skip


# --- setup ------------------------------------------------------------------------------------


def extenders(db: CardDB, decks: dict[str, Deck], generic: set[int]) -> list[int]:
    """Engine main-deck monsters whose script special summons from the hand (a rough 延伸 list)."""
    out = []
    for p in sorted({p for d in decks.values() for p in d.main} - generic):
        card = db.get(p)
        script = paths.card_scripts() / "official" / f"c{p}.lua"
        if card is None or not card.is_monster or not script.exists():
            continue
        src = script.read_text(encoding="utf-8", errors="replace")
        if "CATEGORY_SPECIAL_SUMMON" in src and re.search(r"SetRange\(LOCATION_HAND", src):
            out.append(p)
    return out


def build_meta(db: CardDB, vocab: CardVocab) -> tuple[MetaTable, dict[str, Deck], list[int]]:
    decks = {p.stem: load_ydk(p) for p in sorted((ROOT / "tests" / "decks").glob("*.ydk"))}
    names = sorted(decks)
    meta_decks = {n: decks[n] for n in names[:8]}
    rogue = {n: decks[n] for n in names[8:]}
    rows = json.loads((ROOT / "tests" / "data" / "generic_pool.json").read_text())["cards"]
    generic = [int(c["password"]) for c in rows]
    roles = default_roles(rows, extenders(db, meta_decks, set(generic)))
    meta = MetaTable(vocab, db, meta_decks, shares=1 / np.arange(1, 9), other_share=0.15, generic=generic,
                     roles=roles)  # fmt: skip
    return meta, rogue, generic


def swap_pool(db: CardDB, generic: list[int]) -> tuple[list[int], list[int]]:
    """Cards swapped in as off-meta noise: any main-deck card of the pool, or a generic main-deck card."""
    mains = [p for p, c in db.items() if (c.is_monster and not c.is_extra_deck and not c.is_token)
             or c.is_spell or c.is_trap]  # fmt: skip
    return sorted(set(mains)), [p for p in generic if not db[p].is_extra_deck]


def sample_deck(rng: np.random.Generator, meta: MetaTable, rogue: dict[str, Deck], pools) -> tuple[Deck, int]:
    """A deck and its deck-type label: meta by share (maybe with swaps), or a rogue deck ("other")."""
    k = int(rng.choice(meta.n_deck_types, p=meta.shares))
    if k == meta.n_deck_types - 1:
        return list(rogue.values())[int(rng.integers(len(rogue)))], k
    deck = _meta_deck(meta, k)
    if rng.random() < 0.3:
        main = list(deck.main)
        for _ in range(int(rng.integers(1, 4))):
            pool = pools[0] if rng.random() < 0.5 else pools[1]
            new = int(pool[int(rng.integers(len(pool)))])
            if main.count(new) < 3:
                main[int(rng.integers(len(main)))] = new
        deck = Deck(tuple(main), deck.extra, name=deck.name)
    return deck, k


_META_DECKS: dict[int, Deck] = {}


def _meta_deck(meta: MetaTable, k: int) -> Deck:
    if k not in _META_DECKS:
        cols = np.nonzero(meta.counts[k])[0]
        main, extra = [], []
        for c in cols:
            (extra if meta.is_extra[c] else main).extend([meta.candidates.passwords[c]] * int(meta.counts[k, c]))
        _META_DECKS[k] = Deck(tuple(main), tuple(extra), name=meta.names[k])
    return _META_DECKS[k]


# --- collection -------------------------------------------------------------------------------


def window(events: np.ndarray, n: int, length: int) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros((length, E_EVENT), dtype=np.int32)
    lo = max(0, n - length)
    out[: n - lo] = events[lo:n]
    return out, np.arange(length) < n - lo


def collect(args, db, vocab, meta, rogue, pools) -> dict[str, np.ndarray]:
    env = EncodedVecEnv(args.envs, args.threads, cards=db, vocab=vocab, privileged=True, event_length=STREAM)
    cfg = DuelConfig(max_turns=args.max_turns)
    rows: dict[str, list] = {k: [] for k in ("cards", "globals", "events", "event_mask", "actions", "game", *TARGET_KEYS,
                                             "copies_mask", "evidence", "order")}  # fmt: skip
    specs, deck_labels, rngs = [], [], []
    for i in range(args.games):
        rng = np.random.default_rng([args.seed, i])  # per game: independent of the thread timing
        (da, ka), (db_, kb) = sample_deck(rng, meta, rogue, pools), sample_deck(rng, meta, rogue, pools)
        specs.append(GameSpec(seed=args.seed * 100_000 + i, deck_a=da, deck_b=db_, first=i % 2, config=cfg))
        deck_labels.append((ka, kb))  # deck a, deck b
        rngs.append(rng)  # the same stream then picks the game's actions

    def start(i: int, spec: GameSpec) -> dict:
        # engine player p's opponent's label:
        return {"op_label": [deck_labels[i][spec.deck_of_seat(1 - p)] for p in (0, 1)],
                "trackers": [EvidenceTracker(meta), EvidenceTracker(meta)], "last": [None, None],
                "pending": [[], []], "rng": rngs[i], "step": 0}  # fmt: skip

    def on_result(game: Game, result: dict) -> None:
        st = game.state
        for p in (0, 1):
            last, pending = st["last"][p], st["pending"][p]
            if last is None or not pending:
                continue
            stream, n = last
            labels = responded_labels(stream[:n], [pos for _, pos in pending])
            if n >= STREAM:  # truncated stream: positions are unreliable
                labels[:] = -1
            for (i, _), y in zip(pending, labels):
                rows["responded"][i] = int(y)

    def decide(ready: list[tuple[Game, EncodedEvent]]) -> list[int]:
        actions = []
        for game, ev in ready:
            st, obs, p = game.state, ev.obs, ev.player
            evidence = st["trackers"][p].update(obs["cards"], obs["globals"])
            n_tok = int(obs["event_mask"].sum())
            st["last"][p] = (obs["events"], n_tok)
            legal = np.nonzero(obs["action_mask"])[0]
            rng = st["rng"]
            st["step"] += 1
            a = int(legal[rng.integers(len(legal))])
            activate = obs["actions"][a, 0] == ACTIVATE
            if activate or rng.random() < 0.1:
                i = len(rows["game"])
                ev_win, ev_mask = window(obs["events"], n_tok, EVENT_WINDOW)
                tg = belief_targets(ev.privileged, meta.candidates)
                roles = role_targets(ev.privileged, meta)
                rows["cards"].append(obs["cards"][:CARD_ROWS].copy())
                rows["globals"].append(obs["globals"].copy())
                rows["events"].append(ev_win)
                rows["event_mask"].append(ev_mask)
                rows["actions"].append(obs["actions"][a : a + 1].copy())
                rows["game"].append(game.index)
                rows["order"].append(game.index * 1_000_000 + st["step"])
                rows["deck_type"].append(st["op_label"][p])
                rows["remaining_copies"].append(tg["remaining_copies"].targets.astype(np.int8))
                rows["copies_mask"].append(tg["remaining_copies"].mask)
                rows["hand"].append(tg["hand"].targets.astype(np.int8))
                rows["hand_mask"].append(tg["hand"].mask)
                rows["hand_roles"].append(roles.targets.astype(np.int8))
                rows["hand_roles_mask"].append(roles.mask)
                rows["set_cards"].append(tg["set_cards"].targets.astype(np.int16))
                rows["set_cards_mask"].append(tg["set_cards"].mask)
                rows["responded"].append(-1)
                rows["responded_action"].append(0)
                rows["evidence"].append(evidence)
                if activate:
                    st["pending"][p].append((i, n_tok))
            actions.append(a)
        return actions

    t0 = time.time()
    decisions = drive(env, specs, decide, on_result, start=start)
    print(f"collected {len(rows['game'])} samples from {len(specs)} games / {decisions} decisions "
          f"in {time.time() - t0:.0f}s", flush=True)  # fmt: skip
    ev = Evidence.stack(rows.pop("evidence"))
    data = {k: np.stack(v) if isinstance(v[0], np.ndarray) else np.asarray(v) for k, v in rows.items()}
    data |= {f"ev_{k}": v for k, v in vars(ev).items()}
    order = np.argsort(data.pop("order"))  # canonical sample order (the arrival order depends on thread timing)
    return {k: v[order] for k, v in data.items()}


# --- training / evaluation --------------------------------------------------------------------


def evidence_of(data, idx) -> Evidence:
    return Evidence(
        *(data[f"ev_{k}"][idx] for k in ("seen", "seen_other", "visible", "public_hand", "counts", "set_zones"))
    )


def targets_of(data, idx) -> dict[str, np.ndarray]:
    t = {k: data[k][idx] for k in TARGET_KEYS}
    t["remaining_copies_mask"] = data["copies_mask"][idx]
    t["responded_mask"] = t["responded"] >= 0
    return t


def obs_of(data, idx) -> dict[str, torch.Tensor]:
    b = to_tensors({k: data[k][idx] for k in ("cards", "globals", "events", "event_mask", "actions")})
    b["action_mask"] = torch.ones(len(idx), 1, dtype=torch.bool)
    return b


def precompute_priors(data, meta, chunk: int = 2048) -> dict[str, np.ndarray]:
    """HDT prior of every sample (float32; set cards float16), computed once instead of per batch."""
    parts: dict[str, list] = {}
    for lo in range(0, len(data["game"]), chunk):
        p = hdt_prior(evidence_of(data, np.arange(lo, min(lo + chunk, len(data["game"])))), meta).arrays()
        for k in PRIOR_KEYS:
            v = p[k]
            if v.dtype.kind == "f":  # the [N, 15, C] set-card prior is the bulk: half precision
                v = v.astype(np.float16 if k == "set_cards" else np.float32)
            parts.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in parts.items()}


def forward(model, data, priors, idx):
    prior = prior_tensors({k: v[idx] for k, v in priors.items()})
    if isinstance(model, BeliefPolicy):
        return model(obs_of(data, idx), prior)[1], prior
    return model(prior), prior


def batch_loss(model, data, priors, idx, weights=None) -> dict[str, torch.Tensor]:
    out, prior = forward(model, data, priors, idx)
    tg = {k: torch.as_tensor(v) for k, v in targets_of(data, idx).items()}
    return belief_losses(out, tg, weights, prior=prior)


@torch.no_grad()
def validation_loss(model, data, priors, idx, chunk: int = 512) -> float:
    """Unweighted sum of the per-head validation losses (each head averaged over its entries)."""
    model.eval()
    sums: dict[str, float] = {}
    for lo in range(0, len(idx), chunk):
        part = idx[lo : lo + chunk]
        for k, v in batch_loss(model, data, priors, part).items():
            if k != "total":
                sums[k] = sums.get(k, 0.0) + float(v) * len(part)
    model.train()
    return sum(sums.values()) / len(idx)


def train(model, data, priors, train_idx, val_idx, args) -> dict:
    """Adam on the belief losses for ``args.seconds``; keeps the parameters with the best validation loss."""
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    t0, step, log = time.time(), 0, []
    best = (float("inf"), 0, None)
    while (elapsed := time.time() - t0) < args.seconds:
        idx = np.sort(rng.choice(train_idx, args.batch, replace=False))
        losses = batch_loss(model, data, priors, idx, loss_weights(elapsed / args.seconds))
        opt.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % args.eval_every == 0:
            val = validation_loss(model, data, priors, val_idx)
            if val < best[0]:
                best = (val, step, {k: v.detach().clone() for k, v in model.state_dict().items()})
            row = {k: round(float(v.detach()), 4) for k, v in losses.items()}
            log.append({"step": step, "seconds": round(elapsed, 1), "val": round(val, 4), **row})
            print(f"  step {step} {elapsed:.0f}s val={val:.3f} " + " ".join(f"{k}={v:.3f}" for k, v in row.items()),
                  flush=True)  # fmt: skip
    if best[2] is not None:
        model.load_state_dict(best[2])
    return {"steps": step, "best_step": best[1], "best_val": best[0], "log": log}


@torch.no_grad()
def predict(model, data, priors, idx, chunk: int = 512) -> dict[str, np.ndarray]:
    model.eval()
    parts: dict[str, list] = {h: [] for h in HEADS}
    for lo in range(0, len(idx), chunk):
        out, _ = forward(model, data, priors, idx[lo : lo + chunk])
        for k, v in out.probs().items():
            parts[k].append(v.numpy())
    model.train()
    return {k: np.concatenate(v) for k, v in parts.items()}


def hdt_probs(priors, idx, base_rate: float) -> dict[str, np.ndarray]:
    heads = ("deck_type", "remaining_copies", "hand", "hand_roles", "set_cards")
    out = {k: priors[k][idx].astype(np.float64) for k in heads}
    for k in ("deck_type", "remaining_copies", "set_cards"):  # undo the rounding of the stored copies
        out[k] /= out[k].sum(-1, keepdims=True)
    return out | {"responded": np.full(len(idx), base_rate)}


def per_role_auc(p: np.ndarray, t: dict[str, np.ndarray], meta: MetaTable) -> dict[str, float]:
    m = t["hand_roles_mask"]
    return {
        n: round(roc_auc(p[:, r], t["hand_roles"][:, r], mask=m[:, r]), 4)
        for n, r in zip(meta.role_names, range(meta.n_roles))
    } | {f"{n}/rate": round(float(t["hand_roles"][:, r][m[:, r]].mean()), 4) for r, n in enumerate(meta.role_names)}


def targets_batch(t: dict[str, np.ndarray], meta: MetaTable) -> BeliefBatch:
    """Targets-only batch for ``prior_predictor(fit=...)``; probs are zero-copy placeholders."""
    n = len(t["deck_type"])

    def z(*shape):
        return np.broadcast_to(np.zeros((1, *shape)), (n, *shape))

    sm, rm = t["set_cards_mask"] & (t["set_cards"] >= 0), t["responded_mask"]
    return BeliefBatch(
        deck_type=Head(z(meta.n_deck_types), t["deck_type"]),
        remaining_copies=Head(z(meta.n_cards, 4), t["remaining_copies"], t["remaining_copies_mask"]),
        hand=Head(z(meta.n_cards), t["hand"], t["hand_mask"]),
        hand_roles=Head(z(meta.n_roles), t["hand_roles"], t["hand_roles_mask"]),
        set_cards=Head(z(t["set_cards"].shape[1], meta.n_cards), np.where(sm, t["set_cards"], 0), sm),
        responded=Head(z(), np.where(rm, t["responded"], 0), rm),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--games", type=int, default=1500)
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2, help="engine worker threads and torch threads")
    ap.add_argument("--seconds", type=float, default=300, help="training time per model")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=50, help="steps between validation checks (early stopping)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bins", type=int, default=15)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    out_dir = args.out or Path(tempfile.mkdtemp(prefix="beliefs-"))
    out_dir.mkdir(parents=True, exist_ok=True)

    db = CardDB.load()
    vocab = CardVocab.from_db(db)
    meta, rogue, generic = build_meta(db, vocab)
    print(f"meta: K={meta.n_deck_types - 1} (+other), C={meta.n_cards} candidates, R={meta.n_roles} roles "
          f"({', '.join(f'{n}:{int(meta.roles[i].sum())}' for i, n in enumerate(meta.role_names))}), "
          f"{meta.n_hash} hash buckets, features {meta.feature_dim}", flush=True)  # fmt: skip
    data = collect(args, db, vocab, meta, rogue, swap_pool(db, generic))
    test, val = data["game"] % 5 == 0, data["game"] % 10 == 1  # test games; validation games for early stopping
    train_idx, val_idx, test_idx = np.nonzero(~test & ~val)[0], np.nonzero(val)[0], np.nonzero(test)[0]
    tr, te = targets_of(data, train_idx), targets_of(data, test_idx)
    base_rate = float(tr["responded"][tr["responded_mask"]].mean())
    priors = precompute_priors(data, meta)
    print(f"train {len(train_idx)} / validation {len(val_idx)} / test {len(test_idx)} samples; responded labels "
          f"{int(tr['responded_mask'].sum())} / {int(te['responded_mask'].sum())}, base rate {base_rate:.3f}", flush=True)  # fmt: skip

    hdt_test = hdt_probs(priors, test_idx, base_rate)
    test_batch = evaluation_batch(hdt_test, te)
    fit_batch = targets_batch(tr, meta)
    reports = {
        "uniform": evaluate_beliefs(uniform_predictor(test_batch), n_bins=args.bins),
        "prior": evaluate_beliefs(prior_predictor(test_batch, fit=fit_batch, smoothing=1.0), n_bins=args.bins),
        "hdt": evaluate_beliefs(test_batch, n_bins=args.bins),
    }
    del fit_batch
    runs = {}
    role_auc = {"hdt": per_role_auc(hdt_test["hand_roles"], te, meta)}
    net_cfg = NetConfig(vocab_size=len(vocab), d_model=64, n_heads=4, board_layers=1, history_layers=1, ff_mult=2,
                        history_mem_len=EVENT_WINDOW)  # fmt: skip
    models = {
        "heads": lambda: BeliefHeads(BeliefConfig.from_meta(meta)),
        "trunk+heads": lambda: BeliefPolicy(PolicyNet(net_cfg), BeliefHeads(BeliefConfig.from_meta(
            meta, context_dim=net_cfg.d_model, action_dim=net_cfg.d_model)), feed_policy=False),
    }  # fmt: skip
    for name, make in models.items():
        model = make()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"training {name} ({n_params:,} parameters) for {args.seconds:.0f}s", flush=True)
        runs[name] = train(model, data, priors, train_idx, val_idx, args) | {"parameters": n_params}
        print(f"  best validation loss {runs[name]['best_val']:.3f} at step {runs[name]['best_step']}", flush=True)
        probs = predict(model, data, priors, test_idx)
        reports[name] = evaluate_beliefs(evaluation_batch(probs, te), n_bins=args.bins)
        role_auc[name] = per_role_auc(probs["hand_roles"], te, meta)

    names = list(reports)
    keys = [k for k in reports["hdt"] if not k.endswith("/n")]
    lines = ["| 指标 | " + " | ".join(names) + " | n |", "|---|" + "---:|" * (len(names) + 1)]
    for key in keys:
        n = int(reports["hdt"][key.split("/")[0] + "/n"])
        lines.append(
            f"| {key} | " + " | ".join(f"{reports[m].get(key, float('nan')):.3f}" for m in names) + f" | {n} |"
        )
    table = "\n".join(lines)
    print(table)
    info = {"args": {k: str(v) for k, v in vars(args).items()},
            "samples": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
            "meta": {"names": meta.names, "shares": meta.shares.round(4).tolist(), "n_cards": meta.n_cards,
                     "roles": {n: int(meta.roles[i].sum()) for i, n in enumerate(meta.role_names)}},
            "responded_base_rate": base_rate, "runs": runs, "reports": reports, "role_auc": role_auc}  # fmt: skip
    print("per-role AUC:", json.dumps(role_auc))
    (out_dir / "report.json").write_text(json.dumps(info, indent=1))
    (out_dir / "report.md").write_text(table + "\n")
    print(f"wrote {out_dir}/report.json, report.md")


if __name__ == "__main__":
    main()
