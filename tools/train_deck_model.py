"""Train and evaluate the masked deck model (#150, ygorl.build.deck_model; docs/tuning.md「掩码卡组模型」).

  uv run python tools/train_deck_model.py --env md-2026-09 --out out/deckmodel/split.pt --holdout 0.1 --device cuda
  uv run python tools/train_deck_model.py --env md-2026-09 --out out/deckmodel/full.pt --holdout 0 --device cuda
  uv run python tools/train_deck_model.py --env md-2026-09 --load out/deckmodel/split.pt --holdout 0.1   # evaluate only

Data: every distinct list of the masterduelmeta history whose cards all map, legal now or not: the deck dataset
(``--data``, ``tools/build_deck_dataset.py``), else the raw history (``--raw-dir``, ``tools/build_deck_corpus.py
--fetch``). ``--holdout S`` holds out the lists of a hashed share S of the deck types
(whole types: unseen archetypes); 0 trains on everything (the model the evolution step uses). The vocab is the
environment's card pool plus every card of the lists; the frozen card-text table comes from ``--text-dir``
(``tools/build_text_embeddings.py``).

Evaluation (written to ``--results``, default: next to the model as ``.json``):

- **fill-in** on the held-out types: one random distinct card of each held-out list masked (every copy), its rank
  among the cards the rest of the list lacks; top-k accuracy and MRR of the model, of the type-free card frequency
  (lists of the training set running the card) and, on a ``--rules-sample`` subsample, of the co-occurrence rules
  (ygorl.build.rules over the training lists, archetype strata; cards no rule proposes follow by frequency);
- **known edits** (docs/spikes/deck-evolution.md「真实轮次」): ranks of the known good and bad additions among the cards
  the deck lacks (1 = most wanted) and of the known bad removals among the deck's cards (1 = most out of place),
  for the model, the frequency and the rules, plus the model's top additions and removals.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# (deck file stem, deck type, kind, card, expected): docs/spikes/deck-evolution.md「真实轮次」
KNOWN_EDITS = [
    ("megalith-3", "Megalith", "add", 29876299, "good"),  # Megalith Anastasis: +3.5 pp on 1,000 fresh pairs
    ("therion", "Therion", "add", 73304257, "good"),  # Alpha, the Master of Beasts: +2.2 pp
    ("therion", "Therion", "add", 71734607, "bad"),  # Rikka Petal: another engine
    ("megalith-3", "Megalith", "add", 96026108, "bad"),  # Drytron Zeta Aldhibah: another engine
    ("megalith-3", "Megalith", "remove", 96729612, "bad"),  # Preparation of Rites: engine core
    ("therion", "Therion", "remove", 97526666, "bad"),  # Planet Pathfinder: engine core
    # the removals of the two good edits above (tools/deckevo_eval_compare.py, MVP arm)
    ("megalith-3", "Megalith", "remove", 25726386, "good"),  # Megalith Aratron (out for Anastasis)
    ("therion", "Therion", "remove", 48806195, "good"),  # Therion "Empress" Alasia (out for Alpha)
]
TOP_DECKS = ("therion", "megalith-3", "pendulum-magician-3")  # decks whose top additions and removals are reported
REMOVALS = ("typicality", "support", "combined")  # DeckModel.removal_scores kinds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", required=True)
    ap.add_argument("--data", type=Path, default=None,
                    help="deck dataset directory (default: out/deck_dataset/<env version>, else --raw-dir)")  # fmt: skip
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "out" / "deck_corpus" / "raw")
    ap.add_argument("--text-dir", type=Path, default=ROOT / "out" / "views" / "both")
    ap.add_argument("--out", type=Path, default=None, help="model file to write")
    ap.add_argument("--load", type=Path, default=None, help="evaluate this model instead of training one")
    ap.add_argument("--results", type=Path, default=None)
    ap.add_argument("--holdout", type=float, default=0.1, help="share of deck types held out (0: train on all)")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ff", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--mask-rate", type=float, default=0.15)
    ap.add_argument("--copy-rate", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--rules-sample", type=int, default=1000, help="held-out tasks the rule baseline is scored on")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if args.load is None and args.out is None:
        ap.error("give --out (train) or --load (evaluate)")

    import torch

    from ygorl.build.deck_model import (DeckModel, DeckModelConfig, fill_in_tasks, history_lists, load_deck_model,
                                        load_text_table, ranks, split_by_type, topk, train_deck_model)  # fmt: skip
    from ygorl.build.packages import setcodes_from_db
    from ygorl.build.rules import DeckRules
    from ygorl.cards.ydk import Deck, load_ydk
    from ygorl.data.environment import load_environment
    from ygorl.engine.duel import default_cards

    def say(msg):
        print(msg, flush=True)

    t0 = time.time()
    cards = default_cards()
    env = load_environment(args.env, cards=cards)
    data = args.data or ROOT / "out" / "deck_dataset" / env.version
    lists = history_lists(data if (data / "decks.npz").is_file() else args.raw_dir, cards)
    train, test = split_by_type(lists, args.holdout, args.split_seed) if args.holdout > 0 else (lists, [])
    say(f"{len(lists)} distinct lists of {len({t for t, _ in lists})} types; train {len(train)} lists, held out "
        f"{len(test)} lists of {len({t for t, _ in test})} types ({time.time() - t0:.0f} s)")  # fmt: skip

    if args.load is not None:
        model = load_deck_model(args.load, args.device)
    else:
        vocab = sorted(set(env.card_pool) | {c for _, d in lists for c in d.counts()})
        text = load_text_table(args.text_dir, vocab)
        say(f"vocab {len(vocab)} cards, {int((np.abs(text).sum(1) > 0).sum())} with text ({args.text_dir})")
        cfg = DeckModelConfig(dim=args.dim, layers=args.layers, heads=args.heads, ff=args.ff, dropout=args.dropout,
                              mask_rate=args.mask_rate, copy_rate=args.copy_rate)  # fmt: skip
        model = DeckModel(vocab, text, cfg, environment=env.stamp())
        say(f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
        watch = fill_in_tasks(test, model.index, seed=1)[:2000]

        def on_epoch(epoch, m):
            if watch and (epoch + 1) % 5 == 0:
                lp, _ = m.fill_in([d for d, _ in watch], [c for _, c in watch])
                r = ranks(lp, np.array([m.index[c] for _, c in watch]), _present(m, watch))
                say(f"  held-out fill-in: {_fmt(topk(r))}")
                m.train()

        train_deck_model(model, train, epochs=args.epochs, batch=args.batch, lr=args.lr,
                         weight_decay=args.weight_decay, device=args.device, seed=args.seed, log=say,
                         on_epoch=on_epoch)  # fmt: skip
        model.save(args.out, meta={"lists": len(train), "types": len({t for t, _ in train}), "holdout": args.holdout,
                                   "split_seed": args.split_seed, "text_dir": str(args.text_dir),
                                   "epochs": args.epochs, "trained": time.strftime("%Y-%m-%d"),
                                   "train_seconds": round(time.time() - t0)})  # fmt: skip
        say(f"wrote {args.out} ({time.time() - t0:.0f} s)")
    model.to("cpu")
    torch.set_num_threads(8)

    results: dict = {"model": str(args.load or args.out), "meta": getattr(model, "meta", None),
                     "lists": len(lists), "train_lists": len(train), "test_lists": len(test),
                     "test_types": len({t for t, _ in test})}  # fmt: skip
    freq = np.zeros(len(model.passwords))
    for _, d in train:
        for c in d.counts():
            if c in model.index:
                freq[model.index[c]] += 1
    freq_score = freq - 1e-6 * np.arange(len(freq))  # ties: by password
    say("mining the rules over the training lists")
    rules = DeckRules(train, setcodes=setcodes_from_db(cards))

    def rule_score(deck: Deck, deck_type: str | None) -> np.ndarray:
        s = freq_score / (freq.max() + 1)  # in [0, 1): the fallback order
        for c, p in rules.additions(deck, deck_type).items():
            if c in model.index:
                s[model.index[c]] = 1000 + p.weight + 1e-3 * p.rule.lift
        return s

    if test:
        tasks = fill_in_tasks(test, model.index, seed=args.seed)
        target = np.array([model.index[c] for _, c in tasks])
        present = _present(model, tasks)
        lp, copies = model.fill_in([d for d, _ in tasks], [c for _, c in tasks])
        r_model = ranks(lp, target, present)
        r_freq = ranks(np.broadcast_to(freq_score, lp.shape), target, present)
        true_copies = np.array([min(d.counts()[c], 3) for d, c in tasks])
        results["fill_in"] = {"all": {"model": topk(r_model), "frequency": topk(r_freq),
                                      "model_copies_accuracy": float((copies == true_copies).mean())}}  # fmt: skip
        sub = np.random.default_rng(args.seed).permutation(len(tasks))[: args.rules_sample]
        if len(sub):  # the rules take seconds a task (every antecedent pair of the list)
            t1 = time.time()
            rule_rows = np.stack([rule_score(_without(tasks[i][0], tasks[i][1]), None) for i in sub])
            say(f"rules on {len(sub)} tasks: {time.time() - t1:.0f} s")
            r_rules = ranks(rule_rows, target[sub], [present[i] for i in sub])
            proposed = float(np.mean([rule_rows[k, target[i]] >= 1000 for k, i in enumerate(sub)]))
            results["fill_in"]["rules_sample"] = {"model": topk(r_model[sub]), "frequency": topk(r_freq[sub]),
                                                  "rules": topk(r_rules), "rules_proposed": proposed}  # fmt: skip
        for k, v in results["fill_in"].items():
            say(f"fill-in ({k}): " + "; ".join(f"{m} {_fmt(x)}" for m, x in v.items() if isinstance(x, dict)))

    known = []
    names = {pw: c.name for pw, c in cards.items()}
    decks: dict[str, Deck] = {s: load_ydk(env.artifact_path("decks", f"{s}.ydk")) for s in TOP_DECKS}
    cache: dict[tuple, dict] = {}

    def removal(stem, deck, kind):
        if (stem, kind) not in cache:
            cache[stem, kind] = model.removal_scores(deck, kind)
        return cache[stem, kind]

    for stem, deck_type, kind, card, expected in KNOWN_EDITS:
        path = env.artifact_path("decks", f"{stem}.ydk")
        deck = decks.setdefault(stem, load_ydk(path))
        have = deck.counts()
        in_train = any(t == deck_type for t, _ in train)
        if kind == "add":
            cand = [c for c in env.card_pool if c not in have and c in model.index]
            add = model.addition_scores(deck, cand)
            fs = {c: freq_score[model.index[c]] for c in cand}
            rs_row = rule_score(deck, deck_type if in_train else None)
            rs = {c: rs_row[model.index[c]] for c in cand}
            row = {m: _rank(s, card) for m, s in (("model", add), ("frequency", fs), ("rules", rs))}
            row["of"] = len(cand)
        else:
            fs = {c: -freq_score[model.index[c]] for c in have}  # rarest first
            row = {k: _rank(removal(stem, deck, k), card) for k in REMOVALS}
            row |= {"frequency": _rank(fs, card), "of": len(have)}
        known.append({"deck": stem, "type": deck_type, "type_in_training": in_train, "kind": kind, "card": card,
                      "name": names.get(card, str(card)), "expected": expected, "ranks": row})  # fmt: skip
        say(f"{stem} {'+' if kind == 'add' else '-'}{names.get(card, card)} ({expected}): "
            + ", ".join(f"{k} {v}" for k, v in row.items()))  # fmt: skip
    results["known_edits"] = known
    results["top"] = {}
    for stem, deck in decks.items():
        add = model.addition_scores(deck, [c for c in env.card_pool if c not in deck.counts()])
        results["top"][stem] = {
            "additions": [[names.get(c, str(c)), round(v, 3)] for c, v in sorted(add.items(), key=lambda x: -x[1])[:10]],
            **{f"removals_{k}": [[names.get(c, str(c)), round(v, 3)] for c, v in
                                 sorted(removal(stem, deck, k).items(), key=lambda x: -x[1])[:10]] for k in REMOVALS},
        }  # fmt: skip
        say(f"{stem}: top additions {[n for n, _ in results['top'][stem]['additions'][:5]]}")
        for k in REMOVALS:
            say(f"  first removals ({k}): {[n for n, _ in results['top'][stem][f'removals_{k}'][:5]]}")
    out = args.results or (args.load or args.out).with_suffix(".json")
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    say(f"wrote {out} ({time.time() - t0:.0f} s)")
    return 0


def _present(model, tasks) -> list[list[int]]:
    return [[model.index[x] for x in d.counts() if x != c and x in model.index] for d, c in tasks]


def _without(deck, card):
    from ygorl.cards.ydk import Deck

    return Deck(main=tuple(c for c in deck.main if c != card), extra=tuple(c for c in deck.extra if c != card))


def _rank(scores: dict, card: int) -> int | None:
    """1 + the cards scored at least as high as ``card`` (ties against it); None when ``card`` is not scored."""
    if card not in scores:
        return None
    s = scores[card]
    return 1 + int(sum(v >= s for c, v in scores.items() if c != card))


def _fmt(t: dict) -> str:
    return " ".join(f"{k} {v:.3f}" if k != "n" else f"n {v}" for k, v in t.items())


if __name__ == "__main__":
    sys.exit(main())
