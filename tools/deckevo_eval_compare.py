"""Games per accepted edit (#110, docs/spikes/deck-evolution.md 结论): the deck-tuning MVP against the evolution
evaluator, on the same parent decks and the same policy.

- **mvp** (docs/tuning.md): ``--candidates`` random legal one-card swaps, successive halving from ``--first-pairs``
  down to ``--finalists``, then a fixed ``--validation-pairs`` fresh pairs; accepted when the best finalist's 95%
  interval is above 0.
- **evo**: ``informed_children`` (6 informed + 2 explore, from an empty CardValueModel unless the run carries one:
  its prior is the model's predicted gain), top-two Thompson sampling in batches of ``--batch`` pairs up to
  ``--max-pairs``, then sequential validation of the chosen child on fresh pairs (a look every ``--look`` pairs, at
  most ``--cap``, O'Brien-Fleming alpha spending); ``--control-k K`` adds the critic control variate (K alternative
  shuffles; needs a PPO checkpoint).

Parents are ``N_DECKS`` legal training-split corpus lists (drawn with ``--seed``). Both arms play against the
environment's meta decks by share with the same pilot, each on its own evaluator. Reports, per arm, the games
played, the edits accepted, games per accepted edit, and every accepted edit's validated difference; the edits
themselves are logged by card (#145): per MVP finalist its edit and fresh-pair difference with its standard error,
for the evolution arm the chosen child's edits and its validation's standard error (``ygorl.build.warmstart`` reads
both).

**Re-validation** (``--revalidate N``, #145): every accepted edit is played again, parent and child, on ``N`` fresh
pairs (new evaluator seed, pair indices from 2,000,000: never used by a search or a validation), which measures the
winner's curse (validated minus re-validated difference). ``--from PREV.json`` re-validates the accepted edits of an
earlier run of this tool (arms not replayed; files written before the edits were logged have no MVP edits), and
``--evo-state DIR`` (repeatable) the accepted children of an evolution state (``tools/evolve_decks.py``: manifest
entries with their corpus parent).

Usage: tools/deckevo_eval_compare.py CHECKPOINT N_DECKS OUT.json [--env md-2026-09] [--device cuda] [--arms mvp,evo]
       [--candidates 64] [--first-pairs 25] [--finalists 3] [--validation-pairs 200] [--informed 6] [--explore 2] [--batch 25] [--max-pairs 600]
       [--look 100] [--cap 1000] [--control-k 0] [--control-beta 0.5] [--envs 256] [--seed 0]
       [--revalidate N] [--from PREV.json] [--evo-state DIR ...]"""

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ygorl.build.control import CriticControlVariate, critic_opening_values
from ygorl.build.selection import sequential_validate, top_two_thompson
from ygorl.build.signals import CardValueModel, informed_children, protected_cards
from ygorl.build.tuner import (Edit, PairedEvaluator, apply, neighbors, paired_difference, successive_halving,
                               tech_pool, validate)  # fmt: skip
from ygorl.build.warmstart import FUTILITY_Z, parse_edit
from ygorl.cards.ydk import load_ydk
from ygorl.data.environment import load_environment
from ygorl.engine.duel import default_cards
from ygorl.env.encoded import EncodedVecEnv
from ygorl.train.checkpoint import load_actor, load_actor_critic, torch_device

REVALIDATION_OFFSET = 2_000_000  # first pair index of re-validation: past every search (0..) and validation (1e6..)


def fresh(child, base) -> dict:
    """Mean paired difference over fresh pairs with its standard error and normal 95% interval."""
    mean, lo, hi = paired_difference(child, base)
    d = np.asarray(child, dtype=float) - np.asarray(base, dtype=float)
    d = d[np.isfinite(d)]
    se = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else None
    return {"pairs": int(len(d)), "diff": mean, "stderr": se, "ci": [lo, hi]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("n_decks", type=int)
    ap.add_argument("out")
    ap.add_argument("--env", default="md-2026-09")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--arms", default="mvp,evo")
    ap.add_argument("--candidates", type=int, default=64)
    ap.add_argument("--first-pairs", type=int, default=25)
    ap.add_argument("--finalists", type=int, default=3)
    ap.add_argument("--validation-pairs", type=int, default=200)
    ap.add_argument("--informed", type=int, default=6)
    ap.add_argument("--explore", type=int, default=2)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--max-pairs", type=int, default=600)
    ap.add_argument("--look", type=int, default=100)
    ap.add_argument("--cap", type=int, default=1000)
    ap.add_argument("--control-k", type=int, default=0, help="alternative shuffles of the control variate (0: off)")
    ap.add_argument(
        "--control-beta",
        type=float,
        default=0.5,
        help="control variate coefficient; take it from M5's held-out fit, never from these games",
    )
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--revalidate", type=int, default=0, help="fresh pairs to replay every accepted edit on (0: off)")
    ap.add_argument("--from", dest="prev", type=Path, default=None,
                    help="re-validate the accepted edits of this earlier result instead of running the arms")  # fmt: skip
    ap.add_argument("--evo-state", type=Path, action="append", default=[],
                    help="re-validate the accepted children of this evolution state too (repeatable)")  # fmt: skip
    args = ap.parse_args()
    arms = [a for a in args.arms.split(",") if a]
    device = torch_device(args.device)
    cards = default_cards()
    env = load_environment(args.env, cards=cards)
    names = {pw: c.name for pw, c in cards.items()}
    corpus = json.loads(env.artifact_path("deck_corpus.json").read_text())
    lists = [(e["type"], e["file"], load_ydk(env.artifacts_dir / e["file"])) for e in corpus["decks"]]
    meta = [m.deck for m in env.meta_decks]
    weights = [m.share for m in env.meta_decks]
    pol = load_actor(args.checkpoint)
    net = pol.net.to(device)

    def is_extra(pw):
        return cards[pw].is_extra_deck if pw in cards else False

    def legal(d):
        return not env.validate_deck(d, cards)

    def named(ids):
        out = []
        for text in ids:
            section, o, i = parse_edit(text)
            out.append(Edit(o, i, section).describe(names))
        return out

    control = None
    if args.control_k:
        ac = load_actor_critic(args.checkpoint)
        critic = ac.model.to(device)

        def critic_env(n):
            return EncodedVecEnv(n, 8, cards=cards, vocab=ac.vocab, event_length=ac.event_length, privileged=True,
                                 skip_forced=True)  # fmt: skip

        control = CriticControlVariate(lambda sp: critic_opening_values(critic, critic_env, sp, device=device,
                                                                        num_envs=args.envs), k=args.control_k,
                                       beta=args.control_beta)  # fmt: skip

    def evaluator(seed, control=None):
        return PairedEvaluator(
            env_factory=lambda n: EncodedVecEnv(n, 8, cards=cards, vocab=pol.vocab, event_length=pol.event_length,
                                                skip_forced=True),
            policy=net, opponents=meta, weights=weights, seed=seed, device=args.device, num_envs=args.envs,
            control=control)  # fmt: skip

    if args.prev is not None:
        res = json.loads(args.prev.read_text())
        res["revalidated_from"] = str(args.prev)
        arms = [a for a in ("mvp", "evo") if any(a in p for p in res["parents"])]
        picks = [(p["type"], p["file"], load_ydk(env.artifacts_dir / p["file"])) for p in res["parents"]]
    else:
        train = {p.name for p in Path("out/corpus/train").glob("*.ydk")}
        bases = [t for t in lists if Path(t[1]).name in train and not env.validate_deck(t[2], cards)]
        picks = [bases[i] for i in np.random.default_rng(args.seed).choice(len(bases), args.n_decks, replace=False)]
        res = {"checkpoint": args.checkpoint, "environment": env.version, "parents": []}
    res["args"] = {
        k: [str(x) for x in v] if isinstance(v, list) else str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()
    } | ({"previous": res["args"]} if args.prev is not None and "args" in res else {})
    for di, (dtype, file, base) in enumerate(picks if args.prev is None else []):
        same = [d for t, _, d in lists if t == dtype and d is not base]
        pool = tech_pool(base, same, meta)
        row = {"type": dtype, "file": file}
        if "mvp" in arms:
            ev = evaluator(1000 + di)
            edits = neighbors(base, pool, is_extra, legal, limit=args.candidates, rng=np.random.default_rng(1000 + di))
            base_c, finals = successive_halving(base, edits, ev, first_pairs=args.first_pairs,
                                                finalists=args.finalists)  # fmt: skip
            checked = validate(base, finals, ev, args.validation_pairs) if finals else []
            fins = sorted(({"edit": f.edit.describe(), "named": f.edit.describe(names), **fresh(s, b)}
                           for f, s, b in checked), key=lambda x: -x["diff"])  # fmt: skip
            best = fins[0] if fins else {"diff": 0.0, "ci": [-np.inf, np.inf]}
            accepted = bool(best["ci"][0] > 0)
            row["mvp"] = {"candidates": len(edits), "games": ev.games, "errors": ev.errors, "seconds": ev.seconds,
                          "accepted": accepted, "diff": best["diff"], "ci": best["ci"], "finalists": fins,
                          "edits": [best["edit"]] if accepted else [],
                          "accepted_edits": [best["named"]] if accepted else []}  # fmt: skip
        if "evo" in arms:
            ev = evaluator(2000 + di, control)
            model = CardValueModel()
            children = informed_children(base, dtype, model, pool, legal=legal, is_extra=is_extra,
                                         rng=np.random.default_rng(3000 + di), protected=protected_cards(base),
                                         informed=args.informed, explore=args.explore)  # fmt: skip
            prior = [model.gain([e.into for e in c.edits], [e.out for e in c.edits], dtype) for c in children]
            race = top_two_thompson(base, [c.deck for c in children], ev, prior=prior, batch=args.batch,
                                    max_pairs=args.max_pairs, rng=np.random.default_rng(4000 + di))  # fmt: skip
            search_games = ev.games
            out = {"children": len(children), "search_pairs": race.pairs, "stop": race.stop,
                   "sd_pair": race.sd_pair, "arms": [asdict(a) for a in race.arms]}  # fmt: skip
            if race.best is not None:
                v = sequential_validate(base, children[race.best].deck, ev, look=args.look, cap=args.cap)
                last = v.looks[-1] if v.looks else None
                chosen = [e.describe() for e in children[race.best].edits]
                out.update({"chosen": chosen, "chosen_named": named(chosen), "kind": children[race.best].kind,
                            "decision": v.decision, "validation_pairs": v.pairs, "accepted": v.accepted,
                            "diff": last.mean if last else None, "ci": [last.lower, last.upper] if last else None,
                            "stderr": (last.upper - last.mean) / FUTILITY_Z if last else None,
                            "edits": chosen if v.accepted else [],
                            "accepted_edits": named(chosen) if v.accepted else []})  # fmt: skip
            else:
                out.update({"accepted": False, "edits": [], "accepted_edits": []})
            out.update({"search_games": search_games, "games": ev.games, "errors": ev.errors, "seconds": ev.seconds})
            row["evo"] = out
        res["parents"].append(row)
        print(f"{dtype:28s} " + "  ".join(f"{a}: {row[a]['games']} games, accepted {row[a]['accepted']}"
                                          + (f" ({'; '.join(row[a]['accepted_edits'])})" if row[a]["accepted"] else "")
                                          for a in arms if a in row), flush=True)  # fmt: skip
        Path(args.out).write_text(json.dumps(res, indent=1, default=float))

    def replay(base, child, seed):
        ev = evaluator(seed)
        s = ev.scores([child, base], range(REVALIDATION_OFFSET, REVALIDATION_OFFSET + args.revalidate))
        return {**fresh(s[0], s[1]), "games": ev.games, "errors": ev.errors}

    if args.revalidate:
        for di, ((dtype, _, base), row) in enumerate(zip(picks, res["parents"], strict=True)):
            for k, a in enumerate(arms):
                arm = row.get(a) or {}
                if not arm.get("accepted"):
                    continue
                if not arm.get("edits"):  # a result written before the edits were logged
                    arm["revalidation"] = {"note": "accepted edit not logged"}
                    print(f"{dtype:28s} {a}: accepted edit not logged, not re-validated", flush=True)
                    continue
                child = base
                for text in arm["edits"]:
                    section, o, i = parse_edit(text)
                    child = apply(child, Edit(o, i, section))
                r = arm["revalidation"] = replay(base, child, 5000 + 100 * k + di)
                print(f"{dtype:28s} {a} re-validated on {r['pairs']} pairs: {r['diff']:+.4f} ± {r['stderr']:.4f} "
                      f"(validated {arm['diff']:+.4f}): {'; '.join(named(arm['edits']))}", flush=True)  # fmt: skip
        evolved = []
        for state in args.evo_state:
            manifest = json.loads((state / "manifest.json").read_text())
            by_id = {e["id"]: e for e in manifest["decks"]}
            for e in manifest["decks"]:
                if not e.get("lineage"):  # not an evolution step's accepted child
                    continue
                parent = e["parent"]
                if parent.startswith("corpus:"):
                    pdeck = load_ydk(env.artifacts_dir / "decks" / f"{parent.split(':', 1)[1]}.ydk")
                elif parent.split(":", 1)[-1] in by_id:  # a manifest deck, by id or as a parent file (file:<id>)
                    pdeck = load_ydk(state / by_id[parent.split(":", 1)[-1]]["file"])
                else:
                    print(f"{state}: {e['id']}: parent {parent} not found, skipped", flush=True)
                    continue
                edits = [Edit(x["out"], x["into"], x["section"]).describe() for x in e["edits"]]
                r = replay(pdeck, load_ydk(state / e["file"]), 7000 + len(evolved))
                evolved.append({"state": str(state), "id": e["id"], "parent": parent, "type": e.get("type"),
                                "edits": named(edits), "diff": e.get("diff"), "revalidation": r})  # fmt: skip
                print(f"{e['id']:28s} evolution re-validated on {r['pairs']} pairs: {r['diff']:+.4f} ± "
                      f"{r['stderr']:.4f} (validated {e.get('diff')}): {'; '.join(named(edits))}", flush=True)  # fmt: skip
        if args.evo_state:
            res["evolution"] = evolved
    summary = {}
    for a in arms:
        rows = [p for p in res["parents"] if a in p]
        games = sum(p[a]["games"] for p in rows)
        acc = [(p, p[a]) for p in rows if p[a]["accepted"]]
        summary[a] = {"games": games, "accepted": len(acc), "games_per_accept": games / len(acc) if acc else None,
                      "accepted_diffs": [x["diff"] for _, x in acc],
                      "accepted_edits": [{"type": p["type"], "file": p["file"], "edits": x.get("accepted_edits"),
                                          "diff": x["diff"]} for p, x in acc]}  # fmt: skip
        again = [x for _, x in acc if (x.get("revalidation") or {}).get("diff") is not None]
        if again:
            summary[a]["revalidated"] = [x["revalidation"]["diff"] for x in again]
            summary[a]["winners_curse"] = float(np.mean([x["diff"] - x["revalidation"]["diff"] for x in again]))
    if res.get("evolution"):
        rows = [r for r in res["evolution"] if r["diff"] is not None]
        summary["evolution"] = {"accepted": len(res["evolution"]),
                                "revalidated": [r["revalidation"]["diff"] for r in res["evolution"]],
                                "winners_curse": float(np.mean([r["diff"] - r["revalidation"]["diff"] for r in rows]))
                                if rows else None}  # fmt: skip
    res["summary"] = summary
    print("summary:", json.dumps(summary, default=float))
    Path(args.out).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
