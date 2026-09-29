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
played, the edits accepted, games per accepted edit, and every accepted edit's validated difference.

Usage: tools/deckevo_eval_compare.py CHECKPOINT N_DECKS OUT.json [--env md-2026-09] [--device cuda] [--arms mvp,evo]
       [--candidates 64] [--first-pairs 25] [--finalists 3] [--validation-pairs 200] [--informed 6] [--explore 2] [--batch 25] [--max-pairs 600]
       [--look 100] [--cap 1000] [--control-k 0] [--envs 256] [--seed 0]"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ygorl.build.control import CriticControlVariate, critic_opening_values
from ygorl.build.selection import sequential_validate, top_two_thompson
from ygorl.build.signals import CardValueModel, informed_children, protected_cards
from ygorl.build.tuner import PairedEvaluator, neighbors, paired_difference, successive_halving, tech_pool, validate
from ygorl.cards.ydk import load_ydk
from ygorl.data.environment import load_environment
from ygorl.engine.duel import default_cards
from ygorl.env.encoded import EncodedVecEnv
from ygorl.train.checkpoint import load_actor, load_actor_critic, torch_device


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
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    arms = args.arms.split(",")
    device = torch_device(args.device)
    cards = default_cards()
    env = load_environment(args.env, cards=cards)
    corpus = json.loads(env.artifact_path("deck_corpus.json").read_text())
    lists = [(e["type"], e["file"], load_ydk(env.artifacts_dir / e["file"])) for e in corpus["decks"]]
    train = {p.name for p in Path("out/corpus/train").glob("*.ydk")}
    bases = [t for t in lists if Path(t[1]).name in train and not env.validate_deck(t[2], cards)]
    picks = [bases[i] for i in np.random.default_rng(args.seed).choice(len(bases), args.n_decks, replace=False)]
    meta = [m.deck for m in env.meta_decks]
    weights = [m.share for m in env.meta_decks]
    pol = load_actor(args.checkpoint)
    net = pol.net.to(device)

    def is_extra(pw):
        return cards[pw].is_extra_deck if pw in cards else False

    def legal(d):
        return not env.validate_deck(d, cards)

    control = None
    if args.control_k:
        ac = load_actor_critic(args.checkpoint)
        critic = ac.model.to(device)

        def critic_env(n):
            return EncodedVecEnv(n, 8, cards=cards, vocab=ac.vocab, event_length=ac.event_length, privileged=True,
                                 skip_forced=True)  # fmt: skip

        control = CriticControlVariate(lambda sp: critic_opening_values(critic, critic_env, sp, device=device,
                                                                        num_envs=args.envs), k=args.control_k)  # fmt: skip

    def evaluator(seed, control=None):
        return PairedEvaluator(
            env_factory=lambda n: EncodedVecEnv(n, 8, cards=cards, vocab=pol.vocab, event_length=pol.event_length,
                                                skip_forced=True),
            policy=net, opponents=meta, weights=weights, seed=seed, device=args.device, num_envs=args.envs,
            control=control)  # fmt: skip

    res = {"checkpoint": args.checkpoint, "environment": env.version, "args": vars(args), "parents": []}
    for di, (dtype, file, base) in enumerate(picks):
        same = [d for t, _, d in lists if t == dtype and d is not base]
        pool = tech_pool(base, same, meta)
        row = {"type": dtype, "file": file}
        if "mvp" in arms:
            ev = evaluator(1000 + di)
            edits = neighbors(base, pool, is_extra, legal, limit=args.candidates, rng=np.random.default_rng(1000 + di))
            base_c, finals = successive_halving(base, edits, ev, first_pairs=args.first_pairs,
                                                finalists=args.finalists)  # fmt: skip
            checked = validate(base, finals, ev, args.validation_pairs) if finals else []
            fresh = sorted((paired_difference(f, b) for _, f, b in checked), reverse=True)
            best = fresh[0] if fresh else (0.0, -np.inf, np.inf)
            row["mvp"] = {"candidates": len(edits), "games": ev.games, "errors": ev.errors, "seconds": ev.seconds,
                          "accepted": bool(best[1] > 0), "diff": best[0], "ci": [best[1], best[2]]}  # fmt: skip
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
                out.update({"chosen": [e.describe() for e in children[race.best].edits],
                            "kind": children[race.best].kind, "decision": v.decision, "validation_pairs": v.pairs,
                            "accepted": v.accepted, "diff": last.mean if last else None,
                            "ci": [last.lower, last.upper] if last else None})  # fmt: skip
            else:
                out["accepted"] = False
            out.update({"search_games": search_games, "games": ev.games, "errors": ev.errors, "seconds": ev.seconds})
            row["evo"] = out
        res["parents"].append(row)
        print(f"{dtype:28s} " + "  ".join(f"{a}: {row[a]['games']} games, accepted {row[a]['accepted']}"
                                          for a in arms if a in row), flush=True)  # fmt: skip
        Path(args.out).write_text(json.dumps(res, indent=1, default=float))
    summary = {}
    for a in arms:
        games = sum(p[a]["games"] for p in res["parents"])
        acc = [p[a] for p in res["parents"] if p[a]["accepted"]]
        summary[a] = {"games": games, "accepted": len(acc), "games_per_accept": games / len(acc) if acc else None,
                      "accepted_diffs": [p["diff"] for p in acc]}  # fmt: skip
    res["summary"] = summary
    print("summary:", json.dumps(summary))
    Path(args.out).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
