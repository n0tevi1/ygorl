"""One round of deck evolution (#111, ygorl.build.evolve; docs/tuning.md「进化步骤」).

  uv run python tools/evolve_decks.py --env md-2026-09 --checkpoint CKPT --state out/evo/NAME \\
      [--parent BASE.ydk ...] [--parents 1] [--manifest MANIFEST] [--matrix MATRIX.json] [--nash-share 0.5] \\
      [--budget GAMES] [--rules 0] [--learned-generator 0 --deck-model MODEL.pt] [--value-model VM.pt] \\
      [--warm-start PATH ...] [--device cuda] [--envs 256]

Parents are the ``--parent`` deck files, or else the ``--parents`` pool decks of the manifest used least often as
parents. Per parent: ``informed_children`` from the signal library (6 informed + 2 explore; ``--crossover`` adds
crossover children of the parent and an archive elite, ``ygorl.build.crossover``), top-two Thompson
sampling (``--batch`` pairs a batch, at most ``--max-pairs`` child pairs), sequential validation of the chosen child
on fresh pairs (a look every ``--look`` pairs, at most ``--cap``); the accepted child enters the manifest as a
probation deck. ``--rules N`` adds N children whose additions come from association rules over the environment's
deck corpus (ygorl.build.rules; off by default). ``--learned-generator N --deck-model MODEL.pt`` adds N children
whose additions and removals the masked deck model scores (ygorl.build.learned, #150; ``tools/train_deck_model.py``;
off by default): additions from the environment's whole card pool, removals by ``--learned-removal`` (default
combined: atypical cards first, the cards the rest relies on last), no engine protection; protected cards stay. Every evaluated child's paired difference feeds the card-value
model and the calibration table, every evaluated deck is offered to the MAP-Elites archive, and every child gets a lineage record.

Engine-aware (#145, docs/tuning.md「引擎感知的候选」): additions come from the addition pool (generic cards of
``--generic-pool`` and cards connected to the parent in the synergy graph, fanout <= ``--addition-fanout``;
``--pool tech`` restores the old tech pool); the parent's engine members (in-deck edges, fanout <=
``--engine-fanout``) are protected until the model has ``--engine-evidence`` observations of them. While the model
has fewer than ``--cold-min-obs`` observations of the parent's type, a round screens ``--cold-candidates`` single
swaps at one batch each and races the best ``--cold-keep``. ``--warm-start PATH`` (repeatable: M1 / M2 JSON, an
evolution state or lineage, a tuner comparison; ygorl.build.warmstart) adds earlier paired data to the model first,
each observation once.

``--value-model VM.pt`` (#151, ``tools/fit_value_model.py``; needs the deck model it was fitted with, by default
the one it records) predicts every child's win-rate change: the prediction is mixed into the Thompson prior and ranks
``--value-model-oversample`` times as many learned candidates as are kept, by its calibrated weight (Spearman of its
predictions with the first batches once there are 20; ``--value-model-weight`` before; 0 = card-value model alone).

``--eval factorial`` (#152, docs/tuning.md「析因评估」) replaces the children and the race: ``--factorial-k`` single
edits (informed first, then explore) are played as one fractional factorial design (resolution IV where possible) on
``--factorial-pairs`` pairs; the edits with a positive main effect together are validated; each main effect feeds the
card-value model as a single-edit observation, and the design with its effect estimates goes into the lineage.

Opponents: the environment's meta decks by share, mixed with the Nash weights of a deck matchup matrix
(``--matrix``, a ``ygorl-matchup`` file whose decks are meta decks or manifest decks) by ``--nash-share``.
The checkpoint, the matrix and the state (manifest, archive, signal library) must all belong to ``--env``.

The state directory holds everything (``ygorl.build.evolve``); a killed run resumes where it stopped when rerun with
the same ``--state`` (the unfinished round keeps its parents, settings and opponents). Writes the round report to
``STATE/rounds/NNNN/report.json`` and ``report.txt`` and prints the text.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--state", type=Path, required=True, help="evolution state directory")
    ap.add_argument("--manifest", type=Path, default=None, help="deck-pool manifest (default: STATE/manifest.json)")
    ap.add_argument("--parent", type=Path, action="append", default=[], help="parent deck file (repeatable)")
    ap.add_argument("--parents", type=int, default=1, help="pool decks to evolve when no --parent is given")
    ap.add_argument("--opponent-checkpoint", default=None, help="pilot of the opponents (default: --checkpoint)")
    ap.add_argument("--matrix", type=Path, default=None, help="deck matchup matrix (ygorl-matchup JSON)")
    ap.add_argument("--nash-share", type=float, default=0.5, help="weight of the matrix's Nash mixture vs meta")
    ap.add_argument("--informed", type=int, default=6)
    ap.add_argument("--explore", type=int, default=2)
    ap.add_argument("--max-bundle", type=int, default=3)
    ap.add_argument("--crossover", type=int, default=0,
                    help="crossover children per parent: the parent x an archive elite of a far cell (#141)")  # fmt: skip
    ap.add_argument("--rules", type=int, default=0, help="children from the corpus association rules (0: off)")
    ap.add_argument("--learned-generator", type=int, default=0,
                    help="children from the masked deck model (0: off; needs --deck-model)")  # fmt: skip
    ap.add_argument("--deck-model", type=Path, default=None, help="masked deck model (tools/train_deck_model.py)")
    ap.add_argument("--learned-removal", choices=("combined", "support", "typicality"), default="combined",
                    help="the learned children's removal ranking (ygorl.build.deck_model.DeckModel.removal_scores)")  # fmt: skip
    ap.add_argument("--rule-min-count", type=int, default=3, help="lists a rule needs")
    ap.add_argument("--rule-min-confidence", type=float, default=0.5)
    ap.add_argument("--rule-min-lift", type=float, default=2.0)
    ap.add_argument("--diagnose-pairs", type=int, default=0,
                    help="parent pairs played first (opening-hand effects; the signal weighs 0 until calibrated)")  # fmt: skip
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--max-pairs", type=int, default=600)
    ap.add_argument("--look", type=int, default=100)
    ap.add_argument("--cap", type=int, default=1000)
    ap.add_argument("--min-effect", type=float, default=0.02)
    ap.add_argument("--budget", type=int, default=None, help="games per round: no parent starts past it")
    ap.add_argument("--l0", choices=("off", "shadow", "on"), default="off",
                    help="critic opening-value screen (needs a PPO checkpoint; off until M4 passes)")  # fmt: skip
    ap.add_argument("--l0-min", type=float, default=-0.02)
    ap.add_argument("--l0-hands", type=int, default=256)
    ap.add_argument("--meta-top", type=int, default=60)
    ap.add_argument("--generic-pool", type=Path, default=ROOT / "tests" / "data" / "generic_pool.json",
                    help="generic cards with roles (the hand-trap descriptor, the addition pool's generic cards)")  # fmt: skip
    ap.add_argument("--pool", choices=("engine", "tech"), default="engine",
                    help="addition pool: engine-aware (#145) or the tuner's tech pool (the first rounds)")  # fmt: skip
    ap.add_argument("--engine-fanout", type=int, default=500, help="fanout limit of in-deck edges (engine members)")
    ap.add_argument("--addition-fanout", type=int, default=30, help="fanout limit of edges to cards the deck lacks")
    ap.add_argument("--engine-evidence", type=int, default=2,
                    help="observations of an engine member before it may be taken out")  # fmt: skip
    ap.add_argument("--cold-candidates", type=int, default=40, help="single swaps screened at cold start (0: off)")
    ap.add_argument("--cold-keep", type=int, default=4, help="screened children raced at cold start")
    ap.add_argument("--cold-min-obs", type=int, default=50,
                    help="model observations of the parent's type below which a round is a cold start")  # fmt: skip
    ap.add_argument("--min-confident-pairs", type=int, default=100,
                    help="pairs the leader needs before the search stops as confident")  # fmt: skip
    ap.add_argument("--no-multiplicity", action="store_true",
                    help="confident stop at P > 0.95 regardless of the number of candidates")  # fmt: skip
    ap.add_argument("--eval", dest="evaluation", choices=("thompson", "factorial"), default="thompson",
                    help="children raced by top-two Thompson sampling, or single edits in one fractional factorial "
                    "(#152)")  # fmt: skip
    ap.add_argument("--factorial-k", type=int, default=4, help="edits per fractional factorial (--eval factorial)")
    ap.add_argument("--factorial-pairs", type=int, default=200, help="pairs every variant of the design plays")
    ap.add_argument("--value-model", type=Path, default=None,
                    help="edit-value model (tools/fit_value_model.py, #151): its predictions enter the Thompson prior "
                    "and rank the learned candidates, weighted by the calibration table")  # fmt: skip
    ap.add_argument("--value-model-weight", type=float, default=0.0,
                    help="the value model's weight until the calibration table has enough of its pairs")  # fmt: skip
    ap.add_argument("--value-model-oversample", type=int, default=3,
                    help="learned candidates drawn per learned child kept (ranked by the value model)")  # fmt: skip
    ap.add_argument("--warm-start", type=Path, action="append", default=[],
                    help="earlier paired data for the card-value model (repeatable; ygorl.build.warmstart)")  # fmt: skip
    ap.add_argument("--warm-start-inflate", type=float, default=1.0,
                    help="multiply the warm-start standard errors (data from another checkpoint)")  # fmt: skip
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    if args.learned_generator and args.deck_model is None:
        ap.error("--learned-generator needs --deck-model")

    import torch

    from ygorl.build.archive import deck_descriptors
    from ygorl.build import warmstart
    from ygorl.build.control import critic_opening_values, opening_value_screen
    from ygorl.build.deck_engine import addition_pool, deck_engine, generic_roles
    from ygorl.build.evolve import (Evolution, Lab, Parent, RoundConfig, check_environment, file_sha256, format_report,
                                    opponent_mix)  # fmt: skip
    from ygorl.build.packages import setcodes_from_db
    from ygorl.build.rules import DeckRules, RuleConfig
    from ygorl.build.signals import protected_cards
    from ygorl.build.synergy_graph import load_or_build
    from ygorl.build.tuner import PairedEvaluator, tech_pool
    from ygorl.cards.ydk import load_ydk
    from ygorl.data.environment import load_environment
    from ygorl.engine.duel import default_cards
    from ygorl.env.encoded import EncodedVecEnv
    from ygorl.eval.matchup import MetaGame
    from ygorl.train.checkpoint import load_actor, load_actor_critic

    def say(msg):
        print(msg, flush=True)

    cards = default_cards()
    env = load_environment(args.env, cards=cards)
    evo = Evolution(args.state, env, manifest=args.manifest)
    pol = load_actor(args.checkpoint)
    evo.check_checkpoint(pol.environment, args.checkpoint)
    opp = pol
    if args.opponent_checkpoint:
        opp = load_actor(args.opponent_checkpoint)
        evo.check_checkpoint(opp.environment, args.opponent_checkpoint)
        if why := pol.signature.mismatches(opp.signature):
            raise SystemExit(f"the two checkpoints need the same card vocab and event length: {'; '.join(why)}")

    corpus = json.loads(env.artifact_path("deck_corpus.json").read_text())
    lists = [(e["type"], load_ydk(env.artifacts_dir / e["file"])) for e in corpus["decks"]]

    def nearest_type(deck):
        c = deck.counts()
        return min(lists, key=lambda t: sum(abs(c[k] - t[1].counts()[k]) for k in c.keys() | t[1].counts().keys()))[0]

    if args.parent:
        parents = []
        for f in args.parent:
            deck = load_ydk(f)
            if problems := env.validate_deck(deck, cards):
                raise SystemExit(f"{f}: not legal in {env.version}: {problems}")
            where = "corpus" if f.resolve().is_relative_to(env.artifacts_dir.resolve()) else "file"
            parents.append(Parent(f"{where}:{f.stem}", deck, nearest_type(deck)))
    else:
        pool = [p if p.type else Parent(p.id, p.deck, nearest_type(p.deck)) for p in evo.pool()]
        if not pool:
            raise SystemExit("the manifest has no probation / active decks: give --parent")
        parents = evo.pick_parents(pool, args.parents)

    meta = [(m.name, m.deck, m.share) for m in env.meta_decks]
    nash = None
    if args.matrix:
        nash = MetaGame.load(args.matrix, env).nash_by_deck()
    pool_decks = {p.id: p.deck for p in evo.pool("probation", "active", "history")}
    opponents = opponent_mix(meta, nash=nash, decks=pool_decks, nash_share=args.nash_share)

    # one graph for protection, descriptors (both filter to fanout <= 30), engines and additions
    graph = load_or_build(max_fanout=max(args.engine_fanout, args.addition_fanout, 100)).restrict(env.card_pool,
                                                                                                  env.stamp())  # fmt: skip
    generic = generic_roles(args.generic_pool)
    traps = {c for c, role in generic.items() if role == "hand_trap"}
    meta_decks = [m.deck for m in env.meta_decks]

    def engine(deck):
        return set(deck_engine(deck, graph, generic=generic, max_fanout=args.engine_fanout).members)

    def pool(p):
        tech = tech_pool(p.deck, [d for t, d in lists if t == p.type and d.counts() != p.deck.counts()], meta_decks,
                         meta_top=args.meta_top)  # fmt: skip
        if args.pool == "tech":
            return tech
        return addition_pool(p.deck, graph, generic=generic, candidates=tech, pool=env.card_pool,
                             max_fanout=args.addition_fanout)  # fmt: skip

    for p in parents:
        eng = deck_engine(p.deck, graph, generic=generic, max_fanout=args.engine_fanout)
        say(f"{p.id}: engine {len(eng)} cards ({'; '.join(sorted(cards[c].name for c in eng.members))}), starters "
            f"{'; '.join(cards[c].name for c in eng.starters)}; addition pool {len(pool(p))} cards")  # fmt: skip
    for path in args.warm_start:
        items = warmstart.load(path, inflate=args.warm_start_inflate)
        say(f"warm start {path}: {evo.warm_start(items)} of {len(items)} observations new")
    device = torch.device(args.device)
    net_a = pol.net.to(device)
    net_b = net_a if opp is pol else opp.net.to(device)

    def env_factory(n):
        return EncodedVecEnv(n, args.threads, cards=cards, vocab=pol.vocab, event_length=pol.event_length,
                             skip_forced=True)  # fmt: skip

    def evaluator(seed, opps):
        return PairedEvaluator(env_factory=env_factory, policy=net_a, opponent=net_b, opponents=[o.deck for o in opps],
                               weights=[o.weight for o in opps], seed=seed, device=args.device, num_envs=args.envs)  # fmt: skip

    screen = None
    if args.l0 != "off":
        ac = load_actor_critic(args.checkpoint)
        critic = ac.model.to(device)

        def critic_env(n):
            return EncodedVecEnv(n, args.threads, cards=cards, vocab=ac.vocab, event_length=ac.event_length,
                                 privileged=True, skip_forced=True)  # fmt: skip

        def screen(ev, base, children):
            values = lambda sp: critic_opening_values(critic, critic_env, sp, device=device, num_envs=args.envs)  # noqa: E731
            return opening_value_screen(values, ev, base, children, hands=args.l0_hands)

    rules = None
    if args.rules:
        cfg = RuleConfig(min_count=args.rule_min_count, min_confidence=args.rule_min_confidence,
                         min_lift=args.rule_min_lift)  # fmt: skip
        rules = DeckRules.from_environment(env, cfg, setcodes=setcodes_from_db(cards))
    deck_model = None
    if args.value_model is not None and args.deck_model is None:  # the deck model the value model was fitted with
        meta = torch.load(args.value_model, map_location="cpu", weights_only=True).get("meta", {})
        args.deck_model = Path(meta["deck_model"]["path"])
    if args.learned_generator or args.value_model is not None:
        from ygorl.build.deck_model import load_deck_model

        deck_model = load_deck_model(args.deck_model)  # scored on the CPU: one small batch per parent
        check_environment(env, deck_model.environment, f"deck model {args.deck_model}")
        say(f"deck model {args.deck_model}: {len(deck_model.passwords)} cards, trained on "
            f"{(deck_model.meta or {}).get('lists')} lists")  # fmt: skip
    value_model = None
    if args.value_model is not None:
        from ygorl.build.value_model import load_value_model

        value_model = load_value_model(args.value_model, deck_model)
        want = (value_model.meta.get("deck_model") or {}).get("sha256")
        if want and want != file_sha256(args.deck_model):
            raise SystemExit(
                f"{args.value_model} was fitted with another deck model ({value_model.meta['deck_model']})"
            )
        say(f"value model {args.value_model}: {len(value_model.nets)} members, sigma0 {value_model.sigma0:.4f}, "
            f"target {value_model.clock.run} @ {value_model.clock.update}")  # fmt: skip
    lab = Lab(
        evaluator=evaluator,
        legal=lambda d: not env.validate_deck(d, cards),
        is_extra=lambda pw: cards[pw].is_extra_deck if pw in cards else False,
        pool=pool,
        protected=lambda d: protected_cards(d, graph),
        engine=engine if args.pool == "engine" else (lambda d: set()),
        descriptors=lambda d: deck_descriptors(d, hand_traps=traps, graph=graph),
        screen=screen,
        checkpoint={
            "path": str(args.checkpoint),
            "sha256": file_sha256(args.checkpoint),
            "update": pol.update,
            "opponent": str(args.opponent_checkpoint) if args.opponent_checkpoint else None,
            **(
                {"deck_model": {"path": str(args.deck_model), "sha256": file_sha256(args.deck_model)}}
                if deck_model is not None
                else {}
            ),
            **(
                {"value_model": {"path": str(args.value_model), "sha256": file_sha256(args.value_model)}}
                if value_model is not None
                else {}
            ),
        },  # fmt: skip
        rules=rules,
        deck_model=deck_model,
        card_pool=env.card_pool,
        value_model=value_model,
    )
    config = RoundConfig(informed=args.informed, explore=args.explore, max_bundle=args.max_bundle,
                         crossover=args.crossover, rules=args.rules, learned=args.learned_generator,
                         learned_removal=args.learned_removal,
                         diagnose_pairs=args.diagnose_pairs, batch=args.batch, max_pairs=args.max_pairs,
                         look=args.look, cap=args.cap, min_effect=args.min_effect, budget=args.budget, l0=args.l0,
                         l0_min=args.l0_min, engine_evidence=args.engine_evidence,
                         cold_candidates=args.cold_candidates, cold_keep=args.cold_keep,
                         cold_min_obs=args.cold_min_obs, min_confident_pairs=args.min_confident_pairs,
                         multiplicity=not args.no_multiplicity, evaluation=args.evaluation,
                         factorial_k=args.factorial_k, factorial_pairs=args.factorial_pairs,
                         value_model_weight=args.value_model_weight,
                         value_model_oversample=args.value_model_oversample, seed=args.seed)  # fmt: skip
    say(f"parents: {', '.join(f'{p.id} ({p.type})' for p in parents)}; {len(opponents)} opponents"
        + (f" (Nash share {args.nash_share})" if nash else " (meta shares)"))  # fmt: skip
    report = evo.run_round(parents, lab, config, opponents, log=say)
    print(format_report(report), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
