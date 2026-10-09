"""Behaviour-cloning warm start from solver demonstrations, with its evaluation report (T4a.2, docs/bc.md).

Usage: uv run --extra train python tools/train_bc.py --train out/demos/bc_train.jsonl [--train ...]
           [--heldout out/demos/bc_heldout.jsonl] [--out out/bc] [--epochs 12] [--batch-size 64] [--lr 3e-4]
           [--history transformer|lstm|none] [--d-model 128] [--layers 2] [--event-length 128] [--threads 2]
           [--seed 0] [--baselines]
           [--checkpoint PATH --no-train] [--init-from CKPT] [--report PATH] [--openings all|heldout|none] [--sample-openings]
           [--extra GREEDY.npz --extra-subset all|battle --extra-max N [--extra-heldout GREEDY.npz]]

Replays every verified line of the --train files, encodes each decision of the deck under study with the
reference encoders, trains PolicyNet by cross-entropy over the legal candidates and writes
<out>/policy.pt (loadable by PolicyAgent: ``ygorl arena ... --agent-a policy:<out>/policy.pt``). Then it
reports, into <out>/report.json:

- teacher-forced step accuracy on the training lines and on the held-out lines;
- free-running turn 1 on every training hand (line reproduction, targets reached) and on every held-out
  hand (targets reached, target cards placed; the solver's own result on the same hands for scale);
- with --baselines, the same free-running numbers for RandomAgent and GreedyAgent.

--extra adds heuristic samples beyond turn 1 (tools/greedy_demos.py, ``ygorl.train.heuristic_demos``) to the
solver samples: the --extra-subset of them, a uniform random sample of at most --extra-max. --extra-heldout
reports the teacher-forced step accuracy on the same subset of held-out heuristic games (at most 4,000 samples).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(paths: list[Path]):
    from ygorl.solver import read_jsonl

    return [d for p in paths for d in read_jsonl(p)]


def _environment(demos) -> dict | None:
    stamps = {json.dumps(d.environment, sort_keys=True) for d in demos}
    if len(stamps) > 1:
        raise SystemExit(f"train_bc: error: the demonstrations mix environments: {sorted(stamps)}")
    return json.loads(stamps.pop()) if stamps else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, action="append", required=True, help="training demonstrations (JSONL)")
    parser.add_argument("--heldout", type=Path, action="append", default=[], help="held-out demonstrations (JSONL)")
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "bc", help="output directory (default out/bc)")
    parser.add_argument(
        "--env", default=None, metavar="PATH|VERSION", help="environment the demonstrations are bound to"
    )
    parser.add_argument(
        "--include-synthetic-closing",
        action="store_true",
        help="also imitate automatically appended passive closing (legacy comparisons only)",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--history", default="transformer", choices=("transformer", "lstm", "none"))
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2, help="board and history Transformer layers (default 2)")
    parser.add_argument("--selection-history", action="store_true", help="record actor-private selection history")
    parser.add_argument("--event-length", type=int, default=128, help="event tokens per observation (default 128)")
    parser.add_argument("--threads", type=int, default=2, help="torch threads (default 2)")
    parser.add_argument("--device", default="cpu", help="CPU or CUDA/ROCm device: cpu, cuda, cuda:0 (default cpu)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--text-dir",
        type=Path,
        default=None,
        help="card feature directory: frozen text tables and/or card_facts.npz (docs/nets.md)",
    )
    parser.add_argument("--card-facts", action="store_true", help="use card_facts.npz in --text-dir (experimental)")
    parser.add_argument("--no-text", action="store_true", help="ignore the text tables in --text-dir")
    parser.add_argument("--id-dropout", type=float, default=0.0, help="training: drop each card's ID embedding")
    parser.add_argument("--no-id-embedding", action="store_true", help="drop the per-card ID embedding")
    parser.add_argument("--baselines", action="store_true", help="also report RandomAgent and GreedyAgent openings")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="checkpoint to evaluate (default <out>/policy.pt; with --no-train also a PPO checkpoint)",
    )
    parser.add_argument("--no-train", action="store_true", help="evaluate --checkpoint without training")
    parser.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="fine-tune this checkpoint's actor (PPO or policy checkpoint): its network, card vocab and event "
        "window replace --d-model / --layers / --history / --event-length (the result can be a --bc-prior of a PPO "
        "run over the same vocab)",
    )
    parser.add_argument("--report", type=Path, default=None, help="report file (default <out>/report.json)")
    parser.add_argument(
        "--openings",
        default="all",
        choices=("all", "heldout", "none"),
        help="free-running turn-1 reports on the training and held-out hands (default all)",
    )
    parser.add_argument(
        "--sample-openings",
        action="store_true",
        help="play the free-running openings by sampling at temperature 1 (seed = hand number) instead of argmax",
    )
    parser.add_argument(
        "--extra",
        type=Path,
        action="append",
        default=[],
        help="heuristic samples (.npz of tools/greedy_demos.py) added to the training set",
    )
    parser.add_argument("--extra-subset", default="all", help="subset of the --extra samples (all, battle)")
    parser.add_argument("--extra-max", type=int, default=None, help="at most this many --extra samples")
    parser.add_argument("--extra-heldout", type=Path, default=None, help="held-out heuristic samples (.npz)")
    return parser


def net_config(args: argparse.Namespace, vocab, text):
    """The network for ``args`` (the same card-view switches as tools/train_ppo.py, so PPO can --init-from it)."""
    from ygorl.nets import NetConfig

    return NetConfig(vocab_size=len(vocab), d_model=args.d_model, history=args.history, board_layers=args.layers,
                     history_layers=args.layers, id_embedding=not args.no_id_embedding, card_facts=args.card_facts,
                     card_text=not args.no_text, effect_text=not args.no_text,
                     id_dropout=args.id_dropout, selection_history=args.selection_history).with_text(text)  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    import torch

    from ygorl.agents import GreedyAgent, PolicyAgent, RandomAgent
    from ygorl.cards.cdb import CardVocab
    from ygorl.engine.duel import default_cards
    from ygorl.nets import PolicyNet
    from ygorl.nets.agent import NetPolicy, save_checkpoint
    from ygorl.train.bc import BCConfig, build_dataset, hand_overlap, opening_report, step_accuracy, train_bc

    torch.set_num_threads(args.threads)
    try:
        device = torch.device(args.device)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"train_bc: invalid device: {exc}") from None
    if device.type not in ("cpu", "cuda"):
        raise SystemExit("train_bc: --device must be cpu or cuda[:index] (including ROCm)")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("train_bc: CUDA/ROCm was requested but is unavailable")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise SystemExit(f"train_bc: CUDA/ROCm device index {index} is unavailable")
        device = torch.device("cuda", index)
    else:
        device = torch.device("cpu")
    env = None
    if args.env is not None:
        from ygorl.data import load_environment

        env = load_environment(args.env)
    train_demos, heldout_demos = _load(args.train), _load(args.heldout)
    stamp = _environment(train_demos + heldout_demos)
    expected_stamp = None if env is None else {"version": env.version, "fingerprint": env.fingerprint}
    if stamp != expected_stamp:
        raise SystemExit(f"train_bc: error: the demonstrations are bound to {stamp}; pass the same --env")
    args.out.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.checkpoint or args.out / "policy.pt"
    cards = default_cards()
    if env is not None:
        from ygorl.cards.ydk import Deck

        checked = set()
        for demo in train_demos + heldout_demos:
            deck = Deck(main=tuple(demo.deck["main"]), extra=tuple(demo.deck["extra"]),
                        side=tuple(demo.deck.get("side", ())), name=demo.deck["name"])  # fmt: skip
            if deck in checked:
                continue
            checked.add(deck)
            violations = env.validate_deck(deck, cards)
            if violations:
                raise SystemExit(f"train_bc: illegal demonstration deck {deck.name}: "
                                 + "; ".join(v.message for v in violations))  # fmt: skip
    report: dict = {"train_files": [str(p) for p in args.train], "heldout_files": [str(p) for p in args.heldout],
                    "environment": stamp,
                    "data_policy": {"include_synthetic_closing": args.include_synthetic_closing,
                                    "full_replay_validation": True}}  # fmt: skip
    report["device"] = {
        "requested": args.device,
        "actual": str(device),
        "name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "torch": str(torch.__version__),
        "hip": torch.version.hip,
        "cuda": torch.version.cuda,
    }
    report["opening_device"] = "cpu"

    def count(demos) -> dict:
        solved = [d for d in demos if d.status == "solved"]
        return {"records": len(demos), "plain_hands": sum(d.variant == "plain" for d in demos),
                "solved_records": len(solved), "lines": sum(len(d.lines) for d in solved),
                "steps": sum(len(ln.actions) for d in solved for ln in d.lines)}  # fmt: skip

    report["data"] = {"train": count(train_demos), "heldout": count(heldout_demos),
                      "heldout_hands_also_in_train": hand_overlap(train_demos, heldout_demos)}  # fmt: skip
    t0 = time.time()
    init = None
    if args.no_train:  # a BC policy checkpoint or a PPO training checkpoint
        from ygorl.train.checkpoint import load_actor

        ckpt = load_actor(ckpt_path, args.text_dir)
        net, vocab, event_length = ckpt.net, ckpt.vocab, ckpt.event_length
        net.to(device)
    elif args.init_from is not None:  # fine-tune an existing actor: its vocab indexes its ID embeddings
        from ygorl.train.checkpoint import load_actor

        init = load_actor(args.init_from, args.text_dir)
        vocab, event_length = init.vocab, init.event_length
        report["init_from"] = {"path": str(args.init_from), "updates": init.update, "net": init.net_config.to_dict()}
    else:
        vocab = CardVocab.from_db(cards)
        event_length = args.event_length
    selection_history = (
        ckpt.net_config.selection_history
        if args.no_train
        else init.net_config.selection_history
        if init is not None
        else args.selection_history
    )
    train_data = build_dataset(
        train_demos,
        vocab,
        cards=cards,
        env=env,
        event_length=event_length,
        include_synthetic_closing=args.include_synthetic_closing,
        selection_history=selection_history,
    )
    heldout_data = build_dataset(heldout_demos, vocab, cards=cards, env=env, event_length=event_length,
                                 include_synthetic_closing=args.include_synthetic_closing,
                                 selection_history=selection_history) if any(
        d.status == "solved" for d in heldout_demos) else None  # fmt: skip
    report["data"]["train"]["samples"] = len(train_data)
    report["data"]["train"]["skipped"] = dict(train_data.skipped)
    solver_data, extra_heldout = train_data, None
    if args.extra or args.extra_heldout:
        from ygorl.train.heuristic_demos import concat, load_compatible_data, select

        def extra_data(path):
            return load_compatible_data(
                path, vocab=vocab, event_length=event_length, environment=stamp, selection_history=selection_history
            )[0]

        if args.extra:
            extra = concat([extra_data(p) for p in args.extra])
            extra = select(extra, args.extra_subset, max_samples=args.extra_max, seed=args.seed)
            train_data = concat([solver_data, extra])
            report["data"]["extra"] = {"files": [str(p) for p in args.extra], "subset": args.extra_subset,
                                       "max": args.extra_max, "samples": len(extra),
                                       "solver_fraction": len(solver_data) / len(train_data)}  # fmt: skip
        if args.extra_heldout is not None:
            extra_heldout = select(extra_data(args.extra_heldout), args.extra_subset, max_samples=4000, seed=1)
            report["data"]["extra_heldout"] = {"file": str(args.extra_heldout), "samples": len(extra_heldout)}
    if heldout_data is not None:
        report["data"]["heldout"]["samples"] = len(heldout_data)
    print(f"dataset: {len(train_data)} training samples, {len(heldout_data) if heldout_data else 0} held-out "
          f"({time.time() - t0:.1f}s)", flush=True)  # fmt: skip

    if not args.no_train:
        from ygorl.nets.text import TextFeatures

        torch.manual_seed(args.seed)
        text = TextFeatures.load(args.text_dir, vocab) if args.text_dir else None
        if init is not None:
            net = init.net.requires_grad_(True)
            cfg = net.cfg
        else:
            cfg = net_config(args, vocab, text)
            net = PolicyNet(cfg, text)
        net.to(device)
        bc = BCConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
                      label_smoothing=args.label_smoothing, seed=args.seed)  # fmt: skip
        torch.manual_seed(args.seed)
        evals = {"heldout": heldout_data} if heldout_data is not None else {}
        if extra_heldout is not None:
            evals["extra_heldout"] = extra_heldout
        t1 = time.time()
        history = train_bc(net, train_data, bc, eval_sets=evals,
                           log=lambda e: print(json.dumps(e), flush=True))  # fmt: skip
        report["training"] = {"config": bc.__dict__, "net": cfg.to_dict(), "seconds": round(time.time() - t1, 1),
                              "history": history}  # fmt: skip
        meta = {"trainer": "bc", "task": "T4a.2", "data": report["data"], "bc": bc.__dict__,
                "final": history[-1] if history else {}, "data_policy": report["data_policy"],
                "training_device": report["device"]}  # fmt: skip
        save_checkpoint(ckpt_path, net, vocab, event_length=event_length, environment=stamp, meta=meta)
        print(f"checkpoint: {ckpt_path}", flush=True)

    report["checkpoint"] = str(ckpt_path)
    report["step_accuracy"] = {"train": step_accuracy(net, solver_data)}
    if train_data is not solver_data:
        report["step_accuracy"]["train_all"] = step_accuracy(net, train_data)
    if heldout_data is not None:
        report["step_accuracy"]["heldout"] = step_accuracy(net, heldout_data)
    if extra_heldout is not None:
        report["step_accuracy"]["extra_heldout"] = step_accuracy(net, extra_heldout)
    print("step accuracy:", json.dumps(report["step_accuracy"]), flush=True)
    net.cpu()  # NetPolicy's per-decision observation path runs on CPU.

    def bc_agent(i: int):
        return PolicyAgent(NetPolicy(net, vocab, event_length=event_length, cards=cards), seed=i,
                           greedy=not args.sample_openings)  # fmt: skip

    agents = {"bc": bc_agent}
    if args.baselines:
        agents.update({"random": RandomAgent, "greedy": GreedyAgent})
    report["openings"] = {}
    for name, factory in agents.items():
        for split, demos in (("train", train_demos), ("heldout", heldout_demos)):
            if not demos or args.openings == "none" or (args.openings == "heldout" and split == "train"):
                continue
            t1 = time.time()
            rep = opening_report(demos, factory, env=env, cards=cards)
            report["openings"].setdefault(name, {})[split] = rep
            print(f"openings {name} {split} ({time.time() - t1:.0f}s):", json.dumps(rep["totals"]["total"]), flush=True)
    report_path = args.report or args.out / "report.json"
    report_path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
