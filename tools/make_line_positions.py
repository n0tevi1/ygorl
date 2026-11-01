"""Turn-1-end positions from the solver's lines: training starts for the first seat (out/research/fs3/t1-headroom).

The solver line plays seat 0's turn 1 (the line's deck moves first) against a meta deck on seat 1 that answers
passively; at the first decision of turn 2 the game's action prefix becomes a ``ygorl.train.positions.Position`` with
``learner_seat = 0``. Only completed lines are kept. The env uses the training defaults (``skip_forced``, cancel
budget 32): the trainer replays the prefix in its own env, so they must match (``Position.check``).

  uv run --no-sync python tools/make_line_positions.py --out out/positions/line-t1.jsonl [--opponents 4] [--verify 50]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def passive_row(obs) -> int:
    from ygorl.env.encoding import ACTION_KINDS
    from ygorl.solver.demo import PASSIVE_KINDS

    mask = np.asarray(obs["action_mask"]) != 0
    kinds = np.asarray(obs["actions"])[:, 0]
    for k in PASSIVE_KINDS:
        hit = np.nonzero(mask & (kinds == ACTION_KINDS.index(k) + 1))[0]
        if len(hit):
            return int(hit[0])
    return int(np.nonzero(mask)[0][0])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--lines", default="out/claude-scale/handtrap/lines_train.jsonl")
    p.add_argument("--recovery", action="append", default=[])
    p.add_argument("--decks", default="environments/md-2026-09/meta")
    p.add_argument("--env", default="md-2026-09")
    p.add_argument("--opponents", type=int, default=4, help="opponent decks (and seeds) per line")
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=20261101)
    p.add_argument("--verify", type=int, default=50, help="replay this many positions and check they reach turn 2")
    p.add_argument(
        "--vocab-from",
        default="out/train/hero/checkpoints/latest.pt",
        help="checkpoint whose card vocabulary the lines are matched in (the training run's)",
    )
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    from ygorl.cards.ydk import load_ydk
    from ygorl.data import load_environment
    from ygorl.engine.duel import DuelConfig, default_cards
    from ygorl.env.encoded import EncodedVecEnv
    import torch

    from ygorl.eval.arena import derive_seed
    from ygorl.nets.batch import collate, policy_logits
    from ygorl.train.lines import LineLibrary, LinePlayer
    from ygorl.train.positions import Position, write_positions

    environment = load_environment(args.env)
    cards = default_cards()
    from ygorl.train.checkpoint import load_actor

    actor = load_actor(args.vocab_from)  # also fills seat 0's off-script decisions inside the line (as the harness)
    vocab = actor.vocab
    actor.net.eval()
    lib = LineLibrary.load_cached(
        [args.lines],
        vocab,
        cards=cards,
        env=environment,
        recovery=tuple(args.recovery),
        cache=ROOT / "out" / "linecache" / "positions-lib.pkl",
    )
    decks = [load_ydk(f) for f in sorted(Path(args.decks).glob("*.ydk"))]
    config = DuelConfig.from_environment(environment)
    rng = np.random.default_rng(args.seed)
    jobs = []
    for s in lib.scripts:
        for k in range(args.opponents):
            opp = decks[int(rng.integers(len(decks)))]
            seed = derive_seed(args.seed, s.id, k)
            jobs.append((s, opp, seed, s.spec(opp, seed, config, learner_seat=1)))
    print(f"{len(lib.scripts)} lines x {args.opponents} opponents = {len(jobs)} games", flush=True)

    env = EncodedVecEnv(args.envs, args.threads, cards=cards, vocab=vocab, event_length=64, skip_forced=True,
                        cancel_budget=32, record_actions=True)  # fmt: skip
    slots: dict[int, tuple] = {}
    queue = list(range(len(jobs)))
    out, why = [], {"completed": 0, "broken": 0, "ended": 0, "error": 0}

    def start(e: int) -> None:
        if queue:
            j = queue.pop(0)
            s, _, _, spec = jobs[j]
            slots[e] = (j, LinePlayer(s, True))
            env.reset(e, spec)
        else:
            slots.pop(e, None)

    for e in range(args.envs):
        start(e)
    while slots:
        for ev in env.recv(1):
            e = ev.env_id
            if e not in slots:
                continue
            j, line = slots[e]
            s, opp, seed, spec = jobs[j]
            if ev.result is not None:
                why["ended"] += 1
                start(e)
                continue
            g = ev.obs["globals"]
            if int(g[3]) >= 2:  # the first decision of turn 2: record the position
                fin = line.finish()
                status = fin["status"]
                why.setdefault("status:" + str(status), 0)
                why["status:" + str(status)] += 1
                rest_forced = all(st.forced for st in line.steps[line.pos :])  # never asked under skip_forced
                if status == "completed" or (status == "playing" and rest_forced):
                    why["completed"] += 1
                    out.append(Position.from_spec(spec, list(env.actions(e)), skip_forced=True, cancel_budget=32,
                                                  env=environment, learner_seat=0,
                                                  origin={"source": "make_line_positions", "line_id": s.id,
                                                          "line_deck": s.deck.name, "hand_index": s.hand_index,
                                                          "opponent_deck": opp.name}))  # fmt: skip
                else:
                    why["broken"] += 1
                start(e)
                continue
            row = line.act(ev.obs) if int(ev.player) == 0 else None
            if row is None and int(ev.player) == 0:
                with torch.no_grad():
                    logits = policy_logits(actor.net, collate([ev.obs], None)).float()[0]
                row = int(torch.argmax(logits))
            env.step(e, int(row) if row is not None else passive_row(ev.obs))
    print(json.dumps(why), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_positions(args.out, out)
    print(f"wrote {len(out)} positions to {args.out}", flush=True)

    if args.verify:  # replay a sample by prefix: each must come back at turn 2 with seat 0 still to play turn 3
        check = EncodedVecEnv(1, 1, cards=cards, vocab=vocab, event_length=64, skip_forced=True, cancel_budget=32)
        bad = 0
        for pos in out[: args.verify]:
            pos.start(check, 0, environment)
            ev = check.recv(1)[0]
            if ev.result is not None or int(ev.obs["globals"][3]) != 2:
                bad += 1
        print(f"verified {min(args.verify, len(out))} positions, {bad} did not replay to turn 2", flush=True)
        if bad:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
