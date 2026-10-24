"""P0c: how concentrated is a policy's turn-1 end board for a fixed first-seat hand? (first-seat plan §5.3 C1, §6).

For ``--hands`` fixed hands (solved ``plain`` records of ``--demos``, hand index ``--min-hand`` .. ``--max-hand``,
spread evenly over the decks), the policy ``CKPT`` (sampled, temperature 1) plays turn 1 ``--samples`` times from
the record's exact start against the passive opponent (``ygorl.train.bc.play_opening``, as ``turn1_board.py``); the
only randomness is the policy's (the deck order and core seed are the record's). Each end board is reduced to a
board key:

- ``field_grave`` (the plan's key): the multiset of the learner's monsters and spell / trap cards (password, set or
  face-up) plus its graveyard and banished cards;
- ``field``: the field part only (a coarser key: same pieces, different resources spent).

Per hand: the number of distinct keys and the share of samples the top 1 / top 3 keys cover; overall: the share of
hands whose top 3 keys cover at least ``--cover`` of the samples (the plan's gate: a majority of hands at 50% means
a cache of re-solved lines keyed by the board would hit), and the mean turn-1 interruptions for reference.

  uv run --no-sync python tools/turn1_board_modes.py DEMOS.jsonl CKPT [--hands 50] [--samples 32] [--workers 8] \\
      --out OUT.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase0_common import mean_se, select_hands  # noqa: E402

from ygorl.eval.arena import derive_seed  # noqa: E402

_CTX: dict = {}


def board_keys(board: dict, player: int = 0) -> dict[str, tuple]:
    from ygorl.engine import constants as C

    side = board["players"][player]
    field = sorted((z, int(c["code"]), not c["position"] & C.POS_FACEUP) for z in ("mzone", "szone") for c in side[z])
    piles = sorted(("grave", int(x)) for x in side["grave"]) + sorted(("banished", int(x)) for x in side["banished"])
    return {"field": tuple(field), "field_grave": (tuple(field), tuple(piles))}


def _init(args: dict) -> None:
    import torch

    torch.set_num_threads(1)
    from ygorl.data import load_environment
    from ygorl.engine.duel import default_cards

    _CTX.update(args, env=load_environment(args["env"]), cards=default_cards())


def play_hand(job: tuple[int, object]) -> dict:
    from ygorl.agents.checkpoint import CheckpointAgent
    from ygorl.solver.blocking import board_interruptions
    from ygorl.train.bc import play_opening

    h, demo = job
    ctx = _CTX
    counts = {"field": Counter(), "field_grave": Counter()}
    inter, fails = [], 0
    for s in range(ctx["samples"]):
        agent = CheckpointAgent(ctx["ckpt"], seed=derive_seed(ctx["seed"], h, s))
        try:
            r = play_opening(demo.replay(None), agent, demo.targets, env=ctx["env"], cards=ctx["cards"])
        except Exception:  # noqa: BLE001 - an unhealthy replay counts as a failure, as turn1_board.py does
            fails += 1
            continue
        for name, key in board_keys(r.board).items():
            counts[name][key] += 1
        inter.append(board_interruptions(r.board, ctx["cards"]).interruptions)
    n = ctx["samples"] - fails
    out = {"deck": demo.deck["name"], "hand_index": demo.hand_index, "samples": n, "failures": fails,
           "interruptions": sum(inter) / max(1, len(inter))}  # fmt: skip
    for name, c in counts.items():
        top = [v for _, v in c.most_common(3)]
        out[name] = {"distinct": len(c), "top1": (top[0] if top else 0) / max(1, n), "top3": sum(top) / max(1, n),
                     "top_counts": [v for _, v in c.most_common(5)]}  # fmt: skip
    return out


def summarize(rows: list[dict], cover: float) -> dict:
    out = {"hands": len(rows), "interruptions": mean_se([r["interruptions"] for r in rows])}
    for name in ("field_grave", "field"):
        top3 = [r[name]["top3"] for r in rows]
        out[name] = {
            "hands_top3_covering": sum(t >= cover for t in top3),
            "share_hands_top3_covering": sum(t >= cover for t in top3) / max(1, len(rows)),
            "top3": mean_se(top3), "top1": mean_se([r[name]["top1"] for r in rows]),
            "distinct": mean_se([r[name]["distinct"] for r in rows]),
        }  # fmt: skip
    return out


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("demos", nargs="+")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--hands", type=int, default=50)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--min-hand", type=int, default=40)
    p.add_argument("--max-hand", type=int, default=50)
    p.add_argument("--cover", type=float, default=0.5)
    p.add_argument("--env", default="md-2026-09")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    t0 = time.perf_counter()
    hands = select_hands(a.demos, a.hands, a.min_hand, a.max_hand)
    init = {"ckpt": a.ckpt, "env": a.env, "seed": a.seed, "samples": a.samples}
    jobs = list(enumerate(hands))
    if a.workers <= 1:
        _init(init)
        rows = [play_hand(j) for j in jobs]
    else:
        with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(init,)) as pool:
            rows = pool.map(play_hand, jobs, chunksize=1)
    report = {"ckpt": a.ckpt, "samples": a.samples, "cover": a.cover, "seed": a.seed,
              "summary": summarize(rows, a.cover), "rows": rows}  # fmt: skip
    report["seconds"] = round(time.perf_counter() - t0, 1)
    text = json.dumps(report, indent=1)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n")
    else:
        print(json.dumps(report["summary"], indent=1))
    return report


if __name__ == "__main__":
    main()
