"""Reconstruction fidelity of the Master Duel advisor (design 09「评估」, T7.3 / T7.4).

Usage: uv run python tools/advisor_fidelity.py --env md-2026-09 --policy out/gpu/GF/best.pt [--games 100]

Greedy plays meta decks against each other. At one of each player's own Main Phase 1 idle decisions (turn >= 2),
the board is exported as the player would see it (``snapshot_from_core``) and reconstructed twice: with the
opponent's true deck as the prior (the "oracle": only the reconstruction itself is measured) and with the
environment's meta decks (what the advisor really has). Reported per reconstruction:

- ``jaccard``: overlap of the legal action sets (kind, card, effect description), after the advisor's masks;
- ``top1``: the policy's most probable action is the same as on the real board;
- ``tv``: total variation distance between the two action distributions.

The real board's observation has the game's event history and the reconstruction's has none, so top-1 / tv
measure the whole information loss, not only the rebuild.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path


def normal_summoned_tracker():
    """Feed every point's events; ``used(player)``: a Normal Summon or monster Set happened this turn."""
    from ygorl.engine import constants as C
    from ygorl.engine import messages as M

    used: set[int] = set()

    def feed(events) -> None:
        for e in events:
            if isinstance(e, M.NewTurn):
                used.clear()
            elif isinstance(e, M.Summoning) or (isinstance(e, M.SetCard) and e.loc.location == C.LOCATION_MZONE):
                used.add(e.loc.controller)

    return feed, used


def distribution(policy, obs, actions, masks):
    import torch

    from ygorl.advisor.advise import action_key
    from ygorl.nets import collate

    with torch.no_grad():
        logits = policy.net(collate([obs])).logits[0].float()
    n = min(len(actions), logits.shape[0])
    keep = torch.tensor([not m for m in masks[:n]])
    p = torch.softmax(logits[:n].masked_fill(~keep, float("-inf")), -1).tolist()
    out: Counter = Counter()
    for a, q in zip(actions[:n], p, strict=True):
        out[action_key(a)] += q
    return out


def compare(real: Counter, rec: Counter, real_keys: set, rec_keys: set) -> dict:
    keys = real.keys() | rec.keys()
    return {
        "jaccard": len(real_keys & rec_keys) / len(real_keys | rec_keys),
        "top1": max(real, key=real.get) == max(rec, key=rec.get),
        "tv": 0.5 * sum(abs(real.get(k, 0.0) - rec.get(k, 0.0)) for k in keys),
        "only_real": sorted(real_keys - rec_keys),
        "only_rec": sorted(rec_keys - real_keys),
    }


def run_game(game: int, env, policy, cards, rng: random.Random) -> list[dict]:
    from ygorl.advisor.advise import action_key, masked, observe
    from ygorl.advisor.reconstruct import OpponentPrior, ReconstructError, build, complete
    from ygorl.advisor.snapshot import snapshot_from_core
    from ygorl.agents.greedy import GreedyAgent
    from ygorl.engine import constants as C
    from ygorl.engine import messages as M
    from ygorl.engine.duel import Duel, DuelSession
    from ygorl.env.observer import PointObserver

    metas = env.meta_decks
    weights = [m.share for m in metas]
    a, b = rng.choices(range(len(metas)), weights=weights, k=2)
    duel = Duel(game, env, metas[a].deck, metas[b].deck, first=game % 2)
    session = DuelSession(duel)
    agent = GreedyAgent(game)
    observers = [PointObserver(cards, policy.vocab, policy.event_length, starting_lp=env.player.starting_lp)
                 for _ in range(2)]  # fmt: skip
    feed, used = normal_summoned_tracker()
    start_turn = {0: rng.randint(2, 7), 1: rng.randint(2, 7)}
    rows: list[dict] = []
    meta_prior = OpponentPrior.from_environment(env)
    while not session.done:
        p = session.point
        for o in observers:
            o.observe(p)
        feed(p.events)
        me = p.player
        if (me in start_turn and p.turn >= start_turn[me] and p.player == p.turn_player and p.phase == C.PHASE_MAIN1
                and isinstance(p.decision, M.SelectIdleCmd) and not p.undo):  # fmt: skip
            del start_turn[me]
            snap = replace(snapshot_from_core(session.core, p), normal_summoned=me in used)
            real_obs = observers[me].encode(p, session.core)
            real = distribution(policy, real_obs, p.actions, [masked(snap, x) for x in p.actions])
            real_keys = {action_key(x) for x in p.actions if not masked(snap, x)}
            row = {"game": game, "turn": p.turn, "own": metas[(a, b)[duel.deck_of(me)]].name, "actions": len(real_keys)}
            own = duel.decks[duel.deck_of(me)]
            opp = duel.decks[duel.deck_of(1 - me)]
            for label, prior in (("oracle", OpponentPrior((opp,), (1.0,))), ("meta", meta_prior)):
                try:
                    board = complete(snap, own, prior, cards, rng)
                    rec = build(board, env.rule_flags, cards=cards, seed=rng.randrange(1, 1 << 62))
                except ReconstructError as exc:
                    row[label] = {"error": str(exc)}
                    continue
                obs = observe(rec, PointObserver(cards, policy.vocab, policy.event_length,
                                                 starting_lp=env.player.starting_lp))  # fmt: skip
                acts = rec.point.actions
                dist = distribution(policy, obs, acts, [masked(snap, x) for x in acts])
                rec_keys = {action_key(x) for x in acts if not masked(snap, x)}
                row[label] = compare(real, dist, real_keys, rec_keys) | {"setup_decisions": len(
                    [w for w in rec.warnings if w.startswith("answered")])}  # fmt: skip
            rows.append(row)
            if not start_turn:
                break
        session.act(agent.act(p))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", required=True)
    ap.add_argument("--policy", required=True, type=Path)
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="write every row as JSON lines")
    args = ap.parse_args()

    import torch

    from ygorl.data import load_environment
    from ygorl.engine.duel import default_cards
    from ygorl.train.checkpoint import load_actor

    torch.set_num_threads(1)
    cards = default_cards()
    env = load_environment(args.env, cards=cards)
    policy = load_actor(args.policy)
    rng = random.Random(args.seed)
    rows: list[dict] = []
    for game in range(args.games):
        rows += run_game(args.seed * 1_000_003 + game, env, policy, cards, rng)
    if args.out:
        args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} decision points from {args.games} games ({args.env}, {args.policy})")
    for label in ("oracle", "meta"):
        ok = [r[label] for r in rows if "error" not in r[label]]
        errors = len(rows) - len(ok)
        if not ok:
            print(f"{label}: no reconstruction succeeded")
            continue
        mean = {k: sum(float(o[k]) for o in ok) / len(ok) for k in ("jaccard", "top1", "tv")}
        exact = sum(o["jaccard"] == 1.0 for o in ok) / len(ok)
        setup = sum(o["setup_decisions"] > 0 for o in ok) / len(ok)
        print(f"{label:>6}: jaccard {mean['jaccard']:.3f} (exact {exact:.1%}), top-1 agreement {mean['top1']:.1%}, "
              f"TV {mean['tv']:.3f}, setup decisions {setup:.1%}, errors {errors}")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
