"""Upper bound for decision-time search (#62 stage C): an ORACLE one-step search that sees the true engine state.

The searcher (policy P on deck a) plays P (deck b). At its first idle-command decision of each own main phase 1 it
takes the policy's top-k legal actions (prob >= MIN_P) and, for each, plays R rollouts to the end of the game with P
on both seats (sampled, fresh seeds), starting from the same engine state; it plays the action with the best mean
score. Rollouts rebuild the state by replaying the game's own seed and action prefix in the batched C++ env
(deterministic), so they also know both players' future draws: this cheats on hidden information on purpose. If even
this does not beat P, PIMC (which can only do worse) is not worth building.

``--search`` picks the search points: ``main`` (default, above); ``chain-opp``: the searcher's chain prompts
(``MSG_SELECT_CHAIN``) on the opponent's turns 2-4 (its responses to the opponent's push); ``chain-own``: its chain
prompts on its own turns 1-4 (responses to the opponent's hand traps). At chain prompts "pass" is always a candidate
when legal, on top of the top k. ``--max-searches`` caps the searches per game. Each change of the policy's top choice
is logged by direction (pass -> chain, chain -> pass, chain -> another chain). k = 1 is the no-search control; with
``--control CONTROL.npy`` (its scores) the output adds the paired per-game difference, split by who went first.

The games run in rounds on the game driver (ygorl.env.driver): a real game plays until its next search, where the
driver abandons it; the next round plays the search's rollouts; when the last one is scored the real game resumes in
the round after, by replaying its own prefix plus the chosen action. Every game samples from its own seeded stream,
so the rounds change the timing, not the games.

Usage: tools/oracle_search.py CHECKPOINT SPECS K R OUT.json [DECK_DIR] [--device cuda] [--envs 512]
           [--search main|chain-opp|chain-own] [--max-searches N] [--control CONTROL.npy]"""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from ygorl.cards.ydk import load_ydk
from ygorl.engine import constants as C
from ygorl.engine.duel import DuelConfig, default_cards, shuffle_deck
from ygorl.env import GameSpec
from ygorl.env.driver import ABANDON, drive
from ygorl.env.encoded import EncodedVecEnv
from ygorl.env.encoding import ACTION_KINDS
from ygorl.eval.arena import derive_seed
from ygorl.nets.batch import collate, policy_logits
from ygorl.train.checkpoint import load_actor, torch_device

MIN_P = 0.02
MAIN1 = 3  # globals[4]: phase ordinal + 1 (draw 1, standby 2, main1 3, ...)
PASS = ACTION_KINDS.index("pass") + 1  # actions[:, 0]: action kind + 1
CHAIN_TURNS = {"chain-opp": (0, range(2, 5)), "chain-own": (1, range(1, 5))}  # (globals[2] is_my_turn, turns)


class Job:
    """A game played on the driver: a real game (the searcher's) or one rollout of a candidate. Each time it starts it
    first replays ``replay`` (actions and the (player, turn, legal actions) seen when they were chosen)."""

    def __init__(self, spec, replay, *, parent=None, cand=None, seed=0, real=False):
        self.spec, self.replay, self.parent, self.cand, self.real = spec, list(replay), parent, cand, real
        self.pos = 0  # decisions answered since this job last started
        self.history = []  # (action, signature) of every answered decision (real games: the prefix for rollouts)
        self.rng = np.random.default_rng(seed)
        self.searched_turns = set()
        self.searches = 0
        self.pending = None  # real game waiting on a search: {"cands", "scores", "sig", "left"}
        self.result = None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("specs", type=int, help="deck pairings (each played with both first players)")
    ap.add_argument("k", type=int, help="candidate actions per search")
    ap.add_argument("r", type=int, help="rollouts per candidate")
    ap.add_argument("out")
    ap.add_argument("deck_dir", nargs="?", type=Path, default=Path("out/corpus/train"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--search", choices=["main", *CHAIN_TURNS], default="main", help="search points (see above)")
    ap.add_argument("--max-searches", type=int, default=None, help="cap on searches per game")
    ap.add_argument("--control", type=Path, help="scores (.npy) of the k = 1 control on the same specs: paired diff")
    args = ap.parse_args()
    ckpt, n_specs, k, r, out = args.checkpoint, args.specs, args.k, args.r, args.out
    device = torch_device(args.device)
    cards = default_cards()
    pol = load_actor(ckpt)
    net = pol.net.to(device)
    decks = [load_ydk(p) for p in sorted(args.deck_dir.glob("*.ydk"))]
    rng = np.random.default_rng(0)
    specs = []
    for i in range(n_specs):
        a, b = rng.choice(len(decks), 2, replace=False)
        s = derive_seed(7, i)
        da = replace(decks[a], main=tuple(shuffle_deck(decks[a].main, s, 0)))
        db = replace(decks[b], main=tuple(shuffle_deck(decks[b].main, s, 1)))
        for first in (0, 1):
            specs.append(GameSpec(seed=s, deck_a=da, deck_b=db, first=first,
                                  config=DuelConfig(max_decisions=4000, shuffle_decks=False)))  # fmt: skip
    env = EncodedVecEnv(args.envs, 8, cards=cards, vocab=pol.vocab, event_length=pol.event_length, skip_forced=True)
    real_jobs = [Job(sp, [], seed=derive_seed(8, i), real=True) for i, sp in enumerate(specs)]
    stats = {"searches": 0, "rollouts": 0, "changed": 0}
    directions = {"pass->chain": 0, "chain->pass": 0, "chain->chain": 0, "other": 0}
    turns_searched: dict[int, int] = {}
    t0 = time.time()
    nxt: list[Job] = []  # the jobs of the next round

    def signature(ev):
        g = ev.obs["globals"]
        return ev.player, int(g[3]), int(g[20])

    def on_result(game, result):
        job = game.state
        w = result.get("winner")
        failed = str(result.get("reason", "")) == "error"
        score = np.nan if failed else 0.5 if w is None else float(w == job.spec.seat_of_deck(0))  # deck a: searcher
        if job.real:
            job.result = score
            return
        stats["rollouts"] += 1
        p = job.parent.pending
        p["scores"][job.cand].append(0.5 if np.isnan(score) else score)
        p["left"] -= 1
        if p["left"] == 0:  # the search is over: the real game resumes with the best candidate
            best = int(np.argmax([np.mean(s) for s in p["scores"]]))
            stats["changed"] += best != 0
            if best != 0:
                was, now = p["is_pass"][0], p["is_pass"][best]
                key = ("pass->chain" if was else "chain->pass" if now else "chain->chain") if p["chain"] else "other"
                directions[key] += 1
            parent = job.parent
            parent.pending = None
            parent.history.append((p["cands"][best], p["sig"]))
            parent.replay, parent.pos = list(parent.history), 0
            nxt.append(parent)

    @torch.no_grad()
    def decide(ready):
        actions = [None] * len(ready)
        play = []
        for n, (game, ev) in enumerate(ready):
            job = game.state
            if job.pos < len(job.replay):  # replaying the game so far (a rollout: plus its candidate at the end)
                a, want = job.replay[job.pos]
                if want != signature(ev):
                    stats["desync"] = stats.get("desync", 0) + 1
                job.pos += 1
                actions[n] = a
            else:
                play.append(n)
        if not play:
            return actions
        logits = policy_logits(net, collate([ready[n][1].obs for n in play], device)).float()
        probs = torch.softmax(logits, -1).cpu().numpy().astype(np.float64)
        for n, pr in zip(play, probs):
            game, ev = ready[n]
            job, g = game.state, ev.obs["globals"]
            sig = signature(ev)
            turn = sig[1]
            if job.real and ev.player == job.spec.seat_of_deck(0) and is_search_point(job, g, turn):
                job.searched_turns.add(turn)
                order = [int(i) for i in np.argsort(-pr) if pr[i] >= MIN_P][:k]
                is_pass = np.asarray(ev.obs["actions"])[: len(pr), 0] == PASS
                chain = args.search != "main"
                if chain and k > 1:  # "pass" is always a candidate when legal
                    order += [int(i) for i in np.flatnonzero(is_pass & (pr > 0)) if int(i) not in order][:1]
                if len(order) > 1:
                    job.searches += 1
                    stats["searches"] += 1
                    turns_searched[turn] = turns_searched.get(turn, 0) + 1
                    job.pending = {"cands": order, "scores": [[] for _ in order], "sig": sig, "left": len(order) * r,
                                   "chain": chain, "is_pass": [bool(is_pass[a]) for a in order]}  # fmt: skip
                    seeds = [int(job.rng.integers(1 << 62)) for _ in range(r)]  # shared: paired candidates
                    for c, a in enumerate(order):
                        for j in range(r):
                            nxt.append(Job(job.spec, job.history + [(a, sig)], parent=job, cand=c, seed=seeds[j]))
                    actions[n] = ABANDON  # the real game resumes, by replay, once its rollouts are scored
                    continue
            cdf = np.cumsum(pr)
            a = min(int(np.searchsorted(cdf, job.rng.random() * cdf[-1], side="right")), len(pr) - 1)
            job.history.append((a, sig))
            job.pos += 1
            actions[n] = a
        return actions

    def is_search_point(job, g, turn):
        if args.max_searches is not None and job.searches >= args.max_searches:
            return False
        if args.search == "main":
            return (int(g[18]) == C.MSG_SELECT_IDLECMD and int(g[2]) == 1 and int(g[4]) == MAIN1
                    and turn not in job.searched_turns)  # fmt: skip
        mine, turns = CHAIN_TURNS[args.search]
        return int(g[18]) == C.MSG_SELECT_CHAIN and int(g[2]) == mine and turn in turns

    jobs = real_jobs
    while jobs:  # a round: real games up to their next search, the rollouts of the last round's searches
        nxt = []
        drive(env, [j.spec for j in jobs], decide, on_result, start=lambda i, spec, jobs=jobs: jobs[i])
        jobs = nxt
    scores = np.array([j.result for j in real_jobs], dtype=float)
    ok = np.isfinite(scores)
    np.save(Path(out).with_suffix(".npy"), scores)
    res = {"checkpoint": ckpt, "specs": n_specs, "k": k, "r": r, "games": int(ok.sum()),
           "searcher_win_rate": float(scores[ok].mean()), "ci95": float(1.96 * scores[ok].std() / np.sqrt(ok.sum())),
           "search": args.search, "max_searches": args.max_searches, "seconds": time.time() - t0, **stats,
           "directions": directions, "searches_by_turn": dict(sorted(turns_searched.items()))}  # fmt: skip
    if args.control is not None:
        ctrl = np.load(args.control)
        both = ok & np.isfinite(ctrl)
        first = np.array([sp.seat_of_deck(0) == 0 for sp in specs])  # the searcher went first (seat 0)
        for name, sel in (("all", both), ("searcher_first", both & first), ("searcher_second", both & ~first)):
            d = scores[sel] - ctrl[sel]
            res[f"paired_{name}"] = {"games": int(sel.sum()), "searcher": float(scores[sel].mean()),
                                     "control": float(ctrl[sel].mean()), "diff": float(d.mean()),
                                     "ci95": float(1.96 * d.std(ddof=1) / np.sqrt(len(d))),
                                     "changed_games": int((d != 0).sum())}  # fmt: skip
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
