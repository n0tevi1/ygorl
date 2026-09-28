"""Upper bound for decision-time search (#62 stage C): an ORACLE one-step search that sees the true engine state.

The searcher (policy P on deck a) plays P (deck b). At its first idle-command decision of each own main phase 1 it
takes the policy's top-k legal actions (prob >= MIN_P) and, for each, plays R rollouts to the end of the game with P
on both seats (sampled, fresh seeds), starting from the same engine state; it plays the action with the best mean
score. Rollouts rebuild the state by replaying the game's own seed and action prefix in the batched C++ env
(deterministic), so they also know both players' future draws: this cheats on hidden information on purpose. If even
this does not beat P, PIMC (which can only do worse) is not worth building.

Usage: tools/oracle_search.py CHECKPOINT SPECS K R OUT.json [DECK_DIR]"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from ygorl.cards.ydk import load_ydk
from ygorl.engine import constants as C
from ygorl.engine.duel import DuelConfig, default_cards
from ygorl.env import GameSpec
from ygorl.env.encoded import EncodedVecEnv
from ygorl.eval.arena import derive_seed
from ygorl.nets import collate
from ygorl.train.checkpoint import load_actor

MIN_P = 0.02
MAIN1 = 3  # globals[4]: phase ordinal + 1 (draw 1, standby 2, main1 3, ...)


class Job:
    def __init__(self, spec, prefix, *, parent=None, cand=None, seed=0, real=False):
        self.spec, self.prefix, self.parent, self.cand, self.real = spec, list(prefix), parent, cand, real
        self.pos = 0  # decisions answered so far
        self.actions = []  # every answered decision (the prefix for rollouts)
        self.rng = np.random.default_rng(seed)
        self.searched_turns = set()
        self.pending = None  # real game waiting on a search: {"ev", "cands", "scores", "left"}
        self.result = None
        self.trace = []  # (player, turn, legal actions) at every answered decision: rollouts check their replay


def main():
    ckpt, n_specs, k, r, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    deck_dir = Path(sys.argv[6] if len(sys.argv) > 6 else "out/corpus/train")
    cards = default_cards()
    pol = load_actor(ckpt)
    net = pol.net.to("cuda").eval()
    decks = [load_ydk(p) for p in sorted(deck_dir.glob("*.ydk"))]
    rng = np.random.default_rng(0)
    config = DuelConfig(max_decisions=4000)
    from ygorl.engine.duel import shuffle_deck

    specs = []
    for i in range(n_specs):
        a, b = rng.choice(len(decks), 2, replace=False)
        s = derive_seed(7, i)
        da = decks[a].__class__(main=tuple(shuffle_deck(decks[a].main, s, 0)), extra=decks[a].extra, name=decks[a].name)
        db = decks[b].__class__(main=tuple(shuffle_deck(decks[b].main, s, 1)), extra=decks[b].extra, name=decks[b].name)
        for first in (0, 1):
            specs.append(GameSpec(seed=s, deck_a=da, deck_b=db, first=first,
                                  config=DuelConfig(max_decisions=4000, shuffle_decks=False)))  # fmt: skip
    del config
    num_envs = 512
    env = EncodedVecEnv(num_envs, 8, cards=cards, vocab=pol.vocab, event_length=pol.event_length, skip_forced=True)
    real_jobs = [Job(sp, [], seed=derive_seed(8, i), real=True) for i, sp in enumerate(specs)]
    queue = list(reversed(real_jobs))  # stack: rollouts are pushed on top so searches finish fast
    slots: dict[int, Job] = {}
    free = list(range(num_envs))
    stats = {"searches": 0, "rollouts": 0, "changed": 0}
    t0 = time.time()

    def launch():
        while free and queue:
            job = queue.pop()
            e = free.pop()
            env.reset(e, job.spec)
            slots[e] = job

    def finish_rollout(job, score):
        p = job.parent.pending
        p["scores"][job.cand].append(score)
        p["left"] -= 1
        if p["left"] == 0:
            means = [np.mean(s) for s in p["scores"]]
            best = int(np.argmax(means))
            stats["changed"] += best != 0
            parent = job.parent
            ev_env = p["env"]
            a = p["cands"][best]
            parent.pending = None
            parent.actions.append(a)
            parent.trace.append(p["sig"])
            parent.pos += 1
            env.step(ev_env, a)

    launch()
    while slots:
        evs = env.recv(1)
        ready = []
        for ev in evs:
            job = slots[ev.env_id]
            if ev.result is not None:
                w = ev.result.get("winner")
                me = job.spec.first  # engine seat holding deck a (the searcher)
                failed = str(ev.result.get("reason", "")) == "error"
                score = np.nan if failed else 0.5 if w is None else float(w == me)
                del slots[ev.env_id]
                free.append(ev.env_id)
                if job.real:
                    job.result = score
                else:
                    stats["rollouts"] += 1
                    finish_rollout(job, 0.5 if np.isnan(score) else score)
                continue
            if job.pos < len(job.prefix):  # replaying the parent's history (plus the candidate at the end)
                g = ev.obs["globals"]
                tr = job.parent.trace
                want = tr[job.pos] if job.pos < len(tr) else job.parent.pending["sig"]
                if want != (ev.player, int(g[3]), int(g[20])):
                    stats["desync"] = stats.get("desync", 0) + 1
                a = job.prefix[job.pos]
                job.pos += 1
                env.step(ev.env_id, a)
                continue
            ready.append(ev)
        if ready:
            with torch.no_grad():
                logits = net(collate([ev.obs for ev in ready], "cuda")).logits.float()
            probs = torch.softmax(logits, -1).cpu().numpy().astype(np.float64)
            for ev, pr in zip(ready, probs):
                job = slots[ev.env_id]
                g = ev.obs["globals"]
                me = job.spec.first
                turn = int(g[3])
                if (job.real and ev.player == me and int(g[18]) == C.MSG_SELECT_IDLECMD and int(g[2]) == 1
                        and int(g[4]) == MAIN1 and turn not in job.searched_turns):  # fmt: skip
                    job.searched_turns.add(turn)
                    order = [int(i) for i in np.argsort(-pr) if pr[i] >= MIN_P][:k]
                    if len(order) > 1:
                        stats["searches"] += 1
                        job.pending = {"env": ev.env_id, "cands": order, "scores": [[] for _ in order],
                                       "sig": (ev.player, turn, int(g[20])),
                                       "left": len(order) * r}  # fmt: skip
                        seeds = [int(job.rng.integers(1 << 62)) for _ in range(r)]  # shared: paired candidates
                        for c, a in enumerate(order):
                            for j in range(r):
                                queue.append(Job(job.spec, job.actions + [a], parent=job, cand=c, seed=seeds[j]))
                        continue
                cdf = np.cumsum(pr)
                a = min(int(np.searchsorted(cdf, job.rng.random() * cdf[-1], side="right")), len(pr) - 1)
                job.actions.append(a)
                job.trace.append((ev.player, turn, int(g[20])))
                job.pos += 1
                env.step(ev.env_id, a)
        launch()
    scores = np.array([j.result for j in real_jobs], dtype=float)
    ok = np.isfinite(scores)
    np.save(Path(out).with_suffix(".npy"), scores)
    res = {"checkpoint": ckpt, "specs": n_specs, "k": k, "r": r, "games": int(ok.sum()),
           "searcher_win_rate": float(scores[ok].mean()), "ci95": float(1.96 * scores[ok].std() / np.sqrt(ok.sum())),
           "seconds": time.time() - t0, **stats}  # fmt: skip
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
