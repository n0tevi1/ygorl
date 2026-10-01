"""Plan-level ORACLE search over interruption timing (strength diagnosis, #83).

The one-step chain-prompt search (tools/oracle_search.py ``--search chain-opp``) changes one decision and lets the
policy continue, which then fires at the next chain prompt anyway: "hold the interruption for the key play" is a
plan over several decisions. This tool searches such plans.

Searcher: the first player (seat 0). At its first decision of the opponent's turn ``--turn`` (2 by default; 4 for
the next one), the searcher's real game stops and every plan is evaluated by rollouts to the end of the game:

- ``policy``: no override (the policy's own play, the baseline);
- ``never``: pass every chain prompt of that turn at which an activation is available;
- ``j`` (1..J): pass the first j-1 of those prompts ("opportunities"), then take the policy's most likely
  activation at the j-th; the rest of the turn the policy plays normally.

After the turn the policy plays both seats. Rollouts rebuild the state by replaying the game's own seed and action
prefix (deterministic), so the search knows both players' hands and future draws: an upper bound. Every plan is
rolled out with the same 2R seeds (common random numbers; every decision consumes one random number whether the plan
or the policy answers it, so plans that make the same choices play the same game). The plan is chosen on the first
R seeds ("select") and valued on the other R ("eval"), so the reported headroom is not the optimistic maximum of
noisy means (that one is reported too, as ``in_sample``).

Realised arm: the searcher's real game then resumes and plays the chosen plan. ``--no-search`` plays the same real
games with the policy only (the control: the same random streams, so a game whose chosen plan is ``policy`` is the
same game). With ``--control CONTROL.npy`` the output adds the paired realised difference.

Every turn-``--turn`` activation and summon of either seat (as chosen at a decision; forced triggers skipped by the
environment are not seen) is logged per rollout of the best plan and of ``policy``, with the searcher's firing
point, to read what the best plan's interruption hit (``targets`` in the output).

``--demos OUT.npz`` additionally replays, for every searched situation, ``--demo-rollouts`` fresh rollouts of the
chosen plan up to the end of the turn and saves the searcher's opportunity decisions (observation, plan action):
demonstrations for tools/train_response_prior.py.

``--games LIST.json`` searches only those real games (e.g. the contested situations of a first run, re-valued with
more rollouts); ``--salt`` gives fresh rollout seeds.

Usage: tools/timing_search.py CHECKPOINT SPECS R OUT.json [DECK_DIR] [--turn 2] [--max-j 8] [--device cuda]
           [--envs 512] [--no-search] [--control CONTROL.npy] [--demos OUT.npz --demo-rollouts 8]
           [--games LIST.json] [--salt N]
       tools/timing_search.py CHECKPOINT SPECS 0 OUT.json [DECK_DIR] --effect [--turn 2] [--lines 3] [--conts 12]
           [--max-targets 3] [--pairs 3] [--games LIST.json]

``--effect`` replaces the timing plans by the interruption-effect / choke-point analysis of
tools/interruption_effect.py on the same situations: what each legal interruption (card, effect, follow-up
choice; also pairs of them) does to the opponent's turn and to the win rate, against passing the whole turn."""

from __future__ import annotations

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

SEARCHER = 0  # engine seat of the searcher: the first player, who responds on turns 2 and 4
PASS = ACTION_KINDS.index("pass") + 1  # actions[:, 0]: action kind + 1
LOGGED = {ACTION_KINDS.index(k) + 1: k for k in ("activate", "chain", "summon", "spsummon")}


def opportunity(ev, turn: int) -> bool:
    """A searcher chain prompt on the opponent's turn ``turn`` at which "pass" and an activation are both legal."""
    g = ev.obs["globals"]
    if ev.player != SEARCHER or int(g[18]) != C.MSG_SELECT_CHAIN or int(g[2]) != 0 or int(g[3]) != turn:
        return False
    kinds = np.asarray(ev.obs["actions"])[: int(g[20]), 0]
    return bool((kinds == PASS).any() and (kinds != PASS).any())


class Job:
    """A game on the driver: the searcher's real game, a rollout of one plan, or a demonstration rollout. Each time
    it starts it first replays ``replay`` ((action, signature) pairs)."""

    def __init__(self, spec, replay, *, plan="policy", parent=None, cand=None, seed=0, kind="real", seed_index=0):
        self.spec, self.replay, self.plan, self.parent, self.cand = spec, list(replay), plan, parent, cand
        self.kind, self.seed_index = kind, seed_index
        self.pos = 0  # decisions answered since this job last started
        self.history = []  # (action, signature) of every answered decision (real games: the prefix for rollouts)
        self.rng = np.random.default_rng(seed)
        self.searched = False  # the real game reached the searched turn (its random streams are split there)
        self.history_marked = False  # its rollouts were launched
        self.opps = 0  # opportunities met in the searched turn
        self.fired = False
        self.log = []  # the searched turn: [seat, kind, code, opportunity index or -1]
        self.pending = None
        self.result = None
        self.demo_rows = []  # (obs, action) of the searcher's opportunity decisions (demonstration rollouts)


def build_specs(deck_dir: Path, n_specs: int):
    """The games (random corpus pairings, each with both first players) and their (first, second) deck names."""
    deck_paths = sorted(deck_dir.glob("*.ydk"))
    decks = [load_ydk(p) for p in deck_paths]
    rng = np.random.default_rng(0)
    specs, names = [], []
    for i in range(n_specs):
        a, b = rng.choice(len(decks), 2, replace=False)
        s = derive_seed(7, i)
        da = replace(decks[a], main=tuple(shuffle_deck(decks[a].main, s, 0)))
        db = replace(decks[b], main=tuple(shuffle_deck(decks[b].main, s, 1)))
        for first in (0, 1):
            specs.append(GameSpec(seed=s, deck_a=da, deck_b=db, first=first,
                                  config=DuelConfig(max_decisions=4000, shuffle_decks=False)))  # fmt: skip
            pair = (deck_paths[a].stem, deck_paths[b].stem)
            names.append(pair if first == 0 else pair[::-1])  # (first player's deck, second player's deck)
    return specs, names


def main():  # noqa: C901 - one search loop
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("specs", type=int, help="deck pairings (each played with both first players)")
    ap.add_argument("r", type=int, help="rollouts per plan and half (select / eval): 2R per plan")
    ap.add_argument("out")
    ap.add_argument("deck_dir", nargs="?", type=Path, default=Path("out/corpus/train"))
    ap.add_argument("--turn", type=int, default=2, help="the opponent's turn the plans cover (2 or 4)")
    ap.add_argument("--max-j", type=int, default=8, help="J: plans fire at opportunity 1..J")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--no-search", action="store_true", help="control: the real games with the policy only")
    ap.add_argument("--control", type=Path, help="scores (.npy) of the --no-search run on the same specs")
    ap.add_argument("--demos", type=Path, help="save demonstrations of the chosen plans (.npz)")
    ap.add_argument("--demo-rollouts", type=int, default=8)
    ap.add_argument("--games", type=Path, help="JSON list of real-game indices to search (default: all)")
    ap.add_argument("--salt", type=int, default=0, help="> 0: fresh rollout seeds (same real games and situations)")
    ap.add_argument("--effect", action="store_true", help="interruption-effect / choke-point analysis (see above)")
    ap.add_argument("--lines", type=int, default=3, help="--effect: opponent lines (turn seeds) per situation")
    ap.add_argument("--conts", type=int, default=8, help="--effect: continuations per line and plan (two halves)")
    ap.add_argument("--max-targets", type=int, default=3, help="--effect: follow-up choices tried per activation")
    ap.add_argument("--pairs", type=int, default=3, help="--effect: best singles extended by a second interruption")
    args = ap.parse_args()
    if args.turn % 2:
        ap.error("--turn must be an opponent's turn of the first player (even)")
    if args.effect:
        from interruption_effect import effect_main

        return effect_main(args)
    ckpt, n_specs, r, out, turn = args.checkpoint, args.specs, args.r, args.out, args.turn
    device = torch_device(args.device)
    cards = default_cards()
    pol = load_actor(ckpt)
    net = pol.net.to(device).eval()
    vocab = pol.vocab
    specs, names = build_specs(args.deck_dir, n_specs)
    plans = ["policy", "never", *range(1, args.max_j + 1)]
    env = EncodedVecEnv(args.envs, 8, cards=cards, vocab=vocab, event_length=pol.event_length, skip_forced=True)
    real_jobs = [Job(sp, [], seed=derive_seed(8, i)) for i, sp in enumerate(specs)]
    situations: dict[int, dict] = {}
    demo_obs, demo_act, demo_sit = [], [], []
    only = None if args.games is None else set(json.loads(args.games.read_text()))

    def wanted(job):
        return only is None or real_jobs.index(job) in only

    stats = {"searches": 0, "rollouts": 0, "desync": 0}
    t0 = time.time()
    nxt: list[Job] = []

    def code_of(index: int) -> int:
        try:
            return vocab.password(int(index))
        except (KeyError, IndexError):
            return 0

    def signature(ev):
        g = ev.obs["globals"]
        return ev.player, int(g[3]), int(g[20])

    def finish_search(job):
        p = job.pending
        sc = np.array(p["scores"], dtype=float)  # [plan, 2R]
        sel, ev_ = sc[:, :r].mean(1), sc[:, r:].mean(1)
        best = int(np.argmax(sel))  # ties: the first plan, i.e. policy
        idx = real_jobs.index(job)
        situations[idx] = {"game": idx, "first_deck": names[idx][0], "second_deck": names[idx][1],
                           "select": sel.tolist(), "eval": ev_.tolist(), "all": sc.mean(1).tolist(),
                           "best": best, "best_plan": plans[best], "opportunities": p["opps"],
                           "fired_at": p["fired_at"], "logs": p["logs"]}  # fmt: skip
        job.pending, job.plan = None, plans[best]
        job.replay, job.pos = list(job.history), 0
        nxt.append(job)
        if args.demos is not None and plans[best] != "policy":
            for d in range(args.demo_rollouts):
                nxt.append(Job(job.spec, job.history, plan=plans[best], parent=job, seed=derive_seed(9, idx, d),
                               kind="demo"))  # fmt: skip

    def on_result(game, result):
        job = game.state
        w = result.get("winner")
        failed = str(result.get("reason", "")) == "error"
        score = np.nan if failed else 0.5 if w is None else float(w == SEARCHER)
        if job.kind == "real":
            job.result = score
            return
        if job.kind == "demo":
            return
        stats["rollouts"] += 1
        p = job.parent.pending
        p["scores"][job.cand][job.seed_index] = 0.5 if np.isnan(score) else score
        p["opps"][job.cand].append(job.opps)
        p["fired_at"][job.cand].append(next((e[3] for e in job.log if e[3] >= 0), -1))
        if job.seed_index in (r, r + 1):  # the first two eval seeds: keep the turn's log
            p["logs"][str(plans[job.cand])].append(job.log)
        p["left"] -= 1
        if p["left"] == 0:
            finish_search(job.parent)

    def plan_action(job, pr, ev):
        """The plan's answer at an opportunity, or None for the policy's."""
        if job.plan == "policy" or job.fired:
            return None
        job.opps += 1  # only counted while the plan holds
        kinds = np.asarray(ev.obs["actions"])[: len(pr), 0]
        if job.plan == "never" or job.opps < job.plan:
            return int(np.flatnonzero(kinds == PASS)[0])
        job.fired = True
        acts = np.flatnonzero(kinds != PASS)
        return int(acts[np.argmax(pr[acts])])

    @torch.no_grad()
    def decide(ready):
        actions = [None] * len(ready)
        play = []
        for n, (game, ev) in enumerate(ready):
            job = game.state
            if job.pos < len(job.replay):
                a, want = job.replay[job.pos]
                if want != signature(ev):
                    stats["desync"] += 1
                job.pos += 1
                actions[n] = a
                note(job, ev, a, -1)
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
            t = sig[1]
            if job.kind == "demo" and t > turn:
                actions[n] = ABANDON
                continue
            if job.kind == "real" and not job.searched and ev.player == SEARCHER and t == turn and int(g[2]) == 0:
                job.searched = True
                seeds = [int(job.rng.integers(1 << 62)) for _ in range(2 * r)]  # shared by every plan (CRN)
                if args.salt:
                    seeds = [derive_seed(sd, args.salt) for sd in seeds]
                job.rng = np.random.default_rng(int(job.rng.integers(1 << 62)))  # the real game's own stream
            if job.kind == "real" and job.searched and not job.history_marked and not args.no_search and wanted(job):
                job.history_marked = True
                stats["searches"] += 1
                job.pending = {"scores": [[np.nan] * (2 * r) for _ in plans], "left": len(plans) * 2 * r,
                               "opps": [[] for _ in plans], "fired_at": [[] for _ in plans],
                               "logs": {str(p): [] for p in plans}}  # fmt: skip
                for c, plan in enumerate(plans):
                    for j, sd in enumerate(seeds):
                        nxt.append(Job(job.spec, job.history, plan=plan, parent=job, cand=c, seed=sd,
                                       kind="rollout", seed_index=j))  # fmt: skip
                actions[n] = ABANDON  # resumes, by replay, once its rollouts are scored
                continue
            u = job.rng.random()  # consumed at every decision: common random numbers across plans
            a = None
            opp_idx = -1
            if t == turn and opportunity(ev, turn) and (job.kind != "real" or job.searched):
                was_fired = job.fired
                a = plan_action(job, pr, ev)
                if job.kind == "demo" and a is not None:
                    demo_obs.append({k: np.asarray(v).copy() for k, v in ev.obs.items()})
                    demo_act.append(a)
                    demo_sit.append(real_jobs.index(job.parent))
                if job.fired and not was_fired:
                    opp_idx = job.opps
            if a is None:
                cdf = np.cumsum(pr)
                a = min(int(np.searchsorted(cdf, u * cdf[-1], side="right")), len(pr) - 1)
                if job.plan == "policy" and opportunity(ev, turn) and not job.fired:
                    job.opps += 1
                    if np.asarray(ev.obs["actions"])[a, 0] != PASS:
                        job.fired, opp_idx = True, job.opps
            job.history.append((a, sig))
            job.pos += 1
            actions[n] = a
            note(job, ev, a, opp_idx)
        return actions

    def note(job, ev, a, opp_idx):
        """Log the searched turn's activations and summons (rollouts only)."""
        if job.kind != "rollout" or int(ev.obs["globals"][3]) != turn:
            return
        row = np.asarray(ev.obs["actions"])[a]
        kind = LOGGED.get(int(row[0]))
        if kind is None:
            return
        code = code_of(row[2]) or code_of(row[3])
        job.log.append([int(ev.player), kind, code, opp_idx])

    jobs = real_jobs
    while jobs:  # a round: real games up to their search, the rollouts, the real games' resumption
        nxt = []
        drive(env, [j.spec for j in jobs], decide, on_result, start=lambda i, spec, jobs=jobs: jobs[i])
        jobs = nxt
        print(f"{time.time() - t0:.0f}s: {stats}, next round {len(jobs)} jobs", flush=True)

    scores = np.array([j.result for j in real_jobs], dtype=float)
    ok = np.isfinite(scores)
    np.save(Path(out).with_suffix(".npy"), scores)
    res = {"checkpoint": ckpt, "specs": n_specs, "r": r, "turn": turn, "plans": [str(p) for p in plans],
           "games": int(ok.sum()), "searcher_win_rate": float(scores[ok].mean()),
           "seconds": time.time() - t0, **stats, "no_search": args.no_search}  # fmt: skip
    if situations:
        res["summary"] = summarize(situations, plans, len(specs), cards)
    if args.control is not None:
        ctrl = np.load(args.control)
        both = ok & np.isfinite(ctrl)
        d = scores[both] - ctrl[both]
        res["realised"] = {"games": int(both.sum()), "searcher": float(scores[both].mean()),
                           "control": float(ctrl[both].mean()), "diff": float(d.mean()),
                           "ci95": float(1.96 * d.std(ddof=1) / np.sqrt(len(d))),
                           "changed_games": int((d != 0).sum())}  # fmt: skip
    res["situations"] = [situations[k] for k in sorted(situations)]
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "situations"}, indent=1))
    if args.demos is not None:
        keys = demo_obs[0].keys() if demo_obs else ()
        np.savez_compressed(args.demos, **{f"obs_{k}": np.stack([o[k] for o in demo_obs]) for k in keys},
                            action=np.array(demo_act, dtype=np.int64), situation=np.array(demo_sit, dtype=np.int64),
                            checkpoint=np.array(ckpt), turn=np.array(turn))  # fmt: skip
        print(f"demos: {len(demo_act)} decisions from {len(set(demo_sit))} situations -> {args.demos}")


def _ci(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def summarize(situations: dict, plans: list, n_games: int, cards) -> dict:
    """Headroom of the plans over the policy, which plan wins, and what the chosen interruption hit."""
    sits = list(situations.values())
    P = np.array([s["eval"] for s in sits])  # [situation, plan], eval half
    S = np.array([s["select"] for s in sits])
    A = np.array([s["all"] for s in sits])
    best = np.array([s["best"] for s in sits])
    k = np.arange(len(sits))
    d_eval = P[k, best] - P[:, 0]  # unbiased: chosen on select, valued on eval
    # both halves in turn: chosen on one, valued on the other (twice the rollouts of headroom_eval, still unbiased)
    d_cross = (d_eval + S[k, P.argmax(1)] - S[:, 0]) / 2
    d_in = A.max(1) - A[:, 0]  # optimistic in-sample maximum
    pad = n_games - len(sits)  # unsearched games (ended before the turn) count 0 in the all-games mean
    out = {"situations": len(sits), "games": n_games,
           "policy_value": float(P[:, 0].mean()),
           "headroom_eval": {"diff": float(d_eval.mean()), "ci95": _ci(d_eval),
                             "diff_all_games": float(d_eval.sum() / n_games),
                             "ci95_all_games": _ci(np.concatenate([d_eval, np.zeros(pad)]))},
           "headroom_crossfit": {"diff": float(d_cross.mean()), "ci95": _ci(d_cross)},
           "headroom_in_sample":{"diff": float(d_in.mean()), "ci95": _ci(d_in)},
           "plan_value_minus_policy": {str(p): [float((P[:, i] - P[:, 0]).mean()), _ci(P[:, i] - P[:, 0])]
                                       for i, p in enumerate(plans)},
           "best_plan_counts": {str(p): int((best == i).sum()) for i, p in enumerate(plans)}}  # fmt: skip
    js = [plans[b] for b in best if isinstance(plans[b], int)]
    out["mean_best_j"] = float(np.mean(js)) if js else float("nan")
    # later vs first opportunity: the best of j >= 2 (chosen on select) against j = 1, valued on eval
    later = list(range(3, len(plans)))
    if later:
        bl = np.array(later)[S[:, later].argmax(1)]
        dl = P[k, bl] - P[:, 2]
        has_two = np.array([np.mean(s["opportunities"][1]) >= 2 for s in sits])  # never: every opportunity counted
        out["later_vs_first"] = {"diff": float(dl.mean()), "ci95": _ci(dl), "later_wins": float((dl > 0).mean()),
                                 "first_wins": float((dl < 0).mean()),
                                 "situations_with_2plus_opportunities": int(has_two.sum()),
                                 "diff_2plus": float(dl[has_two].mean()) if has_two.any() else float("nan")}  # fmt: skip
        nv = P[:, 1] - P[:, 2]
        out["never_vs_first"] = {"diff": float(nv.mean()), "ci95": _ci(nv)}
    out["opportunities_mean"] = float(np.mean([np.mean(s["opportunities"][1]) for s in sits]))  # under never
    out["policy_fires_at"] = _hist([f for s in sits for f in s["fired_at"][0]])
    out["best_fires_at"] = _hist([f for s, b in zip(sits, best) for f in s["fired_at"][b]])
    out["targets"] = targets(sits, plans, cards)
    return out


def _hist(xs) -> dict:
    xs = list(xs)
    vals, cnt = np.unique(xs, return_counts=True)
    return {("none" if v < 0 else str(int(v))): round(float(c) / len(xs), 3) for v, c in zip(vals, cnt)}


def frame(cards, code: int) -> str:
    c = cards.get(code) if code else None
    if c is None:
        return "unknown"
    t = c.type
    if t & C.TYPE_MONSTER:
        return "extra_monster" if t & (C.TYPE_FUSION | C.TYPE_SYNCHRO | C.TYPE_XYZ | C.TYPE_LINK) else "main_monster"
    return "spell" if t & C.TYPE_SPELL else "trap"


def targets(sits, plans, cards) -> dict:
    """What the firing hit, for the chosen plan and for the policy: the opponent's last logged play before the
    searcher's interruption, its position among the opponent's plays of the turn, and its card frame."""
    res = {}
    for label, pick in (("best", lambda s: str(plans[s["best"]])), ("policy", lambda s: "policy")):
        rows = []
        for s in sits:
            if label == "best" and s["best"] == 0:
                continue
            for log in s["logs"][pick(s)]:
                opp = [e for e in log if e[0] != SEARCHER]
                fire = next((i for i, e in enumerate(log) if e[0] == SEARCHER and e[3] >= 0), None)
                if fire is None:
                    rows.append({"fired": False, "opp_plays": len(opp)})
                    continue
                before = [e for e in log[:fire] if e[0] != SEARCHER]
                tgt = before[-1] if before else None
                rows.append({"fired": True, "opp_plays": len(opp), "k": len(before),
                             "rel": len(before) / max(1, len(opp)), "after": len(opp) - len(before),
                             "target_kind": tgt[1] if tgt else "none",
                             "target_frame": frame(cards, tgt[2]) if tgt else "none",
                             "target": cards[tgt[2]].name if tgt and tgt[2] in cards else "?",
                             "used": cards[log[fire][2]].name if log[fire][2] in cards else "?"})  # fmt: skip
        fired = [x for x in rows if x["fired"]]
        names: dict[str, int] = {}
        for x in fired:
            names[x["target"]] = names.get(x["target"], 0) + 1
        res[label] = {"logs": len(rows), "fired": len(fired),
                      "mean_k": float(np.mean([x["k"] for x in fired])) if fired else float("nan"),
                      "mean_rel": float(np.mean([x["rel"] for x in fired])) if fired else float("nan"),
                      "mean_opp_plays_after": float(np.mean([x["after"] for x in fired])) if fired else float("nan"),
                      "mean_opp_plays": float(np.mean([x["opp_plays"] for x in rows])) if rows else float("nan"),
                      "target_kind": _count(x["target_kind"] for x in fired),
                      "target_frame": _count(x["target_frame"] for x in fired),
                      "top_targets": sorted(names.items(), key=lambda kv: -kv[1])[:15]}  # fmt: skip
    return res


def _count(xs) -> dict:
    out: dict[str, int] = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out


if __name__ == "__main__":
    main()
