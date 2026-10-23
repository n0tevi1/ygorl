"""Interruption effect and choke points: ``tools/timing_search.py --effect`` (strength diagnosis, #83).

Same situations as the timing search: the first player (the searcher) at its first decision of the opponent's turn
``--turn``, on the real games of ``timing_search.py``. Instead of "fire at the j-th opportunity" it asks what each
interruption DOES to the opponent's turn, and which opponent play is the choke point. ORACLE: rollouts replay the
true game, so the searcher knows the opponent's hand and both decks.

**Lines.** The opponent's turn is fixed per line seed: in the searched turn each seat samples from its own random
stream (seeded by the line), so the opponent makes the same choices as long as the states agree, and a plan that
passes until some point reproduces the line up to there. A line's continuation after the turn (both seats, the
policy) samples from a continuation seed shared by every plan; ``--conts`` continuations per line, split in two
halves (select / eval). The searched turn itself is deterministic per (line, plan), so its outcome is measured once.

**Plans** (per line), the searcher's decisions in the searched turn:

- ``pass``: pass every chain prompt at which an activation is legal (an "opportunity"); other decisions: the
  policy's most likely action. This is line (a).
- ``policy``: the policy's own (sampled) play.
- single ``answer``: in line (a), the first opportunity after each opponent play (activation or summon as chosen at
  a decision; "play 0" is the first opportunity, before any play), EVERY legal activation there and, when the next
  searcher decision is a follow-up choice (target, cost, ...), each of its ``--max-targets`` most likely options;
  then pass every other opportunity of the turn.
- ``pair``: the ``--pairs`` best singles (by the select half) plus a second interruption with another card on a
  later play of line (a) (keyed by the opponent play: its kind, card and occurrence; the card must still be legal).

**Turn outcome**, at the end of the searched turn (the first decision of the next turn, or the game end): the
opponent's plays after the interruption, its monsters, total face-up ATK, Extra Deck monsters and set cards; whether
it reached its payoff (the highest-ATK monster of its line-(a) end board, Extra Deck first); the damage the searcher
took that turn and whether it lost in that turn; the searcher's monsters that survived; then the game result.
"Stopped the combo": at most 1 opponent play after the interruption, or the payoff not reached.

**Choke point** (per line): the single answer with the best win rate on the select half (valued on the eval half);
the "board choke point" is the single answer that leaves the smallest opponent board (total ATK, then monsters).
The policy "hits" the choke point when its first interruption answers the same opponent play (and "exactly" with
the same card). Choke-point plays are classified as: ``none`` (no opponent play yet), ``payoff`` (the line-(a)
payoff itself), ``starter`` (the opponent's first play), ``extra_summon`` (an Extra Deck monster), ``removal``,
``search``, ``extender`` (other main-deck monster plays) or ``other`` (other spells / traps).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from ygorl.engine import constants as C
from ygorl.engine.duel import default_cards
from ygorl.env.driver import ABANDON, drive
from ygorl.env.encoded import EncodedVecEnv
from ygorl.env.encoding import ACTION_KINDS, ATTACK, CARD_INDEX, CONTROLLER, LOCATION, POSITION, TYPE, VISIBLE
from ygorl.eval.arena import derive_seed
from ygorl.nets.batch import collate, policy_logits
from ygorl.train.checkpoint import load_actor, torch_device

SEARCHER = 0
PASS = ACTION_KINDS.index("pass") + 1
LOGGED = {ACTION_KINDS.index(k) + 1: k for k in ("activate", "chain", "summon", "spsummon")}
MZONE, SZONE = 3, 4  # encoding LOCATION_ENUM
EXTRA_TYPES = C.TYPE_FUSION | C.TYPE_SYNCHRO | C.TYPE_XYZ | C.TYPE_LINK
CAT_REMOVAL = 0x1 | 0x2 | 0x80  # cdb category: destroy monster / spell-trap, banish
CAT_SEARCH = 0x20 | 0x200  # to hand, search


def board(obs, viewer: int) -> dict:
    """Both sides' field from one observation (side 0 of the table is the viewer)."""
    t = np.asarray(obs["cards"])
    out = {}
    for who, side in (("me", 0 if viewer == SEARCHER else 1), ("op", 1 if viewer == SEARCHER else 0)):
        rows = t[(t[:, CONTROLLER] == side) & (t[:, CARD_INDEX] > 0)]
        mons = rows[rows[:, LOCATION] == MZONE]
        up = mons[(mons[:, VISIBLE] == 1) & np.isin(mons[:, POSITION], (1, 3))]
        st = rows[rows[:, LOCATION] == SZONE]
        out[who] = {"monsters": int(len(mons)), "atk": int(up[:, ATTACK].sum()),
                    "extra": int(((up[:, TYPE] & EXTRA_TYPES) > 0).sum()),
                    "sets": int((np.isin(st[:, POSITION], (2, 4)) | (st[:, VISIBLE] == 0)).sum()),
                    "codes": [int(c) for c in up[:, CARD_INDEX]], "atks": [int(a) for a in up[:, ATTACK]],
                    "types": [int(x) for x in up[:, TYPE]]}  # fmt: skip
    g = np.asarray(obs["globals"])
    me_lp, op_lp = (int(g[5]), int(g[6])) if viewer == SEARCHER else (int(g[6]), int(g[5]))
    out["lp"] = [me_lp, op_lp]
    return out


class EJob:
    """One game on the driver. ``script``: the searcher's actions at its first decisions of the searched turn (None:
    the default); ``second``: (opponent play key, card code) of a pair's second interruption."""

    def __init__(self, sit, replay, *, kind, line=0, cont=0, plan="pass", script=(), second=None, seed=0):
        self.sit, self.replay, self.kind, self.line, self.cont = sit, list(replay), kind, line, cont
        self.plan, self.script, self.second = plan, list(script), second
        self.pos = 0
        self.rng = np.random.default_rng(seed)  # real games only
        self.history = []
        self.my_i = 0  # searcher decisions since the prefix
        self.decisions = []  # probe: per searcher decision {opp, options, default, plays_before}
        self.plays = []  # opponent plays of the searched turn after the prefix: (kind, code)
        self.prefix_plays = []
        self.fires = []  # (opponent plays seen before it, card code, decision index)
        self.second_done = False
        self.start = None  # board at the search point
        self.end = None  # board at the end of the searched turn
        self.last_obs = None
        self.ended_in_turn = False
        self.win = None
        self.follow = None  # target probe: options of the follow-up decision


def effect_main(args) -> None:  # noqa: C901 - phases of one analysis
    from timing_search import build_specs  # the same real games as the timing search

    t0 = time.time()
    device = torch_device(args.device)
    cards = default_cards()
    pol = load_actor(args.checkpoint)
    net = pol.net.to(device).eval()
    vocab = pol.vocab
    turn, K, M = args.turn, args.lines, args.conts
    specs, names = build_specs(args.deck_dir, args.specs)
    env = EncodedVecEnv(args.envs, 8, cards=cards, vocab=vocab, event_length=pol.event_length, skip_forced=True)
    stats = {"desync": 0, "script_miss": 0, "rollouts": 0}
    only = None if args.games is None else set(json.loads(args.games.read_text()))

    def code_of(index: int) -> int:
        try:
            return vocab.password(int(index))
        except (KeyError, IndexError):
            return 0

    def sig(ev):
        g = ev.obs["globals"]
        return ev.player, int(g[3]), int(g[20])

    def is_opp(ev) -> bool:
        g = ev.obs["globals"]
        if ev.player != SEARCHER or int(g[18]) != C.MSG_SELECT_CHAIN or int(g[2]) != 0 or int(g[3]) != turn:
            return False
        kinds = np.asarray(ev.obs["actions"])[: int(g[20]), 0]
        return bool((kinds == PASS).any() and (kinds != PASS).any())

    def act_code(ev, a) -> int:
        row = np.asarray(ev.obs["actions"])[a]
        return code_of(row[2]) or code_of(row[3])

    def play_key(plays, upto):
        """(kind, code, occurrence) of the opponent play ``plays[upto - 1]``."""
        kind, code = plays[upto - 1]
        return kind, code, sum(1 for p in plays[:upto] if p == (kind, code))

    situations: dict[int, dict] = {}

    # ---------------------------------------------------------------- the driver callbacks
    def on_result(game, result):
        job = game.state
        w = result.get("winner")
        failed = str(result.get("reason", "")) == "error"
        job.win = 0.5 if failed or w is None else float(w == SEARCHER)
        if job.end is None and job.last_obs is not None:
            job.end = board(*job.last_obs)
            job.ended_in_turn = True
        if job.kind == "roll":
            stats["rollouts"] += 1

    @torch.no_grad()
    def decide(ready):
        actions = [None] * len(ready)
        play = []
        for n, (game, ev) in enumerate(ready):
            job = game.state
            if job.pos < len(job.replay):
                a, want = job.replay[job.pos]
                if want != sig(ev):
                    stats["desync"] += 1
                job.pos += 1
                actions[n] = a
                if job.kind != "real" and int(ev.obs["globals"][3]) == turn and ev.player != SEARCHER:
                    kind = LOGGED.get(int(np.asarray(ev.obs["actions"])[a, 0]))
                    if kind:
                        job.prefix_plays.append((kind, act_code(ev, a)))
            else:
                play.append(n)
        if not play:
            return actions
        logits = policy_logits(net, collate([ready[n][1].obs for n in play], device)).float()
        probs = torch.softmax(logits, -1).cpu().numpy().astype(np.float64)
        for n, pr in zip(play, probs):
            game, ev = ready[n]
            job, g = game.state, ev.obs["globals"]
            t = int(g[3])
            if job.kind == "real":
                if ev.player == SEARCHER and t == turn and int(g[2]) == 0:
                    idx = game.index
                    if only is None or idx in only:
                        situations[idx] = {"game": idx, "first_deck": names[idx][0], "second_deck": names[idx][1],
                                           "history": list(job.history)}  # fmt: skip
                    actions[n] = ABANDON
                    continue
                if t > turn:
                    actions[n] = ABANDON
                    continue
                u = job.rng.random()
                a = sample(pr, u)
                job.history.append((a, sig(ev)))
                actions[n] = a
                continue
            if t > turn:  # the searched turn is over
                if job.end is None:
                    job.end = board(ev.obs, ev.player)
                if job.kind != "roll":
                    actions[n] = ABANDON
                    continue
                actions[n] = sample(pr, job.rng_cont.random())
                continue
            job.last_obs = (ev.obs, ev.player)
            if ev.player != SEARCHER:
                a = sample(pr, job.rng_op.random())
                kind = LOGGED.get(int(np.asarray(ev.obs["actions"])[a, 0]))
                if kind:
                    job.plays.append((kind, act_code(ev, a)))
                actions[n] = a
                continue
            if job.start is None:
                job.start = board(ev.obs, ev.player)
            i, opp = job.my_i, is_opp(ev)
            job.my_i += 1
            nleg = int(g[20])  # legal rows (pr is padded to MAX_OPTIONS)
            kinds = np.asarray(ev.obs["actions"])[:nleg, 0]
            u = job.rng_me.random()
            if job.kind == "target" and i == len(job.script):
                if not opp and nleg > 1 and job.fires and job.fires[-1][2] == i - 1:
                    order = [int(x) for x in np.argsort(-pr[:nleg])[: args.max_targets]]
                    job.follow = order
                actions[n] = ABANDON
                continue
            if job.plan == "policy":
                a = sample(pr, u)
            elif i < len(job.script) and job.script[i] is not None:
                a = job.script[i]
                if a >= nleg:
                    stats["script_miss"] += 1
                    a = int(np.argmax(pr[:nleg]))
            elif opp:
                a = int(np.flatnonzero(kinds == PASS)[0])
                if job.second is not None and not job.second_done:
                    key, code = job.second
                    allp = job.prefix_plays + job.plays
                    seen = any(play_key(allp, k) == key for k in range(1, len(allp) + 1))
                    if seen:
                        hits = [x for x in np.flatnonzero(kinds != PASS) if act_code(ev, x) == code]
                        if hits:
                            a = int(max(hits, key=lambda x: pr[x]))
                            job.second_done = True
            else:
                a = int(np.argmax(pr[:nleg]))
            if opp and kinds[a] != PASS:
                job.fires.append((len(job.prefix_plays) + len(job.plays), act_code(ev, a), i))
            if job.kind == "probe":
                job.decisions.append({"opp": opp, "default": a,
                                      "options": [int(x) for x in np.flatnonzero(kinds != PASS)] if opp else [],
                                      "codes": [act_code(ev, x) for x in np.flatnonzero(kinds != PASS)] if opp else [],
                                      "plays_before": len(job.prefix_plays) + len(job.plays)})  # fmt: skip
            actions[n] = a
        return actions

    def sample(pr, u):
        cdf = np.cumsum(pr)
        return min(int(np.searchsorted(cdf, u * cdf[-1], side="right")), len(pr) - 1)

    def run(jobs):
        for j in jobs:
            if j.kind != "real":
                ls = derive_seed(11, j.sit, j.line, args.salt)
                j.rng_op = np.random.default_rng(derive_seed(ls, 1))
                j.rng_me = np.random.default_rng(derive_seed(ls, 2))
                j.rng_cont = np.random.default_rng(derive_seed(ls, 3, j.cont))
        drive(env, [specs[j.sit] for j in jobs], decide, on_result, start=lambda i, spec, jobs=jobs: jobs[i])
        print(f"{time.time() - t0:.0f}s: {len(jobs)} jobs, {stats}", flush=True)

    # ---------------------------------------------------------------- phase 0: the situations
    real = [EJob(i, [], kind="real", seed=derive_seed(8, i)) for i in range(len(specs))]
    run(real)
    sits = sorted(situations)
    hist = {s: situations[s]["history"] for s in sits}

    # ---------------------------------------------------------------- phase 1: line (a) per line seed
    probes = [EJob(s, hist[s], kind="probe", line=k) for s in sits for k in range(K)]
    run(probes)
    lines = {(p.sit, p.line): p for p in probes}

    # ---------------------------------------------------------------- phase 2: singles and their follow-up choices
    singles: dict[tuple, list[dict]] = {}
    tprobes = []
    for (s, k), p in lines.items():
        cands, seen_plays = [], set()
        for i, d in enumerate(p.decisions):
            if not d["opp"] or d["plays_before"] in seen_plays:
                continue
            seen_plays.add(d["plays_before"])
            base = [None] * i
            for x, code in zip(d["options"], d["codes"]):
                cands.append({"script": [*base, x], "at": i, "plays_before": d["plays_before"], "code": code})
        singles[(s, k)] = cands
        for c in cands:
            j = EJob(s, hist[s], kind="target", line=k, script=c["script"])
            c["_probe"] = j
            tprobes.append(j)
    run(tprobes)
    for cands in singles.values():
        more = []
        for c in cands:
            follow = c.pop("_probe").follow
            if follow:
                c["target_rank"] = 0
                c["script"] = [*c["script"], follow[0]]
                for r_, y in enumerate(follow[1:], 1):
                    more.append({**c, "script": [*c["script"][:-1], y], "target_rank": r_})
        cands.extend(more)

    # ---------------------------------------------------------------- phase 3: rollouts of pass, policy and singles
    def plan_jobs(s, k, plan, script=(), second=None):
        return [EJob(s, hist[s], kind="roll", line=k, cont=m, plan=plan, script=script, second=second)
                for m in range(M)]  # fmt: skip

    groups = []  # (key, plan description, jobs)
    for (s, k), cands in singles.items():
        groups.append(((s, k), {"plan": "pass"}, plan_jobs(s, k, "pass")))
        groups.append(((s, k), {"plan": "policy"}, plan_jobs(s, k, "policy")))
        for c in cands:
            groups.append(((s, k), {"plan": "single", **c}, plan_jobs(s, k, "single", c["script"])))
    run([j for g in groups for j in g[2]])

    # ---------------------------------------------------------------- phase 4: pairs
    results: dict[tuple, list[dict]] = {}
    for key, desc, jobs in groups:
        results.setdefault(key, []).append(outcome(desc, jobs, M))
    pair_groups = []
    for (s, k), res in results.items():
        p = lines[(s, k)]
        allp = p.prefix_plays + p.plays
        sing = sorted([r for r in res if r["plan"] == "single"], key=lambda r: -r["win_sel"])
        for first_ in sing[: args.pairs]:
            later = {}
            for r in sing:
                if r["plays_before"] > first_["plays_before"] and r["code"] != first_["code"] and r["plays_before"] > 0:
                    later.setdefault((r["plays_before"], r["code"]), r)
            for (pb, code), r in list(later.items())[:2]:
                second = (play_key(allp, pb), code)
                desc = {"plan": "pair", "script": first_["script"], "at": first_["at"],
                        "plays_before": first_["plays_before"], "code": first_["code"],
                        "second_plays_before": pb, "second_code": code}  # fmt: skip
                pair_groups.append(((s, k), desc, plan_jobs(s, k, "pair", first_["script"], second)))
    run([j for g in pair_groups for j in g[2]])
    for key, desc, jobs in pair_groups:
        results[key].append(outcome(desc, jobs, M))

    # ---------------------------------------------------------------- report
    rows = []
    for (s, k), res in sorted(results.items()):
        p = lines[(s, k)]
        allp = p.prefix_plays + p.plays
        rows.append(line_report(s, k, res, allp, cards, vocab, code_of))
    summary = summarize_effect(rows, len(specs))
    out = {"checkpoint": args.checkpoint, "specs": args.specs, "turn": turn, "lines": K, "conts": M,
           "situations": len(sits), "seconds": time.time() - t0, **stats, "summary": summary, "lines_detail": rows}  # fmt: skip
    Path(args.out).write_text(json.dumps(out, indent=1, default=_jsonable))
    print(json.dumps({k: v for k, v in out.items() if k != "lines_detail"}, indent=1, default=_jsonable))


def _jsonable(x):
    if isinstance(x, np.integer | np.floating):
        return x.item()
    if isinstance(x, tuple):
        return list(x)
    raise TypeError(type(x))


def outcome(desc: dict, jobs: list, M: int) -> dict:
    """A plan's turn outcome (deterministic per line: the first continuation's) and its win rates."""
    j = jobs[0]
    wins = np.array([x.win if x.win is not None else 0.5 for x in jobs], dtype=float)
    end = j.end or {"me": {"monsters": 0, "atk": 0, "extra": 0, "sets": 0, "codes": []},
                    "op": {"monsters": 0, "atk": 0, "extra": 0, "sets": 0, "codes": []}, "lp": [0, 0]}  # fmt: skip
    start = j.start or end
    allp = j.prefix_plays + j.plays
    fire_k = j.fires[0][0] if j.fires else None
    return dict(desc) | {
        "win_sel": float(wins[: M // 2].mean()), "win_eval": float(wins[M // 2 :].mean()), "win": float(wins.mean()),
        "op_plays": len(allp), "fire_k": fire_k, "fired_code": j.fires[0][1] if j.fires else None,
        "fires": len(j.fires), "op_plays_after": len(allp) - fire_k if fire_k is not None else None,
        "op_end": {k: end["op"][k] for k in ("monsters", "atk", "extra", "sets")}, "op_codes": end["op"]["codes"],
        "op_atks": end["op"].get("atks", []), "op_types": end["op"].get("types", []),
        "me_monsters_start": start["me"]["monsters"], "me_monsters_end": end["me"]["monsters"],
        "damage": max(0, start["lp"][0] - end["lp"][0]), "died": bool(j.ended_in_turn and j.win == 0.0),
        "plays": allp, "target": allp[fire_k - 1] if fire_k else None}  # fmt: skip


def line_report(s, k, res, allp_a, cards, vocab, code_of) -> dict:
    a = next(r for r in res if r["plan"] == "pass")
    pol = next(r for r in res if r["plan"] == "policy")
    # payoff of line (a): its highest-ATK face-up monster at the end of the turn, Extra Deck monsters first
    pay = None
    if a["op_codes"]:
        order = sorted(zip(a["op_codes"], a["op_atks"], a["op_types"]),
                       key=lambda x: (-(x[2] & EXTRA_TYPES > 0), -x[1]))  # fmt: skip
        pay = code_of(order[0][0])
    for r in res:
        reached = pay is not None and any(code_of(c) == pay for c in r["op_codes"])
        r["payoff_reached"] = reached
        after = r["op_plays_after"] if r["op_plays_after"] is not None else None
        r["stopped"] = r["fire_k"] is not None and (
            (after is not None and after <= 1) or (pay is not None and not reached)
        )
    sing = [r for r in res if r["plan"] == "single"]
    choke = max(sing, key=lambda r: (r["win_sel"], -r["op_end"]["atk"])) if sing else None
    bchoke = min(sing, key=lambda r: (r["op_end"]["atk"], r["op_end"]["monsters"], -r["win_sel"])) if sing else None
    allc = [r for r in res if r["plan"] in ("single", "pair")]
    best_any = max(allc, key=lambda r: r["win_sel"]) if allc else None
    return {"game": s, "line": k, "payoff": pay, "payoff_name": cards[pay].name if pay in cards else None,
            "pass": a, "policy": pol, "choke": choke, "board_choke": bchoke, "best_any": best_any,
            "n_singles": len(sing), "n_pairs": sum(r["plan"] == "pair" for r in res),
            "choke_class": classify(choke, pay, cards) if choke else None,
            "board_choke_class": classify(bchoke, pay, cards) if bchoke else None,
            "policy_class": classify(pol, pay, cards) if pol["fire_k"] is not None else None,
            "singles": sing}  # fmt: skip


def classify(r, pay, cards) -> str:
    k = r["fire_k"]
    if not k:
        return "none"
    kind, code = r["target"]
    c = cards.get(code)
    if pay is not None and code == pay:
        return "payoff"
    if k == 1:
        return "starter"
    if c is None:
        return "unknown"
    if c.type & C.TYPE_MONSTER and c.type & EXTRA_TYPES:
        return "extra_summon"
    if c.category & CAT_REMOVAL:
        return "removal"
    if c.category & CAT_SEARCH:
        return "search"
    if c.type & C.TYPE_MONSTER:
        return "extender"
    return "other"


def _ci(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def _mean(xs) -> float:
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else float("nan")


def summarize_effect(rows: list[dict], n_games: int) -> dict:
    """Per situation (lines averaged), then over situations."""
    by_sit: dict[int, list[dict]] = {}
    for r in rows:
        if r["choke"] is not None:
            by_sit.setdefault(r["game"], []).append(r)

    def per_sit(f):
        return np.array([np.mean([f(r) for r in rs]) for rs in by_sit.values()])

    out = {"situations": len(by_sit), "lines": sum(len(v) for v in by_sit.values()), "games": n_games}
    d = per_sit(lambda r: r["choke"]["win_eval"] - r["policy"]["win_eval"])
    out["choke_vs_policy_eval"] = {"diff": float(d.mean()), "ci95": _ci(d),
                                   "diff_all_games": float(d.sum() / n_games)}  # fmt: skip
    d = per_sit(lambda r: r["best_any"]["win_eval"] - r["policy"]["win_eval"])
    out["best_incl_pairs_vs_policy_eval"] = {"diff": float(d.mean()), "ci95": _ci(d)}
    d = per_sit(lambda r: r["board_choke"]["win"] - r["policy"]["win"])
    out["board_choke_vs_policy_win"] = {"diff": float(d.mean()), "ci95": _ci(d)}
    d = per_sit(lambda r: r["policy"]["win"] - r["pass"]["win"])
    out["policy_vs_pass_win"] = {"diff": float(d.mean()), "ci95": _ci(d)}
    d = per_sit(lambda r: r["choke"]["win_eval"] - r["pass"]["win_eval"])
    out["choke_vs_pass_eval"] = {"diff": float(d.mean()), "ci95": _ci(d)}
    out["pairs_tried_share"] = float(np.mean([r["n_pairs"] > 0 for rs in by_sit.values() for r in rs]))
    out["best_is_pair_share"] = float(np.mean([r["best_any"]["plan"] == "pair" for rs in by_sit.values() for r in rs]))
    all_rows = [r for rs in by_sit.values() for r in rs]
    fired = [r for r in all_rows if r["policy"]["fire_k"] is not None]
    out["policy_fires_share"] = len(fired) / max(1, len(all_rows))
    out["policy_hits_choke_play"] = _mean([r["policy"]["fire_k"] == r["choke"]["fire_k"] for r in fired])
    out["policy_hits_choke_exact"] = _mean(
        [r["policy"]["fire_k"] == r["choke"]["fire_k"] and r["policy"]["fired_code"] == r["choke"]["code"]
         for r in fired])  # fmt: skip
    out["policy_hits_board_choke_play"] = _mean([r["policy"]["fire_k"] == r["board_choke"]["fire_k"] for r in fired])
    for name in ("pass", "policy", "choke", "board_choke"):
        rs = [r[name] for r in all_rows]
        out[f"turn_{name}"] = {
            "op_end_monsters": _mean(x["op_end"]["monsters"] for x in rs), "op_end_atk": _mean(x["op_end"]["atk"] for x in rs),
            "op_end_extra": _mean(x["op_end"]["extra"] for x in rs), "op_end_sets": _mean(x["op_end"]["sets"] for x in rs),
            "op_plays": _mean(x["op_plays"] for x in rs), "op_plays_after": _mean(x["op_plays_after"] for x in rs),
            "payoff_reached": _mean(x["payoff_reached"] for x in rs), "stopped": _mean(x["stopped"] for x in rs),
            "damage": _mean(x["damage"] for x in rs), "died": _mean(x["died"] for x in rs),
            "me_monsters_kept": _mean(x["me_monsters_end"] for x in rs), "win": _mean(x["win"] for x in rs),
            "fire_k": _mean(x["fire_k"] for x in rs)}  # fmt: skip
    for name in ("choke_class", "board_choke_class", "policy_class"):
        cnt: dict[str, int] = {}
        for r in all_rows:
            if r[name] is not None:
                cnt[r[name]] = cnt.get(r[name], 0) + 1
        out[name] = dict(sorted(cnt.items(), key=lambda kv: -kv[1]))
    return out
