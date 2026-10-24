"""P0b: does a fire / hold counterfactual at a hand-trap window agree with the solver's graded label? (first-seat plan
§5.1, §6 phase 0; the online version is design A).

For every Ash Blossom window of the graded hands (``graded.jsonl`` of ``handtrap_labels_graded.py``, hands
``--min-hand`` .. ``--max-hand``; the window identity is the k-th responder decision with an Ash activation along the
solver's line, as the labels count it): the game starts from the hand's ``fire`` start (the plain start with Ash on
top of the passive responder's deck), the combo seat (engine player 0) plays the solver's line by index and the
responder (player 1, the learner ``CKPT``) passes up to window k. At window k the game branches (K = 1: fire Ash or
pass), and each branch is rolled out ``--rollouts`` times (M, common random numbers: rollout m uses the same learner /
opponent seeds in both branches) to the horizon -- the learner's first decision of turn 2 with a real choice, or the
end of the game. After window k the learner plays its policy (it may fire at a later window of the pass branch).
Once Ash has resolved the line is broken, and the combo seat is played by one of three opponents:

- ``snapshot``: the policy ``--opponent`` (default CKPT), sampled;
- ``recovery``: the solver's recovery line for the window where Ash was fired (``variant="fire"`` records of
  ``--demos``: the line that fires Ash at that window and still reaches the board), else the snapshot;
- ``best_of``: the snapshot rollouts again, the value of a branch being its worst rollout for the learner (the
  ``branch_opponent_best_of`` stand-in for a recovery line).

Two truncations score a rollout: ``critic`` (the learner's privileged critic V at the horizon, +-1 at a game end) and
``board`` (the combo seat's board at the horizon, 3 per interruption + 1 per monster + 0.5 per spell / trap card,
as ``lookahead_oracle.py``). The counterfactual says fire when firing raises V (lowers the board) by more than the
tie margin (``--tie-critic``, ``--tie-board``), hold when it lowers it, tie otherwise. The label says fire when the
graded gain ``g_k`` of :class:`ygorl.train.line_labels.WindowLabel` is positive (fire at the earliest window that
hurts the combo as much as any later one), hold otherwise. Report per (opponent, truncation): agreement (ties count
as hold, as the pilot did), agreement on decided windows, tie rate, the confusion matrix, P(fire | label hold) and
P(hold | label fire), and the mean delta by label; ``rows`` holds every window's deltas.

  uv run --no-sync python tools/window_counterfactual_probe.py GRADED.jsonl CKPT --demos full_fire8.jsonl \\
      [--min-hand 40] [--rollouts 4] [--workers 12] [--seed 0] --out OUT.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase0_common import Brain, LockstepGame, chosen_row, generator, mean_se, sample  # noqa: E402

from ygorl.solver.demo import read_jsonl  # noqa: E402
from ygorl.train.line_labels import FIRE, WindowLabel, fire_start, read_graded  # noqa: E402

MODES = ("snapshot", "recovery", "best_of")
TRUNCATIONS = ("critic", "board")
MAX_DECISIONS = 3000


def ash_index(point, fire: int = FIRE) -> int | None:
    for i, a in enumerate(point.actions):
        if a.card is not None and a.card.code == fire:
            return i
    return None


def passive(point) -> int:
    from ygorl.train.bc import passive_action

    return passive_action(point)


def recovery_lines(fire_demo, env, cards) -> dict[int, list[int]]:
    """window -> the actions (both seats, from the fire start) of the first verified fire line firing Ash there."""
    from ygorl.engine.duel import DuelSession

    out: dict[int, list[int]] = {}
    for ln in fire_demo.lines:
        session = DuelSession(fire_demo.replay(None).duel(env, cards=cards))
        window = 0
        try:
            for step, idx in enumerate(ln.actions):
                point = session.point
                if point is None:
                    break
                if point.player == 1 and ash_index(point) is not None:
                    window += 1
                    a = point.actions[idx]
                    if a.card is not None and a.card.code == FIRE:
                        out.setdefault(window, list(ln.actions))
                        break
                session.act(idx)
        finally:
            session.close()
    return out


def board_score(game, cards) -> dict:
    from ygorl.solver.blocking import board_interruptions
    from ygorl.solver.targets import board_summary

    tr = game.tracker
    b = board_summary(game.session.core, tr.turn, (tr.lp[0], tr.lp[1]))
    bi = board_interruptions(b, cards, 0)
    side = b["players"][0]
    score = 3 * bi.interruptions + len(side["mzone"]) + 0.5 * len(side["szone"])
    return {"score": score, "interruptions": bi.interruptions, "negates": bi.negates, "monsters": len(side["mzone"])}


def rollout(ctx: dict, hand: dict, k: int, fire_now: bool, mode: str, m: int) -> dict:
    """One branch rollout from window ``k``: the learner's critic V and the combo seat's board at the horizon."""
    learner, opponent, env, cards = ctx["learner"], ctx["opponent"], ctx["env"], ctx["cards"]
    seed = ctx["seed"]
    demo = hand["fire_demo"]
    game = LockstepGame(demo.replay(None).duel(env, cards=cards), learner)
    opp_host = None if opponent is learner else opponent.host(game.session.duel)
    learner_gen = generator(seed, hand["id"], k, m, 1)  # CRN: the same streams in both branches of rollout m
    opp_gen = generator(seed, hand["id"], k, m, 2)
    p0 = iter(hand["p0"])
    line, rec = "base", None
    window = 0
    fired_at = None
    prefix_mismatch = recovered = False
    out = {"fired_at": None, "end": "horizon"}
    try:
        for _ in range(MAX_DECISIONS):
            if game.done:
                break
            point = game.point
            if point is None:
                break
            tr = game.tracker
            if tr.turn >= 2 and point.player == 1 and len(point.choosable) > 1:
                break
            if point.player == 0:
                idx = None
                if line == "base":
                    idx = next(p0, None)
                elif line == "recovery":
                    pos = len(game.history)
                    idx = rec[pos] if pos < len(rec) else None
                    if idx is not None and idx >= len(point.actions):
                        idx = None
                if idx is None:
                    line = "policy"
                    if opp_host is None:
                        idx = chosen_row(opponent, game, opp_gen)
                    else:
                        idx = sample(opponent.probs([opp_host.observe()])[0], opp_gen)
            else:
                ai = ash_index(point) if fired_at is None else None
                if ai is not None:
                    window += 1
                    if window < k:
                        idx = passive(point)
                    elif window == k:
                        idx = ai if fire_now else passive(point)
                    else:
                        idx = chosen_row(learner, game, learner_gen)
                    if idx == ai:
                        fired_at = window
                        hist = game.history
                        r = ctx["recovery"].get(hand["id"], {}).get(window) if mode == "recovery" else None
                        if r is not None and r[: len(hist)] == hist and len(r) > len(hist) and r[len(hist)] == ai:
                            line, rec, recovered = "recovery", r, True
                        else:
                            prefix_mismatch = prefix_mismatch or (r is not None)
                            line = "policy"
                elif line == "recovery" and len(game.history) < len(rec) and tr.turn < 2:
                    idx = rec[len(game.history)]
                    if idx >= len(point.actions):
                        idx = passive(point)
                elif tr.turn < 2:
                    idx = passive(point)
                else:
                    idx = chosen_row(learner, game, learner_gen)
            game.act(idx)
            if opp_host is not None:
                opp_host.act(idx)
        out["fired_at"] = fired_at
        out["line"] = line
        out["recovered"] = recovered
        out["prefix_mismatch"] = prefix_mismatch
        out["reached"] = window >= k
        out["board"] = board_score(game, cards)
        if game.done:
            winner = game.session.result().winner
            out["end"] = "game_over"
            out["v"] = 0.0 if winner is None else (1.0 if winner == 1 else -1.0)
        else:
            point = game.point
            if point is None or point.player != 1:
                out["end"] = "cap"
                out["v"] = 0.0
            else:
                out["v"] = float(learner.value([game.obs()], [game.privileged()])[0])
        return out
    finally:
        game.close()


_CTX: dict = {}


def _init(args: dict) -> None:
    import torch

    torch.set_num_threads(1)
    from ygorl.data import load_environment
    from ygorl.engine.duel import default_cards

    learner = Brain(args["ckpt"])
    opponent = learner if args["opponent"] in (None, args["ckpt"]) else Brain(args["opponent"])
    _CTX.update(learner=learner, opponent=opponent, env=load_environment(args["env"]), cards=default_cards(),
                seed=args["seed"], rollouts=args["rollouts"], recovery={})  # fmt: skip


def probe_hand(hand: dict) -> list[dict]:
    ctx = _CTX
    ctx["recovery"][hand["id"]] = (
        recovery_lines(hand["fire_demo"], ctx["env"], ctx["cards"]) if hand["fire_lines"] else {}
    )
    rows = []
    for k in range(1, hand["windows"] + 1):
        runs = {(mode, b): [] for mode in ("snapshot", "recovery") for b in (True, False)}
        for mode in ("snapshot", "recovery"):
            for fire_now in (True, False):
                for m in range(ctx["rollouts"]):
                    runs[(mode, fire_now)].append(rollout(ctx, hand, k, fire_now, mode, m))
        reached = all(r["reached"] for rs in runs.values() for r in rs)
        label = WindowLabel(hand["u"][k - 1], max(hand["u"][k:], default=0.0))
        row = {"deck": hand["deck"], "hand_index": hand["hand_index"], "window": k, "u": label.u, "later": label.later,
               "gain": label.fire_bonus, "label": "fire" if label.fire_bonus > 0 else "hold", "reached": reached,
               "has_recovery": k in ctx["recovery"][hand["id"]], "deltas": {}, "branches": {}}  # fmt: skip
        for mode in MODES:
            src = "recovery" if mode == "recovery" else "snapshot"
            fire, hold = runs[(src, True)], runs[(src, False)]
            if mode == "best_of":
                vf, vh = min(r["v"] for r in fire), min(r["v"] for r in hold)
                bf, bh = max(r["board"]["score"] for r in fire), max(r["board"]["score"] for r in hold)
            else:
                vf, vh = sum(r["v"] for r in fire) / len(fire), sum(r["v"] for r in hold) / len(hold)
                bf = sum(r["board"]["score"] for r in fire) / len(fire)
                bh = sum(r["board"]["score"] for r in hold) / len(hold)
            row["deltas"][mode] = {"critic": vf - vh, "board": bh - bf}
            row["branches"][mode] = {"v_fire": vf, "v_hold": vh, "board_fire": bf, "board_hold": bh}
        rec_runs = runs[("recovery", True)] + runs[("recovery", False)]
        row["recovery_followed"] = sum(r["recovered"] for r in rec_runs)
        row["recovery_mismatch"] = sum(r["prefix_mismatch"] for r in rec_runs)
        row["hold_branch_fired_later"] = sum(r["fired_at"] is not None for r in runs[("snapshot", False)])
        row["ends"] = sorted({r["end"] for rs in runs.values() for r in rs})
        rows.append(row)
    return rows


def load_hands(
    graded: str, demos: list[str], min_hand: int, max_hand: int, limit: int | None, env, cards
) -> list[dict]:
    labels = read_graded(graded)
    plain, fires = {}, {}
    for path in demos:
        for d in read_jsonl(path):
            key = (d.deck["name"], d.hand_index)
            if d.variant == "plain" and d.status == "solved" and d.lines:
                plain.setdefault(key, d)
            elif d.variant == "fire" and d.lines and d.status == "solved":
                fires.setdefault(key, d)
    hands = []
    for key in sorted(labels):
        rec = labels[key]
        if not (min_hand <= key[1] < max_hand) or not rec.get("windows") or key not in plain:
            continue
        best = float(rec.get("best_without") or 0)
        values = {int(k): float(v) for k, v in rec.get("values", {}).items()}
        w = int(rec["windows"])
        if best <= 0 or any(k not in values for k in range(1, w + 1)):
            continue
        base = plain[key]
        start = fire_start(base)
        fd = fires.get(key)
        fire_demo = start if fd is None else _with_start(fd, start)
        p0 = [i for i, pl in zip(base.lines[0].actions, base.lines[0].players) if pl == 0]
        hands.append({"id": len(hands), "deck": key[0], "hand_index": key[1], "windows": w, "p0": p0,
                      "u": [values[k] / best for k in range(1, w + 1)], "fire_demo": fire_demo,
                      "fire_lines": fd is not None})  # fmt: skip
        if limit and len(hands) >= limit:
            break
    return hands


def _with_start(fire_demo, start_demo):
    from dataclasses import replace

    return replace(fire_demo, start=start_demo.start) if fire_demo.start is None else fire_demo


def summarize(rows: list[dict], tie_critic: float, tie_board: float) -> dict:
    out = {}
    use = [r for r in rows if r["reached"]]
    for mode in MODES:
        for trunc in TRUNCATIONS:
            eps = tie_critic if trunc == "critic" else tie_board
            conf = {lab: {"fire": 0, "hold": 0, "tie": 0} for lab in ("fire", "hold")}
            deltas = {"fire": [], "hold": []}
            for r in use:
                d = r["deltas"][mode][trunc]
                call = "fire" if d > eps else ("hold" if d < -eps else "tie")
                conf[r["label"]][call] += 1
                deltas[r["label"]].append(d)
            n = len(use)
            agree = conf["fire"]["fire"] + conf["hold"]["hold"] + conf["hold"]["tie"]
            ties = conf["fire"]["tie"] + conf["hold"]["tie"]
            decided = n - ties
            nf, nh = sum(conf["fire"].values()), sum(conf["hold"].values())
            out[f"{mode}/{trunc}"] = {
                "windows": n, "agreement": agree / max(1, n),
                "agreement_decided": (conf["fire"]["fire"] + conf["hold"]["hold"]) / max(1, decided),
                "tie_rate": ties / max(1, n), "confusion": conf,
                "oracle_fire_rate": (conf["fire"]["fire"] + conf["hold"]["fire"]) / max(1, n),
                "label_fire_rate": nf / max(1, n),
                "p_fire_given_hold": conf["hold"]["fire"] / max(1, nh),
                "p_hold_given_fire": conf["fire"]["hold"] / max(1, nf),
                "mean_delta_label_fire": mean_se(deltas["fire"]), "mean_delta_label_hold": mean_se(deltas["hold"]),
            }  # fmt: skip
    return out


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("graded")
    p.add_argument("ckpt")
    p.add_argument("--demos", nargs="+", required=True, help="plain + fire demonstration JSONL(s) of the graded hands")
    p.add_argument("--opponent", default=None, help="snapshot policy of the combo seat off its line (default: CKPT)")
    p.add_argument("--min-hand", type=int, default=40)
    p.add_argument("--max-hand", type=int, default=10**9)
    p.add_argument("--limit", type=int, default=None, help="first N hands only")
    p.add_argument("--rollouts", type=int, default=4, help="M rollouts per branch (CRN)")
    p.add_argument("--tie-critic", type=float, default=0.02, help="|delta V| at or below this is a tie")
    p.add_argument("--tie-board", type=float, default=0.0, help="|delta board| at or below this is a tie")
    p.add_argument("--env", default="md-2026-09")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    t0 = time.perf_counter()
    from ygorl.data import load_environment
    from ygorl.engine.duel import default_cards

    hands = load_hands(a.graded, a.demos, a.min_hand, a.max_hand, a.limit, load_environment(a.env), default_cards())
    init = {"ckpt": a.ckpt, "opponent": a.opponent, "env": a.env, "seed": a.seed, "rollouts": a.rollouts}
    if a.workers <= 1:
        _init(init)
        per_hand = [probe_hand(h) for h in hands]
    else:
        with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(init,)) as pool:
            per_hand = pool.map(probe_hand, hands, chunksize=1)
    rows = [r for rs in per_hand for r in rs]
    report = {"ckpt": a.ckpt, "opponent": a.opponent or a.ckpt, "graded": a.graded, "hands": len(hands),
              "hands_with_recovery_lines": sum(h["fire_lines"] for h in hands), "windows": len(rows),
              "windows_reached": sum(r["reached"] for r in rows),
              "windows_with_recovery": sum(r["has_recovery"] for r in rows), "rollouts": a.rollouts, "seed": a.seed,
              "tie_critic": a.tie_critic, "tie_board": a.tie_board,
              "label_fire": sum(r["label"] == "fire" for r in rows if r["reached"]),
              "summary": summarize(rows, a.tie_critic, a.tie_board), "rows": rows}  # fmt: skip
    report["seconds"] = round(time.perf_counter() - t0, 1)
    text = json.dumps(report, indent=1)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n")
    else:
        print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1))
    return report


if __name__ == "__main__":
    main()
