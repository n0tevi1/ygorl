"""Turn-1 boards judged by the opponent's turn 2: a policy-played board evaluator (docs/solver.md「存活场面」).

Counting interruptions on a turn-1 end board does not predict who wins (the game reading of #83: the first
player's win rate tracks its monsters on turn 1, the monsters that survive turn 2 and its attacks on turn 3).
This module plays the opponent's turn instead of counting:

* :func:`play_line` follows a verified solver line (``Demonstration.lines[k]``) in a *real* duel whose
  opponent is a real deck -- the record's own start (deck order with the hand on top, core seed words, player
  rules) with ``DUEL_PSEUDO_SHUFFLE`` cleared and the opponent's main deck shuffled by ``order_seed``. Each step
  of the line is matched to the real duel's candidates (:func:`ygorl.build.first_turn.match_action`); every
  decision of the opponent during turn 1 is the policy's (a hand trap may be played). A step without a match
  ends the line there (``deviated``) and the policy finishes player 0's turn. The policy then plays both seats
  through turn 2 (and, with ``to_end``, to the end of the game).
* :func:`board_features` reads a board summary at a turn boundary: the first player's monsters, set spells /
  traps, cards in hand, LP and interruptions; the opponent's cards on the field.
* :func:`survival_score` is the evaluator's score of one sample at the end of turn 2 (first player's
  monsters, cards in hand, LP, the opponent's board); :func:`mean_score` averages samples.

A policy agent (:class:`~ygorl.agents.checkpoint.CheckpointAgent`) is kept in lockstep with the duel; one agent
plays both seats (its host observes whichever seat is to act).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from ygorl import _core
from ygorl.cards.ydk import Deck
from ygorl.data.environment import Environment
from ygorl.engine import constants as C
from ygorl.engine.duel import DuelSession, default_cards, shuffle_deck
from ygorl.solver.demo import PASSIVE_KINDS, DemoError, Demonstration, iter_steps
from ygorl.solver.targets import board_summary

# weights of survival_score, per count at the start of the first player's turn 3 ("lp" per 8000 LP): a logistic fit
# of the game's winner on the pilot's 12,989 games (docs/solver.md「存活场面」), rounded and scaled to monsters = 1.
# The pilot itself chose its lines with PILOT_WEIGHTS.
SCORE_WEIGHTS = {"monsters": 1.0, "hand": 0.5, "lp": 6.0, "opp_board": -0.6, "interruptions": 0.5, "alive": 2.0}
PILOT_WEIGHTS = {"monsters": 1.0, "hand": 0.5, "lp": 1.0, "opp_board": -0.25, "interruptions": 0.0, "alive": 2.0}


@dataclass(frozen=True)
class Opponent:
    """One sample of the opponent: a deck, the shuffle of its main deck and the policy's sampling seed."""

    deck: Deck
    order_seed: int
    agent_seed: int


@dataclass
class LinePlay:
    """How one solver line fared against one opponent sample."""

    followed: int = 0  # line steps of player 0 replayed in the real duel
    line_steps: int = 0  # player-0 steps of the line
    deviated: str = ""  # why the line stopped before its end ("" when it was followed to the end)
    opp_t1_actions: int = 0  # non-passive decisions of the opponent during turn 1 (hand traps and the like)
    t1: dict = field(default_factory=dict)  # board_features at the start of turn 2
    t2: dict = field(default_factory=dict)  # board_features at the start of turn 3 (or the end of the game)
    winner: int | None = None  # engine player who won (to_end), None for a draw or when not played out
    end_turn: int = 0  # turn the game ended on (0: still running)
    error: str = ""

    def to_json(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def _passive(action) -> bool:
    return action.kind in PASSIVE_KINDS


def board_features(board: Mapping, cards=None, previous: Mapping | None = None, *, strict: bool = True) -> dict:
    """Counts of a ``board_summary`` at a turn boundary, from the first player's (engine player 0) side.

    ``monsters`` / ``set_st`` / ``hand`` / ``lp`` / ``interruptions`` of player 0, ``opp_monsters`` and
    ``opp_board`` (monsters + spells / traps on the field) of player 1; with ``previous`` (an earlier summary),
    ``survivors`` is how many of player 0's monsters then are still on the field (by card).
    """
    from ygorl.solver.blocking import board_interruptions

    cards = cards if cards is not None else default_cards()
    me, opp = board["players"][0], board["players"][1]
    st = [c for c in me["szone"] if c["sequence"] < 5]
    out = {
        "monsters": len(me["mzone"]),
        "set_st": sum(1 for c in st if not c["position"] & C.POS_FACEUP),
        "hand": len(me["hand"]),
        "lp": me["lp"],
        "interruptions": board_interruptions(board, cards, strict=strict).interruptions,
        "opp_monsters": len(opp["mzone"]),
        "opp_board": len(opp["mzone"]) + len(opp["szone"]),
        "opp_lp": opp["lp"],
    }
    if previous is not None:
        before = Counter(c["code"] for c in previous["players"][0]["mzone"])
        now = Counter(c["code"] for c in me["mzone"])
        out["survivors"] = sum((before & now).values())
    return out


def survival_score(play: LinePlay | Mapping, weights: Mapping[str, float] | None = None) -> float:
    """The evaluator's score of one sample: the first player's state when its turn 3 starts.

    A weighted sum (:data:`SCORE_WEIGHTS`) of its monsters, cards in hand, LP / 8000 and strict interruptions,
    plus the opponent's cards on the field (a negative weight) and a bonus for being alive; a first player killed
    in turn 2 scores only the opponent's board term.
    """
    w = {**SCORE_WEIGHTS, **(weights or {})}
    t2 = play.t2 if isinstance(play, LinePlay) else play.get("t2", {})
    if not t2:
        return 0.0
    alive = t2.get("lp", 0) > 0
    if not alive:
        return w["opp_board"] * t2.get("opp_board", 0)
    return (w["monsters"] * t2["monsters"] + w["hand"] * t2["hand"] + w["lp"] * min(t2["lp"], 8000) / 8000
            + w["opp_board"] * t2["opp_board"] + w.get("interruptions", 0.0) * t2.get("interruptions", 0)
            + w["alive"])  # fmt: skip


def mean_score(plays: Sequence[LinePlay | Mapping], weights: Mapping[str, float] | None = None) -> float:
    ok = [p for p in plays if not (p.error if isinstance(p, LinePlay) else p.get("error"))]
    return sum(survival_score(p, weights) for p in ok) / len(ok) if ok else float("-inf")


def play_line(demo: Demonstration, line: int, opponent: Opponent, checkpoint: str, *, to_end: bool = False,
              follow: bool = True, env: Environment | None = None, cards=None,
              scripts: _core.ScriptDirectory | None = None, max_turn_steps: int = 600) -> LinePlay:  # fmt: skip
    """Follow ``demo.lines[line]`` against ``opponent`` and let the policy play turn 2 (see the module docstring).

    ``follow=False`` lets the policy play player 0's turn 1 from the same start (the policy's own opening, a
    reference for the lines).
    """
    from ygorl.agents.checkpoint import CheckpointAgent
    from ygorl.build.first_turn import match_action

    cards = cards if cards is not None else default_cards()
    rep = demo.replay(line)
    rep.rule_flags &= ~C.DUEL_PSEUDO_SHUFFLE
    rep.responses = []
    opp = opponent.deck
    rep.decks["b"] = {"name": opp.name or "opponent", "main": shuffle_deck(opp.main, opponent.order_seed, 1),
                      "extra": list(opp.extra), "side": []}  # fmt: skip
    kwargs = {k: v for k, v in (("cards", cards), ("scripts", scripts)) if v is not None}
    duel = rep.duel(env, **kwargs)
    session = DuelSession(duel)
    tracker = session.tracker
    agent = CheckpointAgent(checkpoint, opponent.agent_seed)
    agent.on_duel_start(duel)
    out = LinePlay(line_steps=sum(1 for p in demo.lines[line].players if p == 0))

    def step(point, idx: int) -> None:
        agent.on_decision(point, idx)
        session.act(idx)

    def policy(point) -> int:
        return 0 if len(point.actions) == 1 else agent.act(point)

    try:
        ref = iter_steps(demo, line, env=env, cards=cards, scripts=scripts)
        pending = None
        following = follow
        steps = 0
        while not session.done and tracker.turn < 2 and steps < max_turn_steps:
            point = session.point
            if point is None:
                break
            steps += 1
            if following and pending is None:
                try:
                    pending = next(ref)
                    while pending[0].player == 1 and point.player == 0:
                        pending = next(ref)  # the template opponent's answer: no counterpart here
                except StopIteration:
                    following, pending = False, None
                except DemoError as exc:
                    following, pending, out.deviated = False, None, f"the recorded line does not replay: {exc}"
            if point.player == 1:
                idx = policy(point)
                out.opp_t1_actions += not _passive(point.actions[idx])
                if pending is not None and pending[0].player == 1:
                    pending = None
                step(point, idx)
                continue
            if following and pending is not None:
                want = pending[0].actions[pending[1]]
                real = match_action(want, point.actions)
                if real is not None:
                    pending = None
                    out.followed += 1
                    step(point, real)
                    continue
                following, pending = False, None
                out.deviated = f"step {out.followed}: no {want.kind} of {want.card.code if want.card else '-'}"
            step(point, policy(point))
        ref.close()
        t1_board = board_summary(session.core, tracker.turn, (tracker.lp[0], tracker.lp[1]))
        out.t1 = board_features(t1_board, cards)
        while not session.done and tracker.turn < 3:
            point = session.point
            if point is None:
                break
            step(point, policy(point))
        board = board_summary(session.core, tracker.turn, (tracker.lp[0], tracker.lp[1]))
        out.t2 = board_features(board, cards, t1_board)
        out.t2["turn"] = tracker.turn
        if to_end:
            while not session.done:
                point = session.point
                if point is None:
                    break
                step(point, policy(point))
        if session.done:
            res = session.result()
            out.winner, out.end_turn = res.winner, res.turns
            if res.reason == "error":
                out.error = f"engine error: {res.error}"
    except Exception as exc:  # noqa: BLE001 - one sample; recorded
        out.error = f"{type(exc).__name__}: {exc}"
    finally:
        session.close()
    return out


# ------------------------------------------------------------------ one hand: candidate lines, evaluated


def boss_pieces(main: Sequence[int], extra: Sequence[int], cards, exclude: Sequence[int] = (), n: int = 2) -> list[int]:
    """Plain "boss monster" targets: the deck's Extra Deck monsters tied to its main deck, highest ATK first.

    Tied as in :func:`~ygorl.solver.blocking.deck_pieces` (a shared archetype, or one names the other); without
    any, every Extra Deck monster. Interrupters or not -- the point is a big body that survives.
    """
    from ygorl.solver.blocking import _names

    main_codes = {cards.canonical(c) for c in main}
    main_sets = {s for c in main_codes if c in cards for s in cards[c].setcodes if s}
    main_names = set().union(*(_names(c) for c in main_codes)) if main_codes else set()
    skip = {cards.canonical(c) for c in exclude}
    pool, tied = [], []
    for code in dict.fromkeys(cards.canonical(c) for c in extra):
        card = cards.get(code)
        if card is None or not card.is_monster or code in skip:
            continue
        pool.append(card)
        if set(card.setcodes) & main_sets or code in main_names or _names(code) & main_codes:
            tied.append(card)
    ranked = sorted(tied or pool, key=lambda c: (-c.attack, c.password))
    return [c.password for c in ranked[:n]]


def sample_opponents(decks: Sequence, own: str, n: int, seed: int) -> list[Opponent]:
    """``n`` opponent samples: a random deck of ``decks`` (paths; never ``own``), a main deck shuffle, a policy seed."""
    import random

    from ygorl.cards.ydk import load_ydk

    rng = random.Random(seed)
    out: list[Opponent] = []
    tries = 0
    while len(out) < n and tries < 50 * n:
        tries += 1
        deck = load_ydk(rng.choice(list(decks)))
        if deck.name == own:
            continue
        out.append(Opponent(deck, rng.randrange(1 << 62), rng.randrange(1 << 62)))
    return out


def summarize_plays(plays: Sequence[LinePlay], weights: Mapping[str, float] | None = None) -> dict:
    """Means over the samples of one candidate: the score, turn-1 and turn-2 counts, deviations, the win rate."""
    ok = [p for p in plays if not p.error]
    if not ok:
        return {"n": 0, "errors": len(plays), "score": None}

    def mean(f) -> float:
        return round(sum(f(p) for p in ok) / len(ok), 3)

    out = {
        "n": len(ok), "errors": len(plays) - len(ok), "score": round(mean_score(ok, weights), 3),
        "deviated": mean(lambda p: bool(p.deviated)), "opp_t1_actions": mean(lambda p: p.opp_t1_actions),
        **{f"t1_{k}": mean(lambda p, k=k: p.t1.get(k, 0)) for k in ("monsters", "set_st", "hand", "interruptions")},
        **{f"t2_{k}": mean(lambda p, k=k: p.t2.get(k, 0))
           for k in ("monsters", "survivors", "hand", "lp", "opp_board", "interruptions")},
        "t2_dead": mean(lambda p: p.t2.get("lp", 0) <= 0),
    }  # fmt: skip
    played = [p for p in ok if p.end_turn]
    if played:
        out["games"] = len(played)
        out["win_rate"] = round(sum(p.winner == 0 for p in played) / len(played), 3)
    return out


def solve_survival(job, *, checkpoint: str, opponents: Sequence, samples: int = 8, to_end: int = 0,
                   pieces: int = 3, pair: bool = True, bosses: int = 2, weights: Mapping[str, float] | None = None,
                   seed: int = 0, cards=None, scripts=None) -> tuple[dict, list[dict]]:  # fmt: skip
    """One opening hand: solve candidate lines, play each against ``samples`` opponents, keep the best survivor.

    ``job`` is a :class:`~ygorl.solver.batch.HandJob` (its ``targets`` are ignored, ``lines`` lines are kept per
    attempt). Attempts, each a full solve + verification: ``base + piece`` for the first ``pieces`` field
    interrupters of :func:`~ygorl.solver.blocking.blocking_plan` (strict counter), ``base`` + the two best solved
    pieces (``pair``), ``base`` alone when no piece solved, and ``bosses`` plain boss monsters
    (:func:`boss_pieces`, no base). Every distinct verified line is a candidate; all candidates play the same
    opponent samples (:func:`sample_opponents`, seeded by the hand), the first ``to_end`` of them to the end of
    the game. The policy's own turn 1 plays them too, as a reference. The record keeps the candidate with the
    best :func:`mean_score` (then strict interruptions, then the earlier candidate) as its only line;
    ``solver["survival"]`` describes every candidate, the choice and the old (interruption-count) choice.
    Returns ``(record, sample rows)``.
    """
    import shutil
    import time
    from dataclasses import replace
    from pathlib import Path

    from ygorl.cards.ydk import load_ydk
    from ygorl.engine.duel import DuelConfig
    from ygorl.solver.batch import _config, _load_env, sample_hand, solve_hand
    from ygorl.solver.blocking import blocking_plan, board_interruptions

    cards = cards if cards is not None else default_cards()
    deck = load_ydk(job.deck_path)
    env = _load_env(job.env)
    config = _config(env) if job.env else DuelConfig()
    hand, _ = sample_hand(deck, job.hand_seed, config.player.starting_hand)
    plan = blocking_plan(deck.main, deck.extra, hand, cards, strict=True)
    results: list[dict] = []
    t0 = time.monotonic()

    def attempt(targets: list[str], label: str) -> bool:
        sub = replace(job, targets=tuple(targets), scratch=Path(job.scratch) / f"a{len(results)}")
        demo = solve_hand(sub, cards=cards, scripts=scripts)
        results.append({"label": label, "targets": list(targets), "demo": demo})
        return demo.status == "solved"

    solved_pieces = []
    for p in plan.pieces[:pieces]:
        if attempt([*plan.base, p.target], f"piece:{p.password}"):
            solved_pieces.append(p.target)
    if pair and len(solved_pieces) >= 2:
        attempt([*plan.base, *solved_pieces[:2]], "pair")
    if plan.base and not solved_pieces:
        attempt(list(plan.base), "base")
    for b in boss_pieces(deck.main, deck.extra, cards, [p.password for p in plan.pieces[:pieces]], bosses):
        attempt([str(b)], f"boss:{b}")
    solve_s = time.monotonic() - t0

    cands: list[dict] = []
    seen: set[tuple] = set()
    for a, r in enumerate(results):
        for i, ln in enumerate(r["demo"].lines):
            key = tuple(ln.responses)
            if key in seen:
                continue
            seen.add(key)
            old = board_interruptions(ln.board, cards)
            new = board_interruptions(ln.board, cards, strict=True)
            me = ln.board["players"][0]
            cands.append({"attempt": a, "label": r["label"], "targets": r["targets"], "line": i,
                          "old": list(old.key()), "strict": list(new.key()), "pieces": new.to_json()["pieces"],
                          "monsters": len(me["mzone"]), "hand": len(me["hand"]),
                          "set_st": sum(1 for c in me["szone"] if c["sequence"] < 5 and not c["position"] & C.POS_FACEUP),
                          "board": {"mzone": [c["code"] for c in me["mzone"]],
                                    "szone": [c["code"] for c in me["szone"]], "hand": list(me["hand"])}})  # fmt: skip
    # the old tool: line 0 of each blocking attempt, most interruptions (then negations), earlier attempt on ties
    old_pool = [k for k, c in enumerate(cands) if c["line"] == 0 and not c["label"].startswith("boss")]
    old_choice = max(old_pool, key=lambda k: (tuple(cands[k]["old"]), -k)) if old_pool else None

    rows: list[dict] = []
    opps: list[Opponent] = []
    policy_ref = None
    t1 = time.monotonic()
    if cands:
        opps = sample_opponents(opponents, deck.name, samples, job.hand_seed ^ (seed * 0x9E3779B97F4A7C15 % (1 << 64)))
        ref_demo = results[cands[0]["attempt"]]["demo"]
        for k, c in enumerate([*cands, None]):
            demo = ref_demo if c is None else results[c["attempt"]]["demo"]
            line = 0 if c is None else c["line"]
            plays = [play_line(demo, line, o, checkpoint, to_end=j < to_end, follow=c is not None, env=env,
                               cards=cards, scripts=scripts) for j, o in enumerate(opps)]  # fmt: skip
            summary = summarize_plays(plays, weights)
            if c is None:
                policy_ref = summary
            else:
                c["eval"] = summary
            for j, (o, p) in enumerate(zip(opps, plays)):
                rows.append({"deck": deck.name, "hand_index": job.hand_index, "cand": k if c is not None else -1,
                             "label": c["label"] if c is not None else "policy", "sample": j, "opponent": o.deck.name,
                             "score": round(survival_score(p, weights), 3), **p.to_json()})  # fmt: skip
    eval_s = time.monotonic() - t1

    chosen = None
    if cands:

        def rank(k: int) -> tuple:
            sc = cands[k]["eval"]["score"]
            return (sc if sc is not None else float("-inf"), tuple(cands[k]["strict"]), -k)

        chosen = max(range(len(cands)), key=rank)
        c = cands[chosen]
        base_demo = results[c["attempt"]]["demo"]
        record = replace(base_demo, lines=[base_demo.lines[c["line"]]], targets=list(c["targets"]),
                         solver=dict(base_demo.solver))  # fmt: skip
    elif results:
        record = results[0]["demo"]
    else:
        from ygorl.solver.demo import Demonstration

        record = Demonstration.new(deck, hand, hand_index=job.hand_index, hand_seed=job.hand_seed, variant="plain",
                                   targets=(), environment=env)  # fmt: skip
        record.status, record.error = "unsolved", "no candidate target for this hand"
    record.solver["survival"] = {
        "base": plan.base, "pieces": [p.target for p in plan.pieces[:pieces]],
        "attempts": [{"label": r["label"], "targets": r["targets"], "status": r["demo"].status,
                      "lines": len(r["demo"].lines), "wall_s": r["demo"].solver.get("wall_s", 0.0)} for r in results],
        "candidates": cands, "chosen": chosen, "old_choice": old_choice, "policy": policy_ref,
        "opponents": [o.deck.name for o in opps], "samples": samples, "to_end": to_end,
        "checkpoint": str(checkpoint), "solve_s": round(solve_s, 1), "eval_s": round(eval_s, 1),
    }  # fmt: skip
    record.solver["wall_s"] = round(solve_s + eval_s, 2)
    if not job.keep_files:
        shutil.rmtree(job.scratch, ignore_errors=True)
    return record.to_json(), rows


__all__ = ["PILOT_WEIGHTS", "SCORE_WEIGHTS", "LinePlay", "Opponent", "board_features", "boss_pieces", "mean_score",
           "play_line", "sample_opponents", "solve_survival", "summarize_plays", "survival_score"]  # fmt: skip
