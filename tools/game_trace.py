"""Game traces and per-player statistics (strength diagnosis, #83).

Plays policy self-play games on random corpus deck pairings (``play``), keeps one JSON line per game with the
recorded actions and per-turn statistics of each seat, and summarizes them (``summary``); any recorded game can
be replayed into a human-readable trace (``trace``). Engine seat 0 always moves first ("first"), seat 1 is "second".

Per turn and seat: hand at the start (after the draw, at Main Phase 1) and at the end of the turn; normal / special
summons; effect activations in its own and in the opponent's turn; negations performed and suffered (chain links
negated, attributed to the controller of the resolving link); the opponent's cards it sent to the GY / banished
(by battle, or by an effect of a chain link it controls; costs and materials excluded); monsters, set spells /
traps and face-down cards at the end of the turn; interruptions at the end of the turn
(:func:`ygorl.solver.blocking.board_interruptions`); interruptions used in the opponent's turn (activations of a
card with interruption script facts) and how many were negated; attacks, damage dealt, LP and decisions. The end of
a turn is the state when the next turn starts; the game end closes the last one.

The trace prints, per turn and phase, each decision of either seat (the chosen action; at a chain prompt the
alternatives too), summons, activations (chain links, targets, negations), attacks, damage and cards moved, and both
boards with their interruptions at every turn end.

Usage:
  tools/game_trace.py play CKPT --games 1000 --decks out/corpus/train --out games.jsonl [--workers 12]
  tools/game_trace.py summary games.jsonl [--json summary.json]
  tools/game_trace.py trace games.jsonl --game 17 [--game 18 ...]   (or --select blocked-loss / blocked-win)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path

from ygorl.engine import constants as C
from ygorl.engine import messages as M

STATS = ("hand_start", "hand_end", "normal_summons", "special_summons", "act_own", "act_opp", "negates_done",
         "negates_suffered", "opp_to_gy", "opp_banished", "monsters_end", "set_st_end", "facedown_end",
         "interruptions_end", "intr_used", "intr_negated", "attacks", "damage_dealt", "lp_end", "decisions")  # fmt: skip
SEATS = ("first", "second")
_PHASES = {C.PHASE_DRAW: "Draw", C.PHASE_STANDBY: "Standby", C.PHASE_MAIN1: "Main1", C.PHASE_BATTLE_START: "Battle",
           C.PHASE_MAIN2: "Main2", C.PHASE_END: "End"}  # fmt: skip
_LOC = {C.LOCATION_DECK: "deck", C.LOCATION_HAND: "hand", C.LOCATION_MZONE: "mzone", C.LOCATION_SZONE: "szone",
        C.LOCATION_GRAVE: "GY", C.LOCATION_REMOVED: "banished", C.LOCATION_EXTRA: "extra",
        C.LOCATION_OVERLAY: "material"}  # fmt: skip
_QUIET = {"place", "position", "finish", "sort", "default", "main2"}  # decisions not worth a trace line


class Tracer:
    """Consumes the engine messages and decisions of one duel; per-turn statistics and (``log=True``) a trace."""

    def __init__(self, cards, log: bool = False) -> None:
        self.cards = cards
        self.lines: list[str] | None = [] if log else None
        self.turn, self.tp, self.phase = 0, 0, 0
        self.lp = [8000, 8000]
        self.chain: dict[int, dict] = {}
        self.resolving: list[int] = []
        self.damage_step = False
        self.turns: dict[int, list[dict]] = {}
        self.last_damage: tuple | None = None
        self.lethal: dict | None = None
        self.boards: dict[int, list[dict]] = {}  # turn -> per seat end board (counts, interruption pieces)
        # activations in the opponent's turn: what each answered (the previous chain link, or None for an open
        # chain), as the k-th activation of the turn player in that turn, and whether it was negated
        self.responses: list[dict] = []
        self.turn_acts = 0  # activations of the turn player so far this turn
        # chain prompts outside one's own turn: (turn, player) -> {code: [times offered, times chosen]}
        self.offers: dict[tuple[int, int], dict[int, list[int]]] = defaultdict(dict)
        self.removed: list[dict] = []  # field cards removed by the other player (battle or its chain link)

    # ---------------------------------------------------------------- helpers
    def name(self, code: int) -> str:
        card = self.cards.get(code) if code else None
        return card.name if card is not None else (f"#{code}" if code else "(hidden)")

    def effect(self, desc: int) -> str:
        card = self.cards.get(desc >> 20) if desc > 0xFFFFF else None
        i = desc & 0xFFFFF
        if card is not None and i < len(card.strings) and card.strings[i]:
            return card.strings[i]
        return ""

    def say(self, text: str) -> None:
        if self.lines is not None:
            self.lines.append(text)

    def stats(self, turn: int, seat: int) -> dict:
        if turn not in self.turns:
            self.turns[turn] = [dict.fromkeys(STATS, 0) for _ in SEATS]
        return self.turns[turn][seat]

    def code_at(self, core, loc: M.Location) -> int:
        from ygorl.engine.query import parse_query_location

        try:
            slots = parse_query_location(core.query_location(C.QUERY_CODE, loc.controller, loc.location))
            c = slots[loc.sequence] if loc.sequence < len(slots) else None
            return c.get("code", 0) if c else 0
        except Exception:  # noqa: BLE001 - a name lookup only
            return 0

    def interrupts(self, code: int) -> bool:
        from ygorl.solver.blocking import card_interruptions

        return bool(code) and code in self.cards and bool(card_interruptions(code, self.cards))

    # ---------------------------------------------------------------- boards
    def snapshot(self, core, end_turn: int) -> None:
        """End of ``end_turn``: both boards (the state when the next turn starts or the game ends)."""
        from ygorl.solver.blocking import board_interruptions
        from ygorl.solver.targets import board_summary

        board = board_summary(core, end_turn, (self.lp[0], self.lp[1]))
        out = []
        for p in (0, 1):
            side = board["players"][p]
            score = board_interruptions(board, self.cards, player=p)
            mz = side["mzone"]
            st = [c for c in side["szone"] if c["sequence"] < 5]
            s = self.stats(end_turn, p)
            s["hand_end"] = len(side["hand"])
            s["monsters_end"] = len(mz)
            s["set_st_end"] = sum(1 for c in st if not c["position"] & C.POS_FACEUP)
            s["facedown_end"] = s["set_st_end"] + sum(1 for c in mz if not c["position"] & C.POS_FACEUP)
            s["interruptions_end"] = score.interruptions
            s["lp_end"] = self.lp[p]
            out.append({"interruptions": score.interruptions, "negates": score.negates,
                        "pieces": [[c, w, n] for c, w, n in score.pieces]})  # fmt: skip
            if self.lines is not None:
                face = lambda c: ("" if c["position"] & C.POS_FACEUP else "(set)") + self.name(c["code"])  # noqa: E731
                field = ", ".join(face(c) for c in mz) or "-"
                sts = ", ".join(face(c) for c in side["szone"]) or "-"
                hand = ", ".join(self.name(c) for c in side["hand"]) or "-"
                pieces = ", ".join(f"{self.name(c)}@{w}{'!' if n else ''}" for c, w, n in score.pieces) or "none"
                self.say(f"  [end T{end_turn}] P{p} LP {self.lp[p]} | monsters: {field} | S/T: {sts}")
                self.say(f"               hand({len(side['hand'])}): {hand} | GY {len(side['grave'])}"
                         f" banished {len(side['banished'])} | interruptions {score.interruptions}: {pieces}")  # fmt: skip
        self.boards[end_turn] = out

    # ---------------------------------------------------------------- messages
    def on_message(self, msg, core) -> None:  # noqa: C901 - one branch per message kind
        t, tp = self.turn, self.tp
        if isinstance(msg, M.NewTurn):
            if t:
                self.snapshot(core, t)
            self.turn, self.tp, self.turn_acts = t + 1, msg.player, 0
            self.stats(self.turn, 0), self.stats(self.turn, 1)
            self.say(f"=== Turn {self.turn}: P{msg.player} ({SEATS[msg.player]}) === LP {self.lp[0]}/{self.lp[1]}")
        elif isinstance(msg, M.NewPhase):
            self.phase = msg.phase
            if msg.phase == C.PHASE_MAIN1 and t:
                for p in (0, 1):
                    self.stats(t, p)["hand_start"] = core.query_count(p, C.LOCATION_HAND)
            if msg.phase in (C.PHASE_BATTLE_START, C.PHASE_MAIN2, C.PHASE_END):
                self.say(f" -- {_PHASES[msg.phase]}")
        elif isinstance(msg, (M.Summoning, M.SpSummoning, M.FlipSummoning)):
            p = msg.loc.controller
            kind = {M.Summoning: "normal summons", M.SpSummoning: "special summons", M.FlipSummoning: "flip summons"}
            self.stats(t, p)["special_summons" if isinstance(msg, M.SpSummoning) else "normal_summons"] += 1
            self.say(f"    P{p} {kind[type(msg)]} {self.name(msg.code)}")
        elif isinstance(msg, M.SetCard):
            self.say(f"    P{msg.loc.controller} sets {self.name(msg.code)}")
        elif isinstance(msg, M.Chaining):
            p, n = msg.triggering_controller, msg.chain_count
            intr = p != tp and self.interrupts(msg.code)
            self.chain[n] = {"code": msg.code, "player": p, "intr": intr}
            if p == tp:
                self.turn_acts += 1
            else:
                prev = self.chain.get(n - 1)
                self.chain[n]["resp"] = len(self.responses)
                self.responses.append({"turn": t, "player": p, "code": msg.code, "intr": intr, "link": n,
                                       "to": prev["code"] if prev else None,
                                       "to_player": prev["player"] if prev else None, "after": self.turn_acts,
                                       "negated": False})  # fmt: skip
            s = self.stats(t, p)
            s["act_own" if p == tp else "act_opp"] += 1
            s["intr_used"] += intr
            where = _LOC.get(msg.triggering_location, "?")
            eff = self.effect(msg.description)
            self.say(f"    CL{n} P{p} activates {self.name(msg.code)} @{where}" + (f" [{eff}]" if eff else ""))
        elif isinstance(msg, M.BecomeTarget):
            names = ", ".join(f"P{loc.controller} {self.name(self.code_at(core, loc))}" for loc in msg.locations)
            self.say(f"        targets {names}")
        elif isinstance(msg, M.ChainSolving):
            self.resolving.append(msg.chain_count)
        elif isinstance(msg, M.ChainSolved):
            if self.resolving and self.resolving[-1] == msg.chain_count:
                self.resolving.pop()
        elif isinstance(msg, (M.ChainNegated, M.ChainDisabled)):
            link = self.chain.get(msg.chain_count)
            if link is not None:
                # a link negated while it resolves itself (effect negation applied at resolution, e.g. Infinite
                # Impermanence) names no negater: credit the opponent
                own = not self.resolving or self.resolving[-1] == msg.chain_count
                by = None if own else self.chain.get(self.resolving[-1])
                doer = by["player"] if by else 1 - link["player"]
                self.stats(t, link["player"])["negates_suffered"] += 1
                self.stats(t, doer)["negates_done"] += 1
                if link["intr"]:
                    self.stats(t, link["player"])["intr_negated"] += 1
                if "resp" in link:
                    self.responses[link["resp"]]["negated"] = True
                what = "negated" if isinstance(msg, M.ChainNegated) else "effect negated"
                self.say(f"        CL{msg.chain_count} {self.name(link['code'])} {what}"
                         + (f" by {self.name(by['code'])}" if by else ""))  # fmt: skip
        elif isinstance(msg, M.ChainEnd):
            self.chain.clear()
            self.resolving.clear()
        elif isinstance(msg, M.Move):
            self._move(msg)
        elif isinstance(msg, M.Attack):
            p = msg.card.controller
            self.stats(t, p)["attacks"] += 1
            target = self.name(self.code_at(core, msg.target)) if msg.target.location else "directly"
            self.say(f"    P{p} {self.name(self.code_at(core, msg.card))} attacks {target}")
        elif isinstance(msg, M.DamageStepStart):
            self.damage_step = True
        elif isinstance(msg, M.DamageStepEnd):
            self.damage_step = False
        elif isinstance(msg, (M.Damage, M.PayLpCost)):
            self.lp[msg.player] -= msg.amount
            if isinstance(msg, M.Damage):
                battle = self.damage_step and not self.resolving
                self.stats(t, 1 - msg.player)["damage_dealt"] += msg.amount
                self.last_damage = (1 - msg.player, "battle" if battle else "effect", msg.amount)
                self.say(f"    P{msg.player} takes {msg.amount} {'battle' if battle else 'effect'} damage"
                         f" -> LP {self.lp[0]}/{self.lp[1]}")  # fmt: skip
        elif isinstance(msg, M.Recover):
            self.lp[msg.player] += msg.amount
        elif isinstance(msg, M.LpUpdate):
            self.lp[msg.player] = msg.amount
        elif isinstance(msg, M.Win):
            if self.last_damage is not None and msg.reason == 1:
                by, kind, amount = self.last_damage
                self.lethal = {"by": by, "kind": kind, "amount": amount}
            self.say(f"*** P{msg.player} wins (reason {msg.reason}) on turn {t}")

    def _move(self, msg: M.Move) -> None:
        prev, cur, reason = msg.previous, msg.current, msg.reason
        if prev.location & C.LOCATION_OVERLAY or cur.location == prev.location and cur.controller == prev.controller:
            return
        to_gone = cur.location & (C.LOCATION_GRAVE | C.LOCATION_REMOVED)
        if to_gone and prev.location & (C.LOCATION_ONFIELD | C.LOCATION_HAND):
            victim = prev.controller
            causer = None
            if not reason & (C.REASON_COST | C.REASON_MATERIAL | C.REASON_RELEASE | C.REASON_SUMMON):
                if reason & C.REASON_BATTLE:
                    causer = 1 - victim
                elif reason & C.REASON_EFFECT and self.resolving and self.resolving[-1] in self.chain:
                    causer = self.chain[self.resolving[-1]]["player"]
            if causer is not None and causer != victim:
                key = "opp_to_gy" if cur.location & C.LOCATION_GRAVE else "opp_banished"
                self.stats(self.turn, causer)[key] += 1
        if prev.location & C.LOCATION_ONFIELD and not reason & (C.REASON_COST | C.REASON_MATERIAL |
                                                                 C.REASON_RELEASE | C.REASON_SUMMON):  # fmt: skip
            causer = None
            if reason & C.REASON_BATTLE:
                causer = 1 - prev.controller
            elif reason & C.REASON_EFFECT and self.resolving and self.resolving[-1] in self.chain:
                causer = self.chain[self.resolving[-1]]["player"]
            if causer is not None and causer != prev.controller:
                self.removed.append({"turn": self.turn, "player": prev.controller, "code": msg.code,
                                     "by": self.chain[self.resolving[-1]]["code"] if self.resolving
                                     and self.resolving[-1] in self.chain else None,
                                     "to": _LOC.get(cur.location, "?"), "battle": bool(reason & C.REASON_BATTLE)})  # fmt: skip
        if self.lines is None:
            return
        src, dst = prev.location, cur.location
        if src == C.LOCATION_DECK and dst == C.LOCATION_HAND and not reason & C.REASON_DRAW:
            verb = "adds from deck"
        elif src & (C.LOCATION_ONFIELD | C.LOCATION_HAND) and dst & (C.LOCATION_GRAVE | C.LOCATION_REMOVED |
                                                                     C.LOCATION_HAND | C.LOCATION_DECK |
                                                                     C.LOCATION_EXTRA):  # fmt: skip
            if src == C.LOCATION_HAND and dst == C.LOCATION_GRAVE and reason & C.REASON_COST:
                verb = "discards (cost)"
            else:
                tags = [n for f, n in ((C.REASON_BATTLE, "battle"), (C.REASON_DESTROY, "destroyed"),
                                       (C.REASON_COST, "cost"), (C.REASON_MATERIAL, "material"),
                                       (C.REASON_RELEASE, "tributed"), (C.REASON_EFFECT, "effect")) if reason & f]  # fmt: skip
                verb = f"{_LOC.get(src, '?')} -> {_LOC.get(dst, '?')}" + (f" ({', '.join(tags)})" if tags else "")
        elif src & C.LOCATION_GRAVE and dst & (C.LOCATION_HAND | C.LOCATION_REMOVED):
            verb = f"GY -> {_LOC.get(dst, '?')}"
        else:
            return
        self.say(f"      P{prev.controller} {self.name(msg.code)}: {verb}")

    # ---------------------------------------------------------------- decisions
    def on_decision(self, point, idx: int) -> None:
        self.stats(point.turn, point.player)["decisions"] += 1
        if point.player != point.turn_player and point.decision.TYPE in (C.MSG_SELECT_CHAIN, C.MSG_SELECT_EFFECTYN):
            offers = self.offers[(point.turn, point.player)]
            chosen = point.actions[idx]
            for b in point.actions:
                if b.card is not None and b.card.code and b.kind in ("chain", "yes"):
                    o = offers.setdefault(b.card.code, [0, 0])
                    o[0] += 1
                    o[1] += b is chosen
        if self.lines is None or len(point.actions) < 2:
            return
        a = point.actions[idx]
        if a.kind in _QUIET:
            return
        what = a.kind + (f" {self.name(a.card.code)}" if a.card is not None and a.card.code else "")
        if a.kind == "activate" and a.description:
            eff = self.effect(a.description)
            what += f" [{eff}]" if eff else ""
        if point.decision.TYPE == C.MSG_SELECT_CHAIN:
            alts = sorted({self.name(b.card.code) for b in point.actions if b is not a and b.card is not None})
            self.say(f"  P{point.player} chain? -> {what}" + (f"   (could: {', '.join(alts)})" if alts else ""))
        elif point.decision.TYPE in (C.MSG_SELECT_IDLECMD, C.MSG_SELECT_BATTLECMD):
            self.say(f"  P{point.player} > {what}")
        else:
            self.say(f"  P{point.player}   {point.decision.name.removeprefix('MSG_').lower()}: {what}")

    def finish(self, core, result) -> dict:
        if self.turn:
            self.snapshot(core, self.turn)
        offers = {f"{t}:{p}": {str(c): v for c, v in d.items()} for (t, p), d in sorted(self.offers.items()) if t <= 6}
        return {"turns": {t: v for t, v in sorted(self.turns.items())}, "boards": self.boards, "lethal": self.lethal,
                "responses": [r for r in self.responses if r["turn"] <= 6], "offers": offers,
                "removed": [r for r in self.removed if r["turn"] <= 6]}  # fmt: skip


# -------------------------------------------------------------------- one game


def _session_class():
    from ygorl.engine.duel import DuelSession

    class TracedSession(DuelSession):
        """A :class:`DuelSession` that shows every decoded engine message, with the live core, to a tracer."""

        def __init__(self, duel, tracer: Tracer) -> None:
            self.tracer = tracer
            super().__init__(duel)

        def _advance(self, response):
            from ygorl import _core

            tracker, core = self.tracker, self.core
            while True:
                if response is not None:
                    core.set_response(response)
                    response = None
                try:
                    status = core.process()
                except _core.ScriptBudgetExceeded as e:
                    tracker.stop("error", str(e))
                    return
                buf = core.get_message()
                for msg in M.decode_buffer(buf):
                    self.tracer.on_message(msg, core)
                tracker.on_buffer(buf, status, core.pop_logs())
                if tracker.done:
                    return
                if not tracker.awaiting:
                    continue
                response = tracker.auto_response()
                if response is None:
                    tracker.point()
                    return

    return TracedSession


def play_game(job: dict, log: bool = False) -> tuple[dict, list[str]]:
    """Play (``job["actions"]`` absent) or replay one game; its record and trace lines."""
    from ygorl.agents.checkpoint import CheckpointAgent
    from ygorl.cards.ydk import load_ydk
    from ygorl.engine.duel import Duel, DuelConfig, ScriptedAgent, default_cards, deck_of_seat

    cards = default_cards()
    decks = [load_ydk(p) for p in job["decks"]]
    duel = Duel(job["seed"], None, decks[0], decks[1], config=DuelConfig(max_decisions=job.get("max_decisions", 6000)),
                first=job["first"])  # fmt: skip
    tracer = Tracer(cards, log)
    session = _session_class()(duel, tracer)
    if "actions" in job:
        script = ScriptedAgent(job["actions"])
        agents = [script, script]
    else:
        agents = [CheckpointAgent(job["ckpt"], job["agent_seeds"][i]) for i in (0, 1)]
        for a in agents:
            a.on_duel_start(duel)
    seat = [agents[deck_of_seat(job["first"], p)] for p in (0, 1)]
    actions = []
    try:
        while not session.done:
            point = session.point
            if point is None:
                break
            agent = seat[point.player]
            idx = 0 if len(point.actions) == 1 and "actions" not in job else agent.act(point)
            tracer.on_decision(point, idx)
            for a in agents:
                if hasattr(a, "on_decision"):
                    a.on_decision(point, idx)
            actions.append(idx)
            session.act(idx)
        res = session.result()
        info = tracer.finish(session.core, res)
    finally:
        session.close()
    winner_seat = None if res.winner is None else (0 if res.winner == deck_of_seat(job["first"], 0) else 1)
    t1 = info["turns"].get(1, [{}, {}])[0]
    rec = {k: job[k] for k in ("game", "decks", "seed", "first", "ckpt", "agent_seeds") if k in job}
    rec.update({
        "deck_first": Path(job["decks"][deck_of_seat(job["first"], 0)]).stem,
        "deck_second": Path(job["decks"][deck_of_seat(job["first"], 1)]).stem,
        "winner_seat": winner_seat, "end_turn": res.turns, "reason": res.reason, "win_reason": res.win_reason,
        "lethal": info["lethal"], "t1_decisions": t1.get("decisions", 0),
        "t1_interruptions": info["boards"].get(1, [{}])[0].get("interruptions", 0),
        "t1_pieces": info["boards"].get(1, [{}])[0].get("pieces", []),
        "boards": {t: b for t, b in info["boards"].items() if t <= 6}, "turns": info["turns"], "responses": info["responses"], "offers": info["offers"], "removed": info["removed"],
        "actions": actions,
    })  # fmt: skip
    return rec, tracer.lines or []


def _play_job(job: dict) -> dict:
    try:
        return play_game(job)[0]
    except Exception as exc:  # noqa: BLE001 - recorded
        return {"game": job["game"], "error": f"{type(exc).__name__}: {exc}"}


# -------------------------------------------------------------------- commands


def cmd_play(a) -> None:
    import numpy as np

    from ygorl.eval.arena import derive_seed

    if a.cmd == "replay":  # recompute the records of recorded games (same actions, current statistics)
        jobs = [g for g in load_games(a.games) if "actions" in g]
    else:
        decks = sorted(str(p.resolve()) for p in Path(a.decks).glob("*.ydk"))
        rng = np.random.default_rng(a.seed)
        jobs = []
        for k in range(a.games):
            i, j = (int(x) for x in rng.choice(len(decks), 2, replace=False))
            s = derive_seed(a.seed, k)
            jobs.append({"game": k, "decks": [decks[i], decks[j]], "seed": s, "first": 0, "ckpt": a.ckpt,
                         "agent_seeds": [derive_seed(s, 0), derive_seed(s, 1)]})  # fmt: skip
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    with out.open("w") as f, get_context("spawn").Pool(a.workers) as pool:
        for rec in pool.imap_unordered(_play_job, jobs, chunksize=1):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            done += 1
            if done % 50 == 0:
                print(f"{done}/{len(jobs)} games", file=sys.stderr, flush=True)


def load_games(path) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    for r in rows:
        for key in ("turns", "boards"):
            if key in r:
                r[key] = {int(t): v for t, v in r[key].items()}
    return rows


def select(games: list[dict], kind: str, min_intr: int = 2, by_turn: int = 4) -> list[dict]:
    blocked = [g for g in games if "error" not in g and g["t1_interruptions"] >= min_intr]
    if kind == "blocked-loss":
        return [g for g in blocked if g["winner_seat"] == 1 and g["end_turn"] <= by_turn]
    if kind == "blocked-win":
        return [g for g in blocked if g["winner_seat"] == 0]
    raise ValueError(kind)


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def summarize(games: list[dict], max_turn: int = 6) -> dict:
    """Means per turn and seat, for all games and split by the first player's outcome (and turn-1 blocking)."""
    ok = [g for g in games if "error" not in g and g["reason"] == "win"]
    groups = {"all": ok, "first_wins": [g for g in ok if g["winner_seat"] == 0],
              "first_loses": [g for g in ok if g["winner_seat"] == 1]}  # fmt: skip
    groups["blocked_first_wins"] = [g for g in groups["first_wins"] if g["t1_interruptions"] >= 2]
    groups["blocked_first_loses"] = [g for g in groups["first_loses"] if g["t1_interruptions"] >= 2]
    out: dict = {"games": len(games), "errors": sum("error" in g for g in games),
                 "not_won": sum("error" not in g and g["reason"] != "win" for g in games)}  # fmt: skip
    valid = [g for g in games if "error" not in g]
    out["first_win_rate"] = _mean(g["winner_seat"] == 0 for g in valid if g["winner_seat"] is not None)
    out["blocked_share"] = _mean(g["t1_interruptions"] >= 2 for g in valid)
    blocked = [g for g in valid if g["t1_interruptions"] >= 2 and g["winner_seat"] is not None]
    out["first_win_rate_blocked"] = _mean(g["winner_seat"] == 0 for g in blocked)
    out["first_win_rate_unblocked"] = _mean(g["winner_seat"] == 0 for g in valid
                                            if g["t1_interruptions"] < 2 and g["winner_seat"] is not None)  # fmt: skip
    ends = defaultdict(lambda: [0, 0])
    for g in ok:
        ends[min(g["end_turn"], 9)][g["winner_seat"]] += 1
    out["end_turn"] = {t: {"first": v[0], "second": v[1]} for t, v in sorted(ends.items())}
    lethal = defaultdict(int)
    for g in ok:
        lt = g.get("lethal")
        lethal[f"{SEATS[g['winner_seat']]}:{lt['kind'] if lt else 'other'}"] += 1
    out["lethal"] = dict(sorted(lethal.items()))
    out["t1_decisions"] = {k: _mean(g["t1_decisions"] for g in v) for k, v in groups.items()}
    out["t1_interruptions"] = {k: _mean(g["t1_interruptions"] for g in v) for k, v in groups.items()}
    table: dict = {}
    for name, gs in groups.items():
        table[name] = {"n": len(gs), "turns": {}}
        for t in range(1, max_turn + 1):
            row = {}
            for seat in (0, 1):
                live = [g["turns"][t][seat] for g in gs if t in g["turns"]]
                row[SEATS[seat]] = {k: round(_mean(s[k] for s in live), 3) for k in STATS} | {"n": len(live)}
            table[name]["turns"][t] = row
    out["per_turn"] = table
    return out


def print_summary(s: dict, keys=("hand_start", "hand_end", "special_summons", "act_own", "act_opp", "negates_done",
                                 "opp_to_gy", "monsters_end", "set_st_end", "interruptions_end", "intr_used",
                                 "intr_negated", "attacks", "damage_dealt", "lp_end", "decisions")) -> str:  # fmt: skip
    out = [f"games {s['games']} (errors {s['errors']}, no winner {s['not_won']}); first-player win rate "
           f"{s['first_win_rate']:.3f}; turn-1 >=2 interruptions {s['blocked_share']:.3f} "
           f"(first wins {s['first_win_rate_blocked']:.3f} with, {s['first_win_rate_unblocked']:.3f} without)",
           "end turn (wins first/second): " + ", ".join(f"T{t} {v['first']}/{v['second']}"
                                                         for t, v in s["end_turn"].items()),
           "lethal: " + ", ".join(f"{k} {v}" for k, v in s["lethal"].items()),
           "turn-1 decisions of first: " + ", ".join(f"{k} {v:.1f}" for k, v in s["t1_decisions"].items()),
           "turn-1 interruptions of first: " + ", ".join(f"{k} {v:.2f}" for k, v in s["t1_interruptions"].items())]  # fmt: skip
    for name, block in s["per_turn"].items():
        out.append(f"\n[{name}] n={block['n']}   (F = first player, S = second player; T = turn)")
        out.append(f"{'':18s}" + "".join(f"  T{t}:F   T{t}:S" for t in block["turns"]))
        for k in keys:
            cells = []
            for row in block["turns"].values():
                cells += [row["first"][k], row["second"][k]]
            out.append(f"{k:18s}" + "".join(f"{c:8.2f}" for c in cells))
        out.append(f"{'games alive':18s}" + "".join(f"{row['first']['n']:8d}{row['second']['n']:8d}"
                                                      for row in block["turns"].values()))  # fmt: skip
    return "\n".join(out)


def cmd_summary(a) -> None:
    s = summarize(load_games(a.games))
    text = print_summary(s)
    print(text)
    if a.json:
        Path(a.json).write_text(json.dumps(s, indent=1))


def cmd_trace(a) -> None:
    games = load_games(a.games)
    by_id = {g["game"]: g for g in games}
    chosen = [by_id[k] for k in a.game] if a.game else select(games, a.select)[: a.limit]
    out_dir = Path(a.out_dir) if a.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    for g in chosen:
        rec, lines = play_game(g, log=True)
        head = (f"# game {g['game']}: {rec['deck_first']} (P0, first) vs {rec['deck_second']} (P1, second); "
                f"winner {SEATS[rec['winner_seat']] if rec['winner_seat'] is not None else 'none'} on turn "
                f"{rec['end_turn']} ({rec['reason']}); lethal {rec['lethal']}; turn-1 interruptions "
                f"{rec['t1_interruptions']}")  # fmt: skip
        text = "\n".join([head, *lines]) + "\n"
        if out_dir:
            (out_dir / f"game_{g['game']:05d}.txt").write_text(text)
        else:
            print(text)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("play", help="self-play games of a checkpoint; one JSON line per game")
    pp.add_argument("ckpt")
    pp.add_argument("--games", type=int, default=1000)
    pp.add_argument("--decks", default="out/corpus/train")
    pp.add_argument("--out", required=True)
    pp.add_argument("--workers", type=int, default=8)
    pp.add_argument("--seed", type=int, default=0)
    pr = sub.add_parser("replay", help="recompute the records of recorded games from their actions")
    pr.add_argument("games")
    pr.add_argument("--out", required=True)
    pr.add_argument("--workers", type=int, default=8)
    ps = sub.add_parser("summary", help="per-turn means by seat and outcome")
    ps.add_argument("games")
    ps.add_argument("--json", default=None)
    pt = sub.add_parser("trace", help="replay recorded games into readable traces")
    pt.add_argument("games")
    pt.add_argument("--game", type=int, action="append", default=[])
    pt.add_argument("--select", choices=("blocked-loss", "blocked-win"), default="blocked-loss")
    pt.add_argument("--limit", type=int, default=20)
    pt.add_argument("--out-dir", default=None)
    a = p.parse_args()
    {"play": cmd_play, "replay": cmd_play, "summary": cmd_summary, "trace": cmd_trace}[a.cmd](a)


if __name__ == "__main__":
    main()
