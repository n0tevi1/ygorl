"""Why the BC warm start loses to Random in the Arena: the experiments of docs/bc.md「对 Random 失败的根因分析」.

Usage: uv run --frozen python tools/diagnose_bc.py SUBCOMMAND [options]

  consistency  replay demonstration lines through ``Duel.run`` with ``PolicyAgent(NetPolicy)`` watching and
               compare observations / logits with the training path (``build_dataset`` -> net), step by step
  coverage     what the training samples cover: turn, seat, whose turn, phase, decision type
  play         instrumented paired-seed games of one agent configuration against Random, on the same games as
               ``ygorl arena tests/decks --agent-b random --games 200`` (same cells, seeds and deck orders);
               writes per-game records with per-turn statistics to JSON
  report       tables from ``play`` outputs: win rates with Wilson intervals, paired differences, deck-out
               and battle statistics, per-turn curves, the policy's entropy by decision type

Agent configurations (``play --agent NAME``); "BC" is ``PolicyAgent(NetPolicy)`` of ``--checkpoint``:

  bc              BC everywhere, sampled at temperature 1 (what ``policy:PATH`` plays)
  bc_argmax       BC everywhere, argmax (``policy:PATH@greedy``)
  greedy, random  the baselines
  bc1_greedy      BC decides every decision of its own turn 1 when going first; Greedy decides the rest
  bc1_random      the same with Random for the rest
  bcown1_greedy   BC decides its first own turn (turn 1 going first, turn 2 going second); Greedy the rest
  greedy_bclate   Greedy on the own first turn, BC on everything else (the complement of bcown1_greedy)
  bc_bpd          BC everywhere except battle-phase decisions, which Greedy takes (delegation only)
  bc_bp           bc_bpd, and in main phase 1 an "end phase" choice of BC is turned into "battle phase"
                  whenever the battle phase is offered
  bc_own_greedy   BC in its own turns, Greedy in the opponent's turns (chain responses etc.)

Every game is a copy of the arena game (``Arena.game_specs``): the BC side is agent a, Random agent b, and the
agent seeds are the arena's, so ``play --agent bc`` reproduces ``out/bc12/arena_random.json`` game by game.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECK_DIR = ROOT / "tests" / "decks"
BATTLE_PHASES = 0x08 | 0x10 | 0x20 | 0x40 | 0x80  # PHASE_BATTLE_START .. PHASE_BATTLE
CONFIGS = ("bc", "bc_argmax", "greedy", "random", "bc1_greedy", "bc1_random", "bcown1_greedy", "greedy_bclate",
           "bc_bpd", "bc_bp", "bc_own_greedy")  # fmt: skip


# ------------------------------------------------------------------ instrumented composite agent


def _entropy(probs) -> float:
    return -sum(p * math.log(p) for p in probs if p > 0)


class Mixed:
    """One arena seat: picks the sub-agent for each decision by the configuration's rule and records statistics.

    Sees every decision point of the duel (``observe``), so it also keeps the event stream of the BC policy
    complete when BC decides only some of the decisions.
    """

    name = "mixed"

    def __init__(self, config: str, seed: int, checkpoint: str, temperature: float = 1.0) -> None:
        from ygorl.agents import GreedyAgent, PolicyAgent, RandomAgent

        if config not in CONFIGS:
            raise ValueError(f"unknown configuration {config!r} (one of {', '.join(CONFIGS)})")
        self.config = config
        self.bc = None
        if config.startswith("bc") or config.endswith("bclate"):
            from ygorl.nets.agent import NetPolicy

            self.bc = PolicyAgent(NetPolicy.from_checkpoint(checkpoint), seed=seed, greedy=config == "bc_argmax",
                                  temperature=temperature)  # fmt: skip
        self.greedy = GreedyAgent(seed)
        self.random = RandomAgent(seed)
        self.me: int | None = None
        self.core = None
        self.turn_player = 0
        self.turn = 0
        # statistics (engine players are mapped to 0 = this seat, 1 = the opponent in stats())
        self.draws = [0, 0]
        self.deck_leave: list[Counter] = [Counter(), Counter()]
        self.to_deck = [0, 0]
        self.damage_taken = [0, 0]
        self.attacks = [0, 0]  # MSG_ATTACK by the turn player
        self.bp_entered: list[set] = [set(), set()]  # turns in which the player entered the battle phase
        self.turns: list[dict] = []  # per turn: state at its first decision point
        self.own_turns: dict[int, dict] = {}
        self.decisions: Counter = Counter()  # (who decided, category)
        self.choices: Counter = Counter()  # "who|category|chosen kind" over decisions with >= 2 legal actions
        self.policy: dict[str, list[float]] = defaultdict(lambda: [0, 0.0, 0.0, 0.0])  # n, entropy, max p, log n
        self.close_choice: Counter = Counter()  # MP1 idle with battle phase offered: chosen kind
        self.battle_choice: Counter = Counter()  # SelectBattleCmd with an attack offered: chosen kind
        self.close_mass: Counter = Counter()  # BC's probability on battle / end phase at those MP1 points

    # -- sub-agent selection -------------------------------------------------
    def _own_first(self, point) -> bool:
        return point.turn_player == point.player and point.turn <= 2

    def _pick(self, point):
        c, own = self.config, point.turn_player == point.player
        if c in ("bc", "bc_argmax"):
            return self.bc
        if c == "greedy":
            return self.greedy
        if c == "random":
            return self.random
        if c in ("bc1_greedy", "bc1_random"):
            other = self.greedy if c == "bc1_greedy" else self.random
            return self.bc if point.turn == 1 and own else other
        if c == "bcown1_greedy":
            return self.bc if self._own_first(point) else self.greedy
        if c == "greedy_bclate":
            return self.greedy if self._own_first(point) else self.bc
        if c in ("bc_bpd", "bc_bp"):
            return self.greedy if own and point.phase & BATTLE_PHASES else self.bc
        if c == "bc_own_greedy":
            return self.bc if own else self.greedy
        raise AssertionError(c)

    # -- Agent protocol --------------------------------------------------------
    def observe(self, point, core) -> None:
        self.core = core
        if self.bc is not None:
            self.bc.observe(point, core)
        self._events(point)
        if point.turn != self.turn:
            self.turn = point.turn
            from ygorl.engine import constants as C

            row = {"turn": point.turn, "turn_player": point.turn_player, "lp": list(point.lp)}
            for key, loc in (("deck", C.LOCATION_DECK), ("hand", C.LOCATION_HAND), ("mzone", C.LOCATION_MZONE),
                             ("grave", C.LOCATION_GRAVE), ("removed", C.LOCATION_REMOVED)):  # fmt: skip
                row[key] = [core.query_count(p, loc) for p in (0, 1)]
            self.turns.append(row)

    def _events(self, point) -> None:
        from ygorl.engine import constants as C
        from ygorl.engine import messages as M

        dest = {C.LOCATION_HAND: "hand", C.LOCATION_GRAVE: "grave", C.LOCATION_MZONE: "mzone",
                C.LOCATION_SZONE: "szone", C.LOCATION_REMOVED: "removed", C.LOCATION_EXTRA: "extra"}  # fmt: skip
        for msg in point.events:
            if isinstance(msg, M.NewTurn):
                self.turn_player = msg.player
            elif isinstance(msg, M.Draw) and msg.player in (0, 1):
                self.draws[msg.player] += len(msg.cards)
            elif isinstance(msg, M.Move):
                prev, cur = msg.previous, msg.current
                if prev.location & C.LOCATION_DECK and not cur.location & C.LOCATION_DECK and prev.controller in (0, 1):
                    self.deck_leave[prev.controller][dest.get(cur.location & 0xFF, "other")] += 1
                elif (
                    cur.location & C.LOCATION_DECK and not prev.location & C.LOCATION_DECK and cur.controller in (0, 1)
                ):
                    self.to_deck[cur.controller] += 1
            elif isinstance(msg, M.Damage) and msg.player in (0, 1):
                self.damage_taken[msg.player] += msg.amount
            elif isinstance(msg, M.Attack):
                self.attacks[self.turn_player] += 1
            elif isinstance(msg, M.NewPhase) and msg.phase == C.PHASE_BATTLE_START:
                self.bp_entered[self.turn_player].add(point.turn)

    def _board(self, player: int) -> tuple[int, int]:
        """(face-up attack-position monsters' total ATK, monsters) of ``player``."""
        from ygorl.engine import constants as C
        from ygorl.engine.query import CARD_QUERY_FLAGS, parse_query_location

        cards = [c for c in parse_query_location(self.core.query_location(CARD_QUERY_FLAGS, player, C.LOCATION_MZONE))
                 if c]  # fmt: skip
        atk = sum(max(0, c.get("attack", 0)) for c in cards if c.get("position", 0) & C.POS_FACEUP_ATTACK)
        return atk, len(cards)

    def act(self, point) -> int:
        from ygorl.engine import constants as C

        if self.me is None:
            self.me = point.player
        own = point.turn_player == point.player
        agent = self._pick(point)
        idx = agent.act(point)
        kinds = {a.kind for a in point.actions}
        name = type(point.decision).__name__
        if self.config == "bc_bp" and agent is self.bc and name == "SelectIdleCmd" and point.phase == C.PHASE_MAIN1:
            if point.actions[idx].kind == "end_phase" and "battle_phase" in kinds:
                idx = next(i for i, a in enumerate(point.actions) if a.kind == "battle_phase")
        who = "bc" if agent is self.bc else agent.name
        cat = f"{name}/{'own' if own else 'opp'}/{'first' if point.turn <= 2 else 'later'}"
        self.decisions[(who, cat)] += 1
        if agent is self.bc and len(point.actions) > 1 and self.bc.last_probs is not None:
            s = self.policy[cat]
            s[0] += 1
            s[1] += _entropy(self.bc.last_probs)
            s[2] += max(self.bc.last_probs)
            s[3] += math.log(len(point.actions))
        chosen = point.actions[idx].kind
        if len(point.actions) > 1:
            self.choices[f"{who}|{cat}|{chosen}"] += 1
        if own:
            t = self.own_turns.setdefault(point.turn, {"turn": point.turn, "decisions": 0, "bp_offered": False,
                                                       "attack_offered": 0, "close": None})  # fmt: skip
            t["decisions"] += 1
            if name == "SelectIdleCmd" and point.phase == C.PHASE_MAIN1 and "battle_phase" in kinds:
                t["bp_offered"] = True
                self.close_choice[chosen if chosen in ("battle_phase", "end_phase") else "other"] += 1
                if agent is self.bc and self.bc.last_probs is not None:
                    for i, a in enumerate(point.actions):
                        if a.kind in ("battle_phase", "end_phase"):
                            self.close_mass[a.kind] += self.bc.last_probs[i]
                    self.close_mass["n"] += 1
                if chosen in ("battle_phase", "end_phase") and t["close"] is None:
                    atk, mine = self._board(point.player)
                    _, theirs = self._board(1 - point.player)
                    opp_lp = point.lp[1 - point.player]
                    t["close"] = chosen
                    t["free_damage"] = theirs == 0 and atk > 0
                    t["lethal"] = theirs == 0 and atk >= opp_lp
                    t["my_monsters"], t["opp_monsters"], t["my_atk"], t["opp_lp"] = mine, theirs, atk, opp_lp
            if name == "SelectBattleCmd" and "attack" in kinds:
                t["attack_offered"] += 1
                self.battle_choice[chosen if chosen in ("attack", "main2", "end_phase", "activate") else "other"] += 1
        return idx

    def stats(self) -> dict:
        me = self.me if self.me is not None else 0
        order = (me, 1 - me)
        return {
            "me": me,
            "draws": [self.draws[p] for p in order],
            "deck_leave": [dict(self.deck_leave[p]) for p in order],
            "to_deck": [self.to_deck[p] for p in order],
            "damage_taken": [self.damage_taken[p] for p in order],
            "attacks": [self.attacks[p] for p in order],
            "bp_turns": [sorted(self.bp_entered[p]) for p in order],
            "turns": self.turns,
            "own_turns": [self.own_turns[k] for k in sorted(self.own_turns)],
            "decisions": {f"{w}|{c}": n for (w, c), n in sorted(self.decisions.items())},
            "policy": {k: v for k, v in sorted(self.policy.items())},
            "close_choice": dict(self.close_choice),
            "battle_choice": dict(self.battle_choice),
            "close_mass": dict(self.close_mass),
            "choices": dict(self.choices),
        }


@dataclass(frozen=True)
class MixedFactory:
    config: str
    checkpoint: str
    temperature: float = 1.0

    @property
    def name(self) -> str:
        return self.config

    def __call__(self, seed: int) -> Mixed:
        return Mixed(self.config, seed, self.checkpoint, self.temperature)


def _play_one(spec) -> dict:
    """One arena game (``GameSpec``) with the instrumented seat as agent a."""
    import torch

    torch.set_num_threads(1)
    t0 = time.time()
    duel = spec.duel()
    a = spec.agent_a(spec.agent_seeds[0])
    b = spec.agent_b(spec.agent_seeds[1])
    rec = {"pair": spec.pair, "seed": spec.seed, "first": spec.first, "deck_a": spec.deck_a.name,
           "deck_b": spec.deck_b.name}  # fmt: skip
    try:
        r = duel.run(a, b)
    except Exception as exc:  # noqa: BLE001 - recorded
        return {**rec, "error": f"{type(exc).__name__}: {exc}"}
    rec.update(winner=r.winner, reason=r.reason, win_reason=r.win_reason, turns=r.turns, decisions=r.decisions,
               lp=list(r.lp), seconds=round(time.time() - t0, 2), stats=a.stats())  # fmt: skip
    return rec


def arena_specs(factory, games: int, seed: int = 0, max_turns: int | None = None, max_decisions: int | None = None):
    """The ``GameSpec`` s of ``ygorl arena tests/decks --agent-b random --games N --seed S`` with ``factory`` as a."""
    from ygorl.agents.registry import agent_factory
    from ygorl.commands import duel_config, load_decks
    from ygorl.eval.arena import Arena, derive_seed

    decks = load_decks([DECK_DIR])
    cells = [(a, b, derive_seed(seed, i, j)) for i, a in enumerate(decks) for j, b in enumerate(decks)]
    pairs = max(1, round(games / (2 * len(cells))))
    config = duel_config(None, max_turns)
    if max_decisions is not None:
        config = replace(config, max_decisions=max_decisions)
    arena = Arena(factory, agent_factory("random"), config=config)
    return [spec for a, b, s in cells for spec in arena.game_specs(a, b, pairs, s)]


def cmd_play(args) -> int:
    factory = MixedFactory(args.agent, str(args.checkpoint), args.temperature)
    specs = arena_specs(factory, args.games, args.seed, args.max_turns, args.max_decisions)
    if args.limit:
        specs = specs[: args.limit]
    t0 = time.time()
    if args.workers <= 1:
        records = [_play_one(s) for s in specs]
    else:
        with mp.get_context("fork").Pool(args.workers) as pool:
            records = pool.map(_play_one, specs, chunksize=1)
    out = {"agent": args.agent, "checkpoint": str(args.checkpoint), "temperature": args.temperature,
           "games": len(records), "seed": args.seed, "seconds": round(time.time() - t0, 1), "records": records}  # fmt: skip
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out) + "\n", encoding="utf-8")
    wins = sum(r.get("winner") == 0 for r in records)
    draws = sum(r.get("winner") is None for r in records)
    print(f"{args.agent}: {len(records)} games, {wins} wins, {draws} draws/errors, win rate "
          f"{(wins + draws / 2) / max(1, len(records)):.3f}, {out['seconds']}s -> {args.out}")  # fmt: skip
    return 0


# ------------------------------------------------------------------ report


def wilson(k: float, n: int) -> tuple[float, float]:
    from ygorl.eval.arena import wilson_interval

    return wilson_interval(k, n)


def score(r: dict) -> float:
    w = r.get("winner")
    return 1.0 if w == 0 else 0.5 if w is None else 0.0


def paired_diff(a: list[dict], b: list[dict]) -> tuple[float, float, float, int]:
    """Mean paired score difference a - b over the games both played (same spec), with a normal 95% interval."""
    key = lambda r: (r["deck_a"], r["deck_b"], r["seed"], r["first"])  # noqa: E731
    bm = {key(r): score(r) for r in b}
    d = [score(r) - bm[key(r)] for r in a if key(r) in bm]
    if len(d) < 2:
        return float("nan"), float("nan"), float("nan"), len(d)
    m, sd = statistics.fmean(d), statistics.stdev(d)
    h = 1.96 * sd / math.sqrt(len(d))
    return m, m - h, m + h, len(d)


def _load_runs(paths) -> dict[str, dict]:
    runs = {}
    for p in paths:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        name = data["agent"] + ("" if data.get("temperature", 1.0) == 1.0 else f"@t={data['temperature']}")
        runs[name] = data
    return runs


def _mean(xs) -> float:
    xs = list(xs)
    return statistics.fmean(xs) if xs else float("nan")


def cmd_report(args) -> int:
    runs = _load_runs(args.runs)
    base = runs.get(args.baseline)
    print("## Win rate vs Random (agent a = configuration; paired with the same games)\n")
    print(f"| configuration | games | W/L/D | win rate (95% Wilson) | going first / second | a lost by deck-out / LP "
          f"| a won by deck-out / LP | mean turns | paired diff vs {args.baseline} (95%) |")  # fmt: skip
    print("|---|---|---|---|---|---|---|---|---|")
    for name, run in runs.items():
        rs = [r for r in run["records"] if "error" not in r]
        n = len(rs)
        w = sum(r["winner"] == 0 for r in rs)
        loss = sum(r["winner"] == 1 for r in rs)
        d = n - w - loss
        k = w + d / 2
        lo, hi = wilson(k, n)
        f = [score(r) for r in rs if r["first"] == 0]
        s = [score(r) for r in rs if r["first"] == 1]
        lost_deck = sum(r["winner"] == 1 and r["win_reason"] == 2 for r in rs)
        lost_lp = sum(r["winner"] == 1 and r["win_reason"] == 1 for r in rs)
        won_deck = sum(r["winner"] == 0 and r["win_reason"] == 2 for r in rs)
        won_lp = sum(r["winner"] == 0 and r["win_reason"] == 1 for r in rs)
        diff = ""
        if base is not None and name != args.baseline:
            m, dlo, dhi, _ = paired_diff(rs, base["records"])
            diff = f"{m:+.3f} ({dlo:+.3f}, {dhi:+.3f})"
        errors = len(run["records"]) - n
        print(f"| {name} | {n}{f' (+{errors} err)' if errors else ''} | {w}/{loss}/{d} | {k / n:.3f} ({lo:.3f}–{hi:.3f}) "
              f"| {_mean(f):.3f} / {_mean(s):.3f} | {lost_deck} / {lost_lp} | {won_deck} / {won_lp} "
              f"| {_mean(r['turns'] for r in rs):.1f} | {diff} |")  # fmt: skip

    for name, run in runs.items():
        reasons = Counter(r.get("reason", "exception") for r in run["records"])
        if set(reasons) != {"win"}:
            print(f"- {name}: end reasons {dict(reasons)}")
    print("\n## Resources and battle per game (a = configuration, b = Random)\n")
    print("| configuration | cards leaving a's deck / own turn (a) | same (b) | extra draws / own turn (a / b) "
          "| of which search / GY / summon / banish (a) | entered BP when offered (a) | attacks / game (a / b) "
          "| damage dealt / game (a / b) | free direct damage skipped | lethal skipped |")  # fmt: skip
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, run in runs.items():
        rs = [r for r in run["records"] if "error" not in r]
        per_turn = [[], []]
        extra = [[], []]
        split = Counter()
        bp_offer = bp_in = free = free_skip = lethal = lethal_skip = 0
        for r in rs:
            st = r["stats"]
            me = st["me"]
            own = [sum(1 for t in st["turns"] if t["turn_player"] == (me if i == 0 else 1 - me)) for i in (0, 1)]
            for i in (0, 1):
                leave = sum(st["deck_leave"][i].values()) + st["draws"][i] - 5  # not the opening hand
                per_turn[i].append(leave / max(1, own[i]))
                normal = 5 + own[i] - (1 if (r["first"] == 0) == (i == 0) else 0)  # opening hand + 1 per turn
                extra[i].append((st["draws"][i] - normal) / max(1, own[i]))
            for k, v in st["deck_leave"][0].items():
                split[k] += v
            for t in st["own_turns"]:
                if t["bp_offered"]:
                    bp_offer += 1
                    bp_in += t["turn"] in st["bp_turns"][0]
                if t.get("free_damage"):
                    free += 1
                    free_skip += t["close"] == "end_phase"
                if t.get("lethal"):
                    lethal += 1
                    lethal_skip += t["close"] == "end_phase"
        own_turns = sum(sum(1 for t in r["stats"]["turns"] if t["turn_player"] == r["stats"]["me"]) for r in rs)
        tot = max(1, own_turns)
        print(f"| {name} | {_mean(per_turn[0]):.2f} | {_mean(per_turn[1]):.2f} | {_mean(extra[0]):.2f} / {_mean(extra[1]):.2f} "
              f"| {split['hand'] / tot:.2f} / {split['grave'] / tot:.2f} / {split['mzone'] / tot:.2f} / "
              f"{split['removed'] / tot:.2f} | {bp_in}/{bp_offer} = {bp_in / max(1, bp_offer):.2f} "
              f"| {_mean(r['stats']['attacks'][0] for r in rs):.1f} / {_mean(r['stats']['attacks'][1] for r in rs):.1f} "
              f"| {_mean(r['stats']['damage_taken'][1] for r in rs):.0f} / {_mean(r['stats']['damage_taken'][0] for r in rs):.0f} "
              f"| {free_skip}/{free} | {lethal_skip}/{lethal} |")  # fmt: skip

    print("\n## Per-turn trajectory (mean over games still running; a = configuration)\n")
    marks = (1, 2, 3, 5, 9, 15, 21, 31, 41, 51, 61)
    print("| configuration | " + " | ".join(f"t{t}" for t in marks) + " |")
    print("|---|" + "---|" * len(marks))
    for name, run in runs.items():
        rs = [r for r in run["records"] if "error" not in r]
        for key, label in (("deck", "deck a/b"), ("mzone", "monsters a/b"), ("lp", "LP a/b")):
            cells = []
            for t in marks:
                rows = []
                for r in rs:
                    me = r["stats"]["me"]
                    row = next((x for x in r["stats"]["turns"] if x["turn"] == t), None)
                    if row is not None:
                        rows.append((row[key][me], row[key][1 - me]))
                cells.append(
                    f"{_mean(a for a, _ in rows):.1f}/{_mean(b for _, b in rows):.1f} ({len(rows)})" if rows else "–"
                )
            print(f"| {name} {label} | " + " | ".join(cells) + " |")

    print("\n## Main phase 1 close (battle phase offered) and battle commands (attack offered): a's choices\n")
    for name, run in runs.items():
        cc, bc, mass = Counter(), Counter(), Counter()
        for r in run["records"]:
            if "error" in r:
                continue
            cc.update(r["stats"]["close_choice"])
            bc.update(r["stats"]["battle_choice"])
            mass.update(r["stats"].get("close_mass", {}))
        line = f"- {name}: MP1 idle {dict(cc)}; battle cmd {dict(bc)}"
        if mass.get("n"):
            line += (f"; BC's mean probability at those MP1 points: battle phase {mass['battle_phase'] / mass['n']:.3f}, "
                     f"end phase {mass['end_phase'] / mass['n']:.3f} ({int(mass['n'])} points)")  # fmt: skip
        print(line)

    print("\n## Chosen action kinds by decision category (decisions with >= 2 legal actions, a's side)\n")
    for name, run in runs.items():
        agg: dict[str, Counter] = defaultdict(Counter)
        for r in run["records"]:
            for k, v in r.get("stats", {}).get("choices", {}).items():
                who, cat, kind = k.split("|")
                agg[f"{who} {cat}"][kind] += v
        for cat, kinds in sorted(agg.items(), key=lambda kv: -sum(kv[1].values())):
            n = sum(kinds.values())
            if n >= args.min_decisions:
                top = ", ".join(f"{k} {v / n:.2f}" for k, v in kinds.most_common(6))
                print(f"- {name}: {cat} ({n}): {top}")

    print("\n## BC policy by decision category (decisions BC took with >= 2 legal actions)\n")
    print("| configuration | category | decisions | mean entropy | entropy of uniform | mean max p |")
    print("|---|---|---|---|---|---|")
    for name, run in runs.items():
        agg: dict[str, list[float]] = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
        for r in run["records"]:
            for k, v in r.get("stats", {}).get("policy", {}).items():
                for i in range(4):
                    agg[k][i] += v[i]
        for k, (n, h, pmax, logn) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
            if n >= args.min_decisions:
                print(f"| {name} | {k} | {int(n)} | {h / n:.2f} | {logn / n:.2f} | {pmax / n:.2f} |")
    return 0


# ------------------------------------------------------------------ consistency


def cmd_consistency(args) -> int:
    import numpy as np
    import torch

    from ygorl.agents import PolicyAgent
    from ygorl.engine.duel import default_cards
    from ygorl.nets.agent import NetPolicy, load_checkpoint
    from ygorl.nets.batch import OBS_KEYS, collate, to_tensors
    from ygorl.solver import read_jsonl
    from ygorl.train.bc import line_steps, trim_padding

    torch.set_num_threads(args.threads)
    ckpt = load_checkpoint(args.checkpoint)
    cards = default_cards()
    demos = [d for d in read_jsonl(args.demos) if d.status == "solved"][: args.records]

    class Scripted:
        """Plays a line's actions for both seats; at player 0's non-forced points also asks NetPolicy."""

        def __init__(self, actions, policy):
            self.actions, self.i, self.policy = list(actions), 0, policy
            self.logits, self.obs = [], []

        def observe(self, point, core):
            self.policy.observe(point, core)

        def act(self, point):
            if point.player == 0 and len(point.actions) > 1:
                self.logits.append(np.asarray(self.policy.policy(point)[: min(len(point.actions), 128)]))
                self.obs.append(self.policy.policy._observer.encode(point, self.policy.policy._core))
            idx = self.actions[self.i]
            self.i += 1
            return idx

    worst_logit = worst_prob = 0.0
    mismatched_arrays: Counter = Counter()
    steps = lines = argmax_diff = 0
    for demo in demos:
        for li, ln in enumerate(demo.lines):
            train_steps = [s for s in line_steps(demo, li, ckpt.vocab, cards=cards, event_length=ckpt.event_length)
                           if s.obs is not None]  # fmt: skip
            policy = PolicyAgent(NetPolicy(ckpt.net, ckpt.vocab, event_length=ckpt.event_length, cards=cards), seed=0)
            agent = Scripted(ln.actions, policy)
            duel = demo.replay(li).duel(None, cards=cards)
            try:
                duel.run(agent, agent)
            except IndexError:
                pass  # the line ends before the duel does: the scripted log is exhausted
            if len(agent.logits) < len(train_steps):
                print(f"{demo.deck['name']} {demo.hand_index} line {li}: Duel.run saw {len(agent.logits)} decisions, "
                      f"line_steps {len(train_steps)}")  # fmt: skip
            obs = [s.obs for s in train_steps]
            with torch.no_grad():
                batch = to_tensors(trim_padding({k: np.stack([o[k] for o in obs]) for k in OBS_KEYS if k in obs[0]}))
                train_logits = ckpt.net(batch).logits.numpy()
                single = [ckpt.net(collate([o])).logits[0].numpy() for o in obs]
            for i, s in enumerate(train_steps[: len(agent.logits)]):
                n = min(s.n_legal, 128)
                for k in OBS_KEYS:
                    if k in s.obs and not np.array_equal(s.obs[k], agent.obs[i][k]):
                        mismatched_arrays[k] += 1
                a, b, c = agent.logits[i][:n], train_logits[i][:n], single[i][:n]
                worst_logit = max(worst_logit, float(np.abs(a - b).max()), float(np.abs(a - c).max()))
                pa, pb = np.exp(a - a.max()), np.exp(b - b.max())
                worst_prob = max(worst_prob, float(np.abs(pa / pa.sum() - pb / pb.sum()).max()))
                argmax_diff += int(np.argmax(a) != np.argmax(b))
                steps += 1
            lines += 1
    print(f"{lines} lines, {steps} decisions of player 0 compared (Duel.run + PolicyAgent(NetPolicy) vs "
          f"build_dataset path, batched with trim_padding and one by one)")  # fmt: skip
    print(f"observation arrays that differ: {dict(mismatched_arrays) or 'none'}")
    print(f"max |logit difference| {worst_logit:.2e}, max |probability difference| {worst_prob:.2e}, "
          f"argmax differs at {argmax_diff} decisions")  # fmt: skip
    return 0


# ------------------------------------------------------------------ coverage


def cmd_coverage(args) -> int:
    from ygorl.engine import messages as M
    from ygorl.engine.duel import default_cards
    from ygorl.nets.agent import load_checkpoint
    from ygorl.solver import read_jsonl
    from ygorl.train.bc import build_dataset

    ckpt = load_checkpoint(args.checkpoint)
    demos = read_jsonl(args.demos)
    data = build_dataset(demos, ckpt.vocab, cards=default_cards(), event_length=ckpt.event_length)
    g = data.obs["globals"]
    names = {cls.TYPE: cls.__name__ for cls in vars(M).values() if isinstance(cls, type) and hasattr(cls, "TYPE")
             and issubclass(cls, M.Decision) and cls is not M.Decision}  # fmt: skip
    print(f"{len(data)} samples")
    for col, label in ((0, "viewer (engine player)"), (2, "is my turn"), (3, "turn"), (4, "phase bit index")):
        print(f"{label}: {dict(sorted(Counter(g[:, col].tolist()).items()))}")
    print(f"decision type: {dict(Counter(names.get(t, t) for t in g[:, 18].tolist()).most_common())}")
    from ygorl.env.encoding import ACTION_KINDS

    kinds = data.obs["actions"][..., 0]
    offered = Counter(ACTION_KINDS[k - 1] for k in kinds[data.obs["action_mask"] > 0].tolist())
    chosen = Counter(ACTION_KINDS[kinds[i, a] - 1] for i, a in enumerate(data.actions.tolist()))
    print(f"action kinds offered (candidate rows): {dict(offered.most_common())}")
    print(f"action kinds demonstrated: {dict(chosen.most_common())}")
    idle = g[:, 18] == M.SelectIdleCmd.TYPE
    print(f"SelectIdleCmd samples {int(idle.sum())}: demonstrated "
          f"{dict(Counter(ACTION_KINDS[kinds[i, a] - 1] for i, a in enumerate(data.actions.tolist()) if idle[i]).most_common())}")  # fmt: skip
    ev = data.obs["events"]
    mask = data.obs["event_mask"] > 0
    print(f"event tokens: turn column {dict(sorted(Counter(ev[..., 15][mask].tolist()).items()))}, "
          f"my-turn column {dict(sorted(Counter(ev[..., 17][mask].tolist()).items()))}")  # fmt: skip
    return 0


# ------------------------------------------------------------------ main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    ckpt = ROOT / "out" / "bc12" / "policy.pt"

    p = sub.add_parser("play", help="instrumented arena games against Random")
    p.add_argument("--agent", required=True, choices=CONFIGS)
    p.add_argument("--checkpoint", type=Path, default=ckpt)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-turns", type=int, default=None)
    p.add_argument(
        "--max-decisions", type=int, default=None, help="decision limit per game (default: the arena's 20000)"
    )
    p.add_argument("--limit", type=int, default=0, help="play only the first N games (smoke test)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_play)

    p = sub.add_parser("report", help="tables from play outputs")
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--baseline", default="bc")
    p.add_argument("--min-decisions", type=int, default=200)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("consistency", help="Duel.run + NetPolicy vs the training path on demonstration lines")
    p.add_argument("--demos", type=Path, default=ROOT / "out" / "demos" / "bc_heldout.jsonl")
    p.add_argument("--checkpoint", type=Path, default=ckpt)
    p.add_argument("--records", type=int, default=20)
    p.add_argument("--threads", type=int, default=2)
    p.set_defaults(func=cmd_consistency)

    p = sub.add_parser("coverage", help="what the training samples cover")
    p.add_argument("--demos", type=Path, default=ROOT / "out" / "demos" / "bc_train.jsonl")
    p.add_argument("--checkpoint", type=Path, default=ckpt)
    p.set_defaults(func=cmd_coverage)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
