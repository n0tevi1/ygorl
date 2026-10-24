"""P0a: where and why scripted lines break, and what a break is worth to the learner (first-seat plan §6 phase 0).

Plays ``--games`` scripted-line games per seat as training deals them (``selfplay._deal_lines``: a meta deck for the
learner, a line drawn from the library, ``LineScript.spec``; ``EncodedVecEnv`` with the run's ``skip_forced``; the
opponent seat plays its line with :class:`~ygorl.train.lines.LinePlayer` -- first-seat tolerances when the learner
moves first -- and the snapshot ``--opponent`` plays it otherwise, including the opponent's turn 1 in a first-seat
game). Collection only, no update. Per game: the line's status and break step, the break's cause, the learner's chain
windows on the line's turn (two or more choosable rows, as ``Rollout.line_occurrence`` counts them) and how many it
answered with an activation, and the result. Causes of a break:

- ``learner_interrupt``: the learner activated something at a chain window of the line's turn before the break;
- ``turn2_start``: first seat only, no interruption, the line broke at its first decision (the turn-2 tolerances of
  ``LinePlayer`` found no match: the learner's turn-1 board or the extra card drawn changed the opening);
- ``other``: any other divergence (the learner's board or a non-chain effect changed the line's options).

Writes a JSON report: per seat, the status / cause distribution, the break position (step, fraction of the line,
turn), the learner's win rate by status and cause, windows and fires per game; ``--rows`` adds every game.

  uv run --no-sync python tools/line_break_census.py CKPT --lines out/claude-scale/handtrap/lines_train.jsonl \\
      [--games 512] [--opponent CKPT] [--envs 64] [--threads 8] [--seed 0] --out OUT.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase0_common import CHAIN, Brain, Job, Net, decision, generator, is_pass, legal_rows, mean_se, own_turn, run_jobs, turn  # noqa: E402,E501  # fmt: skip

from ygorl.cards.ydk import load_ydk  # noqa: E402
from ygorl.data import load_environment  # noqa: E402
from ygorl.engine.duel import DuelConfig, default_cards  # noqa: E402
from ygorl.env.encoded import EncodedVecEnv  # noqa: E402
from ygorl.eval.arena import derive_seed  # noqa: E402
from ygorl.train.lines import LineLibrary, LinePlayer  # noqa: E402

CAUSES = ("learner_interrupt", "turn2_start", "other")


class ScriptedGame(Job):
    """One scripted-line game; the learner at ``seat`` (0: it moves first, the line is played on turn 2)."""

    def __init__(self, index: int, spec, script, seat: int, skip_forced: bool, seed: int, *, info: dict) -> None:
        self.index, self.spec, self.script, self.seat = index, spec, script, seat
        self.line = LinePlayer(script, skip_forced, first_seat=seat == 0)
        self.line_turn = 2 if seat == 0 else 1
        self.learner_gen = generator(seed, 1)
        self.opponent_gen = generator(seed, 2)
        self.info = info
        self.windows = self.fires = self.fires_before_break = 0
        self.cause: str | None = None
        self.break_turn: int | None = None
        self.max_turn = 0
        self._window = False

    def _line_turn(self, obs) -> bool:
        return turn(obs) == self.line_turn and not own_turn(obs)

    def decide(self, ev):
        obs = ev.obs
        self.max_turn = max(self.max_turn, turn(obs))
        if ev.player == self.seat:
            self._window = decision(obs) == CHAIN and len(legal_rows(obs)) > 1 and self._line_turn(obs)
            self.windows += self._window
            return Net("learner", self.learner_gen)
        before = self.line.status
        row = self.line.act(obs)
        if before == "playing" and self.line.status == "broken":
            self.break_turn = turn(obs)
            if self.fires_before_break:
                self.cause = "learner_interrupt"
            elif self.seat == 0 and self.line.played == 0:
                self.cause = "turn2_start"
            else:
                self.cause = "other"
        if row is not None:
            return row
        return Net("opponent", self.opponent_gen)

    def chosen(self, ev, row: int) -> None:
        if ev.player == self.seat and self._window and not is_pass(ev.obs, row):
            self.fires += 1
            if self.line.status == "playing":
                self.fires_before_break += 1
        if ev.player == self.seat:
            self._window = False

    def finish(self, result):
        line = self.line.finish()
        winner = None if result is None else result.get("winner")
        score = 0.5 if winner is None else float(winner == self.seat)
        return {"game": self.index, "seat": self.seat, **self.info, "status": line["status"], "cause": self.cause,
                "broken_at": line["broken_at"], "steps": line["steps"], "played": line["played"],
                "skipped": line["skipped"], "extra": line["extra"], "break_turn": self.break_turn,
                "windows": self.windows, "fires": self.fires, "fires_before_break": self.fires_before_break,
                "score": score, "reason": None if result is None else str(result.get("reason", "")),
                "turns": None if result is None else int(result.get("turns", self.max_turn))}  # fmt: skip


def deal(library: LineLibrary, decks, config: DuelConfig, games: int, seat: int, seed: int, skip_forced: bool):
    rng = np.random.default_rng(derive_seed(seed, 17, seat))
    jobs = []
    for g in range(games):
        learner = decks[int(rng.integers(len(decks)))]
        script = library.sample(rng)
        game_seed = derive_seed(seed, seat, g)
        spec = script.spec(learner, game_seed, config, learner_seat=seat)
        info = {"learner_deck": learner.name, "line": script.id, "line_name": script.name}
        jobs.append(ScriptedGame(g, spec, script, seat, skip_forced, game_seed, info=info))
    return jobs


def _rate(rows) -> dict:
    s = mean_se([r["score"] for r in rows])
    return {"games": s["n"], "win_rate": s["mean"], "se": s["se"]}


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    status = Counter(r["status"] for r in rows)
    broken = [r for r in rows if r["status"] == "broken"]
    causes = Counter(r["cause"] for r in broken)
    frac = [r["broken_at"] / max(1, r["steps"]) for r in broken]
    hist = Counter(min(4, int(5 * f)) for f in frac)
    out = {
        "games": n,
        "status": {k: status[k] / n for k in ("completed", "broken", "playing")},
        "status_counts": dict(status),
        "cause": {k: causes[k] / max(1, len(broken)) for k in CAUSES},
        "cause_counts": {k: causes[k] for k in CAUSES},
        "break_step": mean_se([r["broken_at"] for r in broken]),
        "break_fraction": mean_se(frac),
        "break_fraction_hist": {f"{i / 5:.1f}-{(i + 1) / 5:.1f}": hist[i] for i in range(5)},
        "break_turn": dict(Counter(r["break_turn"] for r in broken)),
        "win_rate": _rate(rows),
        "win_rate_completed": _rate([r for r in rows if r["status"] == "completed"]),
        "win_rate_broken": _rate(broken),
        "win_rate_by_cause": {k: _rate([r for r in broken if r["cause"] == k]) for k in CAUSES},
        "win_rate_playing": _rate([r for r in rows if r["status"] == "playing"]),
        "windows_per_game": mean_se([r["windows"] for r in rows]),
        "fires_per_game": mean_se([r["fires"] for r in rows]),
        "fire_rate": sum(r["fires"] for r in rows) / max(1, sum(r["windows"] for r in rows)),
        "games_with_fire": sum(r["fires"] > 0 for r in rows) / max(1, n),
        "turns": mean_se([r["turns"] for r in rows if r["turns"] is not None]),
    }
    return out


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("ckpt")
    p.add_argument("--lines", nargs="+", required=True, help="demonstration JSONL(s) of the line library")
    p.add_argument("--opponent", default=None, help="snapshot that plays the opponent off its line (default: CKPT)")
    p.add_argument("--games", type=int, default=512, help="games per seat")
    p.add_argument("--seats", default="0,1", help="learner seats to run (0: first, 1: second)")
    p.add_argument("--env", default="md-2026-09")
    p.add_argument("--decks", nargs="*", default=None, help="learner decks (default: the environment's meta decks)")
    p.add_argument("--max-turns", type=int, default=None, help="game turn limit (default: the environment's)")
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--torch-threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-lines", type=int, default=None, help="load only the first N lines (smoke tests)")
    p.add_argument("--rows", action="store_true", help="also write every game")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    torch.set_num_threads(a.torch_threads)
    t0 = time.perf_counter()
    env = load_environment(a.env)
    cards = default_cards()
    learner = Brain(a.ckpt)
    brains = {"learner": learner, "opponent": learner if a.opponent in (None, a.ckpt) else Brain(a.opponent)}
    if brains["opponent"].signature != learner.signature:
        raise SystemExit("the opponent's vocabulary / event length differ from the learner's")
    library = LineLibrary.load(a.lines, learner.vocab, cards=cards, env=env)
    if a.max_lines is not None:
        library = LineLibrary(library.scripts[: a.max_lines], library.problems)
    paths = a.decks or sorted(str(x) for x in (Path(env.root) / "meta").glob("*.ydk"))
    decks = [load_ydk(x) for x in paths]
    config = DuelConfig.from_environment(env, **({"max_turns": a.max_turns} if a.max_turns else {}))
    cfg = learner.net_config
    venv = EncodedVecEnv(a.envs, a.threads, cards=cards, vocab=learner.vocab, event_length=learner.event_length,
                         skip_forced=True, selection_history=cfg.selection_history,
                         attack_history=cfg.attack_history)  # fmt: skip
    report = {"ckpt": a.ckpt, "opponent": a.opponent or a.ckpt, "lines": len(library), "line_problems":
              len(library.problems), "decks": [d.name for d in decks], "games_per_seat": a.games, "seed": a.seed,
              "seats": {}}  # fmt: skip
    rows_all = []
    for seat in (int(s) for s in a.seats.split(",")):
        jobs = deal(library, decks, config, a.games, seat, a.seed, venv.skip_forced)
        rows = run_jobs(venv, jobs, brains)
        report["seats"]["first" if seat == 0 else "second"] = summarize(rows)
        rows_all += rows
    if a.rows:
        report["rows"] = rows_all
    report["seconds"] = round(time.perf_counter() - t0, 1)
    text = json.dumps(report, indent=1)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n")
    else:
        print(text)
    return report


if __name__ == "__main__":
    main()
