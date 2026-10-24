"""First-seat line evaluation (docs/training.md 「尺子」; plan docs/spikes/first-seat-interruption-plan-2026-10-24.md T10).

The learner moves first; the opponent seat plays a solver demonstration line on its turn 2 (the scripted-line
opponent of docs/training.md §8.3 in its first-seat form: :class:`ygorl.train.lines.LinePlayer` with
``first_seat=True``) and, once the line is over or broken, is played by ``--opponent`` (default: the first
checkpoint). Reported per checkpoint:

- ``line_completion``: share of games whose line was played to the end (``status`` counts in ``line_status``);
  ``completion_if_fired`` / ``completion_if_held``: the same split by whether the learner activated anything at a
  turn-2 window;
- ``win_rate``: the learner's wins over decided games (``by_line``: split by completed / not completed);
- ``fire_hist``: the turn-2 window (1-based) of the learner's first activation, ``none`` when it never activated;
  ``p_fire_at``: P(activate | window k reached without an earlier activation); ``windows_mean``.

A *window* is a learner chain prompt (MSG_SELECT_CHAIN) with two or more choosable rows on the opponent's turn 2, the
windows ``Rollout.line_window`` counts in a first-seat scripted game; an *activation* is any non-pass row there.

Every checkpoint plays the same games: the same lines, learner decks and deck orders (``--seed``), and each sampled
decision draws its uniform number from (game seed, seat, decision count) -- common random numbers, so two checkpoints
are compared game by game (``paired``: mean score difference with its standard error and the discordant games).

  uv run --no-sync python tools/first_seat_line_eval.py CKPT [CKPT_B] --demos LINES.jsonl [--min-hand 40] \\
      [--games-per-line 2] [--opponent CKPT] [--line-recovery FIRE.jsonl] [--greedy] [--json] [--out report.json]

``--line-recovery`` (repeatable) is passed to ``LineLibrary.load(..., recovery=)`` (the T6 recovery lines).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

LEARNER, LINE_SEAT = 0, 1  # engine seats in a first-seat scripted game (learner_seat 0)


@dataclass
class GameState:
    line: object  # LinePlayer
    decisions: list[int] = field(default_factory=lambda: [0, 0])  # decisions answered per seat
    windows: int = 0  # learner turn-2 windows seen
    fired: list[int] = field(default_factory=list)  # windows the learner activated at


@dataclass
class GameOutcome:
    index: int
    line_id: int
    line_name: str
    learner_deck: str
    winner: int | None  # engine seat; None: draw / truncated / error
    reason: str
    turns: int
    line: dict  # LinePlayer.finish() (+ "completed")
    windows: int
    fired: list[int]

    @property
    def decided(self) -> bool:
        return self.reason == "win" and self.winner in (0, 1)

    @property
    def score(self) -> float:
        """1 / 0.5 / 0 for the learner (0.5: no winner)."""
        return 0.5 if not self.decided else float(self.winner == LEARNER)


def uniform(seed: int, seat: int, step: int) -> float:
    """The sampled decision's uniform number (keyed like ygorl.eval.batched: game seed, first player 0, seat, step)."""
    return float(np.random.default_rng([seed & 0xFFFFFFFFFFFFFFFF, 0, seat, step]).random())


def is_window(obs) -> bool:
    """A learner chain prompt with a real choice on the opponent's turn 2 (Rollout's first-seat window)."""
    from ygorl.engine.constants import MSG_SELECT_CHAIN

    g = obs["globals"]
    return (int(g[18]) == MSG_SELECT_CHAIN and int((obs["action_mask"] != 0).sum()) > 1 and int(g[3]) == 2
            and not int(g[2]))  # fmt: skip


def is_pass(obs, row: int) -> bool:
    from ygorl.env.encoding import ACTION_KINDS

    return int(obs["actions"][row][0]) == ACTION_KINDS.index("pass") + 1


def line_completed(player) -> bool:
    """The line was played to the end: ``completed``, or still ``playing`` with no unforced step left (the seat was not
    asked again before the game ended)."""
    if player.status == "completed":
        return True
    if player.status != "playing" or player.played == 0:
        return False
    rest = player.script.steps[player.pos :]
    return all(player.skip_forced and s.forced for s in rest)


def load_library(paths: Sequence[str], vocab, cards, environment, recovery: Sequence[str], *, min_hand: int = 0,
                 max_hand: int = 10**9, scratch: Path | None = None):  # fmt: skip
    """The line library of the plain records of ``paths`` with ``min_hand <= hand_index < max_hand`` (filtered into
    ``scratch`` first: keying a line replays it, so only the lines in range are keyed)."""
    import tempfile

    from ygorl.train.lines import LineLibrary

    if min_hand > 0 or max_hand < 10**9:
        scratch = Path(scratch or tempfile.mkdtemp(prefix="first-seat-lines-"))
        kept = []
        for i, path in enumerate(paths):
            out = scratch / f"{i}_{Path(path).name}"
            with open(path) as f, out.open("w") as g:
                for line in f:
                    if min_hand <= json.loads(line).get("hand_index", -1) < max_hand:
                        g.write(line)
            kept.append(str(out))
        paths = kept

    kwargs = {}
    if recovery:  # LineLibrary.load(recovery=) takes an iterable of fire-record files (T6)
        kwargs["recovery"] = list(recovery)
    return LineLibrary.load(paths, vocab, cards=cards, env=environment, **kwargs)


def build_games(scripts, decks, games_per_line: int, seed: int, config):
    """(script, learner deck, GameSpec) per game: the learner deck of each game is drawn with ``seed``."""
    from ygorl.eval.arena import derive_seed

    out = []
    for s in scripts:
        for g in range(games_per_line):
            deck = decks[derive_seed(seed, 0, s.id, g) % len(decks)]
            out.append((s, deck, s.spec(deck, derive_seed(seed, 1, s.id, g), config, learner_seat=LEARNER)))
    return out


def play(env, games, learner, opponent, *, greedy: bool = False, device="cpu", min_batch: int | None = None):
    """Play every game with ``learner`` (seat 0) against the line then ``opponent`` (seat 1); one outcome per game."""
    import torch

    from ygorl.env.driver import drive
    from ygorl.nets.batch import collate, policy_logits
    from ygorl.train.lines import LinePlayer

    specs = [spec for _, _, spec in games]
    outcomes: list[GameOutcome | None] = [None] * len(specs)

    def start(i, spec):
        return GameState(LinePlayer(games[i][0], bool(env.skip_forced), first_seat=True))

    def on_result(game, res):
        st: GameState = game.state
        script, deck, _ = games[game.index]
        line = {**st.line.finish(), "completed": line_completed(st.line)}
        w = res.get("winner")
        reason = str(res.get("reason", ""))
        outcomes[game.index] = GameOutcome(game.index, script.id, script.name, deck.name,
                                           None if w is None or reason == "error" else int(w), reason,
                                           int(res.get("turns", 0)), line, st.windows, list(st.fired))  # fmt: skip

    def on_error(i, spec, exc):
        script, deck, _ = games[i]
        outcomes[i] = GameOutcome(i, script.id, script.name, deck.name, None, "exception", 0,
                                  {"status": "error", "completed": False, "error": repr(exc)}, 0, [])  # fmt: skip

    @torch.no_grad()
    def decide(ready):
        actions = [0] * len(ready)
        groups: dict[int, list[int]] = {}
        for k, (game, ev) in enumerate(ready):
            if ev.player == LINE_SEAT:
                row = game.state.line.act(ev.obs)
                if row is not None:
                    actions[k] = int(row)
                    game.state.decisions[LINE_SEAT] += 1
                    continue
            groups.setdefault(LEARNER if ev.player == LEARNER else LINE_SEAT, []).append(k)
        for seat, ks in groups.items():
            module = learner if seat == LEARNER else opponent
            logits = policy_logits(module, collate([ready[k][1].obs for k in ks], device)).float()
            if greedy:
                chosen = logits.argmax(-1).cpu().tolist()
            else:
                probs = torch.softmax(logits, -1).cpu().numpy().astype(np.float64)
                cdf = np.cumsum(probs, -1)
                chosen = []
                for r, k in enumerate(ks):
                    game, ev = ready[k]
                    u = uniform(game.spec.seed, seat, game.state.decisions[seat])
                    chosen.append(min(int(np.searchsorted(cdf[r], u * cdf[r, -1], side="right")), probs.shape[1] - 1))
            for k, a in zip(ks, chosen, strict=True):
                game, ev = ready[k]
                st: GameState = game.state
                st.decisions[seat] += 1
                if seat == LEARNER and is_window(ev.obs):
                    st.windows += 1
                    if not is_pass(ev.obs, int(a)):
                        st.fired.append(st.windows)
                actions[k] = int(a)
        return actions

    for m in {id(learner): learner, id(opponent): opponent}.values():
        m.eval()
    drive(env, specs, decide, on_result, min_batch=max(1, min_batch or env.num_envs // 2), start=start,
          on_error=on_error)  # fmt: skip
    return outcomes


def summarize(outcomes: Sequence[GameOutcome]) -> dict:
    n = len(outcomes)
    decided = [o for o in outcomes if o.decided]
    completed = [o for o in outcomes if o.line.get("completed")]
    fired = [o for o in outcomes if o.fired]
    held = [o for o in outcomes if not o.fired]

    def rate(xs, pred):
        return sum(map(pred, xs)) / len(xs) if xs else float("nan")

    def win(xs):
        d = [o for o in xs if o.decided]
        return {"games": len(xs), "decided": len(d), "win_rate": rate(d, lambda o: o.winner == LEARNER)}

    hist: Counter = Counter(str(o.fired[0]) if o.fired else "none" for o in outcomes)
    top = max((o.windows for o in outcomes), default=0)
    p_fire_at = {}
    for k in range(1, top + 1):
        reached = [o for o in outcomes if o.windows >= k and (not o.fired or o.fired[0] >= k)]
        if reached:
            p_fire_at[str(k)] = {
                "reached": len(reached),
                "p": rate(reached, lambda o: bool(o.fired) and o.fired[0] == k),
            }
    return {
        "games": n, "decided": len(decided), "reasons": dict(Counter(o.reason for o in outcomes)),
        "win_rate": rate(decided, lambda o: o.winner == LEARNER),
        "score": sum(o.score for o in outcomes) / max(1, n),
        "line_completion": len(completed) / max(1, n),
        "line_status": dict(Counter(o.line.get("status") for o in outcomes)),
        "completion_if_fired": rate(fired, lambda o: o.line.get("completed")),
        "completion_if_held": rate(held, lambda o: o.line.get("completed")),
        "by_line": {"completed": win(completed), "not_completed": win([o for o in outcomes if not o.line.get("completed")])},
        "fire_rate": len(fired) / max(1, n),
        "fire_hist": dict(sorted(hist.items(), key=lambda kv: (kv[0] == "none", int(kv[0]) if kv[0] != "none" else 0))),
        "p_fire_at": p_fire_at,
        "windows_mean": sum(o.windows for o in outcomes) / max(1, n),
        "recoveries": sum(int(o.line.get("recoveries", 0) or 0) for o in outcomes),
    }  # fmt: skip


def paired(a: Sequence[GameOutcome], b: Sequence[GameOutcome]) -> dict:
    """B minus A, game by game (the same games): score, completion and activation."""
    d = np.array([y.score - x.score for x, y in zip(a, b, strict=True)])
    n = len(d)
    se = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    comp = np.array(
        [float(y.line.get("completed")) - float(x.line.get("completed")) for x, y in zip(a, b, strict=True)]
    )
    return {"games": n, "score_diff": float(d.mean()) if n else float("nan"), "score_diff_se": se,
            "b_better": int((d > 0).sum()), "a_better": int((d < 0).sum()),
            "completion_diff": float(comp.mean()) if n else float("nan"),
            "fire_rate_diff": (sum(bool(o.fired) for o in b) - sum(bool(o.fired) for o in a)) / max(1, n)}  # fmt: skip


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpts", nargs="+", help="one checkpoint, or two to compare (B - A) on the same games")
    p.add_argument("--demos", nargs="+", required=True, help="demonstration files (solved plain records: the lines)")
    p.add_argument("--min-hand", type=int, default=0, help="only lines of hands with index >= N (40: held out)")
    p.add_argument("--max-hand", type=int, default=10**9)
    p.add_argument("--limit", type=int, default=None, help="at most N lines (file order)")
    p.add_argument("--games-per-line", type=int, default=2)
    p.add_argument("--opponent", default=None, help="checkpoint playing the line seat after the line (default: CKPT A)")
    p.add_argument("--line-recovery", action="append", default=[], help="T6 recovery lines (fire records); repeatable")
    p.add_argument("--env", default="md-2026-09")
    p.add_argument(
        "--decks", default=None, help="directory of learner .ydk decks (default: the environment's meta decks)"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--greedy", action="store_true", help="argmax for both seats instead of sampling")
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--max-decisions", type=int, default=4000)
    p.add_argument("--json", action="store_true", help="print the report as one JSON object")
    p.add_argument("--out", default=None, help="write the report (and per-game outcomes) as JSON")
    a = p.parse_args(argv)
    if len(a.ckpts) > 2:
        p.error("give one or two checkpoints")

    import torch

    torch.set_num_threads(max(1, min(4, a.threads)))
    from ygorl.cards.ydk import load_ydk
    from ygorl.data import load_environment
    from ygorl.engine.duel import DuelConfig, default_cards
    from ygorl.env.encoded import EncodedVecEnv
    from ygorl.train.checkpoint import load_actor

    environment = load_environment(a.env)
    cards = default_cards()
    pols = [load_actor(c) for c in a.ckpts]
    opp = load_actor(a.opponent) if a.opponent else pols[0]
    for other in (*pols[1:], opp):
        if why := pols[0].signature.mismatches(other.signature):
            raise SystemExit(f"the checkpoints need the same card vocab and event length: {'; '.join(why)}")
    ref = pols[0]
    log = sys.stderr if a.json else sys.stdout
    t0 = time.time()
    lib = load_library(a.demos, ref.vocab, cards, environment, a.line_recovery, min_hand=a.min_hand,
                       max_hand=a.max_hand)  # fmt: skip
    scripts = lib.scripts[: a.limit]
    if not scripts:
        raise SystemExit("no line in range")
    if a.decks:
        decks = [load_ydk(f) for f in sorted(Path(a.decks).glob("*.ydk"))]
    else:
        decks = [m.deck for m in sorted(environment.meta_decks, key=lambda m: m.deck.name)]
    config = DuelConfig.from_environment(environment, max_decisions=a.max_decisions)
    games = build_games(scripts, decks, a.games_per_line, a.seed, config)
    print(f"{len(scripts)} lines ({len(lib.problems)} left out), {len(games)} games per checkpoint, "
          f"library {time.time() - t0:.0f}s", file=log, flush=True)  # fmt: skip
    for problem in lib.problems[:5]:
        print(f"  left out: {problem}", file=log)
    env = EncodedVecEnv(min(a.envs, len(games)), a.threads, cards=cards, vocab=ref.vocab, event_length=ref.event_length,
                        skip_forced=True, selection_history=ref.net_config.selection_history,
                        attack_history=ref.net_config.attack_history)  # fmt: skip
    report: dict = {"lines": len(scripts), "games_per_line": a.games_per_line, "seed": a.seed, "greedy": a.greedy,
                    "opponent": a.opponent or a.ckpts[0], "line_recovery": a.line_recovery, "results": {}}  # fmt: skip
    runs = []
    try:
        for ck, pol in zip(a.ckpts, pols, strict=True):
            t1 = time.time()
            outcomes = play(env, games, pol.net, opp.net, greedy=a.greedy)
            runs.append(outcomes)
            s = summarize(outcomes)
            s["s"] = round(time.time() - t1)
            report["results"][ck] = s
            if not a.json:
                print(f"{ck}: games {s['games']} win {s['win_rate']:.3f} completion {s['line_completion']:.3f} "
                      f"(fired {s['completion_if_fired']:.3f} / held {s['completion_if_held']:.3f}) fire_rate "
                      f"{s['fire_rate']:.3f} windows {s['windows_mean']:.2f} fire_hist {s['fire_hist']} "
                      f"status {s['line_status']} ({s['s']}s)", flush=True)  # fmt: skip
    finally:
        env.close() if hasattr(env, "close") else None
    if len(runs) == 2:
        report["paired"] = pr = paired(*runs)
        if not a.json:
            print(f"paired (B - A): score {pr['score_diff']:+.3f} ± {pr['score_diff_se']:.3f}, B better {pr['b_better']}, "
                  f"A better {pr['a_better']}, completion {pr['completion_diff']:+.3f}, fire rate "
                  f"{pr['fire_rate_diff']:+.3f}")  # fmt: skip
    if a.json:
        print(json.dumps(report))
    if a.out:
        full = {**report, "games": {ck: [o.__dict__ for o in r] for ck, r in zip(a.ckpts, runs, strict=True)}}
        Path(a.out).write_text(json.dumps(full, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
