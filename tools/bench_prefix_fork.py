"""Cost of starting a game at the end of turn 1 and of forking it (docs/spikes/first-seat-interruption-plan-2026-10-24.md).

For ``--lines`` solved turn-1 lines (``plain`` records spread evenly over a demonstration file; deck a moves first and
plays its line, deck b passes), measures per line:

- ``py_replay``: Python host (``DuelSession``): build the duel and replay the line's responses to the first decision
  of turn 2, without and with the snapshot arena (``py_replay_arena``);
- ``py_setup``: the part of that spent building the duel and running to its first decision;
- ``py_snapshot`` / ``py_restore``: ``DuelSession.snapshot()`` / ``restore()`` there (core arena image + deep copy of
  the tracker), and the core image size;
- ``core_snapshot`` / ``core_restore``: the core part alone (``_core.Duel.snapshot()`` / ``restore()``);
- ``cpp_replay``: C++ step loop (``EncodedVecEnv``, one env, one thread): reset from the record's exact start and
  play the line with :class:`ygorl.train.lines.LinePlayer`, the other seat passing, to the first decision of turn 2;
- ``cpp_turn2``: from there, both seats choose uniformly at random until turn 3 starts or the game ends (a stand-in
  for the engine cost of one branch rollout through the opponent's turn 2; no network);
- then the throughput of ``cpp_replay`` with ``--envs`` envs on ``--threads`` threads (prefix starts per second).

Usage: uv run --no-sync python tools/bench_prefix_fork.py DEMOS.jsonl [--lines 40] [--env md-2026-09]
           [--learner-deck environments/md-2026-09/meta/yummy.ydk] [--envs 32] [--threads 16] [--out OUT.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from ygorl import _core
from ygorl.cards.cdb import CardVocab
from ygorl.cards.ydk import load_ydk
from ygorl.data import load_environment
from ygorl.engine.duel import DuelSession, default_cards
from ygorl.env.encoded import EncodedVecEnv
from ygorl.env.encoding import ACTION_KINDS
from ygorl.solver.demo import read_jsonl
from ygorl.train.lines import LinePlayer, line_script

_PASS = ACTION_KINDS.index("pass") + 1
_KIND_COL = 0  # actions[:, 0] is the action kind (1-based ACTION_KINDS index; docs/encoding.md)


def _passive(obs) -> int:
    """The pass row when there is one, else the first choosable row."""
    mask = obs["action_mask"]
    acts = obs["actions"]
    rows = np.flatnonzero(mask)
    for r in rows:
        if int(acts[r][_KIND_COL]) == _PASS:
            return int(r)
    return int(rows[0])


def _turn(obs) -> int:
    return int(obs["globals"][3])


def py_side(demo, env, cards) -> dict:
    t0 = time.perf_counter()
    session = DuelSession(demo.replay(0).duel(env, cards=cards))
    session.point
    out = {"py_setup": time.perf_counter() - t0}
    session.close()
    for arena in (False, True):
        t0 = time.perf_counter()
        session = DuelSession(demo.replay(0).duel(env, cards=cards, snapshots=arena))
        steps = 0
        for idx in demo.lines[0].actions:
            if session.point is None:
                break
            session.act(idx)
            steps += 1
        out["py_replay_arena" if arena else "py_replay"] = time.perf_counter() - t0
        out["steps"] = steps
        out["turn"] = session.tracker.turn
        if arena:
            t0 = time.perf_counter()
            snap = session.snapshot()
            out["py_snapshot"] = time.perf_counter() - t0
            out["snapshot_bytes"] = snap.core.nbytes if hasattr(snap, "core") else session.core.arena_bytes()
            t0 = time.perf_counter()
            session.restore(snap)
            out["py_restore"] = time.perf_counter() - t0
            core = session.core
            t0 = time.perf_counter()
            cs = core.snapshot()
            out["core_snapshot"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            core.restore(cs)
            out["core_restore"] = time.perf_counter() - t0
            out["arena_bytes"] = core.arena_bytes()
        session.close()
    return out


def cpp_side(script, learner, vocab, cards, seed: int) -> dict:
    venv = EncodedVecEnv(1, 1, cards=cards, vocab=vocab)
    player = LinePlayer(script, venv.skip_forced)
    t0 = time.perf_counter()
    venv.reset(0, script.spec(learner, seed, learner_seat=1))
    decisions = 0
    ev = venv.recv(1)[0]
    while ev.result is None and _turn(ev.obs) < 2:
        row = player.act(ev.obs) if ev.player == 0 else None
        venv.step(0, row if row is not None else _passive(ev.obs))
        decisions += 1
        ev = venv.recv(1)[0]
    replay = time.perf_counter() - t0
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    turn2 = 0
    while ev.result is None and _turn(ev.obs) < 3:
        venv.step(0, int(rng.choice(np.flatnonzero(ev.obs["action_mask"]))))
        turn2 += 1
        ev = venv.recv(1)[0]
    return {"cpp_replay": replay, "cpp_decisions": decisions, "line_status": player.status,
            "cpp_turn2": time.perf_counter() - t0, "cpp_turn2_decisions": turn2}  # fmt: skip


def cpp_throughput(scripts, learner, vocab, cards, envs: int, threads: int, games: int) -> dict:
    venv = EncodedVecEnv(envs, threads, cards=cards, vocab=vocab)
    players: dict[int, LinePlayer] = {}
    started = done = decisions = 0

    def start(e: int) -> None:
        nonlocal started
        script = scripts[started % len(scripts)]
        players[e] = LinePlayer(script, venv.skip_forced)
        venv.reset(e, script.spec(learner, 1000 + started, learner_seat=1))
        started += 1

    t0 = time.perf_counter()
    for e in range(envs):
        start(e)
    while done < games:
        for ev in venv.recv(1):
            if ev.result is not None or _turn(ev.obs) >= 2:
                done += 1
                if started < games:
                    start(ev.env_id)
                continue
            row = players[ev.env_id].act(ev.obs) if ev.player == 0 else None
            venv.step(ev.env_id, row if row is not None else _passive(ev.obs))
            decisions += 1
    secs = time.perf_counter() - t0
    # envs whose duel reached turn 2 are left mid-game; the pool is dropped with them
    return {"envs": envs, "threads": threads, "prefixes": done, "seconds": secs, "prefixes_per_s": done / secs,
            "decisions_per_s": decisions / secs}  # fmt: skip


def summary(values: list[float], scale: float = 1e3) -> dict:
    v = sorted(values)
    return {"median": statistics.median(v) * scale, "p90": v[int(0.9 * (len(v) - 1))] * scale, "max": v[-1] * scale}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("demos")
    p.add_argument("--lines", type=int, default=40)
    p.add_argument("--env", default="md-2026-09")
    p.add_argument("--learner-deck", default="environments/md-2026-09/meta/yummy.ydk")
    p.add_argument("--envs", type=int, default=32)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    env = load_environment(a.env)
    cards = default_cards()
    vocab = CardVocab.from_db(cards)
    learner = load_ydk(a.learner_deck)
    demos = [d for d in read_jsonl(a.demos) if d.variant == "plain" and d.status == "solved" and d.lines]
    demos = demos[:: max(1, len(demos) // a.lines)][: a.lines]  # spread over the file (all decks)
    rows, scripts = [], []
    for i, demo in enumerate(demos):
        try:
            script = line_script(demo, 0, vocab, cards=cards, env=env, id=i)
        except ValueError as exc:
            print(f"skip {demo.deck['name']}#{demo.hand_index}: {exc}")
            continue
        scripts.append(script)
        row = {"line": f"{demo.deck['name']}#{demo.hand_index}", **py_side(demo, env, cards)}
        row.update(cpp_side(script, learner, vocab, cards, seed=i))
        rows.append(row)
    keys_ms = ["py_setup", "py_replay", "py_replay_arena", "py_snapshot", "py_restore", "core_snapshot", "core_restore",
               "cpp_replay", "cpp_turn2"]  # fmt: skip
    report = {
        "arena_available": bool(_core.ARENA_AVAILABLE),
        "lines": len(rows),
        "ms": {k: summary([r[k] for r in rows]) for k in keys_ms},
        "steps": summary([r["steps"] for r in rows], 1),
        "cpp_decisions": summary([r["cpp_decisions"] for r in rows], 1),
        "cpp_turn2_decisions": summary([r["cpp_turn2_decisions"] for r in rows], 1),
        "arena_mib": summary([r["arena_bytes"] / 2**20 for r in rows], 1),
        "line_broken": sum(r["line_status"] == "broken" for r in rows) / max(1, len(rows)),
        "throughput": cpp_throughput(scripts, learner, vocab, cards, a.envs, a.threads, 4 * a.envs),
    }
    print(json.dumps(report, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps({"report": report, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
