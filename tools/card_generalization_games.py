"""Frozen full-game coverage diagnostic, with role separation and cold replay checks."""

import argparse
import concurrent.futures as cf
import gzip
import json
import multiprocessing as mp
import subprocess
import threading
import time
import traceback
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch

from behavior_replay import public_board, result_fields, sha
from colab_checkpoint import atomic


def play(root, job):
    from ygorl.agents.registry import make_agent
    from ygorl.cards.ydk import Deck
    from ygorl.data import load_environment
    from ygorl.data.environment import PlayerRules
    from ygorl.engine.duel import Duel, DuelConfig, DuelSession, ScriptedAgent
    from ygorl.engine.replay import Replay
    from ygorl.eval.behavior import summarize_trace

    torch.set_num_threads(1)
    root = Path(root)
    output = root / "games" / str(job["id"])
    output.mkdir(parents=True, exist_ok=False)
    atomic(output / "job.json", job)
    session = None
    agents, trace = [], []
    try:
        cfg = DuelConfig(**{**job["config"], "player": PlayerRules(**job["config"]["player"])})
        env = load_environment(job["environment"])
        args = (job["seed"], env, Deck(**job["deck_a"]), Deck(**job["deck_b"]))
        duel = Duel(*args, config=cfg, first=job["first"], record_steps=True)
        agents = [make_agent(s, seed) for s, seed in zip(job["agents"], job["agent_seeds"], strict=True)]
        session = DuelSession(duel)
        for a in agents:
            if hasattr(a, "on_duel_start"):
                a.on_duel_start(duel)
        per_turn, actions = Counter(), Counter()
        max_cancels = 0
        while (p := session.point) is not None:
            for a in agents:
                if hasattr(a, "observe"):
                    a.observe(p, session.core)
            actor = agents[duel.deck_of(p.player)]
            choice = actor.act(p)
            mask = [True] * len(p.actions)
            if hasattr(actor, "host"):
                native = actor.host.observe()["action_mask"]
                mask = [bool(native[i]) if i < len(native) else False for i in range(len(p.actions))]
            assert 0 <= choice < len(mask) and mask[choice], "unsupported action"
            if actor is agents[0]:
                per_turn[p.turn] += 1
                actions[p.actions[choice].kind] += 1
                max_cancels = max(max_cancels, session.tracker._selection_cancels)
            trace.append(
                dict(
                    index=p.index,
                    turn=p.turn,
                    phase=p.phase,
                    player=p.player,
                    turn_player=p.turn_player,
                    lp=p.lp,
                    choice=choice,
                    chosen=asdict(p.actions[choice]),
                    options=[asdict(a) for a in p.actions],
                    probs=getattr(actor, "last_probs", None),
                    mask=mask,
                    undo=p.undo,
                    decision_type=type(p.decision).__name__,
                    board=public_board(session.core),
                    events=[{"type": type(e).__name__, **asdict(e)} for e in p.events],
                )
            )
            session.act(choice)
            for a in agents:
                if hasattr(a, "on_decision"):
                    a.on_decision(p, choice)
        result = session.result()
        trace.append(
            dict(
                index=result.decisions,
                chosen=None,
                events=[{"type": type(e).__name__, **asdict(e)} for e in session.tracker.events],
            )
        )
        Replay.from_duel(duel, result).save(output / "replay.json.gz")
        record = dict(
            id=job["id"],
            candidate=job["candidate"],
            opponent=job["opponent"],
            target=job["target"],
            exposure=job["exposure"],
            role=job["role"],
            pair=job["pair"],
            draw=job["draw"],
            anchor=job["anchor"],
            first=job["first"],
            result=result_fields(result),
            strict_win=result.reason == "win" and result.winner == 0,
            limit=result.reason in ("turn_limit", "decision_limit"),
            max_selection_cancels=max_cancels,
            max_candidate_decisions_in_turn=max(per_turn.values(), default=0),
            candidate_actions=dict(actions),
            replay_sha256=sha(output / "replay.json.gz"),
        )
        atomic(output / "result.json", record)
        assert not any(
            record["result"][k]
            for k in ["error", "retries", "script_errors", "unknown_messages", "undecodable_messages"]
        ), record
        assert result.reason in ("win", "turn_limit", "decision_limit"), record
        session.close()
        session = None
        script = ScriptedAgent([r["choice"] for r in trace if r["chosen"] is not None])
        cold = Duel(*args, config=cfg, first=job["first"], record_steps=True).run(script, script)
        record["cold_replay_equal"] = (
            result_fields(cold) == record["result"]
            and cold.actions == result.actions
            and cold.responses == result.responses
            and cold.steps == result.steps
        )
        assert record["cold_replay_equal"], "cold replay mismatch"
        # Greedy has no probability distribution to report; preserve its raw trace instead.
        if job["candidate"] != "greedy-control":
            atomic(output / "behavior.json", summarize_trace(trace, job["first"]))
        atomic(output / "result.json", record)
        return record
    except BaseException as exc:
        atomic(output / "failure.json", dict(reason=repr(exc), traceback=traceback.format_exc()))
        raise
    finally:
        with gzip.open(output / "trace.jsonl.gz", "wt") as f:
            for row in trace:
                f.write(json.dumps(row) + "\n")
        if session is not None:
            session.close()
        for a in agents:
            if hasattr(a, "host"):
                a.host = None


def verify(root):
    from ygorl import _core, paths
    from ygorl.solver.resume import tree_digest

    repo = Path(__file__).resolve().parents[1]
    identity = json.loads((root / "identity.json").read_text())
    assert not (root / "STOP.json").exists()
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() == identity["commit"]
    for p, digest in identity["files"].items():
        assert sha(Path(p)) == digest, p
    assert tree_digest(repo / "src/ygorl", "*.py") == identity["python"]
    assert sha(_core.__file__) == identity["native"]
    assert sha(paths.cards_cdb()) == identity["cards"]
    assert tree_digest(paths.card_scripts(), "*.lua") == identity["scripts"]
    assert str(torch.__version__) == identity["torch"]


def run(root, worker=play):
    root = root.resolve()
    state = {"stage": "starting", "completed": 0}
    done = threading.Event()

    def heartbeat():
        while not done.is_set():
            atomic(root / "pipeline-status.json", {**state, "updated_unix": time.time()})
            done.wait(15)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        verify(root)
        jobs = json.loads((root / "panel.json").read_text())
        reg = json.loads((root / "registration.json").read_text())
        assert len(jobs) == reg["games"] and len({j["id"] for j in jobs}) == len(jobs)
        deadline = time.time() + reg["max_hours"] * 3600
        rows = []
        canary = set(reg["canary_ids"])
        phases = [
            ([j for j in jobs if j["id"] in canary], "health-canary"),
            ([j for j in jobs if j["id"] not in canary], "evaluation"),
        ]
        with cf.ProcessPoolExecutor(reg["workers"], mp_context=mp.get_context("spawn")) as pool:
            for phase, label in phases:
                state["stage"] = label
                it = iter(phase)
                pending = set()
                for _ in range(reg["workers"]):
                    if (j := next(it, None)) is not None:
                        pending.add(pool.submit(worker, root, j))
                while pending:
                    assert time.time() < deadline, "wall budget expired"
                    assert not (root / "STOP.json").exists(), "external STOP"
                    finished, pending = cf.wait(pending, timeout=10, return_when=cf.FIRST_COMPLETED)
                    for f in finished:
                        rows.append(f.result())
                        if (j := next(it, None)) is not None:
                            pending.add(pool.submit(worker, root, j))
                    state.update(completed=len(rows), total=len(jobs))
                    atomic(root / "progress.json", {**state, "updated_unix": time.time()})
        verify(root)
        atomic(
            root / "report.json",
            dict(
                study_sha256=sha(root / "identity.json"),
                healthy=True,
                records=sorted(rows, key=lambda r: r["id"]),
                scope=reg["scope"],
            ),
        )
        state["stage"] = "complete"
    except BaseException as exc:
        atomic(
            root / ("pipeline-failure.json" if (root / "STOP.json").exists() else "STOP.json"),
            dict(reason=repr(exc), traceback=traceback.format_exc()),
        )
        state["stage"] = "stopped"
        raise
    finally:
        done.set()
        thread.join()
        atomic(root / "pipeline-status.json", {**state, "updated_unix": time.time()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    from colab_checkpoint import run_lock

    with run_lock(args.root):
        run(args.root)
