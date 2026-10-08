"""Read-only reconstruction of frozen evaluation games, with complete final events."""

import argparse
import gzip
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from ygorl.agents.registry import agent_factory
from ygorl.cards.ydk import Deck
from ygorl.data import load_environment
from ygorl.data.environment import PlayerRules
from ygorl.engine import constants as C
from ygorl.engine.duel import Duel, DuelConfig, DuelSession, ScriptedAgent
from ygorl.engine.query import CARD_QUERY_FLAGS, parse_query_location
from ygorl.engine.replay import Replay
from ygorl.eval.behavior import summarize_trace
from ygorl.train.registration import atomic_json


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def public_board(core):
    result = []
    for player in range(2):
        cards = parse_query_location(core.query_location(CARD_QUERY_FLAGS, player, C.LOCATION_MZONE))
        result.append(
            [None if c is None else c if c["position"] & C.POS_FACEUP else {"position": c["position"]} for c in cards]
        )
    return result


def result_fields(result):
    fields = {
        k: getattr(result, k)
        for k in (
            "winner",
            "reason",
            "turns",
            "decisions",
            "win_reason",
            "lp",
            "retries",
            "unknown_messages",
            "undecodable_messages",
            "error",
        )
    }
    fields["script_errors"] = len(result.script_errors)
    return json.loads(json.dumps(fields))


def replay_game(job):
    root, cell, raw, selection = job
    torch.set_num_threads(1)
    out = Path(root) / "replays" / cell / str(raw["game_id"])
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "source.json", {"row": raw, "selection": selection})
    try:
        env = load_environment("md-2026-09")
        cfg = dict(raw["config"])
        cfg["player"] = PlayerRules(**cfg["player"])
        config = DuelConfig(**cfg)
        args = (raw["result"]["seed"], env, Deck(**raw["deck_a"]), Deck(**raw["deck_b"]))
        duel = Duel(*args, config=config, first=raw["result"]["first"], record_steps=True)
        agents = [
            agent_factory(raw[k])(seed) for k, seed in zip(("agent_a", "agent_b"), raw["agent_seeds"], strict=True)
        ]
        assert hasattr(agents[0], "host")
        session = DuelSession(duel)
        trace = []
        try:
            for a in agents:
                if hasattr(a, "on_duel_start"):
                    a.on_duel_start(duel)
            while (p := session.point) is not None:
                for a in agents:
                    if hasattr(a, "observe"):
                        a.observe(p, session.core)
                actor = agents[duel.deck_of(p.player)]
                idx = actor.act(p)
                probs = getattr(actor, "last_probs", None)
                mask = [True] * len(p.actions)
                if actor is agents[0]:
                    m = actor.host.observe()["action_mask"]
                    mask = [bool(m[i]) if i < len(m) else False for i in range(len(p.actions))]
                trace.append(
                    {
                        "index": p.index,
                        "turn": p.turn,
                        "phase": p.phase,
                        "player": p.player,
                        "turn_player": p.turn_player,
                        "lp": p.lp,
                        "choice": idx,
                        "chosen": asdict(p.actions[idx]),
                        "options": [asdict(a) for a in p.actions],
                        "probs": probs,
                        "mask": mask,
                        "undo": p.undo,
                        "decision_type": type(p.decision).__name__,
                        "board": public_board(session.core),
                        "events": [{"type": type(e).__name__, **asdict(e)} for e in p.events],
                    }
                )
                session.act(idx, probs)
                for a in agents:
                    if hasattr(a, "on_decision"):
                        a.on_decision(p, idx)
            result = session.result()
            trace.append(
                {
                    "index": result.decisions,
                    "chosen": None,
                    "events": [{"type": type(e).__name__, **asdict(e)} for e in session.tracker.events],
                }
            )
        finally:
            session.close()
        with gzip.open(out / "trace.jsonl.gz", "wt") as f:
            for row in trace:
                f.write(json.dumps(row) + "\n")
        Replay.from_duel(duel, result).save(out / "replay.json.gz")
        actual = result_fields(result)
        atomic_json(out / "result.json", actual)
        expected = {k: raw["result"][k] for k in actual}
        assert actual == expected, {"expected": expected, "actual": actual}
        cold = Duel(*args, config=config, first=raw["result"]["first"], record_steps=True)
        script = ScriptedAgent([r["choice"] for r in trace if r["chosen"] is not None])
        checked = cold.run(script, script)
        assert (
            result_fields(checked) == actual
            and checked.actions == result.actions
            and checked.responses == result.responses
        )
        report = summarize_trace(trace, raw["result"]["first"])
        report.update(
            cell=cell,
            game_id=raw["game_id"],
            selection=selection,
            result=actual,
            original_result_equal=True,
            cold_replay_equal=True,
            files={
                name: sha(out / name) for name in ("source.json", "trace.jsonl.gz", "replay.json.gz", "result.json")
            },
        )
        atomic_json(out / "report.json", report)
        return report
    except BaseException as e:
        atomic_json(out / "STOP.json", {"error": repr(e)})
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--cell", type=Path, required=True)
    p.add_argument("--game", type=int, required=True)
    a = p.parse_args()
    rows = [json.loads(s) for s in a.cell.read_text().splitlines()]
    row = next(r for r in rows if r["game_id"] == a.game)
    print(json.dumps(replay_game((a.root, a.cell.parent.name + "-" + a.cell.stem, row, "selected-case")), indent=2))
