"""Replay all Runick-facing cases and branch one selected long game's early action."""

import argparse
import gzip
import json
import traceback
from dataclasses import asdict
from pathlib import Path

import torch

from behavior_replay import public_board, result_fields, sha
from card_generalization_games import run
from colab_checkpoint import atomic, run_lock


def public_spells(core):
    from ygorl.engine import constants as C
    from ygorl.engine.query import CARD_QUERY_FLAGS, parse_query_location

    return [
        [
            c
            for c in parse_query_location(core.query_location(CARD_QUERY_FLAGS, player, C.LOCATION_SZONE))
            if c is not None and c["position"] & C.POS_FACEUP
        ]
        for player in range(2)
    ]


def play(root, task):
    from combat_probe import masked_action
    from ygorl.agents.greedy import GreedyAgent
    from ygorl.agents.registry import make_agent
    from ygorl.cards.ydk import Deck
    from ygorl.data import load_environment
    from ygorl.data.environment import PlayerRules
    from ygorl.engine.duel import Duel, DuelConfig, DuelSession, ScriptedAgent
    from ygorl.engine.replay import Replay

    torch.set_num_threads(1)
    root = Path(root)
    directory = root / "games" / str(task["id"])
    directory.mkdir(parents=True, exist_ok=False)
    atomic(directory / "task.json", task)
    source = Path(task["source"])
    raw = json.loads((source / "job.json").read_text())
    expected = json.loads((source / "result.json").read_text())["result"]
    trace = [json.loads(line) for line in gzip.open(source / "trace.jsonl.gz", "rt")]
    cfg = DuelConfig(**{**raw["config"], "player": PlayerRules(**raw["config"]["player"])})
    args = (raw["seed"], load_environment(raw["environment"]), Deck(**raw["deck_a"]), Deck(**raw["deck_b"]))
    duel = Duel(*args, config=cfg, first=raw["first"], record_steps=True)
    session = DuelSession(duel)
    agents, choices, windows, overrides = [], [], [], []
    root_probability = None
    try:
        if task["kind"] == "branch":
            agents = [make_agent(s, seed) for s, seed in zip(raw["agents"], raw["agent_seeds"], strict=True)]
            for a in agents:
                if hasattr(a, "on_duel_start"):
                    a.on_duel_start(duel)
            teacher = GreedyAgent(202610104001 + task["trial"], cards=duel.cards)
        while (p := session.point) is not None:
            if task["kind"] == "snapshot":
                row = trace[p.index]
                assert asdict(p.actions[row["choice"]]) == row["chosen"]
                assert p.turn == row["turn"] and p.player == row["player"] and list(p.lp) == row["lp"]
                choice = row["choice"]
                if p.player == raw["first"] and row["decision_type"] in ("SelectBattleCmd", "SelectIdleCmd"):
                    legal = [a for a, ok in zip(row["options"], row["mask"], strict=True) if ok]
                    windows.append(
                        dict(
                            index=p.index,
                            turn=p.turn,
                            decision_type=row["decision_type"],
                            lp=p.lp,
                            chosen=row["chosen"],
                            legal=legal,
                            board=public_board(session.core),
                            spells=public_spells(session.core),
                        )
                    )
            else:
                for a in agents:
                    if hasattr(a, "observe"):
                        a.observe(p, session.core)
                actor = agents[duel.deck_of(p.player)]
                choice = actor.act(p)
                if p.index <= task["decision"]:
                    row = trace[p.index]
                    assert choice == row["choice"] and asdict(p.actions[choice]) == row["chosen"], "prefix mismatch"
                if p.index == task["decision"]:
                    assert p.player == raw["first"]
                    assert public_board(session.core) == row["board"] and list(p.lp) == row["lp"]
                    assert actor.host.observe()["action_mask"][task["action"]]
                    root_probability = actor.last_probs[task["action"]]
                    choice = task["action"]
                    # Consume the original root draw in every branch. Trial zero
                    # preserves both original streams; other trials share suffix RNG.
                    if task["trial"]:
                        for seat, a in enumerate(agents):
                            seed = 202610104100 + 1009 * task["trial"] + 1000003 * seat
                            if hasattr(a, "generator"):
                                a.generator.manual_seed(seed)
                            if hasattr(a, "rng"):
                                a.rng.seed(seed)
                elif p.index > task["decision"] and actor is agents[0] and task["suffix"] == "greedy":
                    policy = choice
                    choice = masked_action(teacher, p, actor.host.observe()["action_mask"])
                    if choice != policy:
                        overrides.append(
                            dict(
                                index=p.index,
                                turn=p.turn,
                                policy=asdict(p.actions[policy]),
                                teacher=asdict(p.actions[choice]),
                            )
                        )
            choices.append(choice)
            session.act(choice)
            for a in agents:
                if hasattr(a, "on_decision"):
                    a.on_decision(p, choice)
        result = session.result()
        Replay.from_duel(duel, result).save(directory / "replay.json.gz")
        fields = result_fields(result)
        output = dict(
            id=task["id"],
            kind=task["kind"],
            source=task["source"],
            result=fields,
            strict_win=result.reason == "win" and result.winner == 0,
            windows=windows,
            root_probability=root_probability,
            overrides=overrides,
            replay_sha256=sha(directory / "replay.json.gz"),
        )
        atomic(directory / "result.json", output)
        assert not any(
            fields[k] for k in ["error", "retries", "script_errors", "unknown_messages", "undecodable_messages"]
        )
        assert result.reason in ("win", "turn_limit", "decision_limit")
        if task["kind"] == "snapshot" or (
            task["trial"] == 0 and task["suffix"] == "policy" and task["action"] == trace[task["decision"]]["choice"]
        ):
            assert fields == expected and choices == [r["choice"] for r in trace if r["chosen"] is not None]
            output["original_equal"] = True
        session.close()
        session = None
        script = ScriptedAgent(choices)
        cold = Duel(*args, config=cfg, first=raw["first"], record_steps=True).run(script, script)
        output["cold_replay_equal"] = (
            result_fields(cold) == fields
            and cold.actions == choices
            and cold.responses == result.responses
            and cold.steps == result.steps
        )
        assert output["cold_replay_equal"]
        atomic(directory / "result.json", output)
        return output
    except BaseException as exc:
        atomic(directory / "failure.json", dict(reason=repr(exc), traceback=traceback.format_exc()))
        raise
    finally:
        if session is not None:
            session.close()
        for a in agents:
            if hasattr(a, "host"):
                a.host = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    with run_lock(args.root):
        run(args.root, worker=play)
