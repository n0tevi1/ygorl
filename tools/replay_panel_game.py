"""Replay one paired-panel game with a Python tracker and encoded policy host in lockstep.

Preserves slot-keyed policy RNG and skips RNG draws for forced choices, as the batched evaluator does.
Records full action choices and the core's Lua logs, which the batch result does not otherwise preserve.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ygorl import _core
from ygorl.data.environment import load_environment
from ygorl.engine.duel import Duel, DuelConfig, DuelSession
from ygorl.eval.agent_matrix import pairing_slots
from ygorl.eval.batched import _uniform
from ygorl.nets.batch import collate, policy_logits
from ygorl.train.checkpoint import load_actor


def file_hash(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def json_default(value):
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    raise TypeError(f"unsupported JSON type: {type(value).__name__}")


def replay(manifest, candidate, opponent, pairing, game, environment, device, out, script_budget=100000):
    if not 0 <= pairing < len(manifest["pairings"]) or game not in range(4):
        raise ValueError("pairing / game index out of range")
    env = load_environment(environment)
    env.check_stamp(manifest["environment"])
    config = DuelConfig.from_environment(
        env, max_decisions=manifest["config"]["max_decisions"], max_turns=manifest["config"]["max_turns"]
    )
    if asdict(config) != manifest["config"]:
        raise ValueError("unsupported or different duel config")
    decks = [md.deck for md in env.meta_decks]
    slots = pairing_slots(decks, manifest["pairings"], manifest["seed"], config)
    seed, pair_decks, seeds = slots[pairing]
    m, g = divmod(game, 2)
    first = int(m != g)
    a, b = pair_decks[m], pair_decks[1 - m]
    entries = [manifest[k][n] for k, n in (("candidates", candidate), ("opponents", opponent))]
    for entry in entries:
        if file_hash(entry["path"]) != entry["sha256"]:
            raise ValueError(f"checkpoint changed: {entry['path']}")
    policies = [load_actor(entry["path"]) for entry in entries]
    for pol in policies:
        pol.net.to(device).eval()
    if policies[0].signature.mismatches(policies[1].signature):
        raise ValueError("checkpoint signatures differ")
    vocab = policies[0].vocab
    duel = Duel(seed, None, a, b, config=replace(config, shuffle_decks=False), first=first)
    session = DuelSession(duel)
    host = _core.HostDuel(
        duel.cards.to_core(),
        duel.scripts,
        [vocab.password(i) for i in range(vocab.FIRST_INDEX, len(vocab))],
        event_length=policies[0].event_length,
    )
    player = config.player
    options = (player.starting_lp, player.starting_hand, player.draw_per_turn)
    host.start(
        list(duel.core_seed),
        config.rule_flags,
        options,
        options,
        [(list(main), list(extra)) for main, extra in duel.loaded_decks()],
        config.max_turns,
        config.max_decisions,
    )
    steps = [0, 0]
    history = []
    with out.with_suffix(".jsonl").open("w") as log:
        try:
            while not session.done:
                point = session.point
                if host.done() or host.player() != point.player or len(host.actions()) != len(point.actions):
                    raise RuntimeError(f"host desync at {point.index}")
                obs = host.observe()
                legal = np.flatnonzero(obs["action_mask"])
                side = (first + point.player) % 2
                u = None
                if len(legal) == 1:
                    action = int(legal[0])
                else:
                    with torch.no_grad():
                        logits = policy_logits(policies[side].net, collate([obs], device)).float()
                        probs = torch.softmax(logits, -1)[0].cpu().numpy().astype(np.float64)
                    u = _uniform(seeds[m if side == 0 else 1 - m], 0, 0, steps[point.player])
                    cdf = probs.cumsum()
                    action = min(int(np.searchsorted(cdf, u * cdf[-1], side="right")), len(probs) - 1)
                    steps[point.player] += 1

                def describe(a):
                    d = asdict(a)
                    if a.card and a.card.code in duel.cards:
                        d["name"] = duel.cards[a.card.code].name
                    return d

                row = {
                    "index": point.index,
                    "turn": point.turn,
                    "player": point.player,
                    "phase": point.phase,
                    "decision": point.decision.name,
                    "action": action,
                    "chosen": describe(point.actions[action]),
                    "options": [describe(x) for x in point.actions],
                    "uniform": u,
                }
                history.append(action)
                log.write(json.dumps(row) + "\n")
                log.flush()
                session.act(action)
                host.act(action)
            result = asdict(session.result())
            logs = [{"type": kind, "text": text.decode(errors="replace")} for kind, text in session.core.pop_logs()]
            result.update(
                {
                    "lua_logs": logs,
                    "actions": history,
                    "host_result": host.result(),
                    "candidate": candidate,
                    "opponent": opponent,
                    "pairing": pairing,
                    "game": game,
                    "decks": [a.name, b.name],
                    "seed": seed,
                    "environment": env.stamp(),
                    "checkpoint_hashes": [entry["sha256"] for entry in entries],
                    "runtime_core_sha256": file_hash(_core.__file__),
                    "original_core_sha256": manifest["core_sha256"],
                    "tool_sha256": file_hash(__file__),
                    "device": device,
                    "script_budget_thousands": script_budget,
                }
            )
            out.write_text(json.dumps(result, indent=2, default=json_default) + "\n")
            print(json.dumps({k: result[k] for k in ("reason", "error", "turns", "decisions", "lua_logs")}, indent=2))
        finally:
            session.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--opponent", required=True)
    p.add_argument("--pairing", type=int, required=True)
    p.add_argument("--game", type=int, choices=range(4), required=True)
    p.add_argument("--environment", default="environments/md-2026-09")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--script-budget-thousands",
        type=int,
        default=100000,
        help="isolated diagnostic override; does not change other processes or training defaults",
    )
    args = p.parse_args()
    torch.set_num_threads(1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.script_budget_thousands < 1:
        p.error("script budget must be positive")
    old = _core.set_max_script_steps(args.script_budget_thousands)
    try:
        replay(
            json.loads(args.manifest.read_text()),
            args.candidate,
            args.opponent,
            args.pairing,
            args.game,
            args.environment,
            args.device,
            args.out,
            args.script_budget_thousands,
        )
    finally:
        _core.set_max_script_steps(old)


if __name__ == "__main__":
    main()
