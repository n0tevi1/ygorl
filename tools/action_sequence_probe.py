"""Diagnose avoidable initiation errors separately from conditional self-negation rescue.

Frozen actor, controlled native branches, explicit LP overrides; no fitting or
natural-frequency estimate. Reuses the prior registered exception fixture.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from action_filter_exception_probe import ASH, CHANT, LYCORIS, MST, RAIGEKI, decks, event_dict, read, sha
from ygorl import _core
from ygorl.agents.checkpoint import CheckpointAgent
from ygorl.cards.cdb import CardDB
from ygorl.data import load_environment
from ygorl.engine import constants as C
from ygorl.engine.duel import Duel, DuelConfig, DuelSession
from ygorl.nets.batch import to_tensors
from ygorl.train.registration import atomic_json

ROUTES = {
    "own_turn": ["search_pass", "search_ash_clear", "clear_search", "clear_skip"],
    "opponent_turn": ["search_pass", "search_ash", "decline"],
}
HEALTH = ("error", "retries", "unknown_messages", "undecodable_messages", "script_errors")


def capture(p, obs):
    return {
        "decision": p.index,
        "player": p.player,
        "lp": list(p.lp),
        "options": [asdict(a) for a in p.actions],
        "obs": {k: v.tolist() for k, v in obs.items()},
    }


def run_branch(root, source, mode, lp, route):
    env, cards = load_environment("md-2026-09"), CardDB.load()
    a, b = decks(env, cards)
    cfg = DuelConfig.from_environment(env, shuffle_decks=False, max_decisions=250, max_turns=4)
    cfg = replace(cfg, player=replace(cfg.player, starting_lp=lp))
    first = int(mode == "own_turn")
    learner = first
    duel = Duel(2026100780, env, a, b, config=cfg, first=first, validate=True)
    mirror = CheckpointAgent(source, 0)
    mirror.on_duel_start(duel)
    session = DuelSession(duel)
    trace, events, chain, windows = [], [], {}, {}
    summoned = searched = cleared = False
    remaining = []
    critical = {}
    endpoint = None
    try:
        while (p := session.point) is not None:
            assert mirror.host.player() == p.player and len(mirror.host.actions()) == len(p.actions)
            incoming = [event_dict(e) for e in p.events]
            events.extend(incoming)
            for e in incoming:
                if e["type"] == "Chaining":
                    chain = {k: v for k, v in chain.items() if k < e["chain_count"]}
                    chain[e["chain_count"]] = e
                elif e["type"] == "ChainEnd":
                    chain.clear()
            obs = mirror.host.observe()
            legal = np.flatnonzero(obs["action_mask"]).tolist()

            def find(kind, code=None):
                kinds = (kind,) if isinstance(kind, str) else kind
                return next(
                    (
                        i
                        for i in legal
                        if p.actions[i].kind in kinds
                        and (code is None or (p.actions[i].card and p.actions[i].card.code == code))
                    ),
                    None,
                )

            idle = type(p.decision).__name__ == "SelectIdleCmd"
            chant = find(("activate", "chain"), CHANT)
            idx = None
            if "upstream" not in windows and p.player == learner and summoned and p.turn == 2 and chant is not None:
                if mode == "opponent_turn" or (p.phase == C.PHASE_MAIN1 and idle):
                    windows["upstream"] = capture(p, obs)
                    for row in obs["cards"]:
                        if row[0] >= 2 and row[4] == 0:
                            code = mirror.policy.vocab.password(int(row[0]))
                            if code in (ASH, CHANT, RAIGEKI, MST):
                                critical[str(code)] = int(row[1])
                    remaining = {
                        "search_pass": [CHANT] + ([RAIGEKI] if mode == "own_turn" else []),
                        "search_ash_clear": [CHANT, RAIGEKI],
                        "search_ash": [CHANT],
                        "clear_search": [RAIGEKI, CHANT],
                        "clear_skip": [RAIGEKI],
                        "decline": [],
                    }[route].copy()
            active = "upstream" in windows
            top = chain[max(chain)] if chain else None
            ash = find("chain", ASH)
            if active and p.player == learner and ash is not None and top and top["code"] == CHANT:
                windows.setdefault("response", capture(p, obs))
                idx = ash if "ash" in route else find("pass")
                assert idx is not None
            elif active and p.index > windows["upstream"]["decision"] and not remaining and not chain and idle:
                endpoint = {"decision": p.index, "lp": list(p.lp), "done": False}
                break
            elif p.player != learner:
                idx = find("summon", LYCORIS)
                if idx is not None:
                    summoned = True
                else:
                    idx = find("pass")
                    if idx is None:
                        idx = find("end_phase")
            elif active:
                if remaining:
                    # Do not add the next spell to an unresolved prior chain.
                    if not chain and (mode == "opponent_turn" or idle):
                        idx = find(("activate", "chain"), remaining[0])
                        if idx is not None:
                            code = remaining.pop(0)
                            searched |= code == CHANT
                            cleared |= code == RAIGEKI
                else:
                    idx = find("pass")
            elif mode == "opponent_turn" and p.turn == 1:
                idx = find("sset", CHANT)
                if idx is None:
                    idx = find("summon", 69247929)
            if idx is None:
                for kind in ("pass", "end_phase", "main2", "yes"):
                    idx = find(kind)
                    if idx is not None:
                        break
            if idx is None:
                idx = legal[0]
            trace.append(
                {
                    "decision": p.index,
                    "player": p.player,
                    "turn": p.turn,
                    "phase": p.phase,
                    "choice": idx,
                    "action": asdict(p.actions[idx]),
                    "events": incoming,
                }
            )
            session.act(idx)
            mirror.on_decision(p, idx)
        if session.done:
            events.extend(event_dict(e) for e in session.tracker.events)
        result = session.result() if session.done else session.tracker.result
        assert not any(getattr(result, k) for k in HEALTH)
        assert "upstream" in windows
        if endpoint is None:
            assert session.done and result.reason == "win"
            endpoint = {"done": True, "lp": list(session.tracker.lp), "winner_deck": result.winner}
        # Derive resource movements after the common upstream point, including terminal events.
        relevant = []
        for step in trace:
            if step["decision"] > windows["upstream"]["decision"]:
                relevant.extend(step["events"])
        # Endpoint events have not entered trace yet.
        relevant.extend(event_dict(e) for e in (session.tracker.events if session.done else session.point.events))
        locations = {1: 1, 2: 2, 4: 3, 8: 4, 16: 5, 32: 6, 64: 7}
        for e in relevant:
            if e["type"] == "Move" and str(e["code"]) in critical:
                critical[str(e["code"])] = locations.get(e["current"]["location"], 0)
        endpoint["critical_locations"] = critical
        target_lp = windows["upstream"]["lp"][learner]
        assert target_lp == (lp - 200 if mode == "own_turn" else lp)
        expected_damage = 200 if route == "search_pass" else 0
        assert endpoint["lp"][learner] == target_lp - expected_damage
        assert endpoint["done"] == (target_lp == 200 and route == "search_pass")
        if endpoint["done"]:
            assert endpoint["winner_deck"] == 1
        if "ash" in route:
            assert critical[str(ASH)] == 5
        else:
            assert critical[str(ASH)] == 2
        if route in ("clear_skip", "decline"):
            assert critical[str(CHANT)] == (2 if mode == "own_turn" else 4)
        elif not endpoint["done"]:
            assert critical[str(CHANT)] == 5
        if route == "clear_search":
            assert critical[str(MST)] == 2
        record = {
            "mode": mode,
            "starting_lp": lp,
            "route": route,
            "config": asdict(cfg),
            "decks": [asdict(a), asdict(b)],
            "first": first,
            "learner": learner,
            "windows": windows,
            "endpoint": endpoint,
            "searched": searched,
            "cleared": cleared,
            "health": {k: getattr(result, k) for k in HEALTH},
            "events": events,
            "trace": trace,
        }
        atomic_json(root / f"{mode}-{lp}-{route}.json", record)
        return record
    except Exception as exc:
        atomic_json(root / f"{mode}-{lp}-{route}-failure.json", {"error": repr(exc), "trace": trace, "events": events})
        raise
    finally:
        session.close()
        mirror.host = None


@torch.no_grad()
def score(source, window):
    net = CheckpointAgent(source, 0).policy.net.eval()
    obs = to_tensors({k: np.asarray(v)[None] for k, v in window["obs"].items()})
    probs = net(obs).logits.softmax(-1)[0]
    legal = np.flatnonzero(window["obs"]["action_mask"]).tolist()
    return [{"index": i, "probability": float(probs[i]), "action": window["options"][i]} for i in legal]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--native-only", action="store_true")
    a = p.parse_args()
    root = a.output.resolve()
    protocol = read(root / "protocol.json")
    assert not (root / "identity.json").exists(), "preserve existing run"
    source = Path(protocol["source"])
    identity = {
        "driver_sha256": sha(__file__),
        "fixture_sha256": sha(Path(__file__).with_name("action_filter_exception_probe.py")),
        "source_sha256": sha(source),
        "native_sha256": sha(_core.__file__),
        "protocol_sha256": sha(root / "protocol.json"),
        "environment": load_environment("md-2026-09").stamp(),
        "native_only": a.native_only,
    }
    atomic_json(root / "identity.json", identity)
    torch.set_num_threads(1)
    rows = []
    try:
        for mode, lp in protocol["cases"]:
            common = None
            for route in ROUTES[mode]:
                j = run_branch(root, source, mode, lp, route)
                upstream = j["windows"]["upstream"]
                if common is None:
                    common = upstream
                assert upstream == common, "route must branch from exactly the same upstream input"
                row = {
                    "mode": mode,
                    "starting_lp": lp,
                    "route": route,
                    "endpoint": j["endpoint"],
                    "branch_sha256": sha(root / f"{mode}-{lp}-{route}.json"),
                }
                if not a.native_only:
                    row["scores"] = {k: score(source, w) for k, w in j["windows"].items()}
                rows.append(row)
                print(mode, lp, route, j["endpoint"], flush=True)
        atomic_json(
            root / "report.json",
            {
                "identity_sha256": sha(root / "identity.json"),
                "rows": rows,
                "interpretation": protocol["interpretation"],
            },
        )
    except Exception as exc:
        atomic_json(root / "STOP.json", {"error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
