"""Frozen-filter counterexamples: public search-triggered damage at low vs normal LP.

Uses format-legal custom decks and explicit diagnostic starting-LP overrides.
No model training, no population error-rate or natural-game strength estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from ygorl import _core
from ygorl.agents.checkpoint import CheckpointAgent
from ygorl.cards.cdb import CardDB
from ygorl.cards.ydk import Deck
from ygorl.data import load_environment
from ygorl.engine import constants as C
from ygorl.engine.duel import Duel, DuelConfig, DuelSession
from ygorl.nets.action_filter import ActionRiskHead, soft_filter
from ygorl.nets.batch import to_tensors
from ygorl.train.registration import atomic_json

ASH, CHANT, LYCORIS = 14558127, 67115133, 35199656
RAIGEKI, WARWOLF, MST = 12580477, 69247929, 5318639


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def decks(env, cards):
    special = {ASH, CHANT, LYCORIS, RAIGEKI, WARWOLF, MST}
    filler = [
        c.password
        for c in sorted(cards.values(), key=lambda c: c.password)
        if c.password not in special
        and c.alias == 0
        and c.type == (C.TYPE_MONSTER | C.TYPE_NORMAL)
        and c.level <= 4
        and c.password in env.card_pool
        and env.banlist.limit(c.password) > 0
    ][:40]
    assert len(filler) == 40
    # Last five cards are the opening hand. The next draw is ordinary filler.
    a = Deck(tuple([MST] + filler[:34] + [CHANT, ASH, RAIGEKI, WARWOLF, filler[34]]), name="search-side")
    b = Deck(tuple(filler[:35] + [LYCORIS] + filler[35:39]), name="damage-side")
    assert len(a.main) == len(b.main) == 40
    assert not env.validate_deck(a, cards) and not env.validate_deck(b, cards)
    return a, b


def event_dict(e):
    return {"type": type(e).__name__, **asdict(e)}


def branch(root, source, mode, starting_lp, choice):
    env, cards = load_environment("md-2026-09"), CardDB.load()
    a, b = decks(env, cards)
    cfg = DuelConfig.from_environment(env, shuffle_decks=False, max_decisions=250, max_turns=4)
    cfg = replace(cfg, player=replace(cfg.player, starting_lp=starting_lp))
    first = 1 if mode == "own_turn" else 0
    learner = 1 if first else 0
    duel = Duel(2026100780, env, a, b, config=cfg, first=first, validate=True)
    mirror = CheckpointAgent(source, 0)
    mirror.on_duel_start(duel)
    session = DuelSession(duel)
    trace, events, chain = [], [], {}
    lycoris_summoned = False
    target = immediate = None
    target_decision = None
    try:
        while (p := session.point) is not None:
            assert mirror.host.player() == p.player and len(mirror.host.actions()) == len(p.actions)
            incoming = [event_dict(e) for e in p.events]
            events.extend(incoming)
            ended = False
            for e in incoming:
                if e["type"] == "Chaining":
                    chain = {k: v for k, v in chain.items() if k < e["chain_count"]}
                    chain[e["chain_count"]] = e
                elif e["type"] == "ChainEnd":
                    chain.clear()
                    ended = True
            if target is not None and immediate is None and ended:
                immediate = {"done": False, "lp": list(p.lp), "decision": p.index}
                if mode != "own_turn" or starting_lp != 400:
                    break
            obs = mirror.host.observe()
            legal = [i for i in range(min(len(p.actions), len(obs["action_mask"]))) if obs["action_mask"][i]]

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

            ash = find("chain", ASH)
            pas = find("pass")
            top = chain[max(chain)] if chain else None
            idx = None
            if target is None and p.player == learner and ash is not None and top and top["code"] == CHANT:
                assert pas is not None and top["triggering_controller"] == learner
                assert (p.player == p.turn_player) == (mode == "own_turn")
                target_decision = p.index
                target = {
                    "obs": {k: v.tolist() for k, v in obs.items()},
                    "ash": ash,
                    "pass": pas,
                    "lp": list(p.lp),
                    "learner": learner,
                    "own_turn": p.player == p.turn_player,
                    "decision": p.index,
                    "chain": top,
                    "options": [asdict(x) for x in p.actions],
                }
                idx = ash if choice == "ash" else pas
            elif p.player != learner:
                idx = find("summon", LYCORIS)
                if idx is not None:
                    lycoris_summoned = True
                else:
                    idx = find("pass")
                    if idx is None:
                        idx = find("end_phase")
            else:
                if mode == "opponent_turn" and p.turn == 1:
                    idx = find("sset", CHANT)
                    if idx is None:
                        idx = find("summon", WARWOLF)
                elif target is None and lycoris_summoned and p.turn >= 2:
                    if mode == "opponent_turn" or p.phase == C.PHASE_MAIN1:
                        idx = find(("activate", "chain"), CHANT)
                elif target is not None and immediate is not None:
                    for kind, code in [
                        ("activate", RAIGEKI),
                        ("summon", WARWOLF),
                        ("battle_phase", None),
                        ("attack", None),
                    ]:
                        idx = find(kind, code)
                        if idx is not None:
                            break
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
                    "lp": list(p.lp),
                    "choice": idx,
                    "action": asdict(p.actions[idx]),
                    "events": incoming,
                }
            )
            session.act(idx)
            mirror.on_decision(p, idx)
        # Terminal events have no following decision; they must not disappear from branch review.
        events.extend(event_dict(e) for e in session.tracker.events)
        result = session.result() if session.done else session.tracker.result
        assert not any(
            getattr(result, k)
            for k in ("error", "retries", "unknown_messages", "undecodable_messages", "script_errors")
        )
        assert target is not None, "script failed to reach intended Ash window"
        if session.done and immediate is None:
            immediate = {
                "done": True,
                "lp": list(session.tracker.lp),
                "winner_deck": result.winner,
                "reason": result.reason,
                "win_reason": result.win_reason,
            }
        assert immediate is not None
        if session.done:
            assert result.reason == "win"
        record = {
            "mode": mode,
            "starting_lp": starting_lp,
            "choice": choice,
            "config": asdict(cfg),
            "first": first,
            "decks": [asdict(a), asdict(b)],
            "target": target,
            "immediate": immediate,
            "terminal": {
                k: getattr(result, k)
                for k in (
                    "winner",
                    "reason",
                    "win_reason",
                    "lp",
                    "turns",
                    "decisions",
                    "error",
                    "retries",
                    "unknown_messages",
                    "undecodable_messages",
                    "script_errors",
                )
            }
            if session.done
            else None,
            "trace": trace,
            "events": events,
            "target_decision": target_decision,
            "interpretation": "Synthetic controlled state; own-turn low-LP Ash continuation is scripted, not an optimal opponent proof.",
        }
        atomic_json(root / f"{mode}-{starting_lp}-{choice}.json", record)
        return record
    except Exception:
        atomic_json(root / f"{mode}-{starting_lp}-{choice}-failure.json", {"trace": trace, "events": events})
        raise
    finally:
        session.close()
        mirror.host = None


@torch.no_grad()
def score_case(source, candidate, record):
    agent = CheckpointAgent(source, 0)
    net = agent.policy.net.eval()
    t = record["target"]
    obs = to_tensors({k: np.asarray(v)[None] for k, v in t["obs"].items()})
    f = net.features(obs)
    logits = net.logits(f)
    pair = [t["ash"], t["pass"]]
    scores = []
    selection = read(candidate / "selection.json")
    assert selection["threshold"] == 0.5
    for i, selected in enumerate(selection["heads"]):
        path = candidate / f"head-{i}.pt"
        assert sha(path) == selected["sha256"]
        state = torch.load(path, weights_only=True)
        head = ActionRiskHead(state["d_model"])
        head.load_state_dict(state["state"])
        head.eval()
        scores.append(head(f.history, f.actions[:, pair]).sigmoid())
    scores = torch.stack(scores).mean(0)
    risk = torch.zeros_like(logits)
    risk[:, pair] = scores
    output = {
        "original": float(logits.softmax(-1)[0, t["ash"]]),
        "risk_scores_ash_pass": scores[0].tolist(),
        "legal_candidates": int(f.action_mask.sum()),
    }
    for name, indices in [("ash_and_pass", pair), ("ash_only", [t["ash"]])]:
        support = torch.zeros_like(f.action_mask, dtype=torch.bool)
        if f.action_mask.sum() == 2:
            support[:, indices] = True
        filtered = soft_filter(logits, risk, f.action_mask.bool(), support, threshold=0.5)
        output[name] = float(filtered.logits.softmax(-1)[0, t["ash"]])
    assert output["ash_only"] <= output["original"] + 1e-7
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--candidate", required=True, type=Path)
    p.add_argument(
        "--native-only", action="store_true", help="Interface development; no model scoring or formal report"
    )
    a = p.parse_args()
    root, candidate = a.output.resolve(), a.candidate.resolve()
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    protocol = read(root / "protocol.json")
    source = Path(read(candidate / "identity.json")["source"])
    try:
        assert not (root / "identity.json").exists(), "use a new output directory; preserve prior artifacts"
        identity = {
            "driver_sha256": sha(__file__),
            "component_sha256": sha("src/ygorl/nets/action_filter.py"),
            "protocol_sha256": sha(root / "protocol.json"),
            "source_sha256": sha(source),
            "native_sha256": sha(_core.__file__),
            "candidate_selection_sha256": sha(candidate / "selection.json"),
            "environment": load_environment("md-2026-09").stamp(),
            "native_only": a.native_only,
        }
        atomic_json(root / "identity.json", identity)
        results = []
        for mode, lp in protocol["cases"]:
            branches = {kind: branch(root, source, mode, lp, kind) for kind in ("ash", "pass")}
            assert branches["ash"]["target"] == branches["pass"]["target"]
            learner = branches["ash"]["target"]["learner"]
            target_lp = branches["ash"]["target"]["lp"][learner]
            assert target_lp == (lp - 200 if mode == "own_turn" else lp)
            assert branches["ash"]["immediate"]["lp"][learner] == target_lp
            assert branches["pass"]["immediate"]["lp"][learner] == target_lp - 200
            if target_lp == 200:
                assert branches["pass"]["immediate"]["done"] and branches["pass"]["immediate"]["winner_deck"] == 1
                assert not branches["ash"]["immediate"]["done"]
            if mode == "own_turn" and lp == 400:
                assert branches["ash"]["terminal"]["winner"] == 0
            row = {
                "mode": mode,
                "starting_lp": lp,
                "target_lp": target_lp,
                "pass_loses_immediately": branches["pass"]["immediate"]["done"],
                "branch_files": {k: sha(root / f"{mode}-{lp}-{k}.json") for k in branches},
            }
            if not a.native_only:
                row["scores"] = score_case(source, candidate, branches["ash"])
            results.append(row)
            print("CASE", json.dumps(row), flush=True)
        atomic_json(
            root / ("preflight.json" if a.native_only else "report.json"),
            {
                "identity_sha256": sha(root / "identity.json"),
                "cases": results,
                "interpretation": protocol["interpretation"],
            },
        )
    except Exception as exc:
        atomic_json(root / "STOP.json", {"error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
