"""Real GPU restore checks plus historical, explicitly non-study statistical fixtures."""

import copy
import gc
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from analyze import statistics
from audit import audit_statistics
from run import (
    BC,
    DRIVERS,
    HISTORY,
    INITIAL,
    OLD,
    PREVIOUS,
    ROOT,
    inputs,
    play,
    sha,
    source_record,
    verify_restored,
    Watch,
    assert_equal,
)
from ygorl.agents.registry import agent_factory
from ygorl.cards.ydk import Deck
from ygorl.engine.duel import DuelConfig, PlayerRules
from ygorl.eval.arena import GameSpec
from ygorl.train.checkpoint import load_actor, load_checkpoint
from ygorl.train.registration import atomic_json
from ygorl.train.trainer import Trainer


def check_statistics():
    old = np.load(PREVIOUS / "paired-scores.npz")
    scores = {"initial": np.concatenate([old["initial"], old["initial"][:1]])}
    for sid in range(3):
        for arm in ("cold", "warm"):
            for u, previous in ((128, 32), (256, 128), (512, 128)):
                values = old[f"seed-{sid}-{arm}-u{previous}"]
                scores[f"seed-{sid}-{arm}-u{u}"] = np.concatenate([values, values[:1]])
    nodes, growth, proceed = statistics(scores)
    result = {
        "bootstrap_seed": 2026100805,
        "contrasts": {str(k): v for k, v in nodes.items()},
        "growth": growth,
        "criterion_to_test_longer_training": proceed,
    }
    audit_statistics(result, scores)
    tampered = copy.deepcopy(result)
    tampered["growth"]["warm_512_minus_128"]["mean"] += 0.01
    try:
        audit_statistics(tampered, scores)
    except AssertionError:
        pass
    else:
        raise AssertionError("independent auditor accepted altered growth")
    # A flat historical-opponent curve must block budget growth despite primary gains.
    flat = {k: v.copy() for k, v in scores.items()}
    for sid in range(3):
        flat[f"seed-{sid}-warm-u512"][3] = flat[f"seed-{sid}-warm-u128"][3]
    flat_nodes, flat_growth, flat_proceed = statistics(flat)
    assert not flat_proceed and flat_growth["warm_historical_512_minus_128"]["mean"] == 0
    assert flat_nodes == nodes, "secondary opponent contaminated primary endpoint"
    return {
        "historical_schema_fixture_only": True,
        "no_new_training_or_evaluation_results": True,
        "independent_bootstrap_verified": True,
        "tampering_rejected": True,
        "secondary_gate_checked": True,
    }


def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert not (ROOT / "preflight.json").exists()
    checks = []
    with tempfile.TemporaryDirectory(prefix="ygorl-continuation-preflight-", dir="/tmp") as tmp:
        for sid in range(3):
            for arm in ("cold", "warm"):
                source = source_record(sid, arm)
                state = load_checkpoint(source["checkpoint"])
                trainer = Trainer.resume(source["checkpoint"], Path(tmp) / f"{sid}-{arm}", log=None)
                restored = verify_restored(trainer, state)
                assert all(p.requires_grad for p in trainer.model.parameters())
                checks.append({"seed": sid, "arm": arm, "source": source, "restored": restored})
                del trainer, state
                gc.collect()
                torch.cuda.empty_cache()
    # Old, already-opened deal; no new formal panel outcome is inspected before registration.
    row = json.loads((ROOT.parent / "ppo-decision-limits-2026-10-05/identity.json").read_text())["cases"][0]
    cfg = DuelConfig(**{**row["config"], "player": PlayerRules(**row["config"]["player"])})

    def deck(d):
        return Deck(name=d["name"], main=tuple(d["main"]), extra=tuple(d["extra"]), side=tuple(d["side"]))

    env, _ = inputs()
    assert_equal(load_actor(INITIAL).net.state_dict(), load_actor(BC).net.state_dict(), "initial_actor_matches_bc")
    try:
        Watch(agent_factory(f"policy:{BC}")(0))
    except AssertionError:
        pass
    else:
        raise AssertionError("unsupported BC candidate adapter accepted")
    games = []
    for i, opponent in enumerate(("greedy", f"policy:{OLD}", f"policy:{INITIAL}", f"policy:{HISTORY}")):
        spec = GameSpec(
            pair=0,
            first=row["result"]["first"],
            seed=row["result"]["seed"],
            agent_seeds=tuple(row["agent_seeds"]),
            deck_a=deck(row["deck_a"]),
            deck_b=deck(row["deck_b"]),
            agent_a=agent_factory(f"policy:{HISTORY}"),
            agent_b=agent_factory(opponent),
            env=env,
            config=cfg,
        )
        result = play((i, spec, str(ROOT)))
        assert result["result"]["reason"] == "win" and "failure_trace" not in result
        games.append(result)
    failed_path = ROOT.parent / "terminal-critic-continuation-2026-10-08/evaluation/initial/greedy.jsonl"
    failed = json.loads(failed_path.read_text().splitlines()[1])
    assert "has no attribute 'host'" in failed["result"]["error"]
    failed_cfg = DuelConfig(**{**failed["config"], "player": PlayerRules(**failed["config"]["player"])})
    spec = GameSpec(
        pair=failed["result"]["pair"],
        first=failed["result"]["first"],
        seed=failed["result"]["seed"],
        agent_seeds=tuple(failed["agent_seeds"]),
        deck_a=deck(failed["deck_a"]),
        deck_b=deck(failed["deck_b"]),
        agent_a=agent_factory(f"policy:{INITIAL}"),
        agent_b=agent_factory("greedy"),
        env=env,
        config=failed_cfg,
    )
    replay = play((100, spec, str(ROOT)))
    assert replay["result"]["reason"] == "win" and "failure_trace" not in replay
    assert replay["candidate_counts"]["material_cancel_available"] > 0
    games.append(replay)

    from retention import preflight as retention_preflight

    report = {
        "checks": checks,
        "training_rows_generated": 0,
        "historical_nonstudy_games": games,
        "statistics": check_statistics(),
        "analysis_and_auditor_fixture_passed": True,
        "retention": retention_preflight(),
        "gpu": torch.cuda.get_device_name(0),
        "torch": str(torch.__version__),
        "hip": torch.version.hip,
        "driver_sha256": {name: sha(ROOT / name) for name in DRIVERS},
    }
    atomic_json(ROOT / "preflight.json", report)
    print(
        "PREFLIGHT PASSED: six exact GPU restores, four historical games plus failed initial-adapter replay, independent statistics and retention",
        flush=True,
    )


if __name__ == "__main__":
    main()
