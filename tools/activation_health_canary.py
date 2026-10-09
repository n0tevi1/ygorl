"""Bounded recovery validation; never reuse the stopped formal study's identity."""

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch

from critic_continuation.run import assert_equal, unhealthy
from ygorl.solver.resume import implementation_identity
from ygorl.train.checkpoint import load_checkpoint
from ygorl.train.registration import atomic_json
from ygorl.train.trainer import TrainConfig, Trainer


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    identity = json.loads((root / "identity.json").read_text())
    identity_sha = sha(root / "identity.json")
    source = identity["checkpoint"]
    assert sha(source) == identity["checkpoint_sha256"]
    assert sha(identity["stopped_study_stop"]) == identity["stopped_study_stop_sha256"]
    assert implementation_identity(Path(identity["solver"]), Path(__file__)) == identity["implementation"]
    assert sha(Path(__file__).parent / "critic_continuation/run.py") == identity["helper_sha256"]
    assert not (root / "pipeline-status.json").exists(), "retain partial runs; create a new recovery identity"
    for deck, expected in identity["deck_sha256"].items():
        assert sha(deck) == expected
    source_state = load_checkpoint(source)
    start = source_state["counters"]
    atomic_json(root / "pipeline-status.json", {"stage": "starting", "updated_unix": time.time()})

    class CheckedTrainer(Trainer):
        def _collect(self):
            rollout = super()._collect()
            bad = [game for game in rollout.games if unhealthy(game)]
            if bad:
                self._log_errors(bad)
                self._log_games(bad)
                self.save("health-stop.pt")
                raise RuntimeError("unhealthy rollout rejected before optimizer")
            return rollout

        def step(self):
            metrics = super().step()
            assert all(not isinstance(v, float) or math.isfinite(v) for v in metrics.values()), "nonfinite metrics"
            delta = {k: self.counters[k] - start[k] for k in ("updates", "rows", "games", "errors", "truncated")}
            self.save("latest.pt")
            atomic_json(
                root / "pipeline-status.json", {"stage": "running", "updated_unix": time.time(), "delta": delta}
            )
            assert not delta["errors"]
            assert delta["games"] < 100 or delta["truncated"] / delta["games"] <= 0.01, "truncation rate exceeds 1%"
            return metrics

    trainer = None
    try:
        trainer = CheckedTrainer.resume(source, root / "run", log=lambda msg: print(msg, flush=True))
        actual = trainer.state_dict()
        for key in (
            "net_config",
            "vocab",
            "environment",
            "learner",
            "pool",
            "schedule",
            "counters",
            "best",
            "league",
            "rng",
            "evolved",
        ):
            assert_equal(actual[key], source_state[key], key)
        assert trainer.cfg.to_dict() == TrainConfig.from_dict(source_state["config"]).to_dict()
        assert not trainer.cfg.bf16 and trainer.cfg.learner_precision in ("inherit", "fp32")
        atomic_json(
            root / "start-audit.json",
            {
                "source_counters": start,
                "restored_optimizer_pool_schedule_rng": True,
                "active_duels_restored": False,
                "device": torch.cuda.get_device_name(0),
            },
        )
        trainer.train(max_updates=identity["additional_updates"])
        delta = {k: trainer.counters[k] - start[k] for k in ("updates", "rows", "games", "errors", "truncated")}
        assert delta["updates"] == identity["additional_updates"] and not delta["errors"]
        checkpoint = trainer.save("accepted.pt")
        atomic_json(
            root / "report.json",
            {
                "study_sha256": identity_sha,
                "healthy": True,
                "delta": delta,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha(checkpoint),
            },
        )
        atomic_json(root / "pipeline-status.json", {"stage": "complete", "updated_unix": time.time(), "delta": delta})
    except BaseException as exc:
        if trainer is not None:
            trainer.save("health-stop.pt")
        atomic_json(root / "STOP.json", {"reason": repr(exc), "time": time.time()})
        atomic_json(
            root / "pipeline-status.json", {"stage": "stopped", "updated_unix": time.time(), "error": repr(exc)}
        )
        raise
    finally:
        # Rollout workers can be stuck on an exception; all diagnostic writes above
        # finish before the process terminates instead of blocking on native destructors.
        if (root / "STOP.json").exists():
            os._exit(1)


if __name__ == "__main__":
    main()
