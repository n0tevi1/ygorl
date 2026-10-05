"""Immutable training checkpoints and a restartable, independently scheduled matrix consumer (#90)."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import replace
from pathlib import Path

from ygorl.solver.resume import output_lock

FORMAT = "ygorl-training-registration-1"


def digest(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def atomic_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


def register_checkpoint(trainer) -> Path:
    """Publish a checkpoint before its record. Never replace a historical update, including an orphan."""
    from ygorl.train.checkpoint import save_checkpoint

    root = trainer.run_dir.resolve() / "registrations"
    root.mkdir(parents=True, exist_ok=True)
    with output_lock(root / "writer"):
        identity = root / "run.json"
        if not identity.exists():
            atomic_json(identity, {"run_id": uuid.uuid4().hex})
        run_id = json.loads(identity.read_text())["run_id"]
        update = trainer.counters["updates"]
        checkpoint = root / f"update_{update:08d}.pt"
        record = checkpoint.with_suffix(".json")
        if checkpoint.exists() or record.exists():
            raise ValueError(f"registration already exists at update {update}; fork into a new run directory")
        save_checkpoint(trainer.state_dict(), checkpoint)
        data = {
            "format": FORMAT,
            "run_id": run_id,
            "update": update,
            "rows": trainer.counters["rows"],
            "training_seconds": trainer.counters["seconds"],
            "wall_seconds": trainer.counters["wall_seconds"],
            "matrix": str(Path(trainer.cfg.register_matrix).resolve()),
            "checkpoint": str(checkpoint),
            "sha256": digest(checkpoint),
            "environment": trainer.environment.stamp() if trainer.environment else None,
        }
        atomic_json(record, data)
        return record


def _protocol(matrix) -> dict:
    d = matrix.to_dict()
    return {
        k: d[k]
        for k in (
            "decks",
            "deck_hashes",
            "pairings",
            "seed",
            "max_turns",
            "max_decisions",
            "alpha",
            "population_size",
            "confidence",
            "environment",
            "batched",
        )
    }


def consume_registrations(runs, matrix_path, decks, *, env=None, workers=1, device=None, curve_path=None) -> dict:
    """Consume the current published snapshot, atomically saving each completed matrix extension.

    Run directories accumulate in the consumer manifest. Repeating a call also repairs a curve whose write
    was interrupted after the matrix save. Only this consumer may write the matrix while it holds its lock.
    """
    from ygorl.agents.registry import agent_factory
    from ygorl.engine.duel import DuelConfig
    from ygorl.eval.agent_matrix import AgentMatrix, extend_agent_matrix

    target = Path(matrix_path).resolve()
    curve_path = Path(curve_path) if curve_path else target.with_suffix(".strength.json")
    manifest = target.with_suffix(target.suffix + ".registrations.json")
    if workers < 1:
        raise ValueError("workers must be positive")
    with output_lock(target):
        matrix = AgentMatrix.load(target, env=env)
        if matrix.total_errors():
            raise ValueError("matrix contains engine errors; investigate before consuming registrations")
        protocol = _protocol(matrix)
        if manifest.exists():
            binding = json.loads(manifest.read_text())
            if binding["protocol"] != protocol:
                raise ValueError("matrix protocol changed since registration consumption began")
        else:
            binding = {
                "protocol": protocol,
                "baselines": {
                    n: {"spec": s, "sha256": h}
                    for n, s, h in zip(matrix.agents, matrix.specs, matrix.fingerprints, strict=True)
                },
                "runs": [],
            }
        for name, baseline in binding["baselines"].items():
            i = matrix.index(name)
            if matrix.specs[i] != baseline["spec"] or matrix.fingerprints[i] != baseline["sha256"]:
                raise ValueError(f"baseline {name} changed")
        binding["runs"] = sorted(set(binding["runs"]) | {str(Path(r).resolve()) for r in runs})
        # Save the baseline set before adding any agent, so a crash cannot relabel a checkpoint as a baseline.
        atomic_json(manifest, binding)
        records = []
        for run in binding["runs"]:
            for p in sorted((Path(run) / "registrations").glob("update_*.json")):
                r = json.loads(p.read_text())
                if r["format"] != FORMAT:
                    raise ValueError(f"{p}: unsupported registration")
                if r["matrix"] == str(target):
                    records.append(r)
        records.sort(key=lambda r: (r["run_id"], r["update"]))
        config = DuelConfig.from_environment(env) if env else DuelConfig()
        config = replace(config, max_turns=matrix.max_turns, max_decisions=matrix.max_decisions)
        names = set()
        for r in records:
            path = Path(r["checkpoint"])
            if digest(path) != r["sha256"]:
                raise ValueError(f"registered checkpoint changed: {path}")
            if r["environment"] != matrix.environment:
                raise ValueError(f"registered checkpoint environment differs: {path}")
            name = f"run-{r['run_id']}-update-{r['update']:08d}"
            if name in names:
                raise ValueError(f"duplicate registration: {name}")
            names.add(name)
            r["agent"] = name
            updated = extend_agent_matrix(
                matrix,
                {name: agent_factory(f"policy:{path}")},
                decks,
                env=env,
                config=config,
                workers=workers,
                device=device,
            )
            if updated.total_errors():
                rejected = target.with_suffix(target.suffix + ".rejected.json")
                atomic_json(rejected, updated.to_dict())
                raise ValueError(f"evaluation has engine errors; retained at {rejected}; matrix not published")
            if updated is not matrix:
                atomic_json(target, updated.to_dict())
                matrix = updated
        ranking = matrix.ranking()
        rows = []
        for r in records:
            i = matrix.index(r["agent"])
            baselines = {}
            for name in binding["baselines"]:
                j = matrix.index(name)
                baselines[name] = {
                    "win_rate": matrix.win_rate[i][j],
                    "games": matrix.games[i][j],
                    "ci_low": matrix.ci_low[i][j],
                    "ci_high": matrix.ci_high[i][j],
                    "errors": matrix.errors[i][j],
                }
            rows.append({**r, "baselines": baselines, "rank": ranking.index(r["agent"]) + 1})
        curve = {
            "format": "ygorl-training-strength-1",
            "matrix": str(target),
            "matrix_sha256": digest(target),
            "agents": len(matrix.agents),
            "ranking": ranking,
            "records": rows,
        }
        curve_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(curve_path, curve)
        return curve
