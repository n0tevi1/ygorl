"""Co-evolution of decks, policy and edit-value model (#151, design 05 §5.3「与 RL 共同进化」, docs/tuning.md
「共同进化循环」).

:class:`CoEvolution` drives cycles of three stages over one state directory::

    fit 0 (the value model on the paired data so far, for the starting policy)
    cycle c = 1, 2, ...:
      evolve c   one evolution round on the small deck pool (the current version of each deck), with the current
                 policy and value model; accepted children go into the deck-pool manifest and replace their parent
      train c    a training segment: the policy continues to update ``start + c × updates``, with the manifest as its
                 evolved deck pool and its games logged (``--deck-pool``, ``--log-games``)
      fit c      the value model refitted on every paired evaluation (the rounds' lineage included) and the game
                 logs, for the new policy (older labels down-weighted, ``ygorl.build.edit_labels.Clock``)

The stages are a :class:`Stages` object (``tools/coevolve.py`` runs the real ones as subprocesses; tests use fakes).
**Resuming**: ``coevo.json`` records each finished stage (written atomically after the stage); a rerun skips them
and reruns an unfinished stage, which must itself resume: the evolution round resumes from its game logs (and a
finished round is read, not rerun), the training segment from the run's ``latest.pt`` to the same target update,
the fit is deterministic. The settings of the first run are kept (a rerun's are ignored, with a note).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

FORMAT = "ygorl-coevolution"


@dataclass
class CoevoConfig:
    cycles: int = 3
    updates: int = 50  # training updates per cycle
    start_update: int = 0  # the starting checkpoint's update
    decks: dict[str, str] = field(default_factory=dict)  # deck slot -> starting .ydk
    initial_fit: bool = True  # fit a value model before the first round


@dataclass
class EvolveResult:
    round: int
    games: int
    accepted: list[dict]  # {"slot", "parent", "id", "file", "diff"}: accepted children and the parent they replace


class Stages(Protocol):
    def evolve(self, cycle: int, checkpoint: str, value_model: str | None, parents: Mapping[str, str]) -> EvolveResult:
        """Run (or resume, or read if finished) evolution round ``cycle`` on ``parents`` (slot -> deck file)."""

    def train(self, cycle: int, checkpoint: str, until: int) -> str:
        """Train from ``checkpoint`` (or the segment's own latest checkpoint after a crash) to update ``until``;
        returns the checkpoint at ``until``."""

    def fit(self, cycle: int, checkpoint: str) -> str:
        """Fit the value model for the policy ``checkpoint``; returns its path."""


def _write(path: Path, data: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


class CoEvolution:
    """The driver's state in ``directory/coevo.json`` (module docstring)."""

    def __init__(self, directory: str | Path, stages: Stages, config: CoevoConfig, checkpoint: str, *,
                 log: Callable[[str], None] | None = None) -> None:  # fmt: skip
        self.dir = Path(directory)
        self.stages = stages
        self.say = log or (lambda m: None)
        self.path = self.dir / "coevo.json"
        if self.path.is_file():
            self.state = json.loads(self.path.read_text())
            if self.state.get("format") != FORMAT:
                raise ValueError(f"{self.path}: not a {FORMAT} state")
            if asdict(config) != self.state["config"] or checkpoint != self.state["start"]:
                self.say("resuming: the settings of the first run are kept")
        else:
            if not config.decks:
                raise ValueError("give at least one deck")
            self.state = {"format": FORMAT, "version": 1, "config": asdict(config), "start": checkpoint,
                          "checkpoint": checkpoint, "value_model": None, "parents": dict(config.decks),
                          "stages": {}, "cycles": []}  # fmt: skip
            self._save()
        self.config = CoevoConfig(**self.state["config"])

    def _save(self) -> None:
        _write(self.path, self.state)

    def _done(self, key: str) -> dict | None:
        return self.state["stages"].get(key)

    def _record(self, key: str, result: Mapping) -> None:
        self.state["stages"][key] = {**result, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        self._save()

    def run(self) -> dict:
        cfg = self.config
        if cfg.initial_fit and not self._done("fit-0"):
            self.say("fit 0: value model for the starting policy")
            vm = self.stages.fit(0, self.state["checkpoint"])
            self.state["value_model"] = vm
            self._record("fit-0", {"value_model": vm})
        for c in range(1, cfg.cycles + 1):
            if not self._done(f"evolve-{c}"):
                self.say(f"cycle {c}: evolution round with {self.state['checkpoint']}")
                r = self.stages.evolve(c, self.state["checkpoint"], self.state["value_model"], self.state["parents"])
                parents = dict(self.state["parents"])
                for a in r.accepted:
                    parents[a["slot"]] = a["file"]
                self.state["parents"] = parents
                self._record(f"evolve-{c}", {"round": r.round, "games": r.games, "accepted": r.accepted,
                                             "parents": parents})  # fmt: skip
            if not self._done(f"train-{c}"):
                until = cfg.start_update + c * cfg.updates
                self.say(f"cycle {c}: training to update {until}")
                ckpt = self.stages.train(c, self.state["checkpoint"], until)
                self.state["checkpoint"] = ckpt
                self._record(f"train-{c}", {"checkpoint": ckpt, "update": until})
            if not self._done(f"fit-{c}"):
                self.say(f"cycle {c}: refitting the value model")
                vm = self.stages.fit(c, self.state["checkpoint"])
                self.state["value_model"] = vm
                self._record(f"fit-{c}", {"value_model": vm})
        return self.summary()

    def summary(self) -> dict:
        """Per cycle and in total: evolution games, accepted children and their validated gain, games per +1 pp of
        validated gain (the validation is on fresh seeds but selected: re-validate with
        ``tools/deckevo_eval_compare.py --evo-state`` for the unbiased gain)."""
        rows, games, gain, accepted = [], 0, 0.0, 0
        for c in range(1, self.config.cycles + 1):
            e = self._done(f"evolve-{c}")
            if e is None:
                break
            g = sum(a.get("diff") or 0.0 for a in e["accepted"])
            rows.append({"cycle": c, "games": e["games"], "accepted": len(e["accepted"]), "validated_gain": g,
                         "checkpoint": (self._done(f"train-{c}") or {}).get("checkpoint"),
                         "value_model": (self._done(f"fit-{c}") or {}).get("value_model")})  # fmt: skip
            games += e["games"]
            gain += g
            accepted += len(e["accepted"])
        return {"cycles": rows, "games": games, "accepted": accepted, "validated_gain": gain,
                "games_per_pp": games / (gain * 100) if gain > 0 else None, "parents": self.state["parents"],
                "checkpoint": self.state["checkpoint"], "value_model": self.state["value_model"]}  # fmt: skip


def accepted_children(state: str | Path, round_no: int, slots: Mapping[str, str]) -> tuple[int, list[dict]]:
    """(games, accepted children) of evolution round ``round_no`` in the evolution state ``state``: each accepted
    child with the deck slot whose current parent it descends from (the parent id ends in the parent file's stem:
    ``corpus:<stem>`` / ``file:<stem>``) and its deck file in the state's manifest directory."""
    st = Path(state)
    rep = json.loads((st / "rounds" / f"{round_no:04d}" / "report.json").read_text())
    by_stem = {Path(f).stem: slot for slot, f in slots.items()}
    out = []
    for line in (st / "lineage.jsonl").read_text().splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if rec["round"] != round_no or not rec.get("accepted"):
            continue
        slot = by_stem.get(rec["parent"]["id"].split(":", 1)[-1])
        if slot is None:
            continue
        mid = rec["manifest_id"]
        out.append({"slot": slot, "parent": rec["parent"]["id"], "id": mid, "file": str(st / "decks" / f"{mid}.ydk"),
                    "diff": (rec.get("validation") or {}).get("diff"), "edits": rec["edits"]})  # fmt: skip
    return int(rep["games"]), out


def round_done(state: str | Path, round_no: int) -> bool:
    path = Path(state) / "rounds" / f"{round_no:04d}" / "round.json"
    return path.is_file() and bool(json.loads(path.read_text()).get("done"))


def round_count(state: str | Path) -> int:
    return len([p for p in (Path(state) / "rounds").glob("[0-9]*") if (p / "round.json").is_file()])


def check_round(state: str | Path, cycle: int) -> None:
    """Round ``cycle`` is the next one to start or the one in progress: the evolution state has no round past it
    (else it was written by something else than this driver)."""
    n = round_count(state)
    if n not in (cycle - 1, cycle) or (n == cycle - 1 and n and not round_done(state, n)):
        raise ValueError(f"evolution state {state} has {n} rounds: not the state of cycle {cycle}")


def run_command(cmd: Sequence[str], log_path: Path | None = None, *, env: Mapping[str, str] | None = None) -> None:
    """Run a stage's command (appending its output to ``log_path``); a non-zero exit is an error."""
    import subprocess

    out = open(log_path, "a") if log_path is not None else None  # noqa: SIM115
    try:
        if out is not None:
            out.write(f"$ {' '.join(cmd)}\n")
            out.flush()
        res = subprocess.run(list(cmd), stdout=out, stderr=subprocess.STDOUT if out else None,
                             env={**os.environ, **(env or {})})  # fmt: skip
    finally:
        if out is not None:
            out.close()
    if res.returncode != 0:
        raise RuntimeError(f"{cmd[1] if len(cmd) > 1 else cmd[0]} failed with exit code {res.returncode}"
                           + (f" (log: {log_path})" if log_path else ""))  # fmt: skip


__all__ = ["FORMAT", "CoEvolution", "CoevoConfig", "EvolveResult", "Stages", "accepted_children", "check_round",
           "round_count", "round_done", "run_command"]  # fmt: skip
