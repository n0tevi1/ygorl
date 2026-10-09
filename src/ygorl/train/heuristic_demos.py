"""Heuristic demonstrations beyond turn 1 for behaviour cloning (docs/bc.md「补救实验」, issue #28).

The solver demonstrations (:func:`ygorl.train.bc.build_dataset`) cover turn 1 going first only: no battle
phase, no attack, no later turn. This module records a heuristic agent (``GreedyAgent`` by default) in
ordinary duels and turns its decisions into the same ``(observation, action row)`` samples:

- :class:`DemoRecorder` wraps the agent of one seat, sees every decision point of the duel (``observe``,
  the event stream needs them all) and encodes the seat's non-forced decisions from ``min_turn`` on with
  :class:`~ygorl.env.observer.PointObserver`, i.e. the observations ``EncodedVecEnv`` and ``NetPolicy``
  produce. Labels are mapped to the representative row of their equivalent copies (``canonical_action``);
  a masked no-op cancel removes the whole cancelled command's samples, not only the cancel itself.
- :func:`record_games` plays a list of ``GameSpec`` s (both seats via ``first``), optionally in parallel.
- :data:`SUBSETS` select samples by their ``meta``: ``all``, or ``battle`` — the decisions the root-cause
  analysis found missing (own battle-phase decisions, and the main-phase-1 choice to enter the battle phase).
- :func:`concat`, :func:`save_data` / :func:`load_data` combine and store :class:`~ygorl.train.bc.BCData`.
"""

from __future__ import annotations

import json
import hashlib
import multiprocessing as mp
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from ygorl.engine import constants as C
from ygorl.env.encoding import MAX_OPTIONS, canonical_action
from ygorl.env.events import DEFAULT_EVENT_LENGTH
from ygorl.env.observer import PointObserver
from ygorl.nets.batch import OBS_KEYS
from ygorl.train.bc import BCData

BATTLE_PHASES = C.PHASE_BATTLE_START | C.PHASE_BATTLE_STEP | C.PHASE_DAMAGE | C.PHASE_DAMAGE_CAL | C.PHASE_BATTLE


def is_battle_decision(meta: dict) -> bool:
    """Own decision in the battle phase, or the main-phase-1 decision that enters it."""
    if not meta["own_turn"]:
        return False
    if meta["phase"] & BATTLE_PHASES:
        return True
    return meta["phase"] == C.PHASE_MAIN1 and meta["bp_offered"] and meta["kind"] == "battle_phase"


SUBSETS: dict[str, Callable[[dict], bool]] = {"all": lambda m: True, "battle": is_battle_decision}


class DemoRecorder:
    """Agent wrapper that plays ``agent`` and records its decisions as BC samples.

    Recorded: decisions of this seat with ``point.turn >= min_turn``, at least two choosable rows after
    equivalent-copy masking, and the chosen row among the 128 encoded ones.
    """

    def __init__(self, agent, cards, vocab, *, event_length: int = DEFAULT_EVENT_LENGTH, min_turn: int = 2,
                 game: int = 0, include_cancelled_commands: bool = False) -> None:  # fmt: skip
        self.agent = agent
        self.name = f"record({getattr(agent, 'name', type(agent).__name__)})"
        self.observer = PointObserver(cards, vocab, event_length)
        self.min_turn = min_turn
        self.game = game
        self.include_cancelled_commands = include_cancelled_commands
        self._command: tuple[int, int, int] | None = None  # player, turn, first sample of the menu command
        self.core = None
        self.obs: list[dict[str, np.ndarray]] = []
        self.actions: list[int] = []
        self.meta: list[dict] = []
        self.skipped: Counter = Counter()

    def observe(self, point, core) -> None:
        self.core = core
        self.observer.observe(point)
        hook = getattr(self.agent, "observe", None)
        if hook is not None:
            hook(point, core)

    def act(self, point) -> int:
        if point.decision.TYPE in (C.MSG_SELECT_IDLECMD, C.MSG_SELECT_BATTLECMD):
            self._command = (point.player, point.turn, len(self.actions))
        idx = self.agent.act(point)
        if (not self.include_cancelled_commands and point.actions[idx].kind == "cancel" and idx in point.undo
                and self._command is not None and self._command[:2] == (point.player, point.turn)):  # fmt: skip
            # The host only marks cancel as undo if no game event or other player's decision intervened.
            # Teaching the abandoned menu choice is wrong when that same undo is masked for the policy.
            start = self._command[2]
            removed = len(self.actions) - start
            if removed:
                self.skipped["cancelled_command"] += removed
                del self.obs[start:]
                del self.actions[start:]
                del self.meta[start:]
            self._command = None
        if point.turn < self.min_turn:
            self.skipped["early_turn"] += 1
            return idx
        if len(point.actions) < 2:
            self.skipped["forced"] += 1
            return idx
        obs = self.observer.encode(point, self.core)
        n_choices = int(np.count_nonzero(obs["action_mask"]))
        if n_choices < 2:
            self.skipped["forced"] += 1
        elif idx >= MAX_OPTIONS:
            self.skipped["beyond_128"] += 1
        elif not obs["action_mask"][canonical_action(obs, idx)]:
            self.skipped["undo"] += 1  # a no-op undo the mask hides (docs/encoding.md 「撤销类空操作」)
        else:
            kinds = {a.kind for a in point.actions}
            self.obs.append({k: obs[k] for k in OBS_KEYS if k in obs})
            self.actions.append(canonical_action(obs, idx))
            self.meta.append({"source": "heuristic", "game": self.game, "turn": point.turn, "player": point.player,
                              "own_turn": point.turn_player == point.player, "phase": int(point.phase),
                              "decision": type(point.decision).__name__, "kind": point.actions[idx].kind,
                              "bp_offered": "battle_phase" in kinds, "n_legal": len(point.actions),
                              "n_choices": n_choices})  # fmt: skip
        return idx


def _record_one(job) -> dict:
    game, spec, make_agent, vocab, event_length, min_turn, include_cancelled_commands = job
    from ygorl.engine.duel import default_cards

    rec = DemoRecorder(make_agent(spec.agent_seeds[0]), default_cards(), vocab, event_length=event_length,
                       min_turn=min_turn, game=game, include_cancelled_commands=include_cancelled_commands)  # fmt: skip
    try:
        result = spec.duel().run(rec, spec.agent_b(spec.agent_seeds[1]))
    except Exception as exc:  # noqa: BLE001 - reported, the game is dropped
        return {"game": game, "error": f"{type(exc).__name__}: {exc}"}
    health = {"retries": result.retries, "unknown_messages": result.unknown_messages,
              "undecodable_messages": result.undecodable_messages, "script_errors": list(result.script_errors)}  # fmt: skip
    if result.reason == "error" or result.error or any(health.values()):
        return {"game": game, "reason": result.reason, "error": result.error or result.summary(), **health}
    return {"game": game, "obs": rec.obs, "actions": rec.actions, "meta": rec.meta, "skipped": rec.skipped,
            "winner": result.winner, "reason": result.reason, "turns": result.turns,
            "deck_a": spec.deck_a.name, "deck_b": spec.deck_b.name,
            "first": spec.first, "opponent": getattr(spec.agent_b, "name", str(spec.agent_b)), **health}  # fmt: skip


def record_games(specs: Sequence, make_agent: Callable[[int], object], vocab, *,
                 event_length: int = DEFAULT_EVENT_LENGTH, min_turn: int = 2, workers: int = 1,
                 log: Callable[[str], None] | None = None,
                 include_cancelled_commands: bool = False) -> tuple[BCData, list[dict]]:  # fmt: skip
    """Play every ``GameSpec`` with ``make_agent(seed)`` recorded as seat a (``spec.agent_a`` is ignored).

    Returns the samples (``meta`` also carries the deck, opponent and seat of each game) and one summary per game.
    """
    jobs = [(g, spec, make_agent, vocab, event_length, min_turn, include_cancelled_commands)
            for g, spec in enumerate(specs)]  # fmt: skip
    if workers <= 1:
        results = map(_record_one, jobs)
        pool = None
    else:
        pool = mp.get_context("fork").Pool(workers)
        results = pool.imap(_record_one, jobs, chunksize=1)
    columns: dict[str, list[np.ndarray]] = {}
    actions: list[int] = []
    meta: list[dict] = []
    skipped: Counter = Counter()
    games = []
    try:
        for res in results:
            summary = {k: v for k, v in res.items() if k not in ("obs", "actions", "meta", "skipped")}
            summary["samples"] = len(res.get("actions", ()))
            games.append(summary)
            if "error" in res:
                skipped["error_games"] += 1
                continue
            skipped.update(res["skipped"])
            for o in res["obs"]:
                for k, v in o.items():
                    columns.setdefault(k, []).append(v)
            actions.extend(res["actions"])
            extra = {k: summary[k] for k in ("deck_a", "deck_b", "first", "opponent")}
            meta.extend({**m, **extra} for m in res["meta"])
            if log is not None:
                log(f"game {res['game'] + 1}/{len(jobs)}: {summary['samples']} samples, {res['turns']} turns")
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    obs = {k: np.stack(v) for k, v in columns.items()}
    return BCData(obs, np.asarray(actions, dtype=np.int64), meta, skipped), games


def select(data: BCData, subset: str = "all", *, max_samples: int | None = None, seed: int = 0) -> BCData:
    """Samples of ``subset`` (:data:`SUBSETS`); with ``max_samples``, a uniform random sample of at most that many."""
    if subset not in SUBSETS:
        raise ValueError(f"unknown subset {subset!r} (one of {', '.join(SUBSETS)})")
    idx = np.flatnonzero([SUBSETS[subset](m) for m in data.meta])
    if max_samples is not None and len(idx) > max_samples:
        idx = np.sort(np.random.default_rng(seed).choice(idx, max_samples, replace=False))
    return data.subset(idx)


def concat(parts: Sequence[BCData]) -> BCData:
    """One dataset from several with the same observation shapes (e.g. solver turn-1 + heuristic samples)."""
    keys = [k for k in OBS_KEYS if k in parts[0].obs]
    for p in parts[1:]:
        if [k for k in OBS_KEYS if k in p.obs] != keys:
            raise ValueError("the datasets have different observation keys")
        for k in keys:
            if p.obs[k].shape[1:] != parts[0].obs[k].shape[1:]:
                raise ValueError(f"{k}: shapes {p.obs[k].shape[1:]} and {parts[0].obs[k].shape[1:]} differ "
                                 "(same event length?)")  # fmt: skip
    skipped: Counter = Counter()
    for p in parts:
        skipped.update(p.skipped)
    return BCData({k: np.concatenate([p.obs[k] for p in parts]) for k in keys},
                  np.concatenate([p.actions for p in parts]), [m for p in parts for m in p.meta], skipped)  # fmt: skip


def save_data(path: str | Path, data: BCData, **info) -> Path:
    """``np.savez_compressed`` of the observations and actions, the meta and ``info`` as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = json.dumps({"meta": data.meta, "skipped": dict(data.skipped), "info": info})
    np.savez_compressed(path, actions=data.actions, _json=np.frombuffer(extra.encode(), dtype=np.uint8),
                        **{f"obs_{k}": v for k, v in data.obs.items()})  # fmt: skip
    return path


def load_data(path: str | Path) -> tuple[BCData, dict]:
    """(:class:`BCData`, info) written by :func:`save_data`."""
    with np.load(path) as z:
        extra = json.loads(bytes(z["_json"]).decode())
        obs = {k[4:]: z[k] for k in z.files if k.startswith("obs_")}
        data = BCData(obs, z["actions"], extra["meta"], Counter(extra["skipped"]))
    return data, extra["info"]


def data_identity(vocab, event_length: int, environment: dict | None = None) -> dict:
    """Identity of already encoded samples; equal shapes do not imply equal card IDs or rules."""
    from ygorl.nets.agent import vocab_passwords

    mapping = {"first_index": vocab.FIRST_INDEX, "passwords": vocab_passwords(vocab)}
    digest = hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()
    return {"format": "ygorl-bc-data-1", "environment": environment,
            "vocab_sha256": digest, "event_length": event_length}  # fmt: skip


def load_compatible_data(path: str | Path, *, vocab, event_length: int,
                         environment: dict | None = None, selection_history: bool = False) -> tuple[BCData, dict]:  # fmt: skip
    """Load extra BC samples only when their recorded identity matches the solver data / actor."""
    data, info = load_data(path)
    identity = info.get("identity")
    if not isinstance(identity, dict):
        raise ValueError(f"{path}: missing BC data identity; regenerate with tools/greedy_demos.py")
    expected = data_identity(vocab, event_length, environment)
    for key, value in expected.items():
        if key not in identity or identity[key] != value:
            raise ValueError(f"{path}: BC data {key} mismatch; regenerate for the selected environment and actor")
    if ("selection_history" in data.obs) != selection_history:
        raise ValueError(f"{path}: BC data selection-history encoding differs; replay the demonstrations")
    events = data.obs.get("events")
    if events is None or events.ndim != 3 or events.shape[1] != event_length:
        raise ValueError(f"{path}: BC data event_length does not match its encoded observations")
    return data, info


__all__ = ["BATTLE_PHASES", "DemoRecorder", "SUBSETS", "concat", "is_battle_decision", "load_data", "record_games",
           "save_data", "select", "data_identity", "load_compatible_data"]  # fmt: skip
