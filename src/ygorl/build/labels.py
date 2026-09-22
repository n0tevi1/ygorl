"""Real-game labels for the deck surrogate (T5.7).

A label is what the paired-seed :class:`~ygorl.eval.arena.Arena` measures for a
candidate deck against a meta pool: its win rate overall, going first and going
second (pool-weighted means over opponents), the games behind them and the
binomial noise of the estimate. Labels are cached on disk as JSON lines keyed by
``(deck fingerprint, label-config fingerprint)``, so interrupted runs resume and
the same deck is never replayed under the same protocol.

Common random numbers: the games against opponent ``j`` use the seed
``derive_seed(seed, j)`` for *every* candidate, so all candidates face the same
opponent draws and agent seeds; differences between candidates' labels are less
noisy than the labels themselves.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from ygorl.agents.registry import agent_factory
from ygorl.cards.ydk import Deck
from ygorl.data.environment import Environment
from ygorl.engine.duel import DuelConfig
from ygorl.eval.arena import Arena, ArenaReport, derive_seed, wilson_interval

LABEL_VERSION = 1


def deck_fingerprint(deck: Deck) -> str:
    """SHA-256 of the sorted Main and Extra Deck passwords (name, order and Side Deck ignored)."""
    text = ",".join(map(str, sorted(deck.main))) + "|" + ",".join(map(str, sorted(deck.extra)))
    return hashlib.sha256(text.encode()).hexdigest()


def labels_from_reports(reports: Sequence[ArenaReport], weights: Sequence[float] | None = None) -> dict[str, Any]:
    """Pool one candidate's reports (one per meta opponent, candidate = agent a) into a label.

    ``win_rate`` / ``win_rate_first`` / ``win_rate_second`` are ``weights``-weighted
    means of the per-opponent rates (default uniform; draws count half). ``se`` is
    the binomial standard error of ``win_rate`` (per-opponent variance taken at the
    pooled rate); ``ci_half_width`` is the half-width of the 95% Wilson interval at
    the effective sample size ``1 / sum(w_j^2 / n_j)`` (the plain game count for
    uniform weights and equal games per opponent).
    """
    if not reports:
        raise ValueError("no reports to pool")
    w = [1.0] * len(reports) if weights is None else [float(x) for x in weights]
    if len(w) != len(reports) or any(x < 0 for x in w) or sum(w) <= 0:
        raise ValueError(f"need one non-negative weight per report ({len(reports)}), got {weights}")
    total = sum(w)
    w = [x / total for x in w]

    def pooled(rates: list[float]) -> float:
        return sum(wi * r for wi, r in zip(w, rates))

    rate = pooled([r.win_rate for r in reports])
    n_eff = 1.0 / sum(wi * wi / max(r.games, 1) for wi, r in zip(w, reports))
    lo, hi = wilson_interval(rate * n_eff, max(int(round(n_eff)), 1)) if n_eff > 0 else (0.0, 1.0)
    return {
        "win_rate": rate,
        "win_rate_first": pooled([r.as_first.win_rate for r in reports]),
        "win_rate_second": pooled([r.as_second.win_rate for r in reports]),
        "games": sum(r.games for r in reports),
        "draws": sum(r.draws for r in reports),
        "errors": sum(r.errors for r in reports),
        "se": (rate * (1 - rate) / n_eff) ** 0.5,
        "ci_half_width": (hi - lo) / 2,
        "mean_turns": sum(r.mean_turns * r.games for r in reports) / max(sum(r.games for r in reports), 1),
        "per_opponent": {r.deck_b: r.win_rate for r in reports},
    }


@dataclass(frozen=True)
class LabelConfig:
    """Everything that determines a label besides the candidate deck."""

    pool: tuple[tuple[str, str], ...]  # (name, deck fingerprint) per meta opponent
    weights: tuple[float, ...]
    agent: str = "greedy"  # pilots the candidate
    opponent: str = "greedy"  # pilots the meta decks
    pairs: int = 2  # paired seeds per opponent (2 games each)
    seed: int = 0
    max_turns: int | None = None
    environment: tuple[tuple[str, str], ...] | None = None  # Environment.stamp() items
    version: int = LABEL_VERSION

    @classmethod
    def for_pool(cls, pool: Sequence[Deck], *, weights: Sequence[float] | None = None, agent: str = "greedy",
                 opponent: str = "greedy", pairs: int = 2, seed: int = 0, max_turns: int | None = None,
                 env: Environment | None = None) -> LabelConfig:  # fmt: skip
        names = [d.name or f"opponent{j}" for j, d in enumerate(pool)]
        if len(set(names)) != len(names):
            raise ValueError(f"meta pool deck names must be unique: {names}")
        w = tuple(float(x) for x in (weights if weights is not None else [1.0] * len(pool)))
        if len(w) != len(pool):
            raise ValueError(f"{len(pool)} meta decks but {len(w)} weights")
        stamp = tuple(sorted(env.stamp().items())) if env is not None else None
        return cls(tuple((n, deck_fingerprint(d)) for n, d in zip(names, pool)), w, agent, opponent, int(pairs),
                   int(seed), max_turns, stamp)  # fmt: skip

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> LabelConfig:
        env = d.get("environment")
        return cls(
            pool=tuple(tuple(p) for p in d["pool"]), weights=tuple(d["weights"]), agent=d["agent"],
            opponent=d["opponent"], pairs=d["pairs"], seed=d["seed"], max_turns=d.get("max_turns"),
            environment=tuple(tuple(e) for e in env) if env is not None else None, version=d.get("version", LABEL_VERSION),
        )  # fmt: skip

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


class LabelCache:
    """Append-only JSON-lines label store keyed by ``(deck, config)`` fingerprints.

    Each line is a record ``{"deck", "config", "labels", ...}``; later lines win.
    A torn last line (an interrupted write) is ignored on load.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._records: dict[tuple[str, str], dict[str, Any]] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict) and "deck" in rec and "config" in rec:
                    self._records[(rec["deck"], rec["config"])] = rec

    def __len__(self) -> int:
        return len(self._records)

    def get(self, deck: str, config: str) -> dict[str, Any] | None:
        return self._records.get((deck, config))

    def records(self, config: str | None = None) -> list[dict[str, Any]]:
        """All records (of one config), in insertion order."""
        return [r for (_, c), r in self._records.items() if config is None or c == config]

    def put(self, record: Mapping[str, Any]) -> None:
        if "deck" not in record or "config" not in record:
            raise ValueError("a label record needs 'deck' and 'config' fingerprints")
        rec = dict(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        self._records[(rec["deck"], rec["config"])] = rec


def label_decks(decks: Sequence[Deck], pool: Sequence[Deck], *, weights: Sequence[float] | None = None,
                agent: str = "greedy", opponent: str = "greedy", pairs: int = 2, seed: int = 0,
                max_turns: int | None = None, env: Environment | None = None, workers: int = 1,
                cache: LabelCache | None = None, chunk: int = 8,
                progress: Callable[[int, int], None] | None = None) -> list[dict[str, Any]]:  # fmt: skip
    """Label ``decks`` by real games against the meta ``pool``; one record per deck, in order.

    Each candidate (piloted by ``agent``) plays ``2 * pairs`` paired games against
    every meta deck (piloted by ``opponent``); opponent ``j`` always uses the seed
    ``derive_seed(seed, j)`` (common random numbers). Records already in ``cache``
    under the same :class:`LabelConfig` are returned without playing; new ones are
    appended to it ``chunk`` decks at a time (one worker pool per chunk).
    ``progress(done, total)`` is called after each chunk.
    """
    config = LabelConfig.for_pool(pool, weights=weights, agent=agent, opponent=opponent, pairs=pairs, seed=seed,
                                  max_turns=max_turns, env=env)  # fmt: skip
    cfg = config.fingerprint()
    duel_config = DuelConfig.from_environment(env) if env is not None else DuelConfig()
    if max_turns is not None:
        duel_config = replace(duel_config, max_turns=max_turns)
    arena = Arena(agent_factory(agent), agent_factory(opponent), env=env, config=duel_config, workers=workers)
    names = [n for n, _ in config.pool]
    pool = [Deck(d.main, d.extra, d.side, name=n) for d, n in zip(pool, names)]
    fps = [deck_fingerprint(d) for d in decks]
    out: dict[str, dict[str, Any]] = {}
    todo: list[int] = []
    queued: set[str] = set()
    for i, fp in enumerate(fps):
        hit = cache.get(fp, cfg) if cache is not None else None
        if hit is not None:
            out[fp] = hit
        elif fp not in queued:
            queued.add(fp)
            todo.append(i)
    for start in range(0, len(todo), max(chunk, 1)):
        batch = todo[start : start + max(chunk, 1)]
        matchups = [(decks[i], opp, derive_seed(seed, j)) for i in batch for j, opp in enumerate(pool)]
        reports = arena.run_many(matchups, pairs)
        for n, i in enumerate(batch):
            mine = reports[n * len(pool) : (n + 1) * len(pool)]
            rec = {
                "deck": fps[i], "config": cfg, "labels": labels_from_reports(mine, config.weights),
                "main": sorted(decks[i].main), "extra": sorted(decks[i].extra), "name": decks[i].name,
                "label_config": config.to_dict(),
            }  # fmt: skip
            out[fps[i]] = rec
            if cache is not None:
                cache.put(rec)
        if progress is not None:
            progress(min(start + len(batch), len(todo)), len(todo))
    return [out[fp] for fp in fps]
