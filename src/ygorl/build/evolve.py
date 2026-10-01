"""One step of deck evolution (#111, spec #105 「微调」, docs/tuning.md「进化步骤」).

:class:`Evolution` owns the evolution's state in one directory, bound to one environment:

- ``manifest.json``: the deck-pool manifest the trainer reads (``ygorl-deck-pool``, training.md §8.3), plus an
  ``environment`` stamp; accepted children go in as ``probation`` decks, their lists in ``decks/<id>.ydk``;
- ``archive.json``: the MAP-Elites archive (:class:`ygorl.build.archive.DeckArchive`);
- ``signals.json``: the signal library (:class:`ygorl.build.signals.CardValueModel` and
  :class:`ygorl.build.signals.Calibration`);
- ``lineage.jsonl``: one record per evaluated child;
- ``rounds/NNNN/``: a round's settings and per-parent results (``round.json``), every game it played
  (``games-<parent>.jsonl``) and its report (``report.json``, ``report.txt``).

:meth:`Evolution.run_round` runs one round. Per parent: optional diagnosis (the parent's first pairs; its
opening-hand effects only reach the prior through the calibration table, where they weigh 0 until they prove
themselves) → ``informed_children`` → the L0 screen (off by default; ``shadow`` only counts what it would remove) →
top-two Thompson sampling → sequential validation of the chosen child → every evaluated child's paired difference
into the card-value model and the calibration table, every evaluated deck into the archive, the accepted child into
the manifest, and a lineage record per child.

Engine-aware (#145): the parent's engine members (``Lab.engine``) are protected while the model has fewer than
``engine_evidence`` observations of them in the parent's type, and go out after every other card once it has.
While the model has fewer than ``cold_min_obs`` observations of the parent's type (**cold start**), a round makes
``cold_candidates`` single swaps instead of bundles and crossover, plays one batch of each (:func:`screen`), and
races the best ``cold_keep``. The search stops as confident only with ``min_confident_pairs`` pairs on the leader and
a Bonferroni threshold over every candidate it chose among (``multiplicity``).
:meth:`Evolution.warm_start` adds earlier paired data (``ygorl.build.warmstart``) to the model, once per id.
With an edit-value model (``Lab.value_model``, #151), each child's predicted Δ is blended into the Thompson prior
and ranks the learned candidates, by the weight the calibration table gives it (:func:`blend_prior`); at weight 0
the round is the same as without it.

Resuming: every game result is appended to the round's game log as it arrives, and the round replays
deterministically (fixed seeds per round and parent), so a rerun after a crash reads the games it already has
instead of playing them. A parent's result is written to ``round.json`` before its side effects, which are
idempotent (keyed by the child's id), so a crash between the two is repaired by the next run.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from ygorl.build.archive import DESCRIPTORS, DeckArchive
from ygorl.build.crossover import crossover_children
from ygorl.build.diagnose import opening_effects
from ygorl.build.learned import learned_children
from ygorl.build.factorial import LETTERS, evaluate_edits, place, variant
from ygorl.build.rules import rule_children
from ygorl.build.selection import SD_PAIR, screen, sequential_validate, top_two_thompson
from ygorl.build.signals import Calibration, CardValueModel, Observation, combine_prior, informed_children
from ygorl.cards.ydk import Deck, parse_ydk
from ygorl.data.environment import EnvironmentConfigError
from ygorl.eval.arena import derive_seed

MANIFEST_FORMAT = "ygorl-deck-pool"  # ygorl.train.selfplay.MANIFEST_FORMAT (not imported: that module needs PyTorch)
SIGNALS_FORMAT = "ygorl-signal-library"
ROUND_FORMAT = "ygorl-evolution-round"
# per-card signals and their weights before evidence: M3 found no cheap signal that predicts a card's value
SIGNAL_DEFAULTS = {"opening_effect": 0.0}
LIVE = ("probation", "active")
VALIDATION_OFFSET = 1_000_000  # first pair index of sequential validation (fresh seeds, never searched)
MUTATION_KINDS = ("informed", "explore")  # child kinds of the mutation generator (informed_children)


def generator(kind: str) -> str:
    """The generator a child kind belongs to: ``mutation`` for ``informed_children``'s kinds, else the kind itself
    (``crossover``, ``rules``, ``learned``)."""
    return "mutation" if kind in MUTATION_KINDS else kind


@dataclass(frozen=True)
class Parent:
    id: str  # manifest id, or e.g. "corpus:<file stem>"
    deck: Deck
    type: str  # the deck type the card-value model files observations under


@dataclass(frozen=True)
class Opponent:
    name: str
    deck: Deck
    weight: float


@dataclass
class RoundConfig:
    informed: int = 6
    explore: int = 2
    max_bundle: int = 3
    crossover: int = 0  # crossover children per parent (the parent × an archive elite, #141); off by default
    rules: int = 0  # children from the association rules (ygorl.build.rules; needs Lab.rules), off by default
    learned: int = 0  # children from the masked deck model (ygorl.build.learned; needs Lab.deck_model), off by default
    learned_removal: str = "combined"  # the learned children's removal ranking: combined, support or typicality
    diagnose_pairs: int = 0  # the parent's first pairs, played up front (they are the parent's baseline too)
    batch: int = 25
    max_pairs: int = 600
    batches_per_round: int = 4
    confidence: float = 0.95
    look: int = 100
    cap: int = 1000
    alpha: float = 0.025
    min_effect: float = 0.02
    budget: int | None = None  # games per round: no parent starts once the round has played this many
    l0: str = "off"  # "off", "shadow" (count what the screen would remove) or "on"
    l0_min: float = -0.02  # children the screen predicts below this (win-rate units) are removed
    archive_min_pairs: int = 20  # a deck with fewer pairs is not offered to the archive
    archive_z: float = 1.645  # the archive objective is the win rate's one-sided lower bound at this z
    engine_evidence: int = 2  # an engine member is protected until the model has this many observations of it (#145)
    cold_candidates: int = 40  # single swaps screened per parent while its type's evidence is thin (0: off)
    cold_keep: int = 4  # screened children that go on to the search
    cold_min_obs: int = 50  # the model's observations of the parent's type below which a round is a cold start
    min_confident_pairs: int = 100  # pairs the leader needs before the search may stop as confident
    multiplicity: bool = True  # Bonferroni: the confident stop's threshold over every candidate chosen among
    evaluation: str = "thompson"  # "thompson" (children raced, #110) or "factorial" (single edits in one design, #152)
    factorial_k: int = 4  # edits per fractional factorial
    factorial_pairs: int = 200  # pairs every variant of the design plays
    value_model_weight: float = 0.0  # the edit-value model's weight until the calibration table has enough pairs
    value_model_oversample: int = 3  # learned candidates drawn per learned child kept, ranked by the value model
    seed: int = 0


@dataclass
class Lab:
    """What a round needs from the environment and the policy. The command line builds it from ``--env`` and
    ``--checkpoint``; tests build stand-ins.

    ``evaluator(seed, opponents)`` returns a paired evaluator (:class:`ygorl.build.tuner.PairedEvaluator`: ``play(jobs,
    per_game=True)``, optionally ``opening_hands``); ``pool(parent)`` the candidate cards; ``descriptors(deck)`` the
    deck-list descriptors (:func:`ygorl.build.archive.deck_descriptors`); ``screen(evaluator, parent, children)`` the
    L0 screen's predicted difference per child (win-rate units, on the evaluator's pairs: common random numbers);
    ``checkpoint`` what the lineage records about the policy; ``rules`` the association rules
    (:class:`ygorl.build.rules.DeckRules`) behind the ``RoundConfig.rules`` children; ``engine(deck)`` the deck's
    engine members (:func:`ygorl.build.deck_engine.deck_engine`), protected while unproven. ``pool(parent)`` is also
    the rule children's addition filter. ``deck_model`` scores the ``RoundConfig.learned`` children
    (:func:`ygorl.build.learned.learned_children`), whose additions come from ``card_pool`` (the environment's card
    pool; None: ``pool(parent)``). ``value_model`` (``predict(parent, [edits, ...]) -> [(mean, sd), ...]``, e.g.
    :class:`ygorl.build.value_model.EditValueModel`, #151) predicts each child's Δ: blended into the Thompson prior
    and ranking the learned candidates by its calibrated weight (:func:`blend_prior`)."""

    evaluator: Callable[[int, Sequence[Opponent]], Any]
    legal: Callable[[Deck], bool]
    is_extra: Callable[[int], bool]
    pool: Callable[[Parent], Sequence[int]]
    protected: Callable[[Deck], set[int]] = lambda deck: set()
    descriptors: Callable[[Deck], Mapping[str, float]] = lambda deck: {}
    screen: Callable[[Any, Deck, Sequence[Deck]], Sequence[float]] | None = None
    checkpoint: Mapping[str, Any] = field(default_factory=dict)
    rules: Any = None
    engine: Callable[[Deck], set[int]] = lambda deck: set()
    deck_model: Any = None
    card_pool: Collection[int] | None = None
    value_model: Any = None


# ------------------------------------------------------------------ files


def _write(path: Path, text: str) -> None:
    """Atomic replace (a reader never sees half a file: the trainer rereads the manifest while we write it)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=1, allow_nan=False, default=_default) + "\n"


def _default(o):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def _finite(x: float) -> float | None:
    return float(x) if x is not None and math.isfinite(x) else None


def deck_key(deck: Deck) -> str:
    """Identity of a deck list *in order* (in-place edits keep positions, and the order decides the shuffle)."""
    return hashlib.sha1(f"{deck.main}|{deck.extra}".encode()).hexdigest()[:16]


def _deck_json(deck: Deck) -> dict:
    return {"main": list(deck.main), "extra": list(deck.extra)}


def _deck_from(d: Mapping, name: str = "") -> Deck:
    return Deck(main=tuple(d["main"]), extra=tuple(d["extra"]), name=name)


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def append_lines(path: Path, lines: Sequence[str]) -> None:
    """Append whole lines and fsync. A crash can leave a torn last line (no newline): it is cut off first, so the
    new lines never glue onto a fragment (readers skip unparsable lines, so the fragment was never data)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+b") as f:
        size = f.seek(0, os.SEEK_END)
        if size:
            f.seek(max(0, size - (1 << 16)))
            tail = f.read()
            if not tail.endswith(b"\n"):
                cut = tail.rfind(b"\n")
                if cut < 0 and size > len(tail):  # a fragment longer than the tail window: scan the whole file
                    f.seek(0)
                    whole = f.read()
                    f.truncate(whole.rfind(b"\n") + 1)
                else:
                    f.truncate(size - len(tail) + cut + 1)
        f.seek(0, os.SEEK_END)
        f.write("".join(line + "\n" for line in lines).encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())


def check_environment(env, stamp: Mapping | None, what: str) -> None:
    """Raise :class:`EnvironmentConfigError` unless ``stamp`` (a checkpoint's, a manifest's ...) is ``env``'s."""
    try:
        env.check_stamp(stamp or {})
    except EnvironmentConfigError as exc:
        raise EnvironmentConfigError(f"{what}: {exc}") from None


# ------------------------------------------------------------------ games


class GameLog:
    """An evaluator that remembers every game: per deck (in order) and pair, the score going first and going second.
    Results are appended to ``path`` as they arrive, so a rerun of the same calls plays nothing it already has."""

    def __init__(self, evaluator, path: Path | None = None) -> None:
        self.evaluator, self.path = evaluator, path
        self.table: dict[str, dict[int, tuple[float, float]]] = {}
        self.played = 0  # games played by this object (not read back from the file)
        if path is not None and path.is_file():
            for line in path.read_text().splitlines():
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue  # a line cut by a crash
                row = self.table.setdefault(rec["deck"], {})
                for k, (a, b) in zip(rec["pairs"], rec["games"], strict=True):
                    row[int(k)] = (math.nan if a is None else a, math.nan if b is None else b)

    @property
    def games(self) -> int:
        return 2 * sum(len(r) for r in self.table.values())

    def play(self, jobs: Sequence[tuple[Deck, range]]) -> list[np.ndarray]:
        todo = []
        for deck, pairs in jobs:
            have = self.table.get(deck_key(deck), {})
            missing = [k for k in pairs if k not in have]
            if missing:
                todo.append((deck, missing))
        if todo:
            out = self.evaluator.play(todo, per_game=True)
            lines = []
            for (deck, ks), g in zip(todo, out, strict=True):
                g = np.asarray(g, dtype=float).reshape(len(ks), 2)
                row = self.table.setdefault(deck_key(deck), {})
                for k, (a, b) in zip(ks, g.tolist(), strict=True):
                    row[k] = (a, b)
                lines.append(json.dumps({"deck": deck_key(deck), "pairs": ks,
                                         "games": [[_finite(a), _finite(b)] for a, b in g.tolist()]}))  # fmt: skip
                self.played += 2 * len(ks)
            if self.path is not None:
                append_lines(self.path, lines)
        out = []
        for deck, pairs in jobs:
            row = self.table[deck_key(deck)] if len(pairs) else {}
            g = np.array([row[k] for k in pairs], dtype=float).reshape(len(pairs), 2)
            with np.errstate(invalid="ignore"), _quiet():
                out.append(np.nanmean(g, -1) if len(g) else np.zeros(0))
        return out

    def games_of(self, deck: Deck) -> dict[int, tuple[float, float]]:
        return self.table.get(deck_key(deck), {})

    def summary(self, deck: Deck) -> dict[str, float]:
        """Win rate (objective), going first and going second, over every pair ``deck`` played here."""
        g = np.array(list(self.games_of(deck).values()), dtype=float).reshape(-1, 2)
        with _quiet():
            return {"win_rate": float(np.nanmean(g)) if g.size else math.nan,
                    "win_rate_first": float(np.nanmean(g[:, 0])) if g.size else math.nan,
                    "win_rate_second": float(np.nanmean(g[:, 1])) if g.size else math.nan}  # fmt: skip

    def differences(self, child: Deck, base: Deck, pairs: Callable[[int], bool] | None = None) -> np.ndarray:
        """Per shared pair (all of them, or those ``pairs(k)`` selects): child minus base."""
        c, b = self.games_of(child), self.games_of(base)
        keys = [k for k in sorted(c.keys() & b.keys()) if pairs is None or pairs(k)]
        with _quiet():
            d = np.array([np.nanmean(c[k]) - np.nanmean(b[k]) for k in keys])
        return d[np.isfinite(d)]

    def pair_scores(self, deck: Deck) -> np.ndarray:
        with _quiet():
            s = np.array([np.nanmean(g) for g in self.games_of(deck).values()], dtype=float)
        return s[np.isfinite(s)]


class _quiet:
    def __enter__(self):
        import warnings

        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *exc):
        self._w.__exit__(*exc)


def _mean_se(d: np.ndarray, prior_pairs: float = 10.0) -> tuple[float, float]:
    """Mean paired difference and its standard error, the per-pair variance counting ``prior_pairs`` pseudo-pairs of
    SD_PAIR² (as the selection code does), so a short run of identical pairs cannot claim zero error."""
    n = len(d)
    if n == 0:
        return math.nan, math.nan
    ss = float(((d - d.mean()) ** 2).sum()) if n > 1 else 0.0
    var = (prior_pairs * SD_PAIR**2 + ss) / (prior_pairs + n - 1)
    return float(d.mean()), math.sqrt(var / n)


# ------------------------------------------------------------------ opponents


def opponent_mix(meta: Sequence[tuple[str, Deck, float]], *, nash: Mapping[str, float] | None = None,
                 decks: Mapping[str, Deck] | None = None, nash_share: float = 0.5) -> list[Opponent]:  # fmt: skip
    """Opponents drawn by ``nash_share`` × the Nash weights of the deck matchup matrix + (1 − ``nash_share``) × the
    environment's meta shares (each part normalized). ``nash`` maps matrix deck names to weights; a name resolves to
    a meta deck or to ``decks`` (e.g. evolved decks by id). Without a matrix (or with ``nash_share`` 0) the meta shares
    alone decide."""
    if not 0 <= nash_share <= 1:
        raise ValueError("nash_share must be in [0, 1]")
    meta_total = sum(s for _, _, s in meta)
    weights: dict[str, float] = {}
    lists: dict[str, Deck] = {}
    use_nash = bool(nash) and nash_share > 0 and sum(nash.values()) > 0
    for name, deck, share in meta:
        lists[name] = deck
        weights[name] = (1 - nash_share if use_nash else 1.0) * share / meta_total
    if use_nash:
        total = sum(nash.values())
        known = {n for n, _, _ in meta} | set(decks or {})
        unknown = sorted(n for n, w in nash.items() if w > 0 and n not in known)
        if unknown:
            raise ValueError(f"matchup matrix decks with Nash weight that are neither meta nor pool decks: {unknown}")
        for name, w in nash.items():
            if w > 0:
                lists.setdefault(name, (decks or {}).get(name))
                weights[name] = weights.get(name, 0.0) + nash_share * w / total
    return [Opponent(n, lists[n], w) for n, w in weights.items() if w > 0]


# ------------------------------------------------------------------ state


class Evolution:
    """The evolution's persistent state in ``directory``, bound to ``env`` (an :class:`ygorl.data.Environment`, or
    anything with ``stamp()`` / ``check_stamp()``). Every file carries the environment stamp; a file from another
    environment is an error. ``manifest`` defaults to ``directory / "manifest.json"``."""

    def __init__(self, directory: str | Path, env, *, manifest: str | Path | None = None,
                 grid: Mapping | None = None) -> None:  # fmt: skip
        self.dir = Path(directory)
        self.env = env
        self.stamp = dict(env.stamp())
        self.manifest_path = Path(manifest) if manifest is not None else self.dir / "manifest.json"
        self.manifest = self._load_manifest()
        self._added: set[str] = set()  # manifest ids this object added (the rest of the manifest is the file's)
        self.archive, self.archive_applied = self._load_archive(grid)
        self.model, self.calibration, self.signals_applied = self._load_signals()
        self.lineage_path = self.dir / "lineage.jsonl"

    # -- loading
    def _load_manifest(self) -> dict:
        if not self.manifest_path.is_file():
            return {"format": MANIFEST_FORMAT, "version": 1, "environment": self.stamp, "decks": []}
        data = json.loads(self.manifest_path.read_text())
        if data.get("format") != MANIFEST_FORMAT:
            raise ValueError(f"{self.manifest_path}: not a {MANIFEST_FORMAT} manifest")
        if "environment" in data or data.get("decks"):
            check_environment(self.env, data.get("environment"), f"manifest {self.manifest_path}")
        data["environment"] = self.stamp
        return data

    def _load_archive(self, grid) -> tuple[DeckArchive, dict]:
        path = self.dir / "archive.json"
        if not path.is_file():
            return DeckArchive(grid, stamp=self.stamp), {}
        data = json.loads(path.read_text())
        check_environment(self.env, data.get("environment"), f"archive {path}")
        return DeckArchive.from_dict(data), dict(data.get("applied", {}))

    def _load_signals(self) -> tuple[CardValueModel, Calibration, set[str]]:
        path = self.dir / "signals.json"
        if not path.is_file():
            return CardValueModel(), Calibration(SIGNAL_DEFAULTS), set()
        data = json.loads(path.read_text())
        if data.get("format") != SIGNALS_FORMAT:
            raise ValueError(f"{path}: not a {SIGNALS_FORMAT} file")
        check_environment(self.env, data.get("environment"), f"signal library {path}")
        return CardValueModel.from_dict(data["model"]), Calibration.from_dict(data["calibration"]), set(data["applied"])

    def check_checkpoint(self, stamp: Mapping | None, path: str | Path = "checkpoint") -> None:
        check_environment(self.env, stamp, f"checkpoint {path}")

    def warm_start(self, items: Sequence[tuple[str, Observation]]) -> int:
        """Add earlier paired observations (``ygorl.build.warmstart.load``) to the card-value model, each id once
        (a rerun or a resumed round adds nothing twice); saves the signal library. Returns how many were new."""
        new = 0
        for uid, obs in items:
            if uid in self.signals_applied:
                continue
            self.model.add(obs)
            self.signals_applied.add(uid)
            new += 1
        if new:
            self._save()
        return new

    # -- saving
    def _save(self) -> None:
        _write(self.dir / "signals.json", _json({"format": SIGNALS_FORMAT, "version": 1, "environment": self.stamp,
                                                 "model": self.model.to_dict(),
                                                 "calibration": self.calibration.to_dict(),
                                                 "applied": sorted(self.signals_applied)}))  # fmt: skip
        _write(self.dir / "archive.json", _json({**self.archive.to_dict(), "applied": self.archive_applied}))
        self._save_manifest()

    def _save_manifest(self) -> None:
        """Merge into the manifest as it is on disk now: someone (a person, the probation step) may have edited it
        during the round, and those edits win; only the decks this object added are put in (once)."""
        disk = self._load_manifest()
        have = {e["id"] for e in disk["decks"]}
        disk["decks"] += [e for e in self.manifest["decks"] if e["id"] in self._added and e["id"] not in have]
        self.manifest = disk
        _write(self.manifest_path, _json(disk))
        self._added.clear()  # on disk now: a later removal by someone else stands

    # -- queries
    def lineage(self) -> list[dict]:
        if not self.lineage_path.is_file():
            return []
        out = []
        for line in self.lineage_path.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out

    def pool(self, *statuses: str) -> list[Parent]:
        """Manifest decks with the given statuses (default: probation and active), as parents."""
        out = []
        for e in self.manifest["decks"]:
            if e.get("status", "probation") in (statuses or LIVE):
                deck = parse_ydk((self.manifest_path.parent / e["file"]).read_text(), name=e["id"])
                out.append(Parent(e["id"], deck, str(e.get("type", ""))))
        return out

    def pick_parents(self, candidates: Sequence[Parent], k: int) -> list[Parent]:
        """The ``k`` candidates used least often as parents so far (ties: the higher archive objective, then the
        given order): every pool deck gets its turn before any gets a second."""
        uses: dict[str, int] = {}
        for r in {(r["round"], r["parent"]["id"]) for r in self.lineage()}:
            uses[r[1]] = uses.get(r[1], 0) + 1
        best = {e["id"]: e["objective"] for e in self.archive.elites()}
        order = sorted(range(len(candidates)),
                       key=lambda i: (uses.get(candidates[i].id, 0), -best.get(candidates[i].id, -math.inf), i))  # fmt: skip
        return [candidates[i] for i in order[:k]]

    def rounds(self) -> list[int]:
        return sorted(int(p.name) for p in (self.dir / "rounds").glob("[0-9]*") if (p / "round.json").is_file())

    def _round_state(self, n: int) -> dict:
        return json.loads((self.dir / "rounds" / f"{n:04d}" / "round.json").read_text())

    # -- one round
    def run_round(self, parents: Sequence[Parent], lab: Lab, config: RoundConfig, opponents: Sequence[Opponent], *,
                  log: Callable[[str], None] | None = None) -> dict:  # fmt: skip
        """Run (or resume) one round; returns its report. An unfinished round is resumed with the parents, settings
        and opponents it was started with (the arguments are then ignored, with a note in the log)."""
        say = log or (lambda m: None)
        done_rounds = self.rounds()
        last = self._round_state(done_rounds[-1]) if done_rounds else None
        if last is not None and not last.get("done"):
            n, state = last["round"], last
            say(f"resuming round {n} ({len(state['results'])} of {len(state['parents'])} parents done)")
            parents = [Parent(p["id"], _deck_from(p["deck"], p["id"]), p["type"]) for p in state["parents"]]
            config = RoundConfig(**state["config"])
            opponents = [Opponent(o["name"], _deck_from(o["deck"], o["name"]), o["weight"]) for o in state["opponents"]]
            if lab.checkpoint and dict(lab.checkpoint) != state["checkpoint"]:
                raise ValueError(f"round {n} was started with checkpoint {state['checkpoint']}: finish it with the same "
                                 "checkpoint (its logged games were played by that policy)")  # fmt: skip
        else:
            n = (done_rounds[-1] + 1) if done_rounds else 1
            state = {"format": ROUND_FORMAT, "round": n, "environment": self.stamp, "started": _now(),
                     "checkpoint": dict(lab.checkpoint), "config": asdict(config),
                     "parents": [{"id": p.id, "type": p.type, "deck": _deck_json(p.deck)} for p in parents],
                     "opponents": [{"name": o.name, "weight": o.weight, "deck": _deck_json(o.deck)} for o in opponents],
                     "results": [], "done": False}  # fmt: skip
            state["mix"] = mix_fingerprint(opponents, state["checkpoint"])
        if config.l0 not in ("off", "shadow", "on"):
            raise ValueError("l0 must be off, shadow or on")
        if config.l0 != "off" and lab.screen is None:
            raise ValueError(f"l0={config.l0} needs a screen")
        if config.rules > 0 and lab.rules is None:
            raise ValueError("rule children need the association rules (Lab.rules)")
        if config.learned > 0 and lab.deck_model is None:
            raise ValueError("learned children need a deck model (Lab.deck_model)")
        if config.evaluation not in ("thompson", "factorial"):
            raise ValueError("evaluation must be thompson or factorial")
        rdir = self.dir / "rounds" / f"{n:04d}"
        _write(rdir / "round.json", _json(state))
        spent = sum(r["games"] for r in state["results"])
        for r in state["results"]:  # a crash may have cut the side effects short
            self._apply(state, r)
        for i in range(len(state["results"]), len(parents)):
            if config.budget is not None and spent >= config.budget:
                say(f"budget of {config.budget} games spent: {len(parents) - i} parents left for the next round")
                state["budget_stop"] = True
                break
            r = self._parent(n, i, parents[i], lab, config, opponents, rdir, say)
            spent += r["games"]
            state["results"].append(r)
            _write(rdir / "round.json", _json(state))
            self._apply(state, r)
        state["done"], state["finished"] = True, _now()
        report = self._report(state)
        state["report"] = report
        _write(rdir / "report.json", _json(report))
        _write(rdir / "report.txt", format_report(report))
        _write(rdir / "round.json", _json(state))
        return report

    def _parent(self, n: int, i: int, parent: Parent, lab: Lab, config: RoundConfig, opponents: Sequence[Opponent],
                rdir: Path, say) -> dict:  # fmt: skip
        """Play one parent's part of round ``n``; returns its JSON-ready result (no side effects but games)."""
        ev = lab.evaluator(derive_seed(config.seed, n, i), opponents)
        games = GameLog(ev, rdir / f"games-{i}.jsonl")
        rng = np.random.default_rng([config.seed, n, i])
        base = parent.deck
        signals: dict[str, dict[int, float]] = {}
        prior: dict[int, float] = {}
        base_scores: list[float] = []
        if config.diagnose_pairs > 0:
            pairs = range(config.diagnose_pairs)
            base_scores = games.play([(base, pairs)])[0].tolist()
            if hasattr(ev, "opening_hands"):
                effects = opening_effects(ev.opening_hands(base, pairs), base_scores, base.main)
                signals["opening_effect"] = {e.card: e.effect for e in effects if math.isfinite(e.effect)}
            prior = combine_prior(signals, self.calibration)
            if prior:  # also in _apply (from the result), so a crash cannot lose it
                self.model.set_prior({**self.model.prior, **prior})
        engine = set(lab.engine(base))
        unproven = {c for c in engine if self.model.evidence(c, parent.type) < config.engine_evidence}
        protected = set(lab.protected(base)) | unproven
        cold = config.cold_candidates > 0 and self.model.type_evidence(parent.type) < config.cold_min_obs
        pool = lab.pool(parent)
        if config.evaluation == "factorial":  # #152: its own path (no cold start, crossover, rules, L0 or race)
            return self._factorial_parent(n, i, parent, lab, config, games, rng, signals, prior, pool, protected,
                                          engine, say)  # fmt: skip
        if cold:  # single swaps only: half informed by the model (warm start), half random
            half = config.cold_candidates // 2
            children = informed_children(base, parent.type, self.model, pool, legal=lab.legal, is_extra=lab.is_extra,
                                         rng=rng, informed=half, explore=config.cold_candidates - half, max_bundle=1,
                                         protected=protected, engine=engine)  # fmt: skip
        else:
            children = informed_children(base, parent.type, self.model, pool, legal=lab.legal, is_extra=lab.is_extra,
                                         rng=rng, informed=config.informed, explore=config.explore,
                                         max_bundle=config.max_bundle, protected=protected, engine=engine)  # fmt: skip
        mates: dict[int, dict] = {}  # child index -> its second parent (crossover children)
        if config.crossover > 0 and not cold:
            crossed = crossover_children(base, self.archive, descriptors=lab.descriptors(base), legal=lab.legal,
                                         rng=np.random.default_rng([config.seed, n, i, 2]), n=config.crossover,
                                         protected=protected, exclude=[c.deck for c in children])  # fmt: skip
            for c, m in crossed:
                mates[len(children)] = {**m.to_dict(), "key": deck_key(m.deck)}
                children.append(c)
        if config.rules > 0:  # own random stream: the other children stay as they were without it
            children += rule_children(base, parent.type, lab.rules, self.model, legal=lab.legal, is_extra=lab.is_extra,
                                      rng=np.random.default_rng([config.seed, n, i, 3]), children=config.rules,
                                      max_bundle=1 if cold else config.max_bundle, protected=protected,
                                      avoid=[c.deck for c in children], allowed=pool, engine=engine)  # fmt: skip
        vm_weight = self._value_weight(lab, config)
        if config.learned > 0:  # legality and protected cards only: no engine protection, no addition pool (#150)
            drawn = learned_children(base, parent.type, lab.deck_model, self.model,
                                     lab.card_pool if lab.card_pool is not None else pool, legal=lab.legal,
                                     is_extra=lab.is_extra, rng=np.random.default_rng([config.seed, n, i, 5]),
                                     children=config.learned * self._oversample(config, vm_weight),
                                     max_bundle=1 if cold else config.max_bundle, avoid=[c.deck for c in children],
                                     protected=lab.protected(base), removal=config.learned_removal)  # fmt: skip
            children += self._rank_learned(lab, base, drawn, config.learned, vm_weight,
                                           np.random.default_rng([config.seed, n, i, 6]))  # fmt: skip
        predicted = [self.model.gain([e.into for e in c.edits], [e.out for e in c.edits], parent.type)
                     for c in children]  # fmt: skip
        values = self._value_predictions(lab, base, [c.edits for c in children])
        priors = [blend_prior(p, v, vm_weight) for p, v in zip(predicted, values, strict=True)]
        screen_l0 = [None] * len(children)
        if config.l0 != "off" and children:
            screen_l0 = [float(x) for x in lab.screen(ev, base, [c.deck for c in children])]
        screened = [x is not None and x < config.l0_min for x in screen_l0]
        keep = [j for j in range(len(children)) if not (config.l0 == "on" and screened[j])]
        say(f"parent {parent.id} ({parent.type}): {len(children)} children"
            + (" (cold start: single swaps, screened)" if cold else "")
            + f", {len(engine)} engine cards ({len(unproven)} protected)"
            + (f", L0 would remove {sum(screened)}" if config.l0 != "off" else ""))  # fmt: skip
        candidates = len(keep)  # every child the search chooses among, a screened-out one too
        screen_rows: dict[int, dict] = {}
        if cold and len(keep) > config.cold_keep:
            s = screen(base, [children[j].deck for j in keep], games, pairs=config.batch, keep=config.cold_keep,
                       base_scores=base_scores, tiebreak=[priors[j][0] for j in keep],
                       rng=np.random.default_rng([config.seed, n, i, 4]))  # fmt: skip
            for r, j in enumerate(keep):
                screen_rows[j] = {"pairs": s.pairs, "diff": _finite(s.diffs[r]), "kept": r in s.kept}
            base_scores = s.base_scores
            say(f"screen: {len(keep)} children at {s.pairs} pairs, kept "
                + ", ".join(f"c{keep[r]} {s.diffs[r]:+.3f}" for r in s.kept))  # fmt: skip
            keep = [keep[r] for r in s.kept]
        race = top_two_thompson(base, [children[j].deck for j in keep], games,
                                prior=[(m, max(sd, 1e-3)) for m, sd in (priors[j] for j in keep)],
                                batch=config.batch, max_pairs=config.max_pairs,
                                batches_per_round=config.batches_per_round, confidence=config.confidence,
                                base_scores=base_scores, min_pairs=config.min_confident_pairs,
                                multiplicity=max(candidates, 1) if config.multiplicity else 1,
                                rng=np.random.default_rng([config.seed, n, i, 1]), log=say)  # fmt: skip
        chosen = keep[race.best] if race.best is not None else None
        validation = None
        if chosen is not None:
            v = sequential_validate(base, children[chosen].deck, games, look=config.look, cap=config.cap,
                                    alpha=config.alpha, min_effect=config.min_effect, offset=VALIDATION_OFFSET,
                                    log=say)  # fmt: skip
            last = v.looks[-1] if v.looks else None
            validation = {"pairs": v.pairs, "decision": v.decision, "accepted": v.accepted,
                          "diff": last.mean if last else None, "lower": last.lower if last else None,
                          "upper": last.upper if last else None}  # fmt: skip
        arms = {keep[a.index]: a for a in race.arms}
        rows = []
        for j, c in enumerate(children):
            a = arms.get(j)
            d = games.differences(c.deck, base)
            mean, se = _mean_se(d)
            learned = []  # the non-adaptive data only (module docstring): what the signal library learns from
            if a is not None or j in screen_rows:  # every screened or raced child played the first batch
                first = games.differences(c.deck, base, lambda k: k < config.batch)
                learned.append({"source": "first_batch", "pairs": len(first), **_learned(first)})
            if j == chosen:
                fresh = games.differences(c.deck, base, lambda k: k >= VALIDATION_OFFSET)
                learned.append({"source": "validation", "pairs": len(fresh), **_learned(fresh)})
            search = None
            if a is not None:
                ds = np.asarray(race.scores[a.index]) - np.asarray(race.base_scores[: a.pairs])
                _, se_search = _mean_se(ds[np.isfinite(ds)])
                h = 1.96 * se_search
                search = {"pairs": a.pairs, "diff": _finite(a.observed), "ci": [_finite(a.observed - h),
                          _finite(a.observed + h)], "posterior": [a.mean, a.sd], "p_positive": a.p_positive,
                          "p_best": a.p_best}  # fmt: skip
            rows.append({"child": f"r{n:04d}-p{i}-c{j}", "kind": c.kind, "generator": generator(c.kind),
                         "mate": mates.get(j),
                         "edits": [{"out": e.out, "into": e.into, "section": e.section} for e in c.edits],
                         "deck": _deck_json(c.deck), "predicted": {"mean": predicted[j][0], "sd": predicted[j][1]},
                         "value_model": _pair(values[j]),
                         "prior": {"mean": priors[j][0], "sd": priors[j][1], "weight": vm_weight},
                         "l0": screen_l0[j], "screened": screened[j], "screen": screen_rows.get(j),
                         "search": search,
                         "validation": validation if j == chosen else None,
                         "all_pairs": {"pairs": len(d), "diff": _finite(mean), "stderr": _finite(se),
                                       "note": "search + validation pairs: selection-biased, report only"},
                         "learned": [x for x in learned if x["pairs"]],
                         "games": 2 * len(games.games_of(c.deck)),
                         "accepted": bool(j == chosen and validation and validation["accepted"]),
                         "stats": {k: _finite(v) for k, v in games.summary(c.deck).items()},
                         "objective": _objective(games.pair_scores(c.deck), config),
                         "descriptors": _descriptors(lab, c.deck, games)})  # fmt: skip
        return {"index": i, "parent": {"id": parent.id, "type": parent.type, "key": deck_key(base)},
                "signals": {k: {str(c): v for c, v in s.items()} for k, s in signals.items()},
                "prior": {str(c): v for c, v in prior.items()},
                "parent_stats": {k: _finite(v) for k, v in games.summary(base).items()},
                "parent_objective": _objective(games.pair_scores(base), config),
                "parent_descriptors": _descriptors(lab, base, games),
                "parent_games": 2 * len(games.games_of(base)), "stop": race.stop, "search_pairs": race.pairs,
                "cold_start": cold, "candidates": candidates, "stop_threshold": race.threshold,
                "engine": sorted(engine), "protected": sorted(protected),
                "chosen": rows[chosen]["child"] if chosen is not None else None, "children": rows,
                "games": games.games, "played": games.played}  # fmt: skip

    def _factorial_parent(self, n: int, i: int, parent: Parent, lab: Lab, config: RoundConfig, games: GameLog,
                          rng: np.random.Generator, signals: dict, prior: dict, pool: Sequence[int],
                          protected: set[int], engine: set[int], say) -> dict:  # fmt: skip
        """Evaluation by fractional factorial (#152, ``ygorl.build.factorial``): ``factorial_k`` single edits
        (``informed_children`` with bundles of one: the informed ones first, then the explore ones; with
        ``config.learned``, the learned single swaps of the masked deck model before both; each takes its own card copy) in one design on pairs ``0 .. factorial_pairs − 1`` (run 0 is the parent: its diagnosis pairs are
        reused). The edits with a positive main effect together (the smallest effects left out until legal) are the
        chosen child, validated on fresh pairs as a raced child is. The signal library learns each main effect as a
        single-edit observation (every variant plays the same fixed pairs: non-adaptive), not the variants' own
        differences; every variant is a lineage row carrying the design and the effect estimates."""
        base, k = parent.deck, config.factorial_k
        singles = informed_children(base, parent.type, self.model, pool, legal=lab.legal, is_extra=lab.is_extra,
                                    rng=rng, informed=k, explore=k, max_bundle=1, protected=protected, engine=engine)  # fmt: skip
        order = [c for c in singles if c.kind == "informed"] + [c for c in singles if c.kind == "explore"]
        vm_weight = self._value_weight(lab, config)
        if config.learned > 0:  # #150: learned candidates go in the design first
            drawn = learned_children(base, parent.type, lab.deck_model, self.model,
                                     lab.card_pool if lab.card_pool is not None else pool, legal=lab.legal,
                                     is_extra=lab.is_extra, rng=np.random.default_rng([config.seed, n, i, 5]),
                                     children=config.learned * self._oversample(config, vm_weight), max_bundle=1,
                                     protected=lab.protected(base), removal=config.learned_removal)  # fmt: skip
            order = self._rank_learned(lab, base, drawn, config.learned, vm_weight,
                                       np.random.default_rng([config.seed, n, i, 6])) + order  # fmt: skip
        kind_of: dict = {}
        for c in order:
            kind_of.setdefault(c.edits[0], c.kind)
        edits = [pl.edit for pl in place(base, [c.edits[0] for c in order])[0]][:k]
        fid = f"r{n:04d}-p{i}-factorial"
        rows: list[dict] = []
        fac, chosen, validation = None, None, None
        if edits:
            fr = evaluate_edits(base, edits, games, legal=lab.legal, pairs=config.factorial_pairs)
            placed = fr.plan.placed
            main = fr.main()
            predicted = [self.model.gain([pl.edit.into], [pl.edit.out], parent.type) for pl in placed]
            values = self._value_predictions(lab, base, [[pl.edit] for pl in placed])
            good = sorted((e.term[0] for e in main if e.effect > 0), key=lambda j: -main[j].effect)
            while good and not lab.legal(variant(base, [placed[j] for j in good])):
                good.pop()
            fac = {"id": fid, **fr.to_dict(), "kinds": [kind_of[pl.edit] for pl in placed],
                   "predicted": [{"mean": m, "sd": sd} for m, sd in predicted],
                   "value_model": [_pair(v) for v in values], "value_model_weight": vm_weight,
                   "chosen_edits": [LETTERS[j] for j in sorted(good)],
                   "learned": [{"source": "factorial_main", "edit": e.term[0], "into": placed[e.term[0]].edit.into,
                                "out": placed[e.term[0]].edit.out, "section": placed[e.term[0]].edit.section,
                                "diff": e.effect, "stderr": e.stderr, "predicted": predicted[e.term[0]][0],
                                "value_model": None if values[e.term[0]] is None else values[e.term[0]][0]}
                               for e in main if math.isfinite(e.effect) and e.stderr > 0]}  # fmt: skip
            say(f"parent {parent.id} ({parent.type}): factorial {', '.join(fr.design.generators) or 'full'} over "
                f"{fr.design.k} edits, {fr.design.runs} runs × {config.factorial_pairs} pairs; main effects "
                + ", ".join(f"{e.label} {e.effect:+.3f}±{e.stderr:.3f}" for e in main)
                + (f"; {len(fr.plan.repairs)} repairs" if fr.plan.repairs else ""))  # fmt: skip
            variants = [(r, [j for j in range(fr.design.k) if fr.plan.levels[r, j] > 0], d)
                        for r, d in enumerate(fr.plan.decks)]  # fmt: skip
            if good:
                best = variant(base, [placed[j] for j in good])
                if deck_key(best) not in {deck_key(d) for _, _, d in variants}:
                    variants.append((None, sorted(good), best))
            seen = {deck_key(base)}
            for r, on, deck in variants:
                if deck_key(deck) in seen:
                    continue  # the parent (run 0), or a repaired run equal to another
                seen.add(deck_key(deck))
                c_edits = [placed[j].edit for j in on]
                j = len(rows)
                is_best = bool(good) and deck_key(deck) == deck_key(variant(base, [placed[x] for x in good]))
                if is_best:
                    chosen = j
                rows.append({"child": f"r{n:04d}-p{i}-c{j}", "kind": "factorial", "generator": "factorial",
                             "mate": None, "edits": [{"out": e.out, "into": e.into, "section": e.section}
                                                     for e in c_edits],
                             "deck": _deck_json(deck), "l0": None, "screened": False, "screen": None,
                             "predicted": dict(zip(("mean", "sd"), self.model.gain([e.into for e in c_edits],
                                                                                   [e.out for e in c_edits],
                                                                                   parent.type), strict=True)),
                             "factorial": {"run": r, "levels": [1 if x in on else -1 for x in range(fr.design.k)],
                                           **{key: fac[key] for key in ("id", "design", "edits", "effects",
                                                                        "chosen_edits", "repairs", "dropped")}}})  # fmt: skip
        if chosen is not None:
            v = sequential_validate(base, _deck_from(rows[chosen]["deck"]), games,
                                    look=config.look, cap=config.cap, alpha=config.alpha,
                                    min_effect=config.min_effect, offset=VALIDATION_OFFSET, log=say)  # fmt: skip
            last = v.looks[-1] if v.looks else None
            validation = {"pairs": v.pairs, "decision": v.decision, "accepted": v.accepted,
                          "diff": last.mean if last else None, "lower": last.lower if last else None,
                          "upper": last.upper if last else None}  # fmt: skip
        for j, row in enumerate(rows):
            deck = _deck_from(row["deck"])
            d = games.differences(deck, base)
            mean, se = _mean_se(d)
            design_d = games.differences(deck, base, lambda q: q < config.factorial_pairs)
            dm, dse = _mean_se(design_d)
            learned = []
            if j == chosen:
                fresh = games.differences(deck, base, lambda q: q >= VALIDATION_OFFSET)
                learned.append({"source": "validation", "pairs": len(fresh), **_learned(fresh)})
            row.update({"search": {"pairs": len(design_d), "diff": _finite(dm),
                                   "ci": [_finite(dm - 1.96 * dse), _finite(dm + 1.96 * dse)]}
                        if len(design_d) else None,
                        "validation": validation if j == chosen else None,
                        "all_pairs": {"pairs": len(d), "diff": _finite(mean), "stderr": _finite(se),
                                      "note": "design + validation pairs: selection-biased, report only"},
                        "learned": [x for x in learned if x["pairs"]], "games": 2 * len(games.games_of(deck)),
                        "accepted": bool(j == chosen and validation and validation["accepted"]),
                        "stats": {key: _finite(val) for key, val in games.summary(deck).items()},
                        "objective": _objective(games.pair_scores(deck), config),
                        "descriptors": _descriptors(lab, deck, games)})  # fmt: skip
        return {"index": i, "parent": {"id": parent.id, "type": parent.type, "key": deck_key(base)},
                "evaluation": "factorial", "factorial": fac,
                "signals": {key: {str(c): val for c, val in sig.items()} for key, sig in signals.items()},
                "prior": {str(c): val for c, val in prior.items()},
                "parent_stats": {key: _finite(val) for key, val in games.summary(base).items()},
                "parent_objective": _objective(games.pair_scores(base), config),
                "parent_descriptors": _descriptors(lab, base, games),
                "parent_games": 2 * len(games.games_of(base)), "stop": "factorial",
                "search_pairs": sum(r["search"]["pairs"] for r in rows if r["search"]),
                "cold_start": False, "candidates": len(edits), "stop_threshold": None,
                "engine": sorted(engine), "protected": sorted(protected),
                "chosen": rows[chosen]["child"] if chosen is not None else None, "children": rows,
                "games": games.games, "played": games.played}  # fmt: skip

    def _value_weight(self, lab: Lab, config: RoundConfig) -> float:
        """The edit-value model's weight in the prior and the ranking (#151): ``config.value_model_weight`` until the
        calibration table has ``min_pairs`` of its (predicted, first batch) pairs, then their Spearman when its 95%
        interval excludes 0, else 0 (like every signal). 0 without a value model."""
        if lab.value_model is None:
            return 0.0
        self.calibration.defaults["value_model"] = float(config.value_model_weight)
        return max(float(self.calibration.weight("value_model")), 0.0)

    @staticmethod
    def _oversample(config: RoundConfig, weight: float) -> int:
        return max(int(config.value_model_oversample), 1) if weight > 0 else 1

    @staticmethod
    def _value_predictions(lab: Lab, base: Deck, edits: Sequence[Sequence]) -> list[tuple[float, float] | None]:
        if lab.value_model is None or not edits:
            return [None] * len(edits)
        return [(float(m), float(sd)) for m, sd in lab.value_model.predict(base, [list(e) for e in edits])]

    def _rank_learned(self, lab: Lab, base: Deck, drawn: list, keep: int, weight: float,
                      rng: np.random.Generator) -> list:  # fmt: skip
        """The ``keep`` learned candidates to evaluate: with a value model of positive weight, the best by one
        Thompson draw from its predictions (mean + sd · z); otherwise the first ``keep`` drawn (unchanged)."""
        if weight <= 0 or len(drawn) <= keep:
            return drawn[:keep]
        pred = self._value_predictions(lab, base, [c.edits for c in drawn])
        draw = np.array([m + sd * z for (m, sd), z in zip(pred, rng.standard_normal(len(pred)), strict=True)])
        return [drawn[j] for j in np.argsort(-draw, kind="stable")[:keep]]

    def _apply(self, state: dict, r: dict) -> None:
        """A parent's side effects, idempotent per child id: signal library, archive, manifest, lineage."""
        n = state["round"]
        pinfo = next(p for p in state["parents"] if p["id"] == r["parent"]["id"])
        parent = Parent(pinfo["id"], _deck_from(pinfo["deck"], pinfo["id"]), pinfo["type"])
        if r["prior"]:
            self.model.set_prior({**self.model.prior, **{int(c): v for c, v in r["prior"].items()}})
        opening = {int(c): v for c, v in r["signals"].get("opening_effect", {}).items()}
        for row in r["children"]:
            into = tuple(e["into"] for e in row["edits"])
            out = tuple(e["out"] for e in row["edits"])
            for m in row["learned"]:
                uid = f"{row['child']}/{m['source']}"
                if uid in self.signals_applied or m["diff"] is None or not m["stderr"]:
                    continue
                self.model.add(Observation(parent.type, into, out, m["diff"], m["stderr"]))
                if m["source"] == "first_batch":  # one (predicted, measured) pair per child, taken before selection
                    self.calibration.record("model_gain", row["predicted"]["mean"], m["diff"])
                    if row.get("value_model"):  # #151: the edit-value model's prediction, calibrated the same way
                        self.calibration.record("value_model", row["value_model"]["mean"], m["diff"])
                    if opening:  # cards swapped in were never in this deck's openings: count them as average (0)
                        self.calibration.record("opening_effect", sum(opening.get(c, 0.0) for c in into)
                                                - sum(opening.get(c, 0.0) for c in out), m["diff"])  # fmt: skip
                self.signals_applied.add(uid)
        for m in (r.get("factorial") or {}).get("learned", []):  # #152: main effects as single-edit observations
            uid = f"{r['factorial']['id']}/{LETTERS[m['edit']]}"
            if uid in self.signals_applied:
                continue
            self.model.add(Observation(parent.type, (m["into"],), (m["out"],), m["diff"], m["stderr"]))
            self.calibration.record("model_gain", m["predicted"], m["diff"])
            if m.get("value_model") is not None:
                self.calibration.record("value_model", m["value_model"], m["diff"])
            self.signals_applied.add(uid)
        entries = [(f"r{n:04d}-p{r['index']}", parent.id, parent.deck, r["parent_objective"], r["parent_descriptors"])]
        entries += [(row["child"], self._manifest_id(row) if row["accepted"] else row["child"], _deck_from(row["deck"]),
                     row["objective"], row["descriptors"]) for row in r["children"]]  # fmt: skip
        for key, did, deck, objective, desc in entries:
            if key in self.archive_applied:
                continue
            if objective["lower"] is None:  # too few pairs to score
                self.archive_applied[key] = {"status": "too_few_pairs", "cell": None, "replaced": None}
                continue
            desc = {k: math.nan if v is None else v for k, v in desc.items()}
            adm = self.archive.add(did, deck, objective["lower"], desc, mix=state["mix"])
            self.archive_applied[key] = {"status": adm.status, "cell": adm.cell, "replaced": adm.replaced}
        for row in r["children"]:
            if row["accepted"]:
                self._admit(n, parent, row)
        self._save()
        have = {rec["child"] for rec in self.lineage()}
        lines = [json.dumps(self._lineage(state, r, row), ensure_ascii=False) for row in r["children"]
                 if row["child"] not in have]  # fmt: skip
        if lines:
            append_lines(self.lineage_path, lines)

    @staticmethod
    def _manifest_id(row: dict) -> str:
        return f"evo-{row['child']}"

    def _admit(self, n: int, parent: Parent, row: dict) -> None:
        mid = self._manifest_id(row)
        self._added.add(mid)
        if any(e["id"] == mid for e in self.manifest["decks"]):
            return
        deck = replace(_deck_from(row["deck"]), name=mid)
        file = f"decks/{mid}.ydk"
        _write(self.manifest_path.parent / file, deck.to_ydk())
        self.manifest["decks"].append({"id": mid, "file": file, "status": "probation", "weight": 1.0,
                                       "parent": parent.id, "mate": (row.get("mate") or {}).get("id"),
                                       "type": parent.type, "round": n, "lineage": row["child"],
                                       "edits": row["edits"], "diff": row["validation"]["diff"],
                                       "descriptors": row["descriptors"]})  # fmt: skip

    def _lineage(self, state: dict, r: dict, row: dict) -> dict:
        adm = self.archive_applied.get(row["child"], {})
        return {"child": row["child"], "round": state["round"], "time": _now(),
                "parent": {"id": r["parent"]["id"], "type": r["parent"]["type"], "key": r["parent"]["key"]},
                "kind": row["kind"], "generator": row.get("generator", generator(row["kind"])),
                "screen": row.get("screen"), "factorial": row.get("factorial"),
                "mate": row.get("mate"), "edits": row["edits"], "key": deck_key(_deck_from(row["deck"])),
                "predicted": row["predicted"], "value_model": row.get("value_model"), "prior": row.get("prior"),
                "l0": row["l0"], "screened": row["screened"], "search": row["search"],
                "validation": row["validation"], "all_pairs": row["all_pairs"], "learned": row["learned"],
                "games": row["games"],
                "checkpoint": state["checkpoint"], "environment": state["environment"], "accepted": row["accepted"],
                "manifest_id": self._manifest_id(row) if row["accepted"] else None,
                "archive": {k: adm.get(k) for k in ("status", "cell", "replaced")}}  # fmt: skip

    def _report(self, state: dict) -> dict:
        results = state["results"]
        children = [c for r in results for c in r["children"]]
        games = sum(r["games"] for r in results)
        accepted = sum(c["accepted"] for c in children)
        cum_games, cum_accepted = games, accepted
        for m in self.rounds():
            if m != state["round"]:
                rep = self._round_state(m).get("report") or {}
                cum_games += rep.get("games", 0)
                cum_accepted += rep.get("accepted", 0)
        admitted = [self.archive_applied.get(c["child"], {}).get("status") for c in children]
        return {"round": state["round"], "environment": state["environment"], "checkpoint": state["checkpoint"],
                "started": state["started"], "finished": state.get("finished"), "parents": len(results),
                "parents_planned": len(state["parents"]), "budget_stop": bool(state.get("budget_stop")),
                "children": len(children), "l0": state["config"]["l0"],
                "l0_would_remove": sum(bool(c["screened"]) for c in children),
                "games_per_child": {c["child"]: c["games"] for c in children},
                "per_parent": [{"parent": r["parent"]["id"], "children": len(r["children"]), "stop": r["stop"],
                                "cold_start": bool(r.get("cold_start")),
                                "evaluation": r.get("evaluation", "thompson"),
                                "effects": (r.get("factorial") or {}).get("effects"),
                                "search_pairs": r["search_pairs"], "parent_games": r["parent_games"],
                                "chosen": r["chosen"], "games": r["games"],
                                "validation": next((c["validation"] for c in r["children"] if c["validation"]), None),
                                "accepted": [c["child"] for c in r["children"] if c["accepted"]]}
                               for r in results],
                "accepted": accepted, "games": games,
                "generators": {"round": _by_generator(children),
                               "cumulative": _by_generator([c for c in self.lineage() if c["round"] != state["round"]]
                                                           + children)},
                "cumulative": {"games": cum_games, "accepted": cum_accepted,
                               "games_per_accepted": cum_games / cum_accepted if cum_accepted else None},
                "archive": {**self.archive.summary(), "stale": self.archive.stale(state["mix"]),
                            "admitted_this_round": sum(s in ("new", "improved", "refreshed") for s in admitted)},
                "pool": pool_diversity([p.deck for p in self.pool()]),
                "calibration": {k: {**v, "ci": [_finite(x) for x in v["ci"]], "spearman": _finite(v["spearman"])}
                                for k, v in self.calibration.report().items()}}  # fmt: skip


def blend_prior(card: tuple[float, float], value: tuple[float, float] | None, weight: float) -> tuple[float, float]:
    """The Thompson prior of a child (#151): the card-value model's (mean, sd) and the edit-value model's, mixed by
    the value model's calibrated weight ``w``: mean = w·value + (1 − w)·card, variance = w·sd_value² + (1 − w)·sd_card².
    Weight 0 (or no value model) is the card-value model alone."""
    if value is None or weight <= 0:
        return card
    w = min(float(weight), 1.0)
    return (w * value[0] + (1 - w) * card[0], math.sqrt(w * value[1] ** 2 + (1 - w) * card[1] ** 2))


def _pair(v: tuple[float, float] | None) -> dict | None:
    return None if v is None else {"mean": v[0], "sd": v[1]}


def mix_fingerprint(opponents: Sequence[Opponent], checkpoint: Mapping) -> str:
    """Identity of what an archive objective was scored against: the opponents (names, weights, lists) and the
    policy (its checkpoint's hash, else its path)."""
    key = [[o.name, round(float(o.weight), 12), deck_key(o.deck)] for o in opponents]
    pilot = checkpoint.get("sha256") or checkpoint.get("path")
    return hashlib.sha1(json.dumps([key, pilot, checkpoint.get("opponent")]).encode()).hexdigest()[:16]


def _objective(scores: np.ndarray, config: RoundConfig, prior_pairs: float = 10.0) -> dict:
    """The archive objective: the win rate's one-sided lower bound (per-pair variance with ``prior_pairs`` pseudo-pairs
    of 0.125, two independent even games), None below ``config.archive_min_pairs`` pairs."""
    n = len(scores)
    if n == 0:
        return {"pairs": 0, "win_rate": None, "lower": None}
    ss = float(((scores - scores.mean()) ** 2).sum())
    var = (prior_pairs * 0.125 + ss) / (prior_pairs + n - 1)
    lower = float(scores.mean()) - config.archive_z * math.sqrt(var / n)
    return {"pairs": n, "win_rate": float(scores.mean()), "lower": lower if n >= config.archive_min_pairs else None}


def _by_generator(children: Sequence[Mapping]) -> dict[str, dict]:
    """Per generator: children evaluated, chosen for validation, accepted, the accepted share, and the candidate
    quality: the mean first-batch paired difference (non-adaptive, #145) over the children that have one (lineage
    records or round rows)."""
    out: dict[str, dict] = {}
    first: dict[str, list[float]] = {}
    for c in children:
        name = c.get("generator") or generator(c["kind"])
        g = out.setdefault(name, {"children": 0, "chosen": 0, "accepted": 0})
        g["children"] += 1
        g["chosen"] += c.get("validation") is not None
        g["accepted"] += bool(c["accepted"])
        first.setdefault(name, []).extend(m["diff"] for m in c.get("learned", [])
                                          if m["source"] == "first_batch" and m.get("diff") is not None)  # fmt: skip
    for name, g in out.items():
        g["accepted_rate"] = g["accepted"] / g["children"]
        f = first.get(name, [])
        g["first_batch_mean"] = float(np.mean(f)) if f else None
        g["first_batch_children"] = len(f)
    return dict(sorted(out.items()))


def _learned(d: np.ndarray) -> dict:
    mean, se = _mean_se(d)
    return {"diff": _finite(mean), "stderr": _finite(se)}


def _descriptors(lab: Lab, deck: Deck, games: GameLog) -> dict[str, float | None]:
    """The archive descriptors of ``deck``: the two win rates from its games, the rest from the deck list."""
    s = games.summary(deck)
    return {"win_rate_first": _finite(s["win_rate_first"]), "win_rate_second": _finite(s["win_rate_second"]),
            **{k: _finite(float(v)) for k, v in lab.descriptors(deck).items()}}  # fmt: skip


def pool_diversity(decks: Sequence[Deck]) -> dict:
    """Size and mean pairwise card distance (1 − weighted Jaccard of the card counts) of the live pool."""
    counts = [d.counts() for d in decks]
    dist = []
    for a in range(len(counts)):
        for b in range(a + 1, len(counts)):
            x, y = counts[a], counts[b]
            keys = x.keys() | y.keys()
            inter = sum(min(x[k], y[k]) for k in keys)
            union = sum(max(x[k], y[k]) for k in keys)
            dist.append(1 - inter / union if union else 0.0)
    return {"decks": len(decks), "mean_distance": float(np.mean(dist)) if dist else None}


def format_report(r: Mapping) -> str:
    lines = [f"round {r['round']} ({r['environment']['environment']}): {r['parents']} of {r['parents_planned']} "
             f"parents, {r['children']} children, {r['games']} games, {r['accepted']} accepted"
             + (" (budget spent)" if r["budget_stop"] else "")]  # fmt: skip
    if r["l0"] != "off":
        lines.append(f"L0 screen ({r['l0']}): would remove {r['l0_would_remove']} children")
    for p in r["per_parent"]:
        v = p["validation"]
        lines.append(f"  {p['parent']}: {p['children']} children" + (" (cold start)" if p.get("cold_start") else "")
                     + f", search {p['search_pairs']} pairs ({p['stop']}), "
                     f"chosen {p['chosen']}, " + (f"validation {v['decision']} after {v['pairs']} pairs "
                     f"(diff {v['diff']:+.3f}, lower {v['lower']:+.3f})" if v and v["diff"] is not None
                     else "no validation") + f", {p['games']} games")  # fmt: skip
        if p.get("effects"):
            lines.append("    factorial effects: " + ", ".join(f"{e['label']} {e['effect']:+.3f}±{e['stderr']:.3f}"
                                                               for e in p["effects"] if e["effect"] is not None))  # fmt: skip
    gens = r.get("generators", {}).get("cumulative", {})
    if set(gens) - {"mutation"}:
        lines.append("generators (cumulative): " + "; ".join(
            f"{k} {g['accepted']}/{g['children']} accepted ({g['chosen']} validated"
            + (f", first batch {g['first_batch_mean']:+.3f}" if g.get("first_batch_mean") is not None else "") + ")"
            for k, g in gens.items()))  # fmt: skip
    c = r["cumulative"]
    gpa = f"{c['games_per_accepted']:.0f}" if c["games_per_accepted"] else "n/a"
    lines.append(f"cumulative: {c['games']} games, {c['accepted']} accepted, {gpa} games per accepted edit")
    a = r["archive"]
    lines.append(f"archive: {a['elites']} elites of {a['cells']} cells (coverage {a['coverage']:.4f}), "
                 f"{a['admitted_this_round']} admitted this round, QD score {a['qd_score']:.3f}")  # fmt: skip
    p = r["pool"]
    md = f"{p['mean_distance']:.3f}" if p["mean_distance"] is not None else "n/a"
    lines.append(f"pool: {p['decks']} live evolved decks, mean card distance {md}")
    return "\n".join(lines) + "\n"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


__all__ = ["DESCRIPTORS", "Evolution", "GameLog", "Lab", "Opponent", "Parent", "RoundConfig", "blend_prior", "check_environment",
           "deck_key", "file_sha256", "format_report", "generator", "opponent_mix", "pool_diversity"]  # fmt: skip
