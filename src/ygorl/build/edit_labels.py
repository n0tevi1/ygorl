"""Labels of the edit-value model (#151, design 05 §5.3, docs/tuning.md「改动价值模型」).

The edit-value model (:mod:`ygorl.build.value_model`) learns the win-rate change of a deck edit from two kinds of
labels, each bound to the policy checkpoint that played it:

- :class:`PairedLabel`: a clean paired evaluation, child minus parent with a standard error, with the parent's deck
  list and the edits (``ygorl.build.tuner.Edit``, in place). :func:`paired_labels` reads every source the warm start
  of the card-value model reads (``ygorl.build.warmstart``: M1, M2, a lineage or evolution state, a tuner
  comparison) and two more: the factorial measurement (``tools/deckevo_factorial.py``: each design's main effects and
  the one-at-a-time differences of the same edits) and a re-validation (``tools/deckevo_eval_compare.py
  --revalidate``: the MVP arm's accepted edits and the evolution states' accepted children on newer seeds). As in
  the warm start, only non-adaptive pairs count (a lineage's ``learned`` entries and factorial main effects; never
  ``search`` or ``all_pairs``).
- :class:`GameRecord`: one finished training game of a run's game log (``--log-games``, ``games.jsonl.gz``,
  docs/training.md「对局日志」), deck a against deck b; :func:`read_game_log` resolves the deck names (corpus decks
  through the run's ``config.json``, evolved decks through its deck-pool manifest).

:class:`Clock` turns a label's checkpoint into an **age** (training updates between the label's policy and the
target policy; a label of another run counts ``foreign_age``) and a weight ``0.5 ** (age / half_life)``: the
policy changes, so older labels count less. A checkpoint is a :class:`CheckpointRef` (run directory, update).
"""

from __future__ import annotations

import gzip
import json
import math
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ygorl.build.tuner import Edit, apply
from ygorl.build.warmstart import (BLANK, FUTILITY_Z, content_key, lineage_key, parse_edit, pooled_stderr,
                                   round_states)  # fmt: skip
from ygorl.cards.ydk import Deck, load_ydk, parse_ydk

_UPDATE = re.compile(r"update_0*(\d+)\.pt$")


# ------------------------------------------------------------------ checkpoints


@dataclass(frozen=True)
class CheckpointRef:
    """The policy behind a label: its run directory (absolute path text: relative paths are read from the working
    directory) and update (None: unknown)."""

    run: str
    update: int | None = None

    @classmethod
    def from_path(cls, path: str | os.PathLike | None, update: int | None = None) -> CheckpointRef:
        """``RUN/checkpoints/update_N.pt`` -> (RUN, N); a checkpoint elsewhere is its own directory's run. The
        update is taken from the name when ``update`` is not given (``latest.pt`` / ``best.pt``: unknown)."""
        if not path:
            return cls("", update)
        p = Path(os.path.abspath(str(path)))
        run = p.parent.parent if p.parent.name == "checkpoints" else p.parent
        if update is None and (m := _UPDATE.search(p.name)):
            update = int(m.group(1))
        return cls(str(run), update)

    def to_dict(self) -> dict:
        return {"run": self.run, "update": self.update}


@dataclass
class Clock:
    """Age and weight of a label relative to the target policy (``run``, ``update``). ``same`` lists other run
    directories that are the same training (a continuation in another directory, e.g. the co-evolution's
    segments of the base run). A label of another run, or of an unknown update, is ``foreign_age`` updates old."""

    run: str = ""
    update: int | None = None
    half_life: float = 200.0
    foreign_age: float = 400.0
    same: tuple[str, ...] = ()

    @classmethod
    def at(cls, checkpoint: str | os.PathLike, update: int | None = None, **kw) -> Clock:
        ref = CheckpointRef.from_path(checkpoint, update)
        return cls(ref.run, ref.update, **kw)

    def age(self, ref: CheckpointRef) -> float:
        runs = {os.path.abspath(r) for r in (self.run, *self.same) if r}
        if ref.run in runs and ref.update is not None and self.update is not None:
            return float(abs(self.update - ref.update))
        return float(self.foreign_age)

    def weight(self, ref: CheckpointRef) -> float:
        if not math.isfinite(self.half_life):
            return 1.0
        return 0.5 ** (self.age(ref) / self.half_life)

    def to_dict(self) -> dict:
        return {"run": self.run, "update": self.update, "half_life": self.half_life, "foreign_age": self.foreign_age,
                "same": list(self.same)}  # fmt: skip

    @classmethod
    def from_dict(cls, d: Mapping) -> Clock:
        return cls(d.get("run", ""), d.get("update"), float(d.get("half_life", 200.0)),
                   float(d.get("foreign_age", 400.0)), tuple(d.get("same", ())))  # fmt: skip


# ------------------------------------------------------------------ paired labels


@dataclass(frozen=True)
class PairedLabel:
    """A clean paired evaluation: ``edits`` applied in place to ``parent`` (child) minus ``parent``."""

    id: str
    parent: Deck
    deck_type: str
    edits: tuple[Edit, ...]
    diff: float
    stderr: float
    checkpoint: CheckpointRef
    source: str = ""  # m1, m2, lineage, compare, factorial, revalidation
    games: str = ""  # identity of the games measured (unique() keeps one label per identity)

    @property
    def child(self) -> Deck:
        deck = self.parent
        for e in self.edits:
            deck = apply(deck, e)
        return deck

    @property
    def group(self) -> str:
        """The parent deck's identity (leave-one-deck-out folds)."""
        return f"{sorted(self.parent.main)}|{sorted(self.parent.extra)}"

    @property
    def edit_key(self) -> tuple:
        """The parent and the edits as a set: independent measurements of the same edit share it."""
        return self.group, tuple(sorted((e.section, e.out, e.into) for e in self.edits))


def _ok(diff, stderr) -> bool:
    return diff is not None and stderr is not None and math.isfinite(diff) and math.isfinite(stderr) and stderr > 0


def _edits(texts: Iterable[str]) -> tuple[Edit, ...]:
    return tuple(Edit(o, i, s) for s, o, i in (parse_edit(t) for t in texts))


def _legal_edits(parent: Deck, edits: Sequence[Edit]) -> bool:
    """Whether the edits apply in place in order (each card out is still there)."""
    deck = parent
    for e in edits:
        if e.out not in getattr(deck, e.section):
            return False
        deck = apply(deck, e)
    return True


@dataclass
class _Reader:
    artifacts: Path  # the environment's artifacts directory (deck files of M1 / M2 / comparisons are relative to it)
    checkpoint: CheckpointRef | None  # given on the command line: for files that do not record theirs
    out: list[PairedLabel] = field(default_factory=list)
    skipped: int = 0

    def deck(self, file: str) -> Deck:
        return load_ydk(self.artifacts / file)

    def add(self, uid, parent, deck_type, edits, diff, stderr, ckpt, source, games: str | None = None) -> None:
        """A label; its games' identity is ``games`` (a lineage entry with its round's seed and pairs) or else its
        content (the parent's card counts, the edits, the checkpoint, the difference and standard error)."""
        edits = tuple(edits)
        if not edits or not _ok(diff, stderr) or not _legal_edits(parent, edits):
            self.skipped += 1
            return
        counts = sorted(parent.counts().items())
        key = games or content_key(counts, [e.into for e in edits], [e.out for e in edits],
                                   f"{ckpt.run}@{ckpt.update}", diff, stderr)  # fmt: skip
        self.out.append(PairedLabel(uid, parent, str(deck_type), edits, float(diff), float(stderr), ckpt, source,
                                    key))  # fmt: skip

    def ckpt(self, data: Mapping) -> CheckpointRef:
        if self.checkpoint is not None:
            return self.checkpoint
        return CheckpointRef.from_path(data.get("checkpoint"))


def _m2(r: _Reader, data: Sequence[Mapping], name: str) -> None:
    ck = r.checkpoint or CheckpointRef("")
    for d in data:
        parent = r.deck(d["file"])
        for row in d["loo"]:
            r.add(f"m2:{name}:{d['file']}:{row['card']}", parent, d["type"], [Edit(row["card"], BLANK, "main")],
                  -row["value"], row["ci"] / 1.96, ck, "m2")  # fmt: skip


def _m1(r: _Reader, data: Mapping, name: str) -> None:
    ck, pairs = r.ckpt(data), int(data["pairs"])
    for d in data["decks"]:
        parent = r.deck(d["file"])
        for k, row in enumerate(d["children"]):
            r.add(f"m1:{name}:{d['file']}:{k}", parent, d["type"], _edits([row["edit"]]), row["diff"],
                  pooled_stderr(row["sd_diff_pair"], pairs), ck, "m1")  # fmt: skip


def _compare(r: _Reader, data: Mapping, name: str) -> None:
    """Fresh-pair validations (``warmstart.from_compare``) and, when present, the re-validations of the MVP arm's
    accepted edits and of evolution states' accepted children (``--revalidate``)."""
    ck = r.ckpt(data)
    for k, row in enumerate(data.get("parents", [])):
        parent, t = r.deck(row["file"]), row["type"]
        evo = row.get("evo") or {}
        if evo.get("chosen") and evo.get("diff") is not None:
            se = evo.get("stderr")
            if se is None and evo.get("ci") and evo["ci"][1] is not None:
                se = (evo["ci"][1] - evo["diff"]) / FUTILITY_Z
            r.add(f"compare:{name}:{k}:evo", parent, t, _edits(evo["chosen"]), evo["diff"], se, ck, "compare")
        mvp = row.get("mvp") or {}
        for f, fin in enumerate(mvp.get("finalists", [])):
            r.add(f"compare:{name}:{k}:mvp{f}", parent, t, _edits([fin["edit"]]), fin["diff"], fin.get("stderr"), ck,
                  "compare")  # fmt: skip
        rv = mvp.get("revalidation")
        if rv and mvp.get("edits"):
            r.add(f"revalidation:{name}:{k}:mvp", parent, t, _edits(mvp["edits"]), rv.get("diff"), rv.get("stderr"),
                  ck, "revalidation")  # fmt: skip
    for k, e in enumerate(data.get("evolution", [])):  # --evo-state: accepted children re-validated
        rv = e.get("revalidation") or {}
        rec = _lineage_record(Path(e["state"]), e["id"])
        if rec is None:
            r.skipped += 1
            continue
        parent = _lineage_parent(Path(e["state"]), rec, r)
        if parent is None:
            r.skipped += 1
            continue
        edits = [Edit(x["out"], x["into"], x["section"]) for x in rec["edits"]]
        r.add(f"revalidation:{name}:{e['id']}", parent, rec["parent"]["type"], edits, rv.get("diff"), rv.get("stderr"),
              ck, "revalidation")  # fmt: skip


def _factorial(r: _Reader, data: Mapping, name: str) -> None:
    """``tools/deckevo_factorial.py``: each design's main effects (one single-edit label per edit), and the
    one-at-a-time differences of the same edits (their own pairs: independent labels of the same edits)."""
    ck = r.ckpt(data)
    for k, row in enumerate(data.get("parents", [])):
        parent, t = r.deck(row["file"]), row["type"]
        fac = row.get("factorial") or {}
        edits = [Edit(e["out"], e["into"], e["section"]) for e in fac.get("edits", [])]
        for e in fac.get("effects", []):
            if len(e["term"]) == 1:
                j = e["term"][0]
                r.add(f"factorial:{name}:{k}:{e['label']}", parent, t, [edits[j]], e["effect"], e["stderr"], ck,
                      "factorial")  # fmt: skip
        one = (row.get("one_at_a_time") or {}).get("effects", [])
        if len(one) == len(edits) == len(row.get("edits", [])):
            for j, e in enumerate(one):  # same order as the design's edits (row["edits"] names them)
                if e.get("edit") != row["edits"][j]:
                    r.skipped += 1
                    continue
                r.add(f"factorial:{name}:{k}:single{j}", parent, t, [edits[j]], e["diff"], e["stderr"], ck,
                      "factorial")  # fmt: skip


def _lineage_record(state: Path, manifest_id: str) -> dict | None:
    for rec in _read_jsonl(state / "lineage.jsonl"):
        if rec.get("manifest_id") == manifest_id or f"evo-{rec['child']}" == manifest_id:
            return rec
    return None


def _round_parents(state: Path, n: int) -> dict[str, Deck]:
    path = state / "rounds" / f"{n:04d}" / "round.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    return {p["id"]: Deck(main=tuple(p["deck"]["main"]), extra=tuple(p["deck"]["extra"]), name=p["id"])
            for p in data.get("parents", [])}  # fmt: skip


def _lineage_parent(state: Path | None, rec: Mapping, r: _Reader) -> Deck | None:
    """The parent's list: the round's ``round.json``, else a corpus deck by name (``corpus:<stem>``)."""
    pid = rec["parent"]["id"]
    if state is not None:
        deck = _round_parents(state, int(rec["round"])).get(pid)
        if deck is not None:
            return deck
    if pid.startswith("corpus:"):
        f = r.artifacts / "decks" / f"{pid.split(':', 1)[1]}.ydk"
        if f.is_file():
            return load_ydk(f)
    return None


def _lineage(r: _Reader, records: Sequence[Mapping], state: Path | None, name: str) -> None:
    rounds = round_states(state) if state is not None else {}
    designs: set[str] = set()
    cache: dict[tuple, Deck | None] = {}
    for rec in records:
        key = (rec["round"], rec["parent"]["id"])
        if key not in cache:
            cache[key] = _lineage_parent(state, rec, r)
        parent = cache[key]
        if parent is None:
            r.skipped += 1
            continue
        c = rec.get("checkpoint") or {}
        ck = r.checkpoint or CheckpointRef.from_path(c.get("path"), c.get("update"))
        t = rec["parent"]["type"]
        edits = [Edit(e["out"], e["into"], e["section"]) for e in rec["edits"]]
        for m in rec.get("learned", []):
            r.add(f"lineage:{name}:{rec['child']}/{m['source']}", parent, t, edits, m.get("diff"), m.get("stderr"), ck,
                  "lineage", lineage_key(rec, m, rounds))  # fmt: skip
        fac = rec.get("factorial")
        if fac and fac["id"] not in designs:
            designs.add(fac["id"])
            for e in fac["effects"]:
                if len(e["term"]) == 1:
                    x = fac["edits"][e["term"][0]]
                    r.add(f"lineage:{name}:{fac['id']}/{x['letter']}", parent, t,
                          [Edit(x["out"], x["into"], x["section"])], e["effect"], e["stderr"], ck, "lineage",
                          lineage_key(rec, {"source": "factorial", "letter": x["letter"]}, rounds))  # fmt: skip


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue  # a line cut by a crash
    return out


def paired_labels(path: str | Path, artifacts: str | Path, *,
                  checkpoint: str | CheckpointRef | None = None) -> tuple[list[PairedLabel], int]:  # fmt: skip
    """``(labels, skipped)`` of one source, its kind told by its content: an evolution state directory or a
    ``lineage.jsonl``, an M2 list, an M1 run, a tuner comparison or re-validation, a factorial measurement.
    ``artifacts`` is the environment's artifacts directory (deck files are relative to it); ``checkpoint`` overrides
    the policy the file records (M2 files record none). Labels whose edits do not apply to the parent are skipped."""
    ck = CheckpointRef.from_path(checkpoint) if isinstance(checkpoint, str) else checkpoint
    r = _Reader(Path(artifacts), ck)
    p = Path(path)
    if p.is_dir():
        _lineage(r, _read_jsonl(p / "lineage.jsonl"), p, p.name)
        return r.out, r.skipped
    if p.suffix == ".jsonl":
        state = p.parent if (p.parent / "rounds").is_dir() else None
        _lineage(r, _read_jsonl(p), state, p.parent.name if p.name == "lineage.jsonl" else p.stem)
        return r.out, r.skipped
    data = json.loads(p.read_text())
    name = p.stem
    if isinstance(data, list) and (not data or "loo" in data[0]):
        _m2(r, data, name)
    elif isinstance(data, dict) and "decks" in data and "pairs" in data:
        _m1(r, data, name)
    elif isinstance(data, dict) and any("factorial" in row for row in data.get("parents", [])):
        _factorial(r, data, name)
    elif isinstance(data, dict) and ("parents" in data or "evolution" in data):
        _compare(r, data, name)
    else:
        raise ValueError(f"{path}: not an M1, M2, lineage, comparison or factorial file")
    return r.out, r.skipped


def unique(labels: Iterable[PairedLabel]) -> list[PairedLabel]:
    """The first label of each id (the same file given twice counts once) and of each set of games
    (``PairedLabel.games``): rounds and comparisons rerun with the same seed replay the same games (#145 and #150
    share their cold-start children's first batches), and those count once."""
    seen: set = set()
    out = []
    for lab in labels:
        games = ("games", lab.games) if lab.games else None  # "": unknown, the id alone decides
        if lab.id not in seen and (games is None or games not in seen):
            seen.update((lab.id, games))
            out.append(lab)
    return out


# ------------------------------------------------------------------ training games


@dataclass(frozen=True)
class GameRecord:
    """A finished training game: deck ``a`` against deck ``b`` (names), who went first and who won (0 = a, 1 = b,
    None = no winner), played by the policy after ``checkpoint.update`` updates."""

    a: str
    b: str
    first: int
    winner: int | None
    checkpoint: CheckpointRef


@dataclass
class GameData:
    decks: dict[str, Deck]
    games: list[GameRecord]
    skipped: int = 0  # truncated games, games with an unknown deck, duplicated lines


def read_game_log(run_dir: str | Path, *, overlap: bool | None = None) -> GameData:
    """The decided games of a run's ``games.jsonl.gz`` with the decks they name. Corpus decks resolve through the
    run's ``config.json`` (``decks``: .ydk paths, the name is the file stem), evolved and history decks through its
    ``deck_pool`` manifest (ids). Truncated games and games without a winner are left out (not a result), as are
    games naming a deck that no longer resolves and repeated lines (a resumed run replays the updates after its
    last checkpoint: the first record of a (update, seed, decks, first) wins). The policy of a game is the one after
    ``update - 1`` updates (``update - 2`` with the config's ``overlap_collect``)."""
    run = Path(run_dir)
    cfg = json.loads((run / "config.json").read_text())
    lag = 2 if (cfg.get("overlap_collect") if overlap is None else overlap) else 1
    decks: dict[str, Deck] = {}
    for f in cfg.get("decks", []):
        p = Path(f)
        if p.is_file():
            decks.setdefault(p.stem, load_ydk(p))
    if cfg.get("deck_pool"):
        mp = Path(cfg["deck_pool"])
        if mp.is_file():
            for e in json.loads(mp.read_text()).get("decks", []):
                f = mp.parent / e["file"]
                if f.is_file():
                    decks[e["id"]] = parse_ydk(f.read_text(), name=e["id"])
    games, skipped, seen = [], 0, set()
    path = run / "games.jsonl.gz"
    if path.is_file():
        with gzip.open(path, "rt") as fh:
            lines = fh.read().splitlines()
        for line in lines:
            try:
                g = json.loads(line)
            except ValueError:
                skipped += 1
                continue
            a, b = g["decks"]
            key = (g["update"], g.get("seed"), a, b, g["first"])
            if key in seen or g.get("truncated") or g.get("winner") is None or a not in decks or b not in decks:
                skipped += 1
                continue
            seen.add(key)
            ref = CheckpointRef(os.path.abspath(str(run)), max(int(g["update"]) - lag, 0))
            games.append(GameRecord(a, b, int(g["first"]), int(g["winner"]), ref))
    return GameData(decks, games, skipped)


def merge_games(parts: Iterable[GameData]) -> GameData:
    out = GameData({}, [], 0)
    for p in parts:
        out.decks.update(p.decks)
        out.games += p.games
        out.skipped += p.skipped
    return out


__all__ = ["CheckpointRef", "Clock", "GameData", "GameRecord", "PairedLabel", "merge_games", "paired_labels",
           "read_game_log", "unique"]  # fmt: skip
