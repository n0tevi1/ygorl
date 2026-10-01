"""Warm start of the card-value model from paired data measured before (#145, docs/tuning.md「热启动」).

The first rounds started from an empty :class:`ygorl.build.signals.CardValueModel`, so every draw was the prior and
removals were random. :func:`load` turns earlier paired measurements into :class:`Observation`\\ s (child minus
parent, one copy in for one copy out, with a standard error), each with a stable id so that loading the same file
twice adds nothing (:meth:`ygorl.build.evolve.Evolution.warm_start`):

- **M2** leave-one-out (``tools/deckevo_m2_loo.py`` JSON, a list of decks with ``loo`` rows): one copy of the card
  replaced by the blank (Metal Armored Bug); ``value`` is parent minus child, so the observation is
  ``into = (blank,)``, ``out = (card,)``, ``diff = -value``, ``stderr = ci / 1.96``;
- **M1** random swaps (``tools/deckevo_m1_rho.py`` JSON): ``edit`` "main: -OUT +IN", ``diff`` child minus parent,
  ``stderr`` from ``sd_diff_pair`` over the run's ``pairs`` (with the 10 pseudo-pairs of σ = 0.212 the evolution
  uses, so a short run cannot claim zero error);
- **lineage** (an evolution state directory or its ``lineage.jsonl``): each child's ``learned`` entries, the
  non-adaptive pairs only (its first batch, and the validation of a chosen child). The search's later pairs and the
  ``all_pairs`` estimate are selection-biased and never used. A factorial design's main effects (#152, each variant's
  ``factorial`` field) are single-edit observations, taken once per design;
- **tuner comparison** (``tools/deckevo_eval_compare.py`` JSON): the fresh-pair validation of each arm's validated
  edits (the evolution arm's chosen child; the MVP arm's finalists when the file logs them). Its Thompson arm
  estimates are search pairs, never used. Files written before the edits were logged give only the evolution arm;
  its standard error is recovered from the futility bound (``upper = diff + 1.645 se``).

Observations are filed under the deck type the file names; ``inflate`` multiplies every standard error (data played
by another checkpoint informs less).

**The same games count once.** Rounds and comparisons rerun with the same seed replay the same games (#145 and #150
share their cold-start children's first batches; the evolution arm of the first tuner comparison is that of the
second), so every observation carries the identity of the games it measured (``Observation.games``,
:func:`games_key`) and :meth:`ygorl.build.evolve.Evolution.warm_start` adds one observation per identity:

- a lineage entry with its round's ``round.json`` (an evolution state): the opponent mix and policy (``mix``, which
  fingerprints the checkpoint), the round's seed, round and parent index (the evaluator's seed), the parent's and
  the child's lists in order, and the pairs (the first batch, the validation's fresh pairs; a factorial design's
  pairs and edits);
- otherwise (no seed recorded: M1, M2, comparisons, a bare ``lineage.jsonl``) its content: the parent, the edits,
  the checkpoint and the measured difference and standard error, so two files with identical games give equal keys.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from pathlib import Path

from ygorl.build.selection import SD_PAIR
from ygorl.build.signals import Observation

BLANK = 65957473  # Metal Armored Bug: the M2 leave-one-out blank (tools/deckevo_m2_loo.py)
PRIOR_PAIRS = 10.0
FUTILITY_Z = 1.645  # selection.sequential_validate's default futility z
_EDIT = re.compile(r"^\s*(main|extra):\s*-(\d+)\s*\+(\d+)\s*$")


def parse_edit(text: str) -> tuple[str, int, int]:
    """``(section, out, into)`` of an ``Edit.describe()`` string without names ("main: -OUT +IN")."""
    m = _EDIT.match(text)
    if m is None:
        raise ValueError(f"not an edit: {text!r}")
    return m.group(1), int(m.group(2)), int(m.group(3))


def pooled_stderr(sd: float, pairs: int, prior_pairs: float = PRIOR_PAIRS) -> float:
    """Standard error of a mean paired difference over ``pairs`` pairs with per-pair sd ``sd``, the variance pooled
    with ``prior_pairs`` pseudo-pairs of SD_PAIR² (as ``ygorl.build.evolve`` does)."""
    n = max(int(pairs), 1)
    var = (prior_pairs * SD_PAIR**2 + (n - 1) * sd**2) / (prior_pairs + n - 1)
    return math.sqrt(var / n)


def games_key(*parts) -> str:
    """A stable identity of a set of games from the JSON-ready ``parts`` that determine them."""
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:20]


def content_key(parent, into, out, checkpoint, diff, stderr) -> str:
    """The games' identity when no seed is recorded: what was measured (parent, edits as a multiset, checkpoint) and
    the result (difference and standard error, rounded to 1e-9)."""
    return games_key("content", parent, sorted(int(c) for c in into), sorted(int(c) for c in out), checkpoint,
                     round(float(diff), 9), round(float(stderr), 9))  # fmt: skip


def lineage_key(rec: Mapping, entry: Mapping, rounds: Mapping[int, Mapping] | None) -> str | None:
    """The games of one ``learned`` entry of a lineage record (``entry``; a factorial design: ``{"source":
    "factorial", "letter": ...}``) from its round's ``round.json`` (``rounds[round]``), or None without it."""
    r = (rounds or {}).get(rec.get("round"))
    m = re.search(r"-p(\d+)-", rec["child"])
    if r is None or m is None:
        return None
    where = [r.get("mix") or r.get("checkpoint"), r["config"].get("seed"), int(rec["round"]), int(m.group(1)),
             rec["parent"].get("key"), rec.get("key")]  # fmt: skip
    if entry.get("source") == "factorial":
        fac = rec["factorial"]
        return games_key("lineage", *where[:5], fac.get("design"), fac.get("edits"), entry["letter"],
                         r["config"].get("factorial_pairs"))  # fmt: skip
    return games_key("lineage", *where, entry["source"], entry.get("pairs"))


def _checkpoint(c) -> str | None:
    """A checkpoint's fingerprint: its sha256 when recorded, else its path."""
    if isinstance(c, Mapping):
        return c.get("sha256") or c.get("path")
    return c


def _obs(deck_type, into, out, diff, stderr, inflate, *, games: str | None = None, parent=None,
         checkpoint=None) -> Observation | None:  # fmt: skip
    if diff is None or stderr is None or not math.isfinite(diff) or not math.isfinite(stderr) or stderr <= 0:
        return None
    key = games or content_key(parent, into, out, checkpoint, diff, stderr)
    return Observation(str(deck_type), tuple(int(c) for c in into), tuple(int(c) for c in out), float(diff),
                       float(stderr) * inflate, key)  # fmt: skip


def from_m2(data: Iterable[Mapping], name: str = "m2", inflate: float = 1.0) -> list[tuple[str, Observation]]:
    """M2 leave-one-out rows (module docstring)."""
    out = []
    for d in data:
        for row in d["loo"]:
            o = _obs(d["type"], (BLANK,), (row["card"],), -row["value"], row["ci"] / 1.96, inflate, parent=d["file"],
                     checkpoint=d.get("checkpoint"))  # fmt: skip
            if o is not None:
                out.append((f"warm:m2:{name}:{d['file']}:{row['card']}", o))
    return out


def from_m1(data: Mapping, name: str = "m1", inflate: float = 1.0) -> list[tuple[str, Observation]]:
    """M1 random swaps (module docstring)."""
    out = []
    pairs = int(data["pairs"])
    for d in data["decks"]:
        for k, row in enumerate(d["children"]):
            _, o_card, i_card = parse_edit(row["edit"])
            o = _obs(d["type"], (i_card,), (o_card,), row["diff"], pooled_stderr(row["sd_diff_pair"], pairs), inflate,
                     parent=d["file"], checkpoint=data.get("checkpoint"))  # fmt: skip
            if o is not None:
                out.append((f"warm:m1:{name}:{d['file']}:{k}", o))
    return out


def from_lineage(records: Iterable[Mapping], name: str = "lineage", inflate: float = 1.0,
                 rounds: Mapping[int, Mapping] | None = None) -> list[tuple[str, Observation]]:  # fmt: skip
    """The ``learned`` (non-adaptive) entries of lineage records, and the main effects of factorial designs (#152:
    every variant of a design carries it; each main effect once); ``rounds`` (round number -> ``round.json``)
    identifies their games (module docstring)."""
    out = []
    designs: set[str] = set()
    for r in records:
        into = [e["into"] for e in r["edits"]]
        outs = [e["out"] for e in r["edits"]]
        ck = _checkpoint(r.get("checkpoint"))
        for m in r.get("learned", []):
            o = _obs(r["parent"]["type"], into, outs, m.get("diff"), m.get("stderr"), inflate,
                     games=lineage_key(r, m, rounds), parent=r["parent"].get("key"), checkpoint=ck)  # fmt: skip
            if o is not None:
                out.append((f"warm:lineage:{name}:{r['child']}/{m['source']}", o))
        fac = r.get("factorial")
        if fac and fac["id"] not in designs:
            designs.add(fac["id"])
            for e in fac["effects"]:
                if len(e["term"]) == 1:
                    edit = fac["edits"][e["term"][0]]
                    key = lineage_key(r, {"source": "factorial", "letter": edit["letter"]}, rounds)
                    o = _obs(r["parent"]["type"], (edit["into"],), (edit["out"],), e["effect"], e["stderr"], inflate,
                             games=key, parent=r["parent"].get("key"), checkpoint=ck)  # fmt: skip
                    if o is not None:
                        out.append((f"warm:lineage:{name}:{fac['id']}/{edit['letter']}", o))
    return out


def from_compare(data: Mapping, name: str = "compare", inflate: float = 1.0) -> list[tuple[str, Observation]]:
    """Fresh-pair validations of ``tools/deckevo_eval_compare.py`` results (module docstring)."""
    out = []
    ck = data.get("checkpoint")
    for k, row in enumerate(data.get("parents", [])):
        t = row["type"]
        evo = row.get("evo") or {}
        if evo.get("chosen") and evo.get("diff") is not None:
            se = evo.get("stderr")
            if se is None and evo.get("ci") and evo["ci"][1] is not None:
                se = (evo["ci"][1] - evo["diff"]) / FUTILITY_Z
            edits = [parse_edit(e) for e in evo["chosen"]]
            o = _obs(t, [i for _, _, i in edits], [c for _, c, _ in edits], evo["diff"], se, inflate,
                     parent=row["file"], checkpoint=ck)  # fmt: skip
            if o is not None:
                out.append((f"warm:compare:{name}:{k}:evo", o))
        for f, fin in enumerate((row.get("mvp") or {}).get("finalists", [])):
            _, o_card, i_card = parse_edit(fin["edit"])
            o = _obs(t, (i_card,), (o_card,), fin["diff"], fin.get("stderr"), inflate, parent=row["file"],
                     checkpoint=ck)  # fmt: skip
            if o is not None:
                out.append((f"warm:compare:{name}:{k}:mvp{f}", o))
    return out


def round_states(state: str | Path) -> dict[int, dict]:
    """Round number -> ``round.json`` of an evolution state directory (empty without rounds)."""
    out = {}
    for f in sorted(Path(state).glob("rounds/[0-9]*/round.json")):
        try:
            data = json.loads(f.read_text())
        except ValueError:
            continue
        out[int(data["round"])] = data
    return out


def load(path: str | Path, inflate: float = 1.0) -> list[tuple[str, Observation]]:
    """``(id, observation)`` pairs from one warm-start source, its kind told by its content: an evolution state
    directory or a ``.jsonl`` lineage, an M2 list, an M1 run, or a tuner comparison."""
    p = Path(path)
    if p.is_dir():
        p = p / "lineage.jsonl"
        name = p.parent.name
    else:
        name = p.parent.name if p.name == "lineage.jsonl" else p.stem
    if p.suffix == ".jsonl":
        records = []
        for line in p.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except ValueError:
                continue  # a line cut by a crash
        return from_lineage(records, name, inflate, round_states(p.parent))
    data = json.loads(p.read_text())
    if isinstance(data, list) and (not data or "loo" in data[0]):
        return from_m2(data, name, inflate)
    if isinstance(data, dict) and "decks" in data and "pairs" in data:
        return from_m1(data, name, inflate)
    if isinstance(data, dict) and "parents" in data:
        return from_compare(data, name, inflate)
    raise ValueError(f"{path}: not an M1, M2, lineage or tuner-comparison file")


__all__ = ["BLANK", "content_key", "from_compare", "from_lineage", "from_m1", "from_m2", "games_key", "lineage_key", "load",
           "parse_edit", "pooled_stderr", "round_states"]  # fmt: skip
