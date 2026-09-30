"""Deck dataset: every distinct list of the masterduelmeta history, for the learned deck models (docs/data.md「牌组数据集」).

The deck corpus (:mod:`ygorl.data.corpus`) keeps at most three legal lists per type for training games; the learned
deck models (design 05 §5.3: the masked deck model, the Δ surrogate) need all of them. The dataset keeps every list
whose cards all map to passwords, deduplicated by its main + extra card counts (the newest copy stays, ``copies``
counts them), newest first. Lists illegal in the environment are kept and flagged: card relations do not change with
the banlist.

On disk (``<dir>/decks.npz`` + ``<dir>/meta.json``) and in memory (:class:`DeckDataset`), list ``i`` is row ``i`` of a
CSR matrix of card counts over the sorted ``passwords``::

    passwords[indices[indptr[i]:indptr[i + 1]]]   # its distinct cards (main and extra deck together)
    counts[indptr[i]:indptr[i + 1]]               # copies of each (1-3)

and one entry per list in the metadata arrays ``type`` (index into ``types``), ``date``, ``kind``, ``source``,
``event``, ``url``, ``weight``, ``copies``, ``legal``, ``problems``, ``n_main``, ``n_extra``. ``meta.json`` has the
environment stamp the legality refers to, the provenance and the build statistics. :func:`load_deck_dataset` reads
it back (about a second for the full history); ``npz`` needs no pickle.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ygorl.data.masterduelmeta import DeckRecord

FORMAT = "ygorl-deck-dataset"
VERSION = 1
DECKS_FILE = "decks.npz"
META_FILE = "meta.json"
COLUMNS = ("type", "date", "kind", "source", "event", "url", "weight", "copies", "legal", "problems", "n_main",
           "n_extra")  # fmt: skip


@dataclass
class DeckDataset:
    """Card-count rows (CSR over ``passwords``) and one metadata entry per list; see the module docstring."""

    passwords: np.ndarray  # int64 [V], sorted: every card of any list
    indptr: np.ndarray  # int64 [N + 1]
    indices: np.ndarray  # int32 [nnz], into passwords
    counts: np.ndarray  # int8 [nnz]
    types: np.ndarray  # str [K]: deck type names (masterduelmeta deckType), sorted
    type: np.ndarray  # int32 [N], into types
    date: np.ndarray  # str [N]: creation date, YYYY-MM-DD
    kind: np.ndarray  # str [N]: "ranked" or "tournament"
    source: np.ndarray  # str [N]: ranked / tournament type name ("Master I", "Dice Rally", "Meta Weekly")
    event: np.ndarray  # str [N]: a tournament's own name ("Master Cup"), or ""
    url: np.ndarray  # str [N]
    weight: np.ndarray  # float32 [N]: masterduelmeta's statsWeight (0 = not counted in its statistics)
    copies: np.ndarray  # int32 [N]: lists of the history with these exact counts
    legal: np.ndarray  # bool [N]: legal in meta["environment"] (pool, banlist, deck rules)
    problems: np.ndarray  # str [N]: violation codes, comma-separated ("" when legal)
    n_main: np.ndarray  # int16 [N]
    n_extra: np.ndarray  # int16 [N]
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.indptr) - 1

    def cards(self, i: int) -> dict[int, int]:
        """List ``i`` as {password: copies}."""
        lo, hi = self.indptr[i], self.indptr[i + 1]
        return dict(zip(self.passwords[self.indices[lo:hi]].tolist(), self.counts[lo:hi].tolist(), strict=True))

    def type_name(self, i: int) -> str:
        return str(self.types[self.type[i]])

    def matrix(self):
        """The counts as a ``scipy.sparse.csr_matrix`` of shape [N, V] (int8)."""
        from scipy.sparse import csr_matrix

        return csr_matrix((self.counts, self.indices, self.indptr), shape=(len(self), len(self.passwords)))

    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        arrays = {k: getattr(self, k) for k in ("passwords", "indptr", "indices", "counts", "types", *COLUMNS)}
        np.savez_compressed(d / DECKS_FILE, **arrays)
        (d / META_FILE).write_text(json.dumps(self.meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        return d


def load_deck_dataset(directory: str | Path) -> DeckDataset:
    """Read a dataset written by :meth:`DeckDataset.save` (``tools/build_deck_dataset.py``)."""
    d = Path(directory)
    meta = json.loads((d / META_FILE).read_text(encoding="utf-8"))
    if meta.get("format") != FORMAT or meta.get("version") != VERSION:
        raise ValueError(f"{d}: not a {FORMAT} v{VERSION} dataset")
    with np.load(d / DECKS_FILE, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    return DeckDataset(**arrays, meta=meta)


def build_deck_dataset(records: Iterable[DeckRecord], validate: Callable[[DeckRecord], list[str]],
                       meta: Mapping[str, Any] | None = None) -> DeckDataset:  # fmt: skip
    """Every completely mapped record, deduplicated by main + extra counts (newest copy kept), newest first.
    ``validate`` returns a record's violation codes in the environment (empty: legal); ``meta`` is merged into
    ``DeckDataset.meta`` (environment stamp, provenance)."""
    records = sorted(records, key=lambda r: r.created, reverse=True)
    raw, unmapped = len(records), 0
    kept: dict[tuple, int] = {}
    rows: list[DeckRecord] = []
    copies: list[int] = []
    for r in records:
        if r.unmapped:
            unmapped += 1
            continue
        key = (tuple(sorted(Counter(r.main).items())), tuple(sorted(Counter(r.extra).items())))
        if key in kept:
            copies[kept[key]] += 1
            continue
        kept[key] = len(rows)
        rows.append(r)
        copies.append(1)

    passwords = np.array(sorted({p for r in rows for p in (*r.main, *r.extra)}), dtype=np.int64)
    column = {int(p): j for j, p in enumerate(passwords)}
    indptr, indices, counts = [0], [], []
    for r in rows:
        c = sorted(r.counts().items(), key=lambda kv: column[kv[0]])
        indices.extend(column[p] for p, _ in c)
        counts.extend(n for _, n in c)
        indptr.append(len(indices))
    types = sorted({r.type for r in rows})
    type_index = {t: k for k, t in enumerate(types)}
    problems = [",".join(sorted(set(validate(r)))) for r in rows]

    def strings(values) -> np.ndarray:
        values = list(values)
        return np.array(values, dtype=f"<U{max((len(v) for v in values), default=1) or 1}")

    ds = DeckDataset(
        passwords=passwords, indptr=np.array(indptr, dtype=np.int64), indices=np.array(indices, dtype=np.int32),
        counts=np.array(counts, dtype=np.int8), types=strings(types),
        type=np.array([type_index[r.type] for r in rows], dtype=np.int32), date=strings(r.created[:10] for r in rows),
        kind=strings(r.kind for r in rows), source=strings(r.source for r in rows),
        event=strings(r.event for r in rows), url=strings(r.url for r in rows),
        weight=np.array([r.weight for r in rows], dtype=np.float32), copies=np.array(copies, dtype=np.int32),
        legal=np.array([not p for p in problems], dtype=bool), problems=strings(problems),
        n_main=np.array([len(r.main) for r in rows], dtype=np.int16),
        n_extra=np.array([len(r.extra) for r in rows], dtype=np.int16),
    )  # fmt: skip
    ds.meta = {"format": FORMAT, "version": VERSION, **(meta or {}),
               "stats": dataset_stats(ds, raw=raw, unmapped=unmapped)}  # fmt: skip
    return ds


def dataset_stats(ds: DeckDataset, *, raw: int, unmapped: int) -> dict[str, Any]:
    """Counts for the report: lists, types, legal share, lists per type, sources, violation codes."""
    n = len(ds)
    per_type = np.bincount(ds.type, minlength=len(ds.types))
    legal_types = len(set(ds.type[ds.legal].tolist()))
    codes = Counter(c for p in ds.problems.tolist() if p for c in p.split(","))
    return {
        "raw_lists": raw, "unmapped_lists": unmapped, "duplicates": raw - unmapped - n, "lists": n,
        "types": len(ds.types), "legal": int(ds.legal.sum()), "legal_share": round(float(ds.legal.mean()), 4) if n else 0.0,
        "legal_types": legal_types, "cards": len(ds.passwords),
        "lists_per_type": {"median": float(np.median(per_type)) if n else 0.0, "min": int(per_type.min()) if n else 0,
                           "max": int(per_type.max(initial=0))},
        "by_kind": dict(Counter(ds.kind.tolist()).most_common()),
        "top_sources": dict(Counter(ds.source.tolist()).most_common(10)),
        "violations": dict(codes.most_common()),
        "dates": [min(ds.date.tolist()), max(ds.date.tolist())] if n else [],
    }  # fmt: skip


__all__ = ["COLUMNS", "DECKS_FILE", "DeckDataset", "FORMAT", "META_FILE", "build_deck_dataset", "dataset_stats",
           "load_deck_dataset"]  # fmt: skip
