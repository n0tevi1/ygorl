"""Training-mode ground truth about the opponent (T2.5); the specification is docs/encoding.md.

These "privileged" tensors feed the critic and the belief-head losses only
(design I2/I9, docs/design/04-opponent-model.md); the actor never sees them.
They are therefore kept out of the actor's observation dict: environments
return them in a separate field (``EncodedEvent.privileged``) and only when
constructed with ``privileged=True``.

The C++ port (csrc/privileged.cpp) must match :func:`encode_privileged`
element by element; :func:`copy_counts` and :func:`belief_targets` are pure
numpy and shared by both backends.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from ygorl.cards.cdb import CardVocab
from ygorl.engine import constants as C
from ygorl.engine.query import parse_query_location

P_HAND, P_DECK, P_EXTRA, P_SET, P_REMOVED = 32, 64, 32, 15, 64
P_WIDTHS = {"op_hand": P_HAND, "op_deck": P_DECK, "op_extra": P_EXTRA, "op_set": P_SET, "op_removed": P_REMOVED}
PRIVILEGED_KEYS = (*P_WIDTHS, "counts")
P_COLS = 3  # card_index, public, sequence
N_MZONE, N_SZONE = 7, 8
PRIVILEGED_QUERY_FLAGS = C.QUERY_CODE | C.QUERY_POSITION  # QUERY_IS_PUBLIC is always returned
MAX_COPIES = 3  # remaining_copies classes 0..3


def _rows(entries: list[tuple[int, int, int]], width: int) -> np.ndarray:
    out = np.zeros((width, P_COLS), dtype=np.int32)
    n = min(len(entries), width)
    if n:
        out[:n] = entries[:n]
    return out


def encode_privileged(core, viewer: int, vocab: CardVocab) -> dict[str, np.ndarray]:
    """Ground truth of the opponent (``1 - viewer``) hidden zones; see docs/encoding.md.

    Every row is ``(card_index, public, sequence)``; hand and face-down banished cards keep
    their zone order, deck and extra deck are compositions sorted by ``(card_index, public)``,
    and ``op_set`` has one fixed row per monster zone (0-6) and spell/trap zone (7-14).
    """
    op = 1 - viewer

    def query(loc: int) -> list[dict | None]:
        return parse_query_location(core.query_location(PRIVILEGED_QUERY_FLAGS, op, loc))

    def entry(card: dict, seq: int) -> tuple[int, int, int]:
        return vocab.index(card["code"]), int(bool(card["public"])), seq

    def facedown(card: dict) -> bool:
        return bool(card.get("position", 0) & C.POS_FACEDOWN)

    hand = [entry(c, s) for s, c in enumerate(query(C.LOCATION_HAND)) if c]
    deck = sorted(entry(c, 0) for c in query(C.LOCATION_DECK) if c)
    extra = sorted(entry(c, 0) for c in query(C.LOCATION_EXTRA) if c)
    removed = [entry(c, s) for s, c in enumerate(query(C.LOCATION_REMOVED)) if c and facedown(c)]
    field = np.zeros((P_SET, P_COLS), dtype=np.int32)
    n_set = 0
    for loc, base, n in ((C.LOCATION_MZONE, 0, N_MZONE), (C.LOCATION_SZONE, N_MZONE, N_SZONE)):
        for seq, card in enumerate(query(loc)[:n]):
            if card and facedown(card):
                field[base + seq] = entry(card, seq)
                n_set += 1
    return {
        "op_hand": _rows(hand, P_HAND),
        "op_deck": _rows(deck, P_DECK),
        "op_extra": _rows(extra, P_EXTRA),
        "op_set": field,
        "op_removed": _rows(removed, P_REMOVED),
        "counts": np.array([len(hand), len(deck), len(extra), n_set, len(removed)], dtype=np.int32),
    }


# --- belief-head targets (docs/belief-eval.md) ------------------------------------------------


class CandidateCards:
    """The C candidate cards of the belief heads ("meta union + generic cards"), by password.

    Maps vocab indices to columns ``0..C-1``; cards outside the set map to -1.
    """

    def __init__(self, vocab: CardVocab, passwords: Iterable[int]) -> None:
        self.passwords = list(dict.fromkeys(passwords))
        missing = [p for p in self.passwords if p not in vocab]
        if missing:
            raise KeyError(f"candidate passwords not in the vocab: {missing[:5]}")
        self.column = np.full(len(vocab), -1, dtype=np.int64)
        self.column[[vocab.index(p) for p in self.passwords]] = np.arange(len(self.passwords))

    def __len__(self) -> int:
        return len(self.passwords)

    def columns(self, index: np.ndarray) -> np.ndarray:
        """Column of each vocab index (-1 for padding, unknown and non-candidate cards)."""
        index = np.asarray(index)
        valid = (index >= CardVocab.FIRST_INDEX) & (index < len(self.column))
        return np.where(valid, self.column[np.clip(index, 0, len(self.column) - 1)], -1)


def copy_counts(
    rows: np.ndarray, candidates: CandidateCards | CardVocab, *, where: np.ndarray | None = None
) -> np.ndarray:
    """Copies per card in a ``[..., P, 3]`` row list -> ``[..., C]`` int (not clipped).

    With a :class:`CandidateCards` the columns are its candidates; with the :class:`CardVocab`
    itself they are all vocab indices (``[..., len(vocab)]``, special tokens never counted).
    ``where`` (bool ``[..., P]``) restricts which rows count.
    """
    rows = np.asarray(rows)
    index = rows[..., 0].astype(np.int64)
    width = len(candidates)
    if isinstance(candidates, CardVocab):
        col = np.where((index >= CardVocab.FIRST_INDEX) & (index < width), index, -1)
    else:
        col = candidates.columns(index)
    keep = col >= 0
    if where is not None:
        keep &= np.asarray(where, dtype=bool)
    flat_col = col.reshape(-1, col.shape[-1])
    r, p = np.nonzero(keep.reshape(flat_col.shape))
    out = np.zeros((flat_col.shape[0], width), dtype=np.int64)
    np.add.at(out, (r, flat_col[r, p]), 1)
    return out.reshape(*rows.shape[:-2], width)


@dataclass(frozen=True)
class Target:
    """Targets and bool mask (True = train / evaluate), shaped like ygorl.eval.beliefs.Head targets."""

    targets: np.ndarray
    mask: np.ndarray


def belief_targets(priv: Mapping[str, np.ndarray], candidates: CandidateCards) -> dict[str, Target]:
    """Belief-head targets from privileged tensors (optionally batched); see docs/encoding.md.

    - ``hand`` ``[..., C]``: >= 1 copy in the opponent's hand; masked where a public copy is in
      hand (already public -> 1 by construction).
    - ``remaining_copies`` ``[..., C]``: hidden copies left in main + extra deck, clipped to 0..3
      (revealed copies are subtracted); all True.
    - ``set_cards`` ``[..., 15]``: candidate column of each face-down card, -1 for empty zones and
      non-candidate cards; masked to hidden candidate cards.
    """
    hand = priv["op_hand"]
    in_hand = copy_counts(hand, candidates) > 0
    public_in_hand = copy_counts(hand, candidates, where=hand[..., 1] == 1) > 0
    hidden = [copy_counts(priv[k], candidates, where=priv[k][..., 1] == 0) for k in ("op_deck", "op_extra")]
    remaining = np.minimum(hidden[0] + hidden[1], MAX_COPIES)
    field = priv["op_set"]
    col = candidates.columns(field[..., 0])
    return {
        "hand": Target(in_hand.astype(np.int64), ~public_in_hand),
        "remaining_copies": Target(remaining, np.ones(remaining.shape, dtype=bool)),
        "set_cards": Target(col, (col >= 0) & (field[..., 1] == 0)),
    }
