"""Frozen card-text / effect-text vector tables (produced offline by T5.2), aligned to a ``CardVocab``.

On-disk layout of a text directory (all ``.npy``; rows keyed by the 8-digit ``password``, never by name):

* ``card_text.npy`` float ``[N, Dc]`` + ``card_text_passwords.npy`` int ``[N]``: one vector per card text.
* ``effect_text.npy`` float ``[M, De]`` + ``effect_text_keys.npy`` int ``[M, 2]``: one vector per effect
  string, keyed by ``(password, string index 0..15)`` (cdb ``texts.str1..16``; ``aux.Stringid(code, n)``).

Either pair may be missing; that feature is then absent. Passwords not in the vocab are ignored; vocab
entries without a vector (and the padding / unknown indices) get a zero vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ygorl.cards.cdb import CardVocab

EFFECT_SLOTS = 17  # column = string index + 1 (action column 4 / chaining value2); column 0 = no string

CARD_FILES = ("card_text.npy", "card_text_passwords.npy")
EFFECT_FILES = ("effect_text.npy", "effect_text_keys.npy")


@dataclass
class TextFeatures:
    """``card[v]``: card-text vector of vocab index ``v``; ``effect[effect_lookup[v, s]]``: effect string ``s - 1``.

    ``effect`` row 0 is the zero vector (``effect_lookup`` is 0 where the card has no such string).
    """

    card: np.ndarray | None  # [V, Dc] float32
    effect: np.ndarray | None  # [M + 1, De] float32
    effect_lookup: np.ndarray | None  # [V, EFFECT_SLOTS] int64

    def __post_init__(self) -> None:
        if (self.effect is None) != (self.effect_lookup is None):
            raise ValueError("effect and effect_lookup go together")
        if self.effect_lookup is not None and self.effect_lookup.shape[1] != EFFECT_SLOTS:
            raise ValueError(f"effect_lookup must have {EFFECT_SLOTS} columns")

    @property
    def vocab_size(self) -> int | None:
        if self.card is not None:
            return self.card.shape[0]
        return None if self.effect_lookup is None else self.effect_lookup.shape[0]

    def card_effect_mean(self) -> np.ndarray | None:
        """``[V, De]``: mean of each card's effect-string vectors (zero for cards without strings)."""
        if self.effect is None:
            return None
        total = np.zeros((self.effect_lookup.shape[0], self.effect.shape[1]), dtype=np.float32)
        for slot in range(1, EFFECT_SLOTS):  # row 0 of ``effect`` is zero, so missing strings add nothing
            total += self.effect[self.effect_lookup[:, slot]]
        count = np.maximum((self.effect_lookup > 0).sum(1, keepdims=True), 1)
        return (total / count).astype(np.float32)

    @classmethod
    def from_arrays(cls, vocab: CardVocab, card_vectors=None, card_passwords=None, effect_vectors=None,
                    effect_keys=None) -> TextFeatures:  # fmt: skip
        """Align password-keyed vectors to ``vocab`` indices."""
        card = None
        if card_vectors is not None:
            card_vectors = np.asarray(card_vectors, dtype=np.float32)
            card = np.zeros((len(vocab), card_vectors.shape[1]), dtype=np.float32)
            for row, pw in enumerate(np.asarray(card_passwords, dtype=np.int64)):
                if int(pw) in vocab:
                    card[vocab.index(int(pw))] = card_vectors[row]
        effect = lookup = None
        if effect_vectors is not None:
            effect_vectors = np.asarray(effect_vectors, dtype=np.float32)
            effect = np.zeros((effect_vectors.shape[0] + 1, effect_vectors.shape[1]), dtype=np.float32)
            effect[1:] = effect_vectors
            lookup = np.zeros((len(vocab), EFFECT_SLOTS), dtype=np.int64)
            for row, (pw, string) in enumerate(np.asarray(effect_keys, dtype=np.int64).reshape(-1, 2)):
                if int(pw) in vocab and 0 <= string < EFFECT_SLOTS - 1:
                    lookup[vocab.index(int(pw)), string + 1] = row + 1
        return cls(card, effect, lookup)

    @classmethod
    def load(cls, path: str | Path, vocab: CardVocab) -> TextFeatures | None:
        """Read a text directory; None when it holds neither table (the text features are then off)."""
        path = Path(path)
        has_card = all((path / f).is_file() for f in CARD_FILES)
        has_effect = all((path / f).is_file() for f in EFFECT_FILES)
        if not (has_card or has_effect):
            return None
        arrays = {}
        if has_card:
            arrays |= {"card_vectors": np.load(path / CARD_FILES[0]), "card_passwords": np.load(path / CARD_FILES[1])}
        if has_effect:
            arrays |= {"effect_vectors": np.load(path / EFFECT_FILES[0]), "effect_keys": np.load(path / EFFECT_FILES[1])}
        return cls.from_arrays(vocab, **arrays)

    @classmethod
    def random(cls, vocab_size: int, card_dim: int, effect_dim: int, rng: np.random.Generator,
               strings_per_card: int = 3) -> TextFeatures:  # fmt: skip
        """Random tables for tests and smoke runs (every real card gets text and a few effect strings)."""
        card = rng.standard_normal((vocab_size, card_dim)).astype(np.float32)
        card[: CardVocab.FIRST_INDEX] = 0
        lookup = np.zeros((vocab_size, EFFECT_SLOTS), dtype=np.int64)
        rows = 0
        for v in range(CardVocab.FIRST_INDEX, vocab_size):
            for s in rng.choice(EFFECT_SLOTS - 1, size=strings_per_card, replace=False):
                rows += 1
                lookup[v, s + 1] = rows
        effect = rng.standard_normal((rows + 1, effect_dim)).astype(np.float32)
        effect[0] = 0
        return cls(card, effect, lookup)
