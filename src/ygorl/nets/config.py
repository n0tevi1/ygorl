"""Network hyper-parameters (T4b.1 / T4b.2); see docs/nets.md."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ygorl.nets.text import TextFeatures

HistoryKind = Literal["transformer", "lstm", "none"]


@dataclass(frozen=True)
class NetConfig:
    """Everything needed to rebuild a :class:`ygorl.nets.PolicyNet` (plus the frozen text tables, if any).

    ``card_text_dim`` / ``effect_text_dim`` are the widths of the frozen text tables the network is
    built with (0 = feature off). Set them from loaded tables with :meth:`with_text`, which also
    honours the ``card_text`` / ``effect_text`` switches.
    """

    vocab_size: int  # len(CardVocab): embedding rows, including padding (0) and unknown (1)
    d_model: int = 128
    n_heads: int = 4
    ff_mult: int = 2  # feed-forward width = ff_mult * d_model
    board_layers: int = 2  # card-table Transformer depth
    dropout: float = 0.0
    # card identity: structured features are always on; each identity source is switchable
    id_embedding: bool = True  # learned per-card embedding (off = generalise from text + structure only)
    card_text: bool = True  # use the frozen card-text table when one is supplied
    effect_text: bool = True  # use the frozen effect-text table when one is supplied
    card_text_dim: int = 0  # width of the supplied card-text vectors (0 = off)
    effect_text_dim: int = 0  # width of the supplied effect-text vectors (0 = off)
    # card facts (docs/nets.md「卡片事实」; experimental): archetype membership and referenced archetypes through one
    # shared archetype table, effect-category bits, script (action, location) queries; set from tables by with_text
    card_facts: bool = False  # opt-in: use the card facts when the feature directory has them (a file appearing
    # later must not change the network of a run being resumed)
    n_archetypes: int = 0  # archetype table rows - 1 (0 = off)
    n_archetype_slots: int = 0
    n_reference_slots: int = 0
    n_categories: int = 0
    n_queries: int = 0
    # training only: drop each card's ID embedding with this probability, so the other views must carry it
    id_dropout: float = 0.0
    # history over the event token stream (docs/encoding.md "事件 token 流")
    history: HistoryKind = "transformer"
    history_layers: int = 2
    history_mem_len: int = 128  # Transformer: past tokens kept per layer in the streaming state
    gate_bias: float = 2.0  # GTrXL gate bias: > 0 starts every block close to the identity map
    # detached belief features (T4c.1, design 04 channel (a)); 0 = no belief input
    belief_dim: int = 0

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError(f"d_model={self.d_model} is not divisible by n_heads={self.n_heads}")
        if self.history not in ("transformer", "lstm", "none"):
            raise ValueError(f"unknown history module {self.history!r}")
        if self.vocab_size < 2:
            raise ValueError("vocab_size must include the padding and unknown indices")
        if not 0 <= self.id_dropout < 1:
            raise ValueError("id_dropout must be in [0, 1)")

    def with_text(self, text: TextFeatures | None) -> NetConfig:
        """Set the text widths from ``text`` (None = no tables), respecting the on/off switches."""
        card = text.card.shape[1] if text is not None and text.card is not None and self.card_text else 0
        effect = text.effect.shape[1] if text is not None and text.effect is not None and self.effect_text else 0
        facts = {"n_archetypes": 0, "n_archetype_slots": 0, "n_reference_slots": 0, "n_categories": 0, "n_queries": 0}
        if text is not None and text.has_facts and self.card_facts:
            facts = {"n_archetypes": text.n_archetypes, "n_archetype_slots": text.archetypes.shape[1],
                     "n_reference_slots": text.references.shape[1], "n_categories": text.categories.shape[1],
                     "n_queries": text.queries.shape[1]}  # fmt: skip
        return replace(self, card_text_dim=card, effect_text_dim=effect, **facts)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> NetConfig:
        return cls(**data)
