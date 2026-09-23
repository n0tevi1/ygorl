"""Public opponent evidence, the meta prior and the HDT-style filter for the belief heads (T4c.1).

Design 04 ("meta 先验"): the deck-composition heads start from the meta card table, and an
HDT-style exact filter (meta decks consistent with the cards seen so far, share-weighted card
distribution) is both the hard baseline and an input feature of the heads. Everything here reads
only the actor's observation (public information), so its output may enter the policy; the
ground truth for the losses comes from ``ygorl.env.privileged`` (T2.5). Pure numpy; the torch
heads are ``ygorl.nets.belief``. Specification: docs/belief-heads.md.

- :class:`MetaTable`: K meta deck types (+ "other"), their shares and lists over the C candidate
  cards ("meta union + generic cards"), card kinds, hand role bits, hash buckets for the rest.
- :func:`observe` / :class:`EvidenceTracker`: public evidence about the opponent from one
  observation (``cards`` / ``globals`` arrays, docs/encoding.md), running over a game.
- :func:`hdt_prior`: HDT-style posterior for every head plus the constraints the heads apply by
  construction (public hand copies, at most ``3 - visible`` hidden copies).
- :func:`role_targets`, :func:`responded_labels`: targets not covered by ``belief_targets``.
"""

from __future__ import annotations

import zlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields

import numpy as np

from ygorl.cards.cdb import CardDB, CardVocab
from ygorl.cards.ydk import Deck
from ygorl.engine import constants as C
from ygorl.env import events as E
from ygorl.env.privileged import MAX_COPIES, N_MZONE, P_SET, CandidateCards, Target, copy_counts

N_COPY_CLASSES = MAX_COPIES + 1
N_SET_ZONES = P_SET  # op_set rows: monster zones 0-6, spell/trap zones 7-14
FIELD_ZONE = N_MZONE + 5  # spell/trap zone sequence 5
DEFAULT_ROLES = ("hand_trap", "ash", "maxx_c", "nibiru", "extender")
ROLE_PASSWORDS = {"ash": (14558127,), "maxx_c": (23434538,), "nibiru": (27204311,)}
N_COUNTS = 5  # Evidence.counts: op_deck, hidden hand, hidden set, hidden removed, hidden extra

# card-table columns and location codes (docs/encoding.md)
_IDX, _LOC, _SEQ, _CTRL, _OWNER, _VISIBLE = 0, 1, 2, 4, 5, 7
_HAND, _MZONE, _SZONE, _REMOVED, _EXTRA = 2, 3, 4, 6, 7
_OP_DECK, _OP_HAND, _OP_EXTRA = 12, 13, 16  # globals


def password_bucket(password: int, n_buckets: int) -> int:
    """Stable hash bucket of a card password (crc32), for cards outside the candidate set."""
    return zlib.crc32(str(int(password)).encode()) % n_buckets


class MetaTable:
    """Meta deck types, their shares and compositions over the candidate cards, plus hand roles.

    ``decks``: ``name -> Deck`` of the K meta types; ``shares`` their prior weights (default
    uniform); ``other_share`` the prior of the "other" class (last index K). Candidates are the
    meta union followed by ``generic`` (in order, duplicates dropped); other cards are hashed into
    ``n_hash`` buckets on the input side only, so the number of output classes stays C.
    ``roles``: ``role -> passwords``; roles are bits over candidate cards (non-candidates ignored).
    """

    def __init__(self, vocab: CardVocab, db: CardDB, decks: Mapping[str, Deck], *, shares: Sequence[float] | None = None,
                 other_share: float = 0.1, generic: Iterable[int] = (), roles: Mapping[str, Iterable[int]] | None = None,
                 n_hash: int = 64) -> None:  # fmt: skip
        if not decks:
            raise ValueError("a meta table needs at least one deck type")
        if not 0 < other_share < 1:
            raise ValueError("other_share must be in (0, 1)")
        self.vocab = vocab
        self.names = [*decks, "other"]
        k = len(decks)
        w = np.ones(k) if shares is None else np.asarray(shares, dtype=float)
        if w.shape != (k,) or (w <= 0).any():
            raise ValueError(f"shares must be {k} positive weights")
        self.shares = np.append((1 - other_share) * w / w.sum(), other_share)
        union = [p for deck in decks.values() for p in (*deck.main, *deck.extra)]
        self.candidates = CandidateCards(vocab, [*union, *generic])
        n = len(self.candidates)
        self.counts = np.zeros((k, n), dtype=np.int64)
        for i, deck in enumerate(decks.values()):
            for p, m in deck.counts().items():
                self.counts[i, self.candidates.passwords.index(p)] += m
        # "other" hypothesis: the share-weighted mean list, rounded
        self.other_counts = np.rint(self.shares[:k] @ self.counts / self.shares[:k].sum()).astype(np.int64)
        cards = [db[p] for p in self.candidates.passwords]
        self.is_extra = np.array([c.is_extra_deck for c in cards])
        main_monster = np.array([c.is_monster and not c.is_extra_deck for c in cards])
        spell_trap = np.array([bool(c.type & (C.TYPE_SPELL | C.TYPE_TRAP)) for c in cards])
        field = np.array([bool(c.type & C.TYPE_FIELD) for c in cards])
        # which candidates a face-down card in each op_set zone can be
        self.zone_compat = np.zeros((N_SET_ZONES, n), dtype=bool)
        self.zone_compat[:N_MZONE] = main_monster
        self.zone_compat[N_MZONE:] = spell_trap & ~field
        self.zone_compat[FIELD_ZONE] = field
        roles = dict(roles or {})
        self.role_names = list(roles)
        self.roles = np.zeros((len(roles), n), dtype=bool)
        for r, passwords in enumerate(roles.values()):
            cols = self.candidates.columns(np.array([vocab.index(p) for p in passwords], dtype=np.int64))
            self.roles[r, cols[cols >= 0]] = True
        self.n_hash = n_hash
        size = len(vocab)
        self.hash_bucket = np.full(size, -1, dtype=np.int64)  # vocab index -> bucket (non-candidate, non-token)
        is_token = np.zeros(size, dtype=bool)
        for i in range(CardVocab.FIRST_INDEX, size):
            p = vocab.password(i)
            card = db.get(p)
            is_token[i] = card is not None and card.is_token
            if self.candidates.column[i] < 0 and not is_token[i]:
                self.hash_bucket[i] = password_bucket(p, n_hash)
        self.is_token = is_token

    @property
    def n_deck_types(self) -> int:
        """K + 1 (the meta types and "other")."""
        return len(self.names)

    @property
    def n_cards(self) -> int:
        return len(self.candidates)

    @property
    def n_roles(self) -> int:
        return len(self.role_names)

    @property
    def feature_dim(self) -> int:
        """Width of :attr:`BeliefPrior.features`."""
        return self.n_deck_types + 1 + 4 * self.n_cards + self.n_roles + self.n_hash + N_COUNTS

    def deck_type(self, deck: Deck) -> int:
        """Index of the meta type whose list equals ``deck`` (as a multiset), else K ("other")."""
        want = deck.counts()
        for i, name in enumerate(self.names[:-1]):
            have = {p: int(m) for p, m in zip(self.candidates.passwords, self.counts[i]) if m}
            if have == dict(want):
                return i
        return len(self.names) - 1


def default_roles(generic: Iterable[Mapping], extenders: Iterable[int] = ()) -> dict[str, list[int]]:
    """Design 04 role bits: hand_trap (generic-pool role ``hand_trap``), ash, maxx_c, nibiru, extender.

    ``generic``: rows with ``password`` and ``role`` (tests/data/generic_pool.json); ``extenders``:
    the passwords counted as combo extenders (environment-specific, see docs/belief-heads.md).
    """
    roles = {"hand_trap": [int(c["password"]) for c in generic if c.get("role") == "hand_trap"]}
    roles |= {name: list(p) for name, p in ROLE_PASSWORDS.items()}
    roles["extender"] = [int(p) for p in extenders]
    return roles


# --- public evidence --------------------------------------------------------------------------


@dataclass
class Evidence:
    """Public evidence about the opponent (optionally batched along a leading N dimension).

    ``seen``: the most copies of each candidate the opponent owns that were visible at once so far
    (a lower bound on the copies in their deck list); ``seen_other``: the same for non-candidate
    cards, per hash bucket; ``visible``: copies visible now; ``public_hand``: copies visible in
    the opponent's hand now; ``counts``: op_deck, hidden hand, hidden set (face-down on field),
    hidden removed (face-down banished), hidden extra; ``set_zones``: op_set zone holds a hidden card.
    """

    seen: np.ndarray  # [..., C] int
    seen_other: np.ndarray  # [..., H] int
    visible: np.ndarray  # [..., C] int
    public_hand: np.ndarray  # [..., C] int
    counts: np.ndarray  # [..., 5] int
    set_zones: np.ndarray  # [..., 15] bool

    @staticmethod
    def stack(items: Sequence[Evidence]) -> Evidence:
        return Evidence(**{f.name: np.stack([getattr(e, f.name) for e in items]) for f in fields(Evidence)})

    def index(self, rows) -> Evidence:
        return Evidence(**{f.name: getattr(self, f.name)[rows] for f in fields(Evidence)})


def observe(cards: np.ndarray, globals_: np.ndarray, meta: MetaTable) -> Evidence:
    """Evidence from one observation (``obs["cards"]`` ``[N_CARDS, 23]``, ``obs["globals"]`` ``[22]``).

    ``seen`` equals ``visible`` here; :class:`EvidenceTracker` keeps the running maximum.
    """
    cards = np.asarray(cards)
    idx = cards[:, _IDX].astype(np.int64)
    loc, ctrl, owner = cards[:, _LOC], cards[:, _CTRL], cards[:, _OWNER]
    vis = (cards[:, _VISIBLE] == 1) & (idx >= CardVocab.FIRST_INDEX) & (idx < len(meta.is_token))
    safe = np.where(vis, idx, 0)
    mine = vis & (owner == 1) & ~meta.is_token[safe]
    col = np.where(mine, meta.candidates.column[safe], -1)
    visible = np.bincount(col[col >= 0], minlength=meta.n_cards)
    bucket = np.where(mine, meta.hash_bucket[safe], -1)
    other = np.bincount(bucket[bucket >= 0], minlength=meta.n_hash)
    in_hand = (loc == _HAND) & (col >= 0)
    public_hand = np.bincount(col[in_hand], minlength=meta.n_cards)
    op = ctrl == 1
    hidden = op & ~(cards[:, _VISIBLE] == 1) & (idx != 0)
    zones = np.zeros(N_SET_ZONES, dtype=bool)
    for zone_loc, base, n in ((_MZONE, 0, N_MZONE), (_SZONE, N_MZONE, N_SET_ZONES - N_MZONE)):
        seq = cards[hidden & (loc == zone_loc), _SEQ]
        zones[base + seq[seq < n]] = True
    visible_hand = int((op & (cards[:, _VISIBLE] == 1) & (loc == _HAND)).sum())
    visible_extra = int((op & (cards[:, _VISIBLE] == 1) & (loc == _EXTRA)).sum())
    counts = np.array([
        int(globals_[_OP_DECK]),
        max(int(globals_[_OP_HAND]) - visible_hand, 0),
        int(zones.sum()),
        int((hidden & (loc == _REMOVED)).sum()),
        max(int(globals_[_OP_EXTRA]) - visible_extra, 0),
    ], dtype=np.int64)  # fmt: skip
    return Evidence(visible.copy(), other, visible, public_hand, counts, zones)


class EvidenceTracker:
    """Running evidence for one viewer in one game: ``seen`` is the maximum of ``visible`` so far."""

    def __init__(self, meta: MetaTable) -> None:
        self.meta = meta
        self.reset()

    def reset(self) -> None:
        self._seen = np.zeros(self.meta.n_cards, dtype=np.int64)
        self._other = np.zeros(self.meta.n_hash, dtype=np.int64)

    def update(self, cards: np.ndarray, globals_: np.ndarray) -> Evidence:
        ev = observe(cards, globals_, self.meta)
        np.maximum(self._seen, ev.visible, out=self._seen)
        np.maximum(self._other, ev.seen_other, out=self._other)
        ev.seen, ev.seen_other = self._seen.copy(), self._other.copy()
        return ev


# --- HDT-style filter -------------------------------------------------------------------------


@dataclass
class BeliefPrior:
    """HDT-style posterior per head (shapes of ygorl.eval.beliefs, batched) plus construction constraints.

    ``public_hand`` / ``public_roles``: already public in hand -> 1 by construction;
    ``max_copies``: at most ``3 - visible`` copies can remain hidden (the copies seen are deducted).
    """

    deck_type: np.ndarray  # [N, K+1]
    remaining_copies: np.ndarray  # [N, C, 4]
    hand: np.ndarray  # [N, C]
    hand_roles: np.ndarray  # [N, R]
    set_cards: np.ndarray  # [N, S, C]
    features: np.ndarray  # [N, F] input features of the learned heads
    public_hand: np.ndarray  # [N, C] bool
    public_roles: np.ndarray  # [N, R] bool
    max_copies: np.ndarray  # [N, C] int in 0..3
    set_zones: np.ndarray  # [N, S] bool
    n_consistent: np.ndarray  # [N] meta types consistent with the cards seen

    def arrays(self) -> dict[str, np.ndarray]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


_LOG_FACTORIAL = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, 1025)))])


def _log_choose(n: np.ndarray, k: np.ndarray) -> np.ndarray:
    ok = (k >= 0) & (k <= n)
    n, k = np.clip(n, 0, 1024), np.clip(k, 0, 1024)
    return np.where(ok, _LOG_FACTORIAL[n] - _LOG_FACTORIAL[k] - _LOG_FACTORIAL[np.clip(n - k, 0, 1024)], -np.inf)


def hypergeometric(total: np.ndarray, success: np.ndarray, draws: np.ndarray, j: np.ndarray) -> np.ndarray:
    """P(j successes in ``draws`` draws without replacement from ``total`` holding ``success``) (broadcast)."""
    total, success, draws, j = np.broadcast_arrays(*(np.asarray(x, dtype=np.int64) for x in (total, success, draws, j)))
    with np.errstate(invalid="ignore"):
        log_p = _log_choose(success, j) + _log_choose(total - success, draws - j) - _log_choose(total, draws)
    return np.where(np.isfinite(log_p), np.exp(log_p), 0.0)


def hdt_prior(ev: Evidence, meta: MetaTable, *, eps: float = 1e-4) -> BeliefPrior:
    """HDT-style posterior (Bursztein 2016 / Hearthstone Deck Tracker) for a batch of evidence.

    Deck type: meta types whose list holds every card seen (``seen <= list`` and no off-list card
    seen), weighted by share; "other" keeps its share and takes everything when no type is
    consistent. For each hypothesis the unseen copies ``u = clip(list - visible, 0, 3 - visible)``
    of a main-deck card are spread over the hidden main-deck pool (deck + hidden hand + face-down
    field / banished) uniformly, so the copies left in the deck and the copies in hand are
    hypergeometric; extra-deck copies stay in the extra deck. Hand roles: >= 1 of the role's unseen
    copies in hand. Face-down cards: share-weighted unseen copies among the cards that can occupy
    that zone. The mixture over hypotheses is the posterior; multiclass outputs are smoothed by
    ``eps`` so that no class has probability 0.
    """
    seen, visible = ev.seen, ev.visible
    batched = seen.ndim == 2
    if not batched:
        return _unbatch(hdt_prior(Evidence(*(x[None] for x in (ev.seen, ev.seen_other, ev.visible, ev.public_hand,
                                                               ev.counts, ev.set_zones))), meta, eps=eps))  # fmt: skip
    n_types = meta.n_deck_types
    consistent = (seen[:, None, :] <= meta.counts[None]).all(-1) & (ev.seen_other.sum(-1) == 0)[:, None]  # [N, K]
    w = np.concatenate([consistent * meta.shares[:-1], np.full((len(seen), 1), meta.shares[-1])], 1)
    post = w / w.sum(1, keepdims=True)
    comp = np.concatenate([meta.counts, meta.other_counts[None]], 0)  # [K+1, C]
    cap = np.clip(MAX_COPIES - visible, 0, MAX_COPIES)  # [N, C]
    u = np.minimum(np.clip(comp[None] - visible[:, None], 0, None), cap[:, None])  # [N, K+1, C]
    deck, hand_n, set_n, removed_n, extra_n = (ev.counts[:, i, None, None] for i in range(N_COUNTS))
    pool = deck + hand_n + set_n + removed_n  # hidden main-deck cards
    main_u = np.minimum(np.where(meta.is_extra, 0, u), pool)
    extra_u = np.minimum(np.where(meta.is_extra, u, 0), extra_n)
    j = np.arange(N_COPY_CLASSES)
    in_deck = hypergeometric(pool[..., None], main_u[..., None], deck[..., None], j)  # [N, K+1, C, 4]
    copies = np.where(meta.is_extra[..., None], j == extra_u[..., None], in_deck)
    copies = np.einsum("nh,nhcj->ncj", post, copies)
    p_none = hypergeometric(pool, main_u, hand_n, 0)  # [N, K+1, C]
    hand = np.einsum("nh,nhc->nc", post, 1 - p_none)
    public_hand = ev.public_hand > 0
    hand = np.where(public_hand, 1.0, hand)
    role_u = np.minimum(np.einsum("nhc,rc->nhr", main_u, meta.roles.astype(np.int64)), pool)
    roles = np.einsum("nh,nhr->nr", post, 1 - hypergeometric(pool, role_u, hand_n, 0))
    public_roles = (public_hand.astype(np.int64) @ meta.roles.T.astype(np.int64)) > 0
    roles = np.where(public_roles, 1.0, roles)
    expected_u = np.einsum("nh,nhc->nc", post, main_u)
    set_w = expected_u[:, None, :] * meta.zone_compat[None]  # [N, S, C]
    fallback = np.where(meta.zone_compat.any(1, keepdims=True), meta.zone_compat, True).astype(float)
    total = set_w.sum(-1, keepdims=True)
    set_p = np.where(total > 0, set_w / np.maximum(total, 1e-300), fallback / fallback.sum(-1, keepdims=True))

    def smooth(p):
        return (1 - eps) * p + eps / p.shape[-1]

    copies_mean = (copies * j).sum(-1)
    feats = np.concatenate([
        post,
        np.log1p(consistent.sum(1, keepdims=True)) / np.log(n_types),
        hand, copies_mean / MAX_COPIES,
        roles,
        np.minimum(seen, MAX_COPIES) / MAX_COPIES, np.minimum(visible, MAX_COPIES) / MAX_COPIES,
        np.minimum(ev.seen_other, MAX_COPIES) / MAX_COPIES,
        ev.counts / np.array([40.0, 10.0, 10.0, 10.0, 15.0]),
    ], 1).astype(np.float32)  # fmt: skip
    return BeliefPrior(
        deck_type=smooth(post), remaining_copies=smooth(copies), hand=hand, hand_roles=roles, set_cards=smooth(set_p),
        features=feats, public_hand=public_hand, public_roles=public_roles, max_copies=cap, set_zones=ev.set_zones,
        n_consistent=consistent.sum(1),
    )  # fmt: skip


def _unbatch(p: BeliefPrior) -> BeliefPrior:
    return BeliefPrior(**{k: v[0] for k, v in p.arrays().items()})


# --- extra targets ----------------------------------------------------------------------------


def role_targets(priv: Mapping[str, np.ndarray], meta: MetaTable) -> Target:
    """``hand_roles`` ``[..., R]``: >= 1 card with the role in the opponent's hand (privileged ``op_hand``).

    Masked where a public hand card already has the role (1 by construction, like ``hand``).
    """
    hand = priv["op_hand"]
    roles = meta.roles.T.astype(np.int64)
    have = (copy_counts(hand, meta.candidates) @ roles) > 0
    public = (copy_counts(hand, meta.candidates, where=hand[..., 1] == 1) @ roles) > 0
    return Target(have.astype(np.int64), ~public)


_SOLVED = (E.EV["chain_solving"], E.EV["chain_negated"], E.EV["chain_disabled"])


def responded_labels(stream: np.ndarray, positions: Sequence[int]) -> np.ndarray:
    """Was the viewer's next activation responded to? One label per stream position (-1 unknown).

    ``stream``: the viewer's complete event-token stream ``[T, 20]`` (docs/encoding.md). For a
    position p (the number of tokens before a decision), take the first ``chaining`` token of the
    viewer (``player`` = 1) at index >= p, with chain number n; the label is 1 if the opponent
    (``player`` = 2) chains anything before link n resolves (``chain_solving`` / ``chain_negated``
    / ``chain_disabled`` of link n, or ``chain_end``), else 0; -1 if the stream ends first or the
    viewer never activates again.
    """
    kind, player, v1 = stream[:, E.TYPE], stream[:, E.PLAYER], stream[:, E.VALUE1]
    chaining = kind == E.EV["chaining"]
    mine = np.nonzero(chaining & (player == 1))[0]
    outcome: dict[int, int] = {}
    out = np.full(len(positions), -1, dtype=np.int64)
    for i, p in enumerate(positions):
        k = np.searchsorted(mine, p)
        if k == len(mine):
            continue
        start = int(mine[k])
        if start not in outcome:
            link, res = v1[start], -1
            for t in range(start + 1, len(stream)):
                if chaining[t] and player[t] == 2:
                    res = 1
                    break
                if (kind[t] in _SOLVED and v1[t] == link) or kind[t] == E.EV["chain_end"]:
                    res = 0
                    break
            outcome[start] = res
        out[i] = outcome[start]
    return out
