"""Suggestions for a board snapshot: the policy's action distribution averaged over hidden-card samples (design 10).

For each sample, :func:`ygorl.advisor.reconstruct.complete` fills in the hidden cards and
:func:`~ygorl.advisor.reconstruct.build` puts the board into the core; the network (a PPO or BC checkpoint,
``load_actor``) scores the legal actions of the advisor's idle decision. Actions are matched across samples by
(kind, card password, effect description), since the samples differ in the hidden cards but not in the
advisor's own options. The event history is empty (a reconstructed game has no past), and what the core cannot
be told is applied as a mask on the result: a used Normal Summon, monsters that cannot change position, the
Battle Phase in Main Phase 2.

Read-only: nothing here talks to a game client.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from ygorl.advisor.reconstruct import OpponentPrior, Reconstruction, build, complete
from ygorl.advisor.snapshot import BoardSnapshot
from ygorl.cards.ydk import Deck
from ygorl.engine import constants as C
from ygorl.engine.actions import Action
from ygorl.env.observer import PointObserver

GLOBAL_WENT_FIRST, GLOBAL_TURN, GLOBAL_PHASE = 1, 3, 4  # docs/encoding.md globals
NORMAL_SUMMON_KINDS = ("summon", "mset")


def action_key(action: Action) -> tuple[str, int, int]:
    return action.kind, action.card.code if action.card is not None else 0, action.description


def masked(snapshot: BoardSnapshot, action: Action) -> bool:
    """True for an action the core offers but the real game does not allow (what the core cannot be told)."""
    if action.kind in NORMAL_SUMMON_KINDS and snapshot.normal_summoned:
        return True
    if action.kind == "battle_phase" and snapshot.phase == "main2":
        return True
    if action.kind == "attack" and action.card is not None:
        loc = action.card.loc
        for card in snapshot.sides[0].cards:
            if card.location == "mzone" and card.sequence == loc.sequence and card.attacked:
                return True
    if action.kind == "reposition" and action.card is not None:
        loc = action.card.loc
        for card in snapshot.sides[0].cards:
            if card.location == "mzone" and card.sequence == loc.sequence and card.position_locked:
                return True
    return False


def observe(rec: Reconstruction, observer: PointObserver) -> dict[str, np.ndarray]:
    """The network input of a reconstruction, with the real turn, phase and first player restored."""
    snap = rec.board.snapshot
    obs = observer.encode(rec.point, rec.core)
    g = obs["globals"]
    g[GLOBAL_WENT_FIRST] = int(snap.went_first)
    g[GLOBAL_TURN] = min(snap.turn, 999)
    phase = {"main2": C.PHASE_MAIN2, "battle": C.PHASE_BATTLE}.get(snap.phase, C.PHASE_MAIN1)
    g[GLOBAL_PHASE] = (phase & -phase).bit_length()
    return obs


@dataclass
class Suggestion:
    action: Action  # from the first sample where it was legal
    prob: float  # mean over the samples (0 in samples where it was not legal)
    spread: float  # standard deviation over the samples
    legal_in: int  # samples in which it was legal


@dataclass
class Advice:
    snapshot: BoardSnapshot
    suggestions: list[Suggestion]  # most probable first
    samples: int
    consistent: int  # samples whose opponent deck held every public opponent card
    opponent_decks: dict[int, int] = field(default_factory=dict)  # prior deck index -> samples
    warnings: list[str] = field(default_factory=list)
    policy: bool = True  # False: no network, uniform over the legal actions


def _logits(rec: Reconstruction, policy, cards, starting_lp: int) -> np.ndarray:
    if policy is None:
        return np.zeros(len(rec.point.actions))
    import torch

    from ygorl.nets import collate

    obs = observe(rec, PointObserver(cards, policy.vocab, policy.event_length, starting_lp=starting_lp))
    with torch.no_grad():
        return policy.net(collate([obs])).logits[0].float().numpy()


def softmax_kept(logits: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Softmax over the ``keep`` entries (0 elsewhere)."""
    z = np.where(keep, logits, -np.inf)
    e = np.exp(z - z[keep].max())
    return e / e.sum()


def advise(snapshot: BoardSnapshot, own_deck: Deck, policy, prior: OpponentPrior, rule_flags: int, *, cards,
           scripts=None, samples: int = 8, seed: int = 0, starting_lp: int = 8000) -> Advice:  # fmt: skip
    """Average the policy's distribution over ``samples`` completions of ``snapshot`` (``policy``: ``load_actor``).

    Without a policy every legal (unmasked) action gets the same probability: a listing of the reconstruction's
    options, to check a snapshot (no PyTorch needed).
    """
    snapshot.check()
    rng = random.Random(seed)
    probs: dict[tuple, list[float]] = defaultdict(list)
    legal: dict[tuple, int] = defaultdict(int)
    first: dict[tuple, Action] = {}
    warnings: list[str] = []
    advice = Advice(snapshot, [], samples, 0, policy=policy is not None)
    for k in range(samples):
        board = complete(snapshot, own_deck, prior, cards, rng)
        rec = build(board, rule_flags, cards=cards, scripts=scripts, seed=rng.randrange(1, 1 << 62))
        warnings += [w for w in rec.warnings if w not in warnings]
        advice.consistent += board.consistent
        advice.opponent_decks[board.opponent_deck] = advice.opponent_decks.get(board.opponent_deck, 0) + 1
        logits = _logits(rec, policy, cards, starting_lp)
        actions = rec.point.actions
        n = min(len(actions), logits.shape[0])
        keep = np.array([not masked(snapshot, a) for a in actions[:n]])
        if not keep.any():
            raise ValueError("every legal action of the reconstruction is masked by the snapshot")
        p = softmax_kept(logits[:n], keep).tolist()
        seen = set()
        for a, q, ok in zip(actions[:n], p, keep, strict=True):
            if not ok:  # the real game forbids it: never suggested, never listed
                continue
            key = action_key(a)
            if key in seen:  # copies of one card: merge
                probs[key][-1] += q
                continue
            seen.add(key)
            legal[key] += 1
            first.setdefault(key, a)
            probs[key].extend([0.0] * (k - len(probs[key])))
            probs[key].append(q)
    for key, ps in probs.items():
        full = ps + [0.0] * (samples - len(ps))
        arr = np.asarray(full)
        advice.suggestions.append(Suggestion(first[key], float(arr.mean()), float(arr.std()), legal[key]))
    advice.suggestions.sort(key=lambda s: -s.prob)
    advice.warnings = warnings
    return advice


def format_advice(advice: Advice, cards: Mapping, *, top: int = 10, deck_names: list[str] | None = None) -> str:
    from ygorl.commands.branch import describe_action

    snap = advice.snapshot
    lines = [f"turn {snap.turn} {snap.phase}, LP {snap.sides[0].lp} vs {snap.sides[1].lp}, "
             f"{advice.samples} samples ({advice.consistent} with a consistent opponent deck)"]  # fmt: skip
    if deck_names:
        guesses = sorted(advice.opponent_decks.items(), key=lambda kv: -kv[1])
        lines.append("opponent deck samples: " + ", ".join(f"{deck_names[i]} {n}" for i, n in guesses))
    if advice.policy:
        lines.append(f"{'prob':>6} {'±':>5}  action")
        for s in advice.suggestions[:top]:
            lines.append(f"{s.prob:6.3f} {s.spread:5.3f}  {describe_action(s.action, cards)}")
    else:
        lines.append(f"no policy: the reconstruction's legal actions ({len(advice.suggestions)}), legal in k/N samples")
        for s in sorted(advice.suggestions, key=lambda s: -s.legal_in):
            lines.append(f"{s.legal_in:>3}/{advice.samples}  {describe_action(s.action, cards)}")
    lines.append("no event history (reconstructed board); once-per-turn usage is not modelled")
    lines += [f"warning: {w}" for w in advice.warnings]
    return "\n".join(lines)


__all__ = ["Advice", "Suggestion", "action_key", "advise", "format_advice", "masked", "observe", "softmax_kept"]
