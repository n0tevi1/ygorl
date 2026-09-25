"""Curriculum modes: restrict the opponent's responses during the learner's turn (T2.6).

Design challenge C3 (docs/design/02-challenges.md): long first-turn combos are
learned in stages, 单人展开 (``solo``) -> 仅手坑 (``handtrap``) -> 完整对局
(``full``). A mode restricts only the **opponent** (the deck that is not
``DuelConfig.learner``) and only **while the learner is the turn player**; the
opponent's own turn is always played in full. See docs/curriculum.md.

The functions here are pure: given a decision and its legal actions they say
which actions the restricted opponent may choose (:func:`allowed_actions`) and
whether the host answers for it (:func:`auto_action`). ``DuelTracker`` applies
them; the host's answers still go through ``set_response`` and into the
response log, so replays reproduce the game exactly.

Rules, for a decision that has a passive answer (``pass`` / ``no`` /
``cancel``):

- ``solo``: only the passive answer is allowed, so the host always answers.
- ``handtrap``: activation choices keep only effects activated from the hand
  (``SELECT_CHAIN`` options and ``SELECT_EFFECTYN`` prompts whose card is in
  ``LOCATION_HAND``: Ash Blossom, Maxx "C", Effect Veiler, Nibiru, Infinite
  Impermanence from the hand, ...) plus the passive answer. Other prompts
  (``SELECT_YESNO``, cancelable card selections) belong to effects that are
  already resolving and are left to the agent.
- ``full``: nothing is restricted.

A decision without a passive answer (a forced chain of mandatory triggers, a
card the core makes the opponent pick, a zone, a position...) is never
restricted: the opponent's agent answers it in every mode.
"""

from __future__ import annotations

from collections.abc import Sequence

from ygorl.engine import constants as C
from ygorl.engine import messages as M
from ygorl.engine.actions import Action

FULL, SOLO, HANDTRAP = "full", "solo", "handtrap"
MODES = (FULL, SOLO, HANDTRAP)
PASSIVE_KINDS = frozenset({"pass", "no", "cancel"})
_ACTIVATION_KINDS = {M.SelectChain: "chain", M.SelectEffectYn: "yes"}


def is_hand_activation(action: Action) -> bool:
    """The action activates an effect of a card in the hand (the ``handtrap`` criterion)."""
    return action.card is not None and action.card.loc.location == C.LOCATION_HAND


def allowed_actions(mode: str, decision: M.Decision, actions: Sequence[Action]) -> list[int]:
    """Indices of ``actions`` the restricted opponent may choose in the learner's turn."""
    everything = list(range(len(actions)))
    passive = [i for i, a in enumerate(actions) if a.kind in PASSIVE_KINDS]
    if mode == FULL or not passive:
        return everything
    if mode == SOLO:
        return passive
    if mode == HANDTRAP:
        kind = _ACTIVATION_KINDS.get(type(decision))
        if kind is None:
            return everything
        return [
            i for i, a in enumerate(actions) if a.kind in PASSIVE_KINDS or (a.kind == kind and is_hand_activation(a))
        ]
    raise ValueError(f"unknown curriculum mode {mode!r} (expected one of {MODES})")


def auto_action(mode: str, decision: M.Decision, actions: Sequence[Action]) -> int | None:
    """The index the host answers with on the opponent's behalf, or None to ask its agent."""
    if mode == FULL:
        return None
    allowed = allowed_actions(mode, decision, actions)
    if len(allowed) == 1 and actions[allowed[0]].kind in PASSIVE_KINDS:
        return allowed[0]
    return None


__all__ = ["FULL", "HANDTRAP", "MODES", "PASSIVE_KINDS", "SOLO", "allowed_actions", "auto_action", "is_hand_activation"]
