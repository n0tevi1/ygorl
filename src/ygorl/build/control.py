"""Critic control variate for paired deck evaluation (#110, docs/spikes/deck-evolution.md §4, docs/tuning.md).

A game's result is partly the luck of the opening hand. The privileged critic reads that luck at the game's first
decision: ``luck = V(actual opening) - mean_j V(alternative opening j)``, where the alternatives are ``k`` other
shuffles of the same deck in the same game (same opponent order, same first player, same engine seed). Each
alternative shuffle is a uniform shuffle, like the actual one, so ``E[luck] = 0`` for any fixed critic, and
``score - beta * luck`` has the score's expectation whatever the critic's quality (a simplified AIVAT, Burch et al.
2018); it only lowers the variance as far as the critic predicts the outcome from the opening.

- Parent and child each use their own deck. The alternative shuffles re-permute the dealt deck by a permutation
  that depends on the game seed only, so a parent and an in-place child (:func:`ygorl.build.tuner.apply`) get the same
  alternative hands but for the edited card: their lucks correlate like their games do.
- ``beta`` is fixed, not fitted on the same games (fitting it would bias the estimate slightly); 0.5 turns the
  critic's +1 / -1 scale into the score's 1 / 0.
- The critic's side: V is from the deciding player's view, so it is negated when the first decision is the
  opponent's (zero-sum). A game with no decision (NaN) gets no adjustment.

:class:`CriticControlVariate` is the evaluator's pluggable ``control``
(:class:`ygorl.build.tuner.PairedEvaluator`); :func:`critic_opening_values` is its engine-backed value reader
(forward passes only: every game is abandoned at its first decision).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from ygorl.engine.duel import shuffle_deck
from ygorl.eval.arena import derive_seed

SALT = 3  # derive_seed stream of the alternative shuffles (1 and 2 are the evaluator's game seeds and opponents)


@dataclass
class CriticControlVariate:
    """Per game ``beta * (V(actual) - mean over k alternative shuffles of V)``, to subtract from the game's score.

    ``values`` maps game specs to the critic's value at each spec's first decision from deck a's side
    (:func:`critic_opening_values` in games; any function of the spec in tests)."""

    values: Callable[[Sequence[object]], np.ndarray]
    k: int = 8
    beta: float = 0.5
    seed: int = 0
    readings: int = 0  # value readings so far (1 + k per game)

    def alternatives(self, specs: Sequence[object]) -> list[object]:
        """``k`` specs per spec (spec-major): deck a's dealt main deck re-permuted by a seed-only permutation."""
        return [replace(sp, deck_a=replace(sp.deck_a, main=tuple(shuffle_deck(sp.deck_a.main,
                                                                              derive_seed(self.seed, SALT, sp.seed, j), 0))))
                for sp in specs for j in range(self.k)]  # fmt: skip

    def luck(self, specs: Sequence[object]) -> np.ndarray:
        """``V(actual) - mean_j V(alternative j)`` per spec (NaN when no value could be read)."""
        if not specs:
            return np.zeros(0)
        v = np.asarray(self.values([*specs, *self.alternatives(specs)]), dtype=float)
        self.readings += len(v)
        n = len(specs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN alternatives stay NaN
            return v[:n] - np.nanmean(v[n:].reshape(n, self.k), -1)

    def __call__(self, specs: Sequence[object]) -> np.ndarray:
        return self.beta * np.nan_to_num(self.luck(specs), nan=0.0)


def critic_opening_values(model, env_factory: Callable[[int], object], specs: Sequence[object], *,
                          device="cpu", num_envs: int = 256, min_batch: int = 64) -> np.ndarray:  # fmt: skip
    """The privileged critic's V at each spec's first decision, from deck a's side (NaN if the game never reached
    a decision or could not start). ``env_factory(n)`` builds an ``EncodedVecEnv`` with ``privileged=True`` over ``n`` games; no game is
    played past its first decision."""
    import torch

    from ygorl.env.driver import ABANDON, drive
    from ygorl.nets.actor_critic import collate_privileged
    from ygorl.nets.batch import collate

    values = np.full(len(specs), np.nan)
    if not specs:
        return values
    env = env_factory(min(num_envs, len(specs)))
    model.eval()

    @torch.no_grad()
    def decide(ready):
        out = model(
            collate([ev.obs for _, ev in ready], device), collate_privileged([ev.privileged for _, ev in ready], device)
        )
        for (game, ev), v in zip(ready, out.v.float().cpu().tolist(), strict=True):
            values[game.index] = v if ev.player == game.spec.seat_of_deck(0) else -v
        return [ABANDON] * len(ready)

    drive(env, specs, decide, lambda game, result: None, min_batch=min_batch,
          on_error=lambda i, spec, exc: None)  # fmt: skip
    return values


__all__ = ["CriticControlVariate", "critic_opening_values"]
