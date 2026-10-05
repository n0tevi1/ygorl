"""Policy-vs-policy games on the C++ stepping path with batched inference (docs/evaluation.md「批量评估」).

:class:`ygorl.eval.arena.Arena` plays every game in a Python :class:`ygorl.engine.duel.Duel` and asks each policy
agent for one decision at a time on the CPU. :func:`play_policies` runs the games on
:class:`ygorl.env.encoded.EncodedVecEnv` instead and scores all the ready decisions of a policy in one forward pass
(on a GPU, if given), which is what deck tuning needs: many games of network policies on many decks.

- **Pairing** is the arena's: :func:`paired_specs` gives each deck pairing ``pairs`` seeds, both first players per
  seed, the same shuffled deck order in both games.
- **Sampling** draws each decision from the policy's masked softmax with a uniform number derived from
  ``(game seed, first player, seat, decision number)``, so the result does not depend on how the games were batched
  or on thread timing (the arena's per-agent RNG streams are different: same distribution, different samples).
- **Records** are :class:`ygorl.eval.arena.GameRecord` from deck a's side, so :func:`ygorl.eval.arena.summarize`
  and the paired comparisons work unchanged.
- **Scheduling** (spec queue, launch failures, slot reuse, ``min_batch``) is :func:`ygorl.env.driver.drive`'s; this
  module answers a round's decisions and turns results into records.

Only network policies play here (greedy / random agents live in Python; use the arena for them).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import torch
from torch import nn

from ygorl.cards.ydk import Deck
from ygorl.engine.duel import DuelConfig, shuffle_deck
from ygorl.env import GameSpec
from ygorl.env.driver import Game, drive
from ygorl.env.encoded import EncodedEvent, EncodedVecEnv
from ygorl.eval.arena import GameRecord, derive_seed
from ygorl.nets.batch import collate, policy_logits


def paired_specs(deck_a: Deck, deck_b: Deck, pairs: int, seed: int, config: DuelConfig) -> list[GameSpec]:
    """The arena's paired games of one deck pairing: per seed, deck a first then deck b first, same deck order."""
    base = replace(config, shuffle_decks=False)
    specs = []
    for p in range(pairs):
        s = derive_seed(seed, p)
        a, b = deck_a, deck_b
        if config.shuffle_decks:
            a = replace(deck_a, main=tuple(shuffle_deck(deck_a.main, s, 0)))
            b = replace(deck_b, main=tuple(shuffle_deck(deck_b.main, s, 1)))
        specs.extend(GameSpec(seed=s, deck_a=a, deck_b=b, first=first, config=base) for first in (0, 1))
    return specs


def _uniform(seed: int, first: int, seat: int, step: int) -> float:
    return float(np.random.default_rng([seed & 0xFFFFFFFFFFFFFFFF, first, seat, step]).random())


@torch.no_grad()
def play_policies(env: EncodedVecEnv, specs: Sequence[GameSpec], policy_a: nn.Module, policy_b: nn.Module | None = None,
                  *, device: torch.device | str = "cpu", min_batch: int | None = None, greedy: bool = False,
                  temperature: float = 1.0, pairs_per_spec: int = 2,
                  sample_seeds: Sequence[tuple[int, int]] | None = None,
                  sampling: tuple[tuple[bool, float], tuple[bool, float]] | None = None) -> tuple[list[GameRecord], dict]:  # fmt: skip
    """Play ``specs`` with ``policy_a`` on deck a and ``policy_b`` (default: the same module) on deck b.

    Returns one record per spec (spec order; ``pair`` = spec index // ``pairs_per_spec``) and timing stats. The
    policies must share the environment's card vocab and event length (they are not checked here).

    ``sample_seeds[i]`` (optional) seeds the decisions of deck a's and deck b's player in spec ``i`` instead of
    ``(game seed, first player, seat)``, e.g. to key them by something other than the seat (the agent matrix keys
    them by deck slot). ``sampling`` (optional) gives deck a's and deck b's player their own ``(greedy,
    temperature)`` instead of the shared ``greedy`` / ``temperature``.
    """
    policy_b = policy_a if policy_b is None else policy_b
    side_sampling = sampling if sampling is not None else ((greedy, temperature), (greedy, temperature))
    device = torch.device(device)
    for m in {id(policy_a): policy_a, id(policy_b): policy_b}.values():
        m.eval()
    min_batch = max(1, min_batch if min_batch is not None else env.num_envs // 2)
    records: list[GameRecord | None] = [None] * len(specs)
    t0, forwards, forward_s = time.perf_counter(), 0, 0.0

    def on_error(i: int, spec: GameSpec, exc: Exception) -> None:
        """A spec whose duel cannot start is recorded as an exception (as in the arena)."""
        records[i] = GameRecord(pair=i // pairs_per_spec, seed=spec.seed, first=spec.first, winner=None,
                                reason="exception", error=f"{type(exc).__name__}: {exc}")  # fmt: skip

    def on_result(game: Game, res: dict) -> None:
        spec = game.spec
        w = res.get("winner")
        lp = res.get("lp", (0, 0))
        reason = str(res.get("reason", ""))
        failed = reason == "error"  # the engine raised: an error, not a draw (the arena's "exception")
        records[game.index] = GameRecord(pair=game.index // pairs_per_spec, seed=spec.seed, first=spec.first,
                                         winner=None if w is None or failed else spec.deck_of_seat(w),
                                         reason="exception" if failed else reason,
                                         turns=int(res.get("turns", 0)), decisions=int(res.get("decisions", 0)),
                                         win_reason=res.get("win_reason"),
                                         lp=(lp[spec.seat_of_deck(0)], lp[spec.seat_of_deck(1)]),
                                         retries=int(res.get("retries", 0)),
                                         unknown_messages=int(res.get("unknown_messages", 0)),
                                         undecodable_messages=int(res.get("undecodable_messages", 0)),
                                         script_errors=len(res.get("script_errors", [])),
                                         error=str(res.get("error", "")))  # fmt: skip

    def decide(ready: list[tuple[Game, EncodedEvent]]) -> list[int]:
        """One forward pass per (module, greedy, temperature) group of the ready decisions."""
        nonlocal forwards, forward_s
        groups: dict[tuple, tuple[nn.Module, list[int]]] = {}
        for k, (game, ev) in enumerate(ready):
            side = game.spec.deck_of_seat(ev.player)  # 0: the player holding deck a
            module = policy_a if side == 0 else policy_b
            groups.setdefault((id(module), *side_sampling[side]), (module, []))[1].append(k)
        actions = [0] * len(ready)
        for (_, greedy_s, temp_s), (module, ks) in groups.items():
            f0 = time.perf_counter()
            logits = policy_logits(module, collate([ready[k][1].obs for k in ks], device)).float()
            if greedy_s:
                chosen = logits.argmax(-1).cpu().tolist()
            else:
                probs = torch.softmax(logits / temp_s, -1).cpu().numpy().astype(np.float64)
                cdf = np.cumsum(probs, -1)
                chosen = []
                for row, k in enumerate(ks):
                    game, ev = ready[k]
                    sp, n = game.spec, game.state[ev.player]  # decisions this seat answered before this one
                    if sample_seeds is not None:
                        u = _uniform(sample_seeds[game.index][sp.deck_of_seat(ev.player)], 0, 0, n)
                    else:
                        u = _uniform(sp.seed, sp.first, ev.player, n)
                    chosen.append(min(int(np.searchsorted(cdf[row], u * cdf[row, -1], side="right")),
                                      probs.shape[1] - 1))  # fmt: skip
            forward_s += time.perf_counter() - f0
            forwards += 1
            for k, a in zip(ks, chosen, strict=True):
                actions[k] = int(a)
                game, ev = ready[k]
                game.state[ev.player] += 1
        return actions

    decisions = drive(env, specs, decide, on_result, min_batch=min_batch, start=lambda i, spec: [0, 0],
                      on_error=on_error)  # fmt: skip
    seconds = time.perf_counter() - t0
    stats = {"games": len(specs), "decisions": decisions, "seconds": seconds, "forwards": forwards,
             "forward_s": forward_s, "games_per_s": len(specs) / seconds if seconds else 0.0,
             "decisions_per_s": decisions / seconds if seconds else 0.0}  # fmt: skip
    return records, stats  # type: ignore[return-value]


__all__ = ["paired_specs", "play_policies"]
