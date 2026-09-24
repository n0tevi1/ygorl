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
from ygorl.env.encoded import EncodedVecEnv
from ygorl.eval.arena import GameRecord, derive_seed
from ygorl.nets.batch import collate


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
                  temperature: float = 1.0, pairs_per_spec: int = 2) -> tuple[list[GameRecord], dict]:  # fmt: skip
    """Play ``specs`` with ``policy_a`` on deck a and ``policy_b`` (default: the same module) on deck b.

    Returns one record per spec (spec order; ``pair`` = spec index // ``pairs_per_spec``) and timing stats. The
    policies must share the environment's card vocab and event length (they are not checked here).
    """
    policy_b = policy_a if policy_b is None else policy_b
    device = torch.device(device)
    for m in {id(policy_a): policy_a, id(policy_b): policy_b}.values():
        m.eval()
    min_batch = max(1, min_batch if min_batch is not None else env.num_envs // 2)
    records: list[GameRecord | None] = [None] * len(specs)
    queue = iter(enumerate(specs))
    running: dict[int, list[int]] = {}  # env -> [spec index, decisions answered by seat 0, by seat 1]
    t0, forwards, decisions, forward_s = time.perf_counter(), 0, 0, 0.0

    def launch(env_id: int) -> bool:
        nxt = next(queue, None)
        if nxt is None:
            return False
        running[env_id] = [nxt[0], 0, 0]
        env.reset(env_id, nxt[1])
        return True

    active = sum(launch(e) for e in range(env.num_envs))
    while active:
        groups: dict[int, list] = {}
        for ev in env.recv(min(min_batch, active)):
            i = running[ev.env_id][0]
            spec = specs[i]
            if ev.result is not None:
                res = ev.result
                w = res.get("winner")
                deck_of_seat = [(spec.first + p) % 2 for p in (0, 1)]  # engine seat p holds deck (first + p) % 2
                lp = res.get("lp", (0, 0))
                records[i] = GameRecord(pair=i // pairs_per_spec, seed=spec.seed, first=spec.first,
                                        winner=None if w is None else deck_of_seat[w], reason=str(res.get("reason", "")),
                                        turns=int(res.get("turns", 0)), decisions=int(res.get("decisions", 0)),
                                        win_reason=res.get("win_reason"),
                                        lp=(lp[deck_of_seat.index(0)], lp[deck_of_seat.index(1)]),
                                        error=str(res.get("error", "")))  # fmt: skip
                if not launch(ev.env_id):
                    active -= 1
                continue
            module = policy_a if (spec.first + ev.player) % 2 == 0 else policy_b
            groups.setdefault(id(module), [module, []])[1].append(ev)
        for module, events in groups.values():
            f0 = time.perf_counter()
            batch = collate([ev.obs for ev in events], device)
            fn = getattr(module, "policy_logits", None)
            logits = (fn(batch) if fn is not None else module(batch).logits).float()
            if greedy:
                actions = logits.argmax(-1).cpu().numpy()
            else:
                probs = torch.softmax(logits / temperature, -1).cpu().numpy().astype(np.float64)
                cdf = np.cumsum(probs, -1)
                actions = np.empty(len(events), dtype=np.int64)
                for k, ev in enumerate(events):
                    slot = running[ev.env_id]
                    u = _uniform(specs[slot[0]].seed, specs[slot[0]].first, ev.player, slot[1 + ev.player])
                    actions[k] = min(int(np.searchsorted(cdf[k], u * cdf[k, -1], side="right")), probs.shape[1] - 1)
            forward_s += time.perf_counter() - f0
            forwards += 1
            for ev, a in zip(events, actions.tolist(), strict=True):
                running[ev.env_id][1 + ev.player] += 1
                decisions += 1
                env.step(ev.env_id, int(a))
    seconds = time.perf_counter() - t0
    stats = {"games": len(specs), "decisions": decisions, "seconds": seconds, "forwards": forwards,
             "forward_s": forward_s, "games_per_s": len(specs) / seconds if seconds else 0.0,
             "decisions_per_s": decisions / seconds if seconds else 0.0}  # fmt: skip
    return records, stats  # type: ignore[return-value]


__all__ = ["paired_specs", "play_policies"]
