# Budget1 confirmation: fewer cancellations, no demonstrated strength gain

The confirmation supports a narrower conclusion than “better play”: budget1
reduces cancellation overhead, but this panel does not show stronger or shorter
ordinary games. Do not change the default mask. The single newly lost win was
reproduced and traced to downstream sampling alignment in that specific game,
not a necessary second target revision. Further blind budget evaluation is not
the next priority; investigate the remaining long-game behavior directly.

## Completed and independently checked

Root: `out/research/cancel-budget-confirmation-2026-10-09/`.
Identity: `fd440b0f121cfd26d84486b486f21e47ea18dc4d315a74d37e3f9fec63723d42`.
All1,536 games completed healthy. Independent `review.py` / `completion-review.json`
checks exact panel coverage, raw record and replay hashes, baseline unwrapped
identity, cold replay and cancellation-only mask interventions with nonempty
legal choices. Every cell's wins, turns and cancellations are recomputed.

Three frozen checkpoints, two opponents, 32 new shared deal blocks with four
assignments/starts give768 inputs per budget. These are correlated checkpoints
and shared deals, not768 independent samples from a learner population.

| Candidate / opponent | Wins32 →1 (/128) | Cancels32 →1 | Mean turns32 →1 |
| --- | ---: | ---: | ---: |
| seed0 cold u512 / Greedy | 97 →97 | 48 →7 | 8.5234 →8.5469 |
| seed0 cold u512 / historical RL | 97 →97 | 3 →1 | 8.5156 →8.5625 |
| seed0 warm u512 / Greedy | 89 →88 | 101 →11 | 8.2813 →8.3672 |
| seed0 warm u512 / historical RL | 87 →87 | 76 →8 | 8.6250 →8.6406 |
| seed1 cold u256 / Greedy | 86 →86 | 0 →0 | 10.1641 →10.1641 |
| seed1 cold u256 / historical RL | 80 →80 | 0 →0 | 9.8750 →9.8750 |

Descriptive totals: 536 →535 wins/768, 228 →27 cancellations (88.2% fewer),
8.9974 →9.0260 mean turns and436.19 →438.11 mean decisions. Only23 restricted
runs have a mask intervention,25 intervention decisions in total. No sole-exit
fallback occurs. In particular, the seed1 cold cells contain no intervention,
so their unchanged results supply no evidence about cancellation safety.

There is one lost win and no gained wins. The warm/Greedy difference is
-0.7813 percentage points, shared-deal bootstrap interval [-2.3438,0].
Other cell win intervals collapse to[0,0] because there are no observed winner
changes; this is not a guarantee of population equivalence. Sparse discordance
cannot rule out unseen harmful cases. These data do not establish that the
mask causes systematic regression either.

## Lost-game root cause: controlled continuation experiment

Warm/Greedy game66 has336 common decisions before the first difference. At turn8,
the baseline repeatedly selects87746184 and cancels the subsequent target window.
It eventually selects53927851 at decision390. Budget1 makes the same final
selection at336, omitting54 intermediate decisions. This is28 cancellations
versus1 in that window, not the full32-cancellation cap.

The ordinary baseline wins at turn16 after702 decisions; ordinary budget1 loses
at turn15 after638. `rng-alignment-audit.py` independently reruns both and requires
exact agreement with their original recorded actions and responses.

At the shared continuation (baseline391, restricted337), all actor input arrays
are equal but the candidate's sampling-generator states differ. In a separate
selected-case diagnostic, restoring only the baseline candidate generator state
at that boundary makes **every subsequent action and semantic step identical**.
The original win, turn16 and final LP[4500,-2200] return, with648 decisions:
exactly702 minus the54 removed steps.

This establishes the sampling-stream explanation for this selected discrepancy.
It does not overwrite the observed loss, alter the evaluation totals, justify
RNG surgery in deployment, or establish general safety of restricting target
revisions. Artifacts: `rng-alignment-audit.json` and all three `alignment-replays/`.

## Legal target-change completion check

The older known counterexample had only verified entry into a different target's
material prompt. `target-completion-audit.py` extends it using real engine
snapshots and bounded forward-only selection search. At the recorded decision479,
budgets32 and1 allow the first cancel; all three offered targets87758525,
46759931 and22908820 then admit material choices that reach `MSG_SELECT_PLACE`
with movement events and no retry/error. Budget0 blocks that initial revision.

Thus budget1 preserves complete material-selection routes in this one real
state, including the alternate target. This is not an exhaustive proof for other
states, optimal materials or full duel outcomes. Repeated legitimate revisions
and sole-exit exceptions remain distinct concerns. No full cancellation ban.

## Next diagnostic running: long-game battle exits

Root: `out/research/longgame-battle-audit-2026-10-09/`.
Identity: `c5a9c9ddb3472d0d85e3546dd08caddeec0be9b52a549f7ac711f0d2d2fc53b3`.
Reconstruct all768 budget32 controls without resampling the policy. Count battle
exits with an encoded legal attack, main-phase exits with proactive options and
per-turn attacks. An available attack is not necessarily good; these are replay
signals, not automatic error labels.

For >=16-turn games, probe only the first two such battle-exit windows and up to
four initial attacks, then at most128 decisions of Greedy continuation against
a passive opponent in that turn. Save any conditional winning lines and whether
they pass the existing event filter. Realized hidden state and a passive opponent
mean these are **not guaranteed lethal lines or safe training labels**, even if
the event filter passes. The finite search cannot prove that a win is absent.

A separate real-game preflight passed. Two CPU workers, frozen runtime/data,
per-game artifacts, three-hour cap and an independent watchdog are active.
Require exact original responses and steps after restoring all diagnostic
snapshots. Stop on health, identity or replay mismatch. Formal training and
policy/mask/reward defaults remain unchanged. Review positive cases and actual
opponent response possibilities before deciding on a tactical learning change.

## Battle-exit follow-up completed

All768 replays were independently verified. Finite continuation search found no
same-turn wins in the40 tested long-game windows. The completion-report schema
issue was repaired with a separate audited proof. See [results and phase-entry
follow-up](longgame-battle-audit-2026-10-09.md).
