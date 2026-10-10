# Fresh sampling confirmation: reject global cooling, retain bounded argmax mitigation

The T0.5 win improvement did not reproduce on fresh deals. Argmax wins more
on this panel, but per-cell intervals remain unresolved. Budget1 removes its
repeated-selection overhead without changing observed terminal outcomes;
this is a bounded mitigation result, not proof of universally stronger play.
Keep formal training and inference defaults unchanged.

## Independent review

Root: `out/research/policy-sampling-confirmation-2026-10-09/`.
Identity: `7f68eb7b214bf7c1084bdedcf198328de1cd8460e351eb349799654ad8e5a262`.
All 3,072 games ended normally. Independent review verifies identity files,
complete panel coverage, records, replay hashes, cold action/response/step
replays, and all 21 mask interventions. No engine errors, termination limits
or sole-exit fallbacks. Completion was acknowledged after review.

Thirty-two new shared deal blocks, three frozen checkpoints, two unchanged
opponents and four seat/deck assignments per cell. Engine seeds are disjoint
from the earlier development panel; the deck pool and fitted models are shared.
Each mode has 768 games. Intervals condition on these fixed models and resample
32 paired deal blocks within cell; they are not independent training-seed
inference and are unadjusted across contrasts.

| Mode | Wins /768 | Mean turns | Games >20 turns | Cancels | Games reaching cancel32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| T1 | 548 | 9.4857 | 16 | 150 | 1 |
| T0.5 | 529 | 9.1589 | 6 | 197 | 3 |
| argmax | 574 | 9.4635 | 15 | 672 | 20 |
| argmax + budget1 | 574 | 9.4635 | 15 | 21 | 0 |

| Candidate / opponent | Wins T1 / T0.5 / argmax (each /128) |
| --- | ---: |
| seed0 cold u512 / Greedy | 99 / 98 / 102 |
| seed0 cold u512 / historical RL | 90 / 85 / 94 |
| seed0 warm u512 / Greedy | 103 / 100 / 108 |
| seed0 warm u512 / historical RL | 85 / 95 / 93 |
| seed1 cold u256 / Greedy | 91 / 88 / 92 |
| seed1 cold u256 / historical RL | 80 / 63 / 85 |

T0.5 loses 19 wins descriptively despite shorter mean games. Seed1 cold versus
historical RL loses 17/128 wins, paired difference −13.281pp with conditional
interval [−21.875,−5.469]. The other temperature win intervals cross zero.
This contradicts adopting cooling as a general inference improvement. Do not
select it because the long-game tail shrank.

Argmax gains 26 wins descriptively, with positive point differences in all six
cells, but every cell's paired win interval crosses zero. Its mean duration is
almost unchanged from T1; the earlier shorter-game effect is not consistently
reproduced. Against Greedy, all three point win differences now improve, whereas
they declined on the earlier panel. Treat opponent-specific conclusions from
one small panel cautiously. No default promotion.

## Exact cancellation-overhead audit

`audit-loop-removal.py` checks all 768 argmax/budget1 pairs. Every terminal result
field other than decision count is identical, including winner, turns, LP and
health. In the 20 changed games, comparing full step sequences removes only
select/cancel pairs; every retained semantic step is identical. There are
21 intervention windows, 651 fewer cancellations and **1,302 fewer decisions**.
Mean decisions fall 448.0781→446.3828. Remaining 21 cancellations preserve the
first cancellation in each window; there are no sole-exit cases in this panel.

This establishes removal of repeated-choice overhead on these observed
trajectories. Only 20 games actually change, so unchanged results elsewhere and
a zero-width empirical outcome-difference interval are not a universal safety
bound. It does not prove that a second target revision is always unnecessary.
It also does not imply sampled-policy invariance: [the seed1 warm loop audit](seed-1-warm-u512-2026-10-10.md)
shows that fewer sampled decisions can change the RNG continuation and duration.
No engine/script changes or tactical training labels follow from this finding.

Evidence: `reviewed-completion.json`, `loop-removal-audit.json`, source report,
all game/replay files and the two independently executable review scripts.

## Running broader matched-u512 comparison

Root: `out/research/policy-inference-broad-confirmation-2026-10-10/`.
Identity: `47b95caa92f63adb656c045b0893f830ceb7c2a785a68b08bfabc407d26fb589`.
Default T1 versus argmax-budget1, four matched u512 checkpoints (seed0/seed1,
cold/warm), four opponents (Greedy, old-256x2, initial-128x2, historical RL),
32 new shared deal blocks, four assignments per cell: **4,096 games**.
Engine seeds are disjoint from both previous sampling panels. This expands
checkpoint and opponent coverage, not the deck pool or independent seed count.

Preregistered primary contrast: equally weighted fixed-model win difference
across the three primary opponents. Resample shared deal blocks jointly across
all cells; report historical/all-four aggregates separately and every per-cell
harm, turn/decision distribution, cancellation and termination-limit result.
An aggregate is conditional on these four correlated checkpoints, not a claim
about the distribution of trained agents. No automatic adoption threshold.

All 32 first-game/cell/mode preflights passed full cold replay. Protocol and
mode choice preceded preflight outcomes. Two CPU workers, per-game artifacts,
a three-hour cap and an independent watchdog are active. Each game requires
cold replay; intervention masks must remain nonempty native subsets removing
only cancellations. Limits remain non-wins and alert; engine, identity, mask
or replay failures STOP. Formal seed2 training continues unchanged.
