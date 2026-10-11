# Opponent shift: resource gains persist, opponent prior remains wrong

The [paired opponent-shift study](critic-residual-ablation-confirmation-2026-10-10.md)
completed 512 healthy games / 128 fresh openings. All games have eligible idle
samples. Independent verification reproduces all twelve prediction groups and
aggregate/role metrics exactly. The learner and critic coefficients remain frozen.
Old-256x2 and mirror-u528 each play 256 games with paired openings/seats.

## Transfer results

Errors are averaged across the three auxiliary initializations per game before
opening resampling. These estimators do not add games or independent policy seeds.

| Opponent | Frozen Q/V MSE | Resources | Without LP | Frozen V bias | Resources V bias |
|---|---:|---:|---:|---:|---:|
| Old-256x2 | 0.35706 | 0.34093 | 0.34207 | -0.12723 | -0.13779 |
| Mirror-u528 | 0.84355 | 0.79940 | 0.79249 | +0.32521 | +0.29462 |
| Pooled | 0.60031 | 0.57016 | 0.56728 | +0.09899 | +0.07841 |

Resource-minus-frozen differences, with descriptive unadjusted 95% intervals
from 20,000 stratified paired opening bootstrap replicates:

| Slice | Q/V MSE difference | Interval |
|---|---:|---:|
| Pooled | -0.03014 | [-0.05219, -0.00976] |
| Old opponent | -0.01614 | [-0.03750, +0.00300] |
| Mirror opponent | -0.04415 | [-0.07256, -0.01817] |
| Mirror, facing targets | -0.07860 | [-0.13233, -0.02964] |
| Mirror, operating targets | -0.00969 | [-0.02823, +0.00885] |

The primary transfer contrast is `(resources - frozen) under mirror -
(resources - frozen) under old`, paired by opening and learner seat. It is
**-0.02801 [-0.05091, -0.00625]**, from 256 paired seat outcomes / 128 openings.
Thus the resource gain survives this opponent change. It does not establish
transfer after changing the learner policy or encoder.

Own-role improvement remains uncertain. Pooled Tearlaments error improves
0.46176 to 0.43371, difference -0.02805 [-0.04360, -0.01075]. Its prior regressions
are distribution-dependent, not established as universal or resolved for every
opponent. Facing Runick bias falls +0.35267 to +0.09201 overall. Under the mirror
alone it falls +0.35464 to +0.04866, but its MSE difference interval spans zero:
only 16 games / eight openings contribute. Small signed bias is insufficient to
establish individual-state accuracy.

The frozen learner wins 225/256 against old-256x2 (mean return +0.75781) and
129/256 against its mirror (+0.00781). Mean lengths are 10.441 and 10.813 turns.
All estimators score those same outcomes; these numbers do not measure a gain
from critic adaptation. The unresolved finding is the remaining **+0.29462**
mirror-opponent V bias.

## Identical-input audit

After confirmation, `critic-opponent-input-audit-2026-10-10` registered an
exploratory comparison of first eligible V inputs across paired opponents.
Cache hashes are verified; public context, privileged tensors and all thirteen
resource inputs must match exactly. Candidate-action embeddings are not needed
for this V-only check. Pairs are selected by inputs, never by outcomes.

Of 256 opening/seat pairs, **77** have identical V inputs. All are learner-first,
turn-one samples. Every estimator's paired V agrees to below 1e-7. For the middle
auxiliary initialization:

| Quantity on 77 input-identical pairs | Old opponent | Mirror opponent |
|---|---:|---:|
| Mean terminal return | +0.71429 | -0.14286 |
| Frozen mean V | +0.44867 | +0.44867 |
| Resource-corrected mean V | +0.43776 | +0.43776 |

The paired return difference is -0.85714, descriptive stratified opening interval
[-1.06494, -0.64935]. Individual outcomes are noisy, not exact state values, but
the aggregate contrast supports an opponent-dependent shift that these identical
inputs cannot distinguish. This remains an exploratory, input-conditioned subset.

The current interface has no explicit opponent-policy identity. Past actions may
eventually reveal behavior; these early inputs do not. A value predictor can
learn the expectation under its training opponent mixture, but cannot produce
different opponent-conditional values for exactly the same input. Updating the
mixture can improve pooled calibration while leaving opposing subgroup biases.
This is an input/distribution limitation, not evidence of a duel-engine error or
that LP alone explains the Runick problem.

## Next: equal-budget coverage and critic-only opponent conditioning

`critic-opponent-coverage-data-2026-10-10` completed **1,024** healthy games: 512
training and 512 validation, each balanced across old/mirror opponents, target
lists, roles and seats. Its 256 openings are disjoint from 1,344 prior openings
and across splits. Exact replays and critic feature caches are retained; no
fitting occurred during collection. All replay/cache hashes were independently
verified. Training game 507 has no eligible idle sample: old-only fitting has
255 eligible games versus 256 in the balanced arms.

The subsequent registered fit compares three arms: old-only resource residual,
balanced resource residual, and balanced resource residual plus a known-opponent
indicator for the **training critic only**. Actor and middle-initialization base
critic stay frozen. Each arm plans 256 training games: all old-opponent games,
or 128 old + 128 mirror games from draws 0/1. Other mirror training games remain
recorded but are excluded from primary arms. Zero-idle games stay in planned
totals; any effective imbalance must be reported.

All arms use training-only normalization, the existing ridge grid and the same
512-game validation Q/V MSE for selection. The [verified fit and fresh
confirmation](critic-opponent-coverage-fit-2026-10-10.md) now follow this corpus.
The opponent indicator adds one feature and is a diagnostic control,
not unseen-opponent generalization or proof of an architecture-independent causal
effect. Opponent/role/target retention, particularly Tearlaments, remains required.
No actor, production reward or PPO checkpoint changes. Learner-policy-shift
validation and matched RL continuation remain necessary before adoption.

Evidence in the three named `out/research/` studies retains registrations,
identities, verification, predictions, paired analysis and input-matched rows.
Original completion records and study evidence are preserved.
