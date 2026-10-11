# Frozen context preserves low-deck gains but does not resolve retention

The [context pilot](critic-lowdeck-mixture-confirmation-2026-10-10.md) completed
fresh confirmation on 768 healthy games /192 openings, disjoint from 2,048 prior
openings. Independent verification checks every replay/cache hash and exactly
reproduces frozen/control/mixed50/context50 predictions and metrics. Runick game
568 has no eligible idle sample: its outcome is retained, while critic metrics
use 512 broad and 255 Runick games. The completion event was acknowledged only
after verification.

## Confirmation results

These are equal-game Q/V terminal-return MSE, not policy performance. The
original conditioned control is the current comparison, and Mixed50 is a
same-data, same-mixture resource-only control.

| Estimator | Broad | Broad familiar | Runick | Facing Runick | Operating Runick |
|---|---:|---:|---:|---:|---:|
| Frozen base | 0.62074 | 0.49768 | 0.84706 | 0.83340 | 0.86060 |
| Original control | 0.57978 | 0.48026 | 0.72959 | 0.81732 | 0.64255 |
| Mixed50 | 0.53582 | 0.46960 | 0.71632 | 0.74093 | 0.69190 |
| Context50 | 0.54218 | 0.49394 | 0.69744 | 0.75801 | 0.63733 |

Descriptive unadjusted 95% intervals use 20,000 stratified paired-opening
bootstrap replicates, retaining opponent/seat outcomes together:

| Context50 comparison | Q/V MSE difference | Interval |
|---|---:|---:|
| Original control, broad | -0.03761 | [-0.07394, +0.00062] |
| Original control, Runick | -0.03216 | [-0.12972, +0.05766] |
| Original control, own Runick | -0.00521 | [-0.06717, +0.05599] |
| Mixed50, broad | +0.00636 | [-0.01318, +0.02528] |
| Mixed50, Runick | -0.01889 | [-0.06033, +0.02098] |
| Mixed50, own Runick | -0.05457 | [-0.12535, +0.00541] |
| Mixed50, familiar | +0.02434 | [+0.00508, +0.04403] |
| Mixed50, Tearlaments | +0.07232 | [+0.01967, +0.12864] |

Context50's own-Runick point estimate returns near the original control, but
uncertainty does not establish non-inferiority or superiority. Familiar and
Tearlaments error worsen against the same-data resource control. Versus the
original control, familiar difference is +0.01368 [-0.00698, +0.03414], and
Tearlaments +0.04580 [-0.00516, +0.09495]. Failure to reject zero is not evidence
of retention. The validation 2% allowance did not establish a robust guarantee.

Mixed50 again harms own Runick versus the original control: +0.04936
[+0.00217, +0.09968]. However its familiar point estimate improves in this batch,
unlike the previous confirmation. This opening-to-opening variation is a reason
to keep all studies visible, not selectively report the favorable batch.

## Low-deck outcomes

The Runick panel contains 66 low-own-deck games /36 openings, using the first
sampled idle state with own deck <=5. This subset has observed mean return
-0.12121, compared with -0.53425 in the previous mixture confirmation; each
batch contains different fresh openings.

| Estimator | Low-deck V MSE | Signed V bias |
|---|---:|---:|
| Original control | 1.42292 | +0.47383 |
| Mixed50 | 0.97864 | -0.13533 |
| Context50 | 0.93955 | -0.03491 |

Context50 minus original-control V MSE is -0.48338 [-0.87828, -0.09581]. Thus
low-deck improvement persists, but it does not remove the retention tradeoff.
Improvement over the older frozen base is also insufficient to select Context50
over the current conditioned control.

The unchanged actor wins 346/512 broad games and 186/256 Runick games; mean
lengths are 11.963 and 22.566 turns. Every critic scores the same trajectories,
so these figures are not critic-driven gameplay gains or shorter-game progress.

## Decision and next investigation

Context50 is not adopted into PPO. Appending frozen public context has not
demonstrated an overall improvement over the same-data resource residual, and
its familiar/Tearlaments regressions remain. No additional tuning on this
confirmation is allowed.

The next diagnostic inspects the relationship between both players' remaining
deck counts and outcomes in the original training/validation corpora. A low
own-deck count can coexist with an even more depleted opposing deck; the current
linear correction cannot express arbitrary nonlinear interactions between
these counts. This is a hypothesis to check, not an established explanation.
No universal low-deck penalty or card-specific reward is introduced.

The audit found an observational split in original Runick validation. Among
low-own-deck games operating Runick, all 13 with opponent deck <=5 won, whereas
the eight with opponent deck >5 have mean return -0.5. Training has the same
direction (+0.857 on 14 games versus -0.429 on seven). These small subgroups
support testing deck-count interactions, not a causal conclusion or a new
win-condition rule.

`critic-deckrace-mixture-fit-2026-10-10` adds three generic features: own and
opponent pressure `clip((6 - deck_count) / 6, 0, 1)` and their product. The
boundary follows the previously defined <=5 slice and is not searched. No
card IDs or role labels enter these features. Twenty-six related helper tests
pass, including raw-count scaling, player-swap symmetry and invalid counts.
All twelve fitted candidates/metrics/gates independently reproduce. Neither
mixture arm passes own-Runick retention at any ridge setting, so neither is
advanced to full-game confirmation. At the lowest validation objective,
own-Runick MSE is 0.73382 /0.72359 versus control 0.67532. The simple feature
addition is insufficient; the negative result is retained.

The completed study, `critic-head-refresh-pilot-2026-10-10`, moves from a
linear output correction to a bounded nonlinear critic-only update. Both arms
start at the original conditioned control, keep its 14-feature correction
fixed, and train the private critic encoder, critic trunk/heads and resource
MLP. Actor/public encoder parameters remain frozen, with actor hashes checked
after every update. The same 511 eligible games and 50/50 corpus mass are used.

The two arms compare terminal-return MSE alone against the same loss plus a
coefficient-1 prediction anchor on broad-training states. The anchor preserves
the initial critic's predictions, not labels from confirmation. Both use
32 full-batch Adam updates (lr=1e-4, norm clip=1), with model/optimizer
checkpoints at 0/4/8/16/32. Unchanged step zero remains eligible; selection
uses the existing validation/gates, including own-Runick, and can choose no
adaptation. No production PPO/actor change is made.
The larger trainable critic capacity and repeated historical validation use
make new confirmation essential before adopting any selected checkpoint.

Evidence is retained in `critic-context-mixture-confirmation-2026-10-10`
(`analysis.json`, `completion-verification.json`, prediction/cache artifacts)
and `critic-deckrace-audit-2026-10-10`. Actor/base parameters and all original
sealed study reports remain unchanged.


## Nonlinear critic pilot verification and fresh confirmation

Independent verification reloads all ten checkpoints and exactly recomputes
validation predictions/metrics, retention gates, and the eligible minimum-score
selection. Step-zero predictions equal the original control, and actor hashes
and saved actor tensors remain unchanged throughout the checkpoint series.
This verification does not rerun the optimizer. Both arms select update 32:

| Validation estimator | Broad Q/V MSE | Runick Q/V MSE | Own Runick Q/V MSE | Familiar Q/V MSE | Low-deck V MSE | Low-deck V bias |
|---|---:|---:|---:|---:|---:|---:|
| Original control | 0.58013 | 0.87981 | 0.67532 | 0.46763 | 1.56449 | +0.53912 |
| Terminal only, u32 | 0.56352 | 0.74775 | 0.65817 | 0.47370 | 1.01969 | +0.26834 |
| Anchored, u32 | 0.56312 | 0.75854 | 0.65799 | 0.46991 | 1.07220 | +0.28854 |

All five registered validation gates pass at these selected checkpoints.
Earlier checkpoints do not improve monotonically: terminal-only updates
4/8/16 and anchored update 8 fail at least one gate. Anchoring has slightly
better familiar error at u32 but slightly worse Runick/low-deck error. This
single training/validation split does not establish a superior objective.

`critic-head-refresh-confirmation-2026-10-10` is now running under a separate
study service and watchdog. Its 512 broad plus 256 Runick games use 192 new
openings, with zero overlap against 2,240 prior openings. It compares the frozen
base, original conditioned control and both selected nonlinear critics on the
same fixed-actor trajectories. All checkpoints and input dependencies are
sealed before collection. The first 72 canary games precede the rest of the
panel. Zero-idle outcomes, engine/replay health, familiar/old-opponent retention,
each Runick role and paired-opening uncertainty remain mandatory.

The head-pilot completion event was acknowledged after verification. The
confirmation results are pending; no gameplay improvement, shorter-game effect,
or PPO promotion is established. Full artifacts and the independent verifier
are in `critic-head-refresh-pilot-2026-10-10`.
