# Wider battle-root coverage does not establish a critic improvement

The fixed 32-to-128-root training expansion fails to establish an improvement on its primary validation comparison. Both fitted heads have worse validation point estimates than the unchanged initialization. No model is promoted. These are privileged-critic action-ranking diagnostics with frozen continuation policies, not new public-actor gameplay results.

## Verified collection and fitting

All 7,408 registered branch tasks ended normally. The full action/suffix grid, 128 original controls, per-task result/replay hashes and recorded cold replay equality pass verification. This verification did not independently rerun every game. The fixed split contains 96 new training roots and 32 separate validation roots, from distinct source games; each of eight strata contributes 12 and four respectively.

The baseline uses the original 32 roots with pooled 32-suffix targets. The expanded arm adds 96 roots with 16-suffix targets. Initialization, full Q head, source-prediction anchor, centered action loss, Adam settings and 32 optimizer updates are identical. More data costs more sampling; this is not a compute-matched comparison. Actor, critic trunk, V and resource correction stay fixed.

All eight checkpoints were saved before new validation labels were aggregated. The original arm exactly reproduces previous pooled32 parameters at steps 0/8/16/32. Independent checkpoint loading recomputes predictions, legal choices, retention metrics and the registered bootstrap. No non-Q parameter changes occur; all pre-existing retention gates pass. This does not make the new action-ranking result acceptable automatically.

## New 32-root validation

| Fixed checkpoint | Centered action MSE | Conditional choice wins |
|---|---:|---:|
| Unchanged initialization | .034699 | 72.656% |
| Original32, step16 | .038452 | 72.070% |
| Original32, step32 | .039693 | 70.703% |
| Expanded128, step16 | .039161 | 68.555% |
| Expanded128, step32 | .037898 | 70.313% |

Primary expanded32 minus original32 MSE: **−.001795006**, 95% interval **[−.003781193, +.000122074]**. Conditional win difference: **−0.390625 percentage points**, interval **[−2.539063, +1.757813]**. The models change 13/32 root choices. This does not establish a useful improvement.

Intervals use 50,000 shared suffix-block bootstrap samples, seed 2026101119, conditional on the fixed roots and continuation policies. They do not cover population-wide root variability. A critic selects one root action, then the frozen public actor continues; the percentages are not full-game win rates of a newly trained actor. This panel differs from the preceding 64-root panel, so their raw percentages are not a learning curve.

Secondary step16 expanded minus original32 conditional wins are −2.148438 points, interval [−4.101563, −.390625]. Original step16 minus original32 is +1.367188 points [.195313, 2.539063]. These exploratory intervals are not multiplicity-adjusted and do not license choosing a different checkpoint after inspection. The unchanged model's MSE is better than original32 by .004993816, interval [−.008350113, −.001690350]; its win interval still crosses zero. Keep every checkpoint visible rather than reporting only the most favorable one.

## Post-hoc fit and uncertainty audit

The expanded head lowers MSE on its 96 added training roots from .028929 to .022002, but worsens the new validation MSE from .034699 to .037898. Its performance on the old32 training roots is .011209 versus the original-only fit's .008708. Training therefore runs and trades off groups; this is not an optimizer-no-op or missing-data result.

The estimated centered target sampling variance is .018477 on the 96 new training roots, versus observed centered target energy .023998: about 77%. On validation the estimates are .020853 and .042281. This is a descriptive finite-suffix estimate retaining paired-action covariance, not a proven noise floor or a causal attribution. Noise subtraction on fitted training predictions is especially not an unbiased generalization estimate.

The expanded head also reduces mean squared centered predictions on validation from .006486 to .001453. This suggests a tendency toward smaller action differences, but does not establish that model capacity is the bottleneck. Fit noise, feature generalization, group conflicts and the source anchor remain competing mechanisms.

## Next fixed-model label replication

`critic-ranking-coverage-label-replication-2026-10-11` freezes the unchanged, original32 and expanded128 model predictions at all 128 roots before collecting independent suffix streams 1401–1416. It contains another 7,408 tasks including 128 original controls, with two workers, a six-hour cap and an independent monitor.

The primary diagnostic asks whether the expanded model's apparent training-MSE gain over initialization reproduces on independent labels at the same 96 training states. Estimate the new-minus-old change in this gain with independent old/new shared-suffix bootstrap draws (50,000, seed 2026101120). Report the original-only model and the 32 previously inspected validation roots separately. These are new random continuations, **not new roots or fresh population confirmation**. No fitting or checkpoint selection occurs in this study; do not automatically average the labels into another training run.

Collection: `critic-ranking-coverage-branches-2026-10-11/verification.json`, completion event `c579d0b510b7de20ca3f77ad0c8d65580ae2ef9b285e3bf52ff68721ab06ad15` acknowledged after verification.

Fit: `critic-ranking-coverage-fit-2026-10-11/`, identity `9d72a0072d420013cf2afdfda01b331991fac3782b79071c29f3076ef9083535`; `verification.json`, `validation.json`, `fit-diagnostic.json` and executable verification/diagnostic scripts. Completion event `8889e36501b84dab614760dfddd9d2f01b7216b31bda62cb8472edd696774170` acknowledged after independent metric checks. Replication identity: `048fe5a29a5ba8c4e13602ef1a28e022e5821e4b20f6c6462aa14b5dba574a8a`.

The matched experiment retains frozen runtime `ae09594`; development/reporting is synchronized with main `56c29d6`. Preserve earlier STOP evidence. Issues #61/#83/#36 remain active, and declaration coverage #319 remains unresolved.
