# Relative-Q confirmation: no demonstrated action-selection gain

The 64-root, 3,536-game conditional confirmation is complete. Relative-Q supervision and unchanged initialization both achieve **70.703%** conditional wins. The preregistered difference is **0.000 percentage points**, with a paired95% interval of **−1.074 to +0.977 pp**. No model is promoted.

## Verification and fixed comparison

Artifact: `out/research/critic-battle-relative-confirmation-2026-10-11`. Identity SHA256: `baf59366ed7143ec14c52bf7d38fce774037578399f7225411384765eaab0bcb`. The audit verifies sealed runtime and inputs, final report identity, exact task grids, task/result agreement, replay hashes, recorded cold replay equality, and all64 original-action/original-RNG controls. All3,536 games ended with a winner. Interventions were not independently rerun during this audit. Completion `0187ddde83ca35124a79be5ff19eafac5d9a89dad2c5174cfb50bae596228c07` was acknowledged after verification.

Models and predictions were frozen before suffix outcomes. Every one of the64 registered roots and217 supported actions was retained; each action used16 new shared suffix streams. Primary inference weights fixed roots equally and bootstraps suffix indices jointly across actions and roots:50,000 replicates, seed2026101103. Earlier pilot outcomes are not pooled.

| Frozen critic | Conditional win fraction | Difference vs unchanged | Paired95% interval | Centered action MSE |
|---|---:|---:|---:|---:|
| Unchanged | 70.703% | — | — | .024368 |
| Relative-step32 (primary) | 70.703% | 0.000 pp | −1.074 to +0.977 pp | .022262 |
| Absolute-step32 (secondary) | 70.801% | +0.098 pp | −1.367 to +1.563 pp | .027249 |

The absolute arm previously failed familiar-card retention and remains only a comparison. Its small point advantage does not establish improvement. Q-argmax is a private-critic diagnostic, not a deployable public-information actor. These root interventions do not measure RL actor improvement or full-game playing strength.

The64 openings are distinct from all32 branch-training and eight pilot-validation roots. However, source trajectories previously appeared in calibration/retention panels; this is not globally untouched game data. Models do not train on these64 branch outcomes. Source learner/opponent policies remain fixed after the root action.

## Changed actions and uncertainty

Relative supervision changes13/64 root choices: two roots gain wins, four lose wins, and seven have equal win counts. Five of those seven have identical paired outcomes throughout; two have offsetting wins and losses. Across the13 roots there are15 favorable and15 unfavorable paired outcomes, explaining the zero net result. Examples are diagnostic, not certified tactical labels:

- Root16: old action6/16 wins, changed action10/16; five favorable versus one unfavorable paired result.
- Root28: old16/16, changed14/16; zero favorable versus two unfavorable paired results.

The centered-MSE point improvement is8.64%, but an exploratory shared-suffix bootstrap gives a difference of **−.002106**,95% interval **[−.004917, +.000492]** (50,000 replicates, seed2026101104). The interval includes zero. `audit_noise.py` uses the fact that squared targets cancel in the model-loss difference, preserving paired action/stream covariance.

An action-centered covariance estimate puts the mean-label sampling-noise contribution at **.016142**, compared with unchanged observed centered MSE of .024368. This suggests16 suffixes per action can be a substantial source of measurement noise. It does not establish that noise caused the training failure, and the estimate itself is uncertain. Subtracting such noise from training error would be biased because the model was fitted to those labels.

Artifacts: `analysis.json`, `analyze.py`, `changed-actions-audit.json`, `audit_changes.py`, `noise-audit.json`, `audit_noise.py`.

## Next: independently resample every training target

On the original32 training roots, relative supervision reduces centered error **.021531 → .015286** and raises descriptive conditional wins **69.531% → 72.461%**. Those are fitted-label results, not validation. The next study distinguishes suffix-noise fitting from genuine root-specific learning that fails to transfer.

`critic-battle-target-replication-2026-10-11` is running with **1,904 games**: all117 supported actions at every one of the32 original training roots ×16 new suffix streams701–716, plus32 original controls. Models/checkpoint hashes and predictions are frozen before collection; no optimization or checkpoint selection occurs. Every training root is retained regardless of previous result or disagreement.

Primary: relative-step32 minus unchanged centered action MSE on **new outcomes only**, with50,000 shared-suffix-block bootstrap replicates (seed2026101105). Old-target fit is descriptive; do not pool it into the primary. Secondary diagnostics compare argmax outcomes and old/new label variability.

- If improvement on old training labels fails to repeat with new suffixes, investigate label noise, target shrinkage or additional continuation samples.
- If training-root improvement repeats but fails on other roots, prioritize state coverage and representation generalization.

Neither explanation is automatically proven by a single small replication. Any subsequent refit using these new labels needs a new registered protocol and separate validation.

Study identity: `332bb74d180419b692480b2e74fa982fd35e119c0797b61c7e587c8f46325178`. Two workers, three-hour limit, original controls before interventions, cold replay, per-game artifacts and independent watchdog. Previous runtime, native engine, checkpoints and STOPs remain untouched; issue#319 is still open.

This change records analyses and experiment registration only. Documentation diff checks pass; no production code or model defaults change.
