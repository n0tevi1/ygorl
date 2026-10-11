# Equal-budget critic role-coverage experiment

The [resource confirmation](critic-resource-confirmation-2026-10-10.md) found
opposite Runick biases by learner role, while the preceding critic corpus only
contained `face-target` games. This experiment tests that coverage gap without
increasing the training-game budget or changing the behavior actor.

The new own-target corpus is complete: **512 healthy games, 2,843 cached critic
states, no zero-idle games**, with 256 training and 256 validation games. Its 128
ordered openings have no overlap against 576 prior openings. Training and
validation opening sets are disjoint. Runtime identity, response replay,
cached critic parity, actor probabilities, feature hashes and replay hashes
were checked before fitting.

## Registered comparison

| Arm | Training games | Calibration coefficient |
|---|---|---:|
| Face MSE | 512 original face-target | 0 |
| Face calibration | 512 original face-target | 0.5 |
| Balanced MSE | 256 original face-target + 256 new own-target | 0 |
| Balanced calibration | 256 original face-target + 256 new own-target | 0.5 |

Original face draws 0–3 were selected for the balanced arms before the new data
were collected. This compares replacing half the face data with own-role data,
not simply adding games. Every game has equal total loss weight despite different
numbers of states: face training has 3,115 states and balanced training 3,026.
Calibration averages squared signed residuals over target-list × learner-role
groups (eight observed groups in face-only, sixteen in balanced). These labels
never enter inference features.

All arms use the same 13-feature public-resource branch and private critic,
frozen actor/public encoder, 32 full-corpus Adam updates at learning rate 1e-4,
epsilon 1e-5, gradient clipping 0.5, and three auxiliary initializations. Exact
two-pass gradient accumulation preserves the full-corpus calibration objective.
The registered initialization seeds are 202610120701–703.

Selection uses one common **512-game, 3,031-state balanced validation set**:
256 original face-target and 256 new own-target games. Each arm selects update
0, 4, 16 or 32 by Q/V MSE plus 0.5 times target/role conditional bias penalty.
No training or validation games have zero eligible states. Planned totals and
zero-idle cases remain explicit in the implementation for future data.

`tools/critic_role_probe.py` composes caches with namespaced source IDs, remapped
sample-to-game indices, rebuilt equal-game weights and contiguous target/role
loss groups. Tests cover colliding local IDs, unequal trajectory lengths,
subsetting, zero-idle accounting and duplicate-source rejection. Existing
resource-branch and exact calibration-gradient tests also pass (six total).

## Validation: better overall fit, worse facing-Runick bias

Fitting completed; all 12 arm/initialization selections chose update 32. Selected
checkpoint metrics were recomputed exactly, their selection minima checked, and
actor hashes verified unchanged. The following ranges span auxiliary
initializations, not independent training runs:

| Arm | All Q/V MSE | Face-role Q/V MSE | Own-role Q/V MSE | Facing Runick V bias | Operating Runick V bias |
|---|---:|---:|---:|---:|---:|
| Face MSE | 0.47722–0.47735 | 0.62473–0.62491 | 0.32967–0.32998 | +0.2134–+0.2149 | -0.4523–-0.4510 |
| Face calibration | 0.47466–0.47480 | 0.62435–0.62450 | 0.32497–0.32525 | +0.2046–+0.2064 | -0.4448–-0.4434 |
| Balanced MSE | 0.45560–0.45567 | 0.63494–0.63509 | 0.27622–0.27640 | +0.3491–+0.3500 | -0.3103–-0.3093 |
| Balanced calibration | 0.45668–0.45676 | 0.63517–0.63531 | 0.27815–0.27834 | +0.3312–+0.3322 | -0.3206–-0.3195 |

Replacing half the face-only corpus improves own-role validation substantially,
but slightly worsens face-role MSE and substantially increases facing-Runick
overestimation. Role-conditional calibration only partially limits the latter.
Coverage is therefore not a sufficient correction for this bias. These are
model-selection data, not confirmation evidence; no arm is promoted.

## Independent confirmation

The confirmation is complete; see the
[paired results and gradient-conflict audit](critic-role-conflict-2026-10-10.md).
Facing-Runick regression reproduces, so these critics are not promoted.

All input files and the registered plan are sealed. The previous 512-game
confirmation is excluded from fitting and selection. After selection and
independent verification, `critic-role-coverage-confirmation-2026-10-10` launched
512 new both-role games on 128 openings, disjoint from 704 prior openings. The
same fixed actor plays once per job; all 12 selected critics, the unchanged
source and a fixed previous resource/calibration reference score the same states.
Collection and final scoring have separate completion status, with an outer
heartbeat and independent watchdog. Checkpoints and health-failure evidence are
retained.

The primary contrast is balanced MSE versus face MSE. Calibration is a secondary
factor. Report paired opening-deal uncertainty, both roles separately, every
target list, familiar retention, and low-deck/LP-leading slices. Three auxiliary
initializations are not independent policy-training runs. No improved gameplay
or corrected Runick bias is claimed from validation; deployment still requires
matched RL continuation and retention tests.

Artifacts: `out/research/critic-role-coverage-data-2026-10-10/` retains the data,
pre-collection `fit-plan.json` and completion verification.
`out/research/critic-role-coverage-fit-2026-10-10/` retains the frozen fitting
script/helper, registration, runtime identity, dataset audit, checkpoints,
selection/report, completion verification, and monitor. Original studies and STOP evidence are
preserved.
