# Critic opponent coverage: fit verified, fresh confirmation running

The [opponent-distribution audit](critic-opponent-distribution-2026-10-10.md)
found that identical early critic inputs can precede different return
distributions under different opponents. This follow-up separates changing the
training opponent mixture from supplying a known opponent identity. It changes
only a diagnostic critic residual, with the actor and base critic frozen.

## Corpus and controls

`critic-opponent-coverage-data-2026-10-10` completed 1,024 healthy games:
512 training and 512 validation, each with 256 old-256x2 and 256 mirror-u528
games. Its 256 unique openings have no overlap with 1,344 prior openings or
between training and validation. Independent verification checked every full
game, replay and critic cache hash, including replay/actor-probability parity.

Training game 507 has no eligible idle sample. It remains in outcome totals,
but contributes no critic error or training row. Thus the old-only arm has
255 eligible games out of 256 planned, while both balanced arms have 256.
This small effective-budget mismatch is retained and reported rather than
silently resampling or assigning a zero error.

The three arms use the same middle-initialization frozen base critic:

| Arm | Training games | Residual inputs |
|---|---|---|
| Old resources | All 256 planned old-opponent games | 13 resource features |
| Balanced resources | 128 old + 128 mirror, draws 0/1 | Same 13 features |
| Balanced conditioned | Same balanced 256 games | Resources + known opponent indicator |

Other mirror training draws remain recorded but are excluded from these fits.
Each arm fits separate Q/V ridge residuals with equal game weighting,
training-only normalization and an unpenalized intercept. All use the same
six-value ridge grid and select by Q/V MSE on the same balanced 512-game
validation set. All selected lambda = 0.0001, the smallest registered value;
this does not establish an optimum outside the tested grid.

`tools/critic_opponent_probe.py` supplies the optional metadata feature. The
unconditioned control never reads opponent metadata. The diagnostic accepts
only the two registered policies, validates game-index alignment and rejects
unknown labels. It is not an actor feature or an unseen-opponent inference
mechanism. The indicator adds model capacity as well as information, so this
is not an architecture-independent causal estimate.

## Validation results, not confirmation

Errors and signed biases use terminal returns in [-1, +1], with equal weight
per eligible game. Lower MSE is better. Frozen-base pooled Q/V MSE is 0.63612.

| Arm | Pooled Q/V MSE | Mirror Q/V MSE | Mirror V bias | Old Q/V MSE | Old V bias |
|---|---:|---:|---:|---:|---:|
| Old resources | 0.67168 | 0.92016 | +0.43719 | 0.42320 | +0.02598 |
| Balanced resources | 0.60255 | 0.74501 | +0.23337 | 0.46009 | -0.09159 |
| Balanced conditioned | 0.58013 | 0.71815 | +0.12832 | 0.44210 | +0.00346 |

Independent refitting reproduces all selected coefficients and validation
metrics exactly and verifies the registered selection minima. These results
support testing both broader opponent coverage and explicit conditioning.
They also expose a retention tradeoff: conditioned old-opponent error remains
higher than the old-only arm. The validation set selected hyperparameters;
these numbers are not an independent generalization result.

## Sealed fresh confirmation

`critic-opponent-coverage-confirmation-2026-10-10` has started with an independent
watchdog. It evaluates the frozen base and all three sealed residuals on the
same 512 games / 128 new openings, paired across opponents and seats. None of
these openings overlaps the 1,600 prior openings. No confirmation outcomes
enter fitting or selection.

The primary comparisons are balanced minus old-only and conditioned minus
balanced. Analyze paired opening uncertainty, both opponents, both roles,
individual targets, familiar retention, Tearlaments and low-deck states. Retain
zero-idle games in game totals but exclude them from critic-error denominators.
Input-matched opponent analysis must allow the conditioned estimator to see
different opponent indicators; identical base inputs no longer imply identical
conditioned values. Verify collection, replay/cache identity and all four
prediction groups before interpreting the completion notification.

This remains a fixed-learner, two-known-opponent diagnostic. It does not prove
adaptation to unknown opponents, correction of every Runick state, or stronger
play. Learner-policy-shift validation and matched RL continuation are still
required before adoption. No production checkpoint, reward or PPO behavior
has changed.

Local validation: 15 opponent/residual/role helper tests pass; full repository
Ruff lint and format checks pass. The named research directories retain
registrations, immutable dependency identities, coefficients, data audits,
verification and monitor handoffs; original study evidence is preserved.
