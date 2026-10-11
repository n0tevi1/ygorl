# Critic resource readout: objective tradeoff and frozen residual pilot

The [group-loss experiment](critic-role-conflict-2026-10-10.md) completed
512 healthy confirmation games on 128 new openings. Independent checkpoint
forwards and aggregate/role metrics reproduce exactly. All games use the same
frozen actor; these are critic errors against terminal returns, not changes in
playing strength. No candidate is promoted to PPO.

## Group weighting moves the error between roles

| Critic | Overall Q/V MSE | Face-role MSE | Own-role MSE | Facing Runick Q/V MSE | Facing Runick V bias |
|---|---:|---:|---:|---:|---:|
| Ordinary MSE | 0.44119 | 0.58620 | 0.29619 | 1.17413 | +0.57173 |
| Group maximum, tau 0.25 | 0.44954 | 0.57410 | 0.32499 | 0.92590 | +0.36112 |
| Group maximum, tau 0.5 | 0.44301 | 0.57256 | 0.31345 | 0.98667 | +0.42075 |

For tau 0.25 minus ordinary MSE, 20,000 stratified paired opening-deal bootstrap
replicates give these descriptive, unadjusted 95% intervals:

| Slice | Q/V MSE change | Interval |
|---|---:|---:|
| Overall | +0.00835 | [-0.00805, +0.02452] |
| Facing targets | -0.01210 | [-0.04046, +0.01485] |
| Operating targets | +0.02880 | [+0.01062, +0.04632] |
| Facing Runick | -0.24823 | [-0.40039, -0.10703] |

Resampling preserves target/anchor/role strata and keeps all four opponent/seat
games from an opening together. This is one auxiliary initialization, not a
multi-seed result. Facing Runick has 32 games / eight openings. The improvement
there comes with an own-role regression, so group weighting is not an accepted
general fix.

At the first eligible state with own deck <=5 and LP ahead, ten facing-Runick
games have mean return -0.8. Ordinary-MSE mean V is +0.60079; tau 0.25 reduces it
to +0.14977, still too optimistic. Corresponding V MSE falls 2.39170 to 1.34674.
This small trajectory-conditioned slice does not justify a universal LP rule.

## Can the existing representation distinguish the roles?

A subsequent exploratory audit fitted equal-game, training-normalized ridge
readouts on the balanced training corpus, selecting regularization on its
validation corpus. It then inspected the already-used confirmation panel.
These diagnostic results are not a new independent confirmation.

A linear probe of the existing public context distinguishes face/own roles with
93.6% validation and 94.4% diagnostic accuracy. Adding existing scalar resources
raises diagnostic accuracy to 96.4%. Thus the representation contains substantial
role information, although decodability does not prove that the trained critic
uses it correctly or that it captures all relevant strategy.

An unordered own-starting-deck ID-embedding average decodes the role perfectly
on this particular ten-list panel. That is a panel-specific shortcut, not
unseen-deck generalization. Own remaining-deck cards are already encoded, sorted
by card ID; the premise that the actor has no own-deck information is incorrect.

The same audit fitted a **V-only** residual on the frozen ordinary-MSE critic:

| Residual inputs | Diagnostic V MSE | Facing Runick V MSE | Facing Runick V bias |
|---|---:|---:|---:|
| No correction | 0.44426 | 1.19523 | +0.57173 |
| Public context | 0.44068 | 1.09896 | +0.52514 |
| Existing scalar resources | 0.41192 | 0.73469 | +0.27259 |
| Context + resources | 0.42150 | 0.71144 | +0.23935 |
| Own-deck embedding | 0.44472 | 1.07104 | +0.47108 |
| Context + own-deck embedding | 0.44204 | 0.98283 | +0.43063 |

Scalar-resource correction changes mean V on the ten low-deck/LP-leading games
to -0.17332 and V MSE to 0.78113. Explicit own-deck identity provides no aggregate
improvement. These observations motivate testing how the critic uses available
resources rather than assuming that missing deck identity is the main cause.
They do not isolate LP causally, establish a universal LP coefficient, or prove
that the residual improves actor learning.

## Registered Q/V residual fit and fresh confirmation

`critic-residual-fit-2026-10-10` freezes the entire selected critic and actor.
It fits both Q and V residuals on the same 512 balanced training games, using
training-only weighted normalization and ridge regression. Four input views
are registered: intercept only, resources, context, and context + resources.
The intercept control tests whether a global return-prior shift suffices.

Each view selects its ridge parameter independently from
`[0.0001, 0.001, 0.01, 0.1, 1, 10]`, by equal-game Q/V MSE on the same 512
validation games. The fit minimizes unconstrained residual error with an
unpenalized intercept; selection and evaluation clip predictions to [-1,1].
All four selected views proceed to confirmation; no view is selected using the
preceding exploratory diagnostic results.

| View | Selected ridge | Validation Q/V MSE | Facing Runick V bias |
|---|---:|---:|---:|
| Frozen critic | — | 0.45560 | +0.34998 |
| Intercept | 0.0001 | 0.45546 | +0.35095 |
| Resources | 0.0001 | 0.44062 | +0.07893 |
| Context | 1 | 0.45223 | +0.29577 |
| Context + resources | 0.01 | 0.44007 | +0.03226 |

Independent verification refits every selected coefficient exactly, reproduces
validation metrics, verifies registered minima and the frozen actor hash.
The helper `tools/critic_residual_probe.py` does not modify network parameters;
its saved coefficients are research artifacts, not PPO checkpoints.
Six focused tests cover weighted intercept fitting, bounds, normalization
immutability, duplicated-state weight invariance and invalid regularization.
Together with resource, role and group-objective tests, 19 tests pass.

`critic-residual-confirmation-2026-10-10` is running 512 fresh games / 128
openings, disjoint from 960 prior openings. It evaluates all four selected
views, the frozen ordinary-MSE critic and the fixed tau-0.25 reference. The
primary comparison is resources against frozen MSE and intercept; report paired
opening intervals, both roles, targets and low-deck/LP-leading slices. A separate
watchdog checks liveness and verified completion. Engine failures or truncations
stop the study while retaining evidence. No automatic promotion is authorized
by these metrics alone.

## Evidence and limits

Under `out/research/`, the group-loss confirmation retains its independent
verification, predictions and paired analysis. The context-readout audit retains
its exploratory registration, selected readouts and original/extended reports.
The residual fit and confirmation retain immutable input identities, coefficients,
panels, verification records and monitoring. A stale final child progress count
of 511 in the earlier confirmation did not represent missing data: all 512 game
IDs and outputs were verified. The new runner derives final collection progress
from the completed report.

Development was synchronized to main `758a638`, including PRs #291 and #296.
The sealed experiment runtime remains at `ae09594`; changing it mid-study would
invalidate the controlled comparison. Further actor-training experiments must
separately test retention, calibration under policy shift and playing strength.
