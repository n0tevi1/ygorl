# Private critic: public resources × conditional calibration

The [Runick bias audit](runick-critic-bias-2026-10-10.md) found conflicting
cross-matchup gradients and misleading LP/deck-count sensitivity. This bounded
factorial separates an extra public-resource input path from a change to the
critic objective. The actor, its shared encoder, and gameplay behavior are fixed.

## Registered implementation

`tools/critic_resource_probe.py` adds a 13→32→128 public MLP before the private
critic trunk's first ReLU. Inputs are turn, both LP totals, and both players'
deck/hand/graveyard/banished/extra-deck counts, with fixed scaling. The final
projection starts at zero, preserving the original Q/V outputs exactly. Every
arm has the same auxiliary architecture; controls receive zero public inputs.
This controls capacity and update budget while testing the extra information.
There is no terminal outcome or target-deck identifier in inference features.

All arms train the original private encoder and critic plus this branch:

| Arm | Direct public resources | Calibration penalty coefficient |
|---|---|---:|
| Control | No | 0 |
| Resources | Yes | 0 |
| Calibration | No | 0.5 |
| Both | Yes | 0.5 |

The main loss is game-weighted Q/V squared terminal-return error. The extra
penalty is mean squared **per-target-list mean signed error**, averaged over
eight target lists and Q/V. Target identifiers group training examples only.
This does not hard-code a negative Runick value or turn LP into a shaped reward.
The numerical input path also does not yet model attack legality or persistent
effect interactions directly; it is a minimal resource-representation test.

There are 32 full-corpus Adam updates at LR 1e-4, epsilon 1e-5, gradient norm
limit 0.5, and three auxiliary initializations (202610120301–303). Each update
uses an exact two-pass gradient: current predictions determine group residuals,
then minibatch backpropagation accumulates their derivative and MSE before one
optimizer step. Parameters remain unchanged between passes; dropout is disabled.
A regression test compares this with direct full-batch autograd under unequal
game lengths and group sizes.

512 training and 256 validation games were replayed to recover the public
inputs at the exact cached prediction indices. Result and response parity was
checked. The previous test split is not loaded. Each arm/initialization selects
update 0, 4, 16, or 32 by the **same** validation criterion: mean Q/V MSE plus
0.5 times the conditional bias penalty. Three initializations of one actor and
one corpus are not independent policy-training replications.

These are experimental weights, not production PPO checkpoints. No actor
parameters, deployment defaults, reward functions, or action masks change.

## Validation result: small incremental effect, unresolved bias

All 12 selections chose update 32. Scores and selection minima were independently
reproduced; all actor hashes match. The ranges below span auxiliary initializations:

| Arm | All Q/V MSE | Familiar Q/V MSE | Runick V bias | Runick Q/V MSE |
|---|---:|---:|---:|---:|
| Unchanged source | 0.69136 | 0.65124 | +0.16378 | 1.01979 |
| Matched control | 0.62521–0.62523 | about 0.5829 | about +0.2161 | 0.95572–0.95581 |
| Resources | 0.62480–0.62502 | 0.5829–0.5831 | +0.2135–+0.2145 | 0.95319–0.95398 |
| Calibration | 0.62480–0.62482 | about 0.5831 | +0.2075–+0.2077 | 0.96042–0.96046 |
| Both | 0.62440–0.62466 | 0.5830–0.5832 | +0.2047–+0.2059 | 0.95793–0.95874 |

Most improvement versus the unchanged source is ordinary private fitting.
The two proposed changes add little on this validation set. Combining them
reduces Runick bias by roughly 0.01 relative to the matched control, but bias
remains **higher than before fitting**. Calibration slightly worsens Runick MSE
relative to the matched control even while lowering bias. No correction or
generalization success is claimed from these selection data.

## Independent confirmation launched

`critic-resource-confirmation-2026-10-10` was prepared before fitting and launched
only after model selection and its verification were sealed. It contains 512
games on **128 new ordered deck pairs and 128 new opening-hand pairs**, with zero
order overlap against 448 distinct prior deals. Eight target lists, two anchor
lists, four draws, two opponents and both seats cover both `face-target` and
`own-target` roles. The same frozen actor plays once per job; every selected
critic is scored on the same resulting states. The unchanged source and the
previous refresh are additional references.

All games receive health checks, cold response replay and cached critic replay.
Zero-idle games remain explicit; limits and engine failures stop the study with
evidence retained. The outer pipeline keeps its own heartbeat while its child
collection completes, so collection completion cannot be mistaken for completed
critic scoring. An independent invocation-bound watchdog reports failures and
completion. Checkpoint files and selection metadata are retained locally.

Report overall and familiar MSE, per-target signed bias, low-deck calibration
and each role separately. These outcomes belong to the unchanged behavior
actor; no win-rate or turn-length gain can be attributed to the new critic.
Any promising result still needs paired confirmation uncertainty and matched
RL continuation with retention checks before promotion.

Artifacts: `out/research/critic-resource-factorial-2026-10-10/` contains the
sealed registration/identity, replayed resource cache, candidates, selection,
report and completion verification. The confirmation directory contains its
separate registration, frozen panel, collection, monitor, and eventual scores.
The original experiment implementation remains frozen; the PR copy is
format-normalized with identical Python AST, and its tests skip when torch is
absent in core CI.

Validation before publication: three focused tests pass (initial forward
identity and actor freeze; control input isolation and feature orientation;
exact calibration gradient); lint/format checks pass; all 12 selected validation
scores reproduce exactly. Confirmation results are pending.
