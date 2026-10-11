# Nonlinear critic refresh transfers to fresh openings

The [head pilot](critic-context-confirmation-2026-10-10.md) now has independently
verified fresh confirmation. All 768 full games completed healthily, with exact
cold replay and critic-cache checks. The panel has 192 new openings, disjoint
from 2,240 earlier openings. Games 459 (broad) and 537 (Runick) contain no eligible
learner idle sample. Their outcomes remain in the totals; error metrics use
511 broad and 255 Runick games, not imputed zero errors.

All four estimators' predictions and metrics reproduce exactly after loading
sealed checkpoints. The behavior actor is unchanged and all estimators score
the same trajectories. These are Q/V errors against terminal returns in [-1,1],
not win probabilities or critic-driven gameplay gains. Sampling takes the first
learner idle decision per turn and its chosen action, with equal game weights;
this does not establish calibration for every chain/selection or unchosen action.

## Independent confirmation

| Estimator | Broad Q/V MSE | Familiar | Runick | Facing Runick | Operating Runick | Low-deck V MSE | Low-deck V bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| control | 0.58255 | 0.48835 | 1.08083 | 1.35670 | 0.80711 | 1.91462 | +0.96279 |
| terminal_only | 0.55679 | 0.46360 | 0.90704 | 1.01317 | 0.80173 | 1.14236 | +0.61813 |
| anchored | 0.56119 | 0.47130 | 0.92442 | 1.05136 | 0.79848 | 1.23016 | +0.66067 |

The low-deck slice uses each game's first eligible idle with own deck <=5:
78 games /38 opening clusters, observed mean return -0.56410. Terminal-only
reduces low-deck V MSE by 0.77226 (95% interval [-0.94512, -0.59259]); anchored
reduces it by 0.68446 [-0.84900, -0.51629]. Bias remains +0.61813 /+0.66067, so
this problem is improved, not solved. In particular the terminal-only mean V
is still +0.05402 on this negative-return subset.

Descriptive, unadjusted 95% intervals below use 20,000 stratified paired-opening
bootstrap replicates, preserving target/anchor/role and clustering opponent/seat
outcomes together. Unequal eligible counts use sum/count ratios and fixed
observed stratum weights. There is one behavior learner, not multiple independent
policy-training seeds.

| Comparison vs original control | Scope | Q/V MSE difference | 95% interval |
|---|---|---:|---:|
| terminal_only | broad/all | -0.02576 | [-0.03800, -0.01227] |
| terminal_only | broad/familiar | -0.02475 | [-0.03939, -0.01002] |
| terminal_only | broad/old-256x2 | -0.02404 | [-0.05015, +0.00034] |
| terminal_only | runick/all | -0.17379 | [-0.24967, -0.10055] |
| terminal_only | runick/own-target | -0.00537 | [-0.02427, +0.01265] |
| anchored | broad/all | -0.02136 | [-0.03119, -0.01064] |
| anchored | broad/familiar | -0.01704 | [-0.02612, -0.00791] |
| anchored | broad/old-256x2 | -0.02345 | [-0.04547, -0.00405] |
| anchored | runick/all | -0.15640 | [-0.22422, -0.09111] |
| anchored | runick/own-target | -0.00863 | [-0.02383, +0.00648] |

Both arms improve broad, familiar and enriched Runick errors on this batch.
Own-Runick intervals straddle zero; they do not prove improvement. Their upper
bounds are small, but formal non-inferiority is not claimed from these descriptive
intervals. Tearlaments also improves: terminal-only -0.02615 [-0.04248, -0.01093],
anchored -0.01377 [-0.02604, -0.00151].

Anchored minus terminal-only Runick error is +0.01739 [+0.00776, +0.02735], while
broad difference is +0.00440 [-0.00016, +0.00888]. This batch does not support a
benefit from the broad prediction anchor; both preselected arms remain visible.
There is no confirmation-based parameter tuning or replacement of the validation
selection rule.

The fixed actor wins 335/512 broad games and 151/256 Runick games, with mean
11.117 and 24.723 turns. These opening-dependent outcomes are shared by all four
critics, and cannot establish shorter games or stronger play from the refresh.

## Next: actual learner-policy shift

`critic-head-learner-shift-2026-10-10` is running with 1,536 games and a separate
heartbeat/failure monitor. It pairs the existing fixed-pool and expanded-pool
u528 actor checkpoints on identical new deck orders, seats, opponents and RNG
seeds: 512 broad plus 256 Runick games per behavior. There are 192 independent
opening clusters, disjoint from 2,432 earlier openings. Shared openings are not
counted as twice as many independent samples. A 144-game health canary precedes
the remainder.

Critic weights and the reference public encoder remain frozen. A reference actor
host follows each recorded trajectory to generate inputs in the original feature
space; a separate behavior host verifies the actual changed actor's probabilities
against the trace. The ordered vocabularies must match. Both seats passed a
preflight: the original behavior reproduces prior cached features/predictions
exactly, and the changed actor has nonzero probability differences while its
replay/probability and reference-critic cache checks pass. An initial preflight
incorrectly compared vocabulary object identity; it was corrected to compare
ordered card passwords, with the failed preflight retained separately.

The primary question is whether each refreshed critic's error improvement over
the original conditioned control persists under the changed learner, and how
that improvement differs between the two paired behavior distributions. Report
both policies, both roles, familiar/old-opponent retention and low-deck errors;
keep the enriched Runick panel separate. No new fitting occurs. This measures
transfer across trajectories generated by changed actor weights, not successful
migration of critic weights into a new encoder or an on-policy PPO update.

If transfer is healthy, the next engineering step is an explicit critic/encoder
adapter with parity tests before matched RL continuation. If transfer fails,
inspect the changed-policy error slices before refreshing data or changing the
loss. No automatic production promotion follows this offline result.

## Evidence and validation

Artifacts: `critic-head-refresh-confirmation-2026-10-10/analysis.json`,
`completion-verification.json`, sealed predictions and replay/cache files;
`critic-head-learner-shift-2026-10-10` contains the registration, sealed identity,
worker preflight and monitored run. The completion event was acknowledged only
after the independent confirmation audit. This documentation changes no
production actor, reward or PPO implementation.
