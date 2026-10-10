# Full-game sampling: modest outcome changes, deterministic cancellation loops

Lower temperature has not yet demonstrated a general strength gain. Argmax
reduces mean duration on this panel, but increases cancellations tenfold and
repeatedly reaches the existing cancellation guard. Keep inference defaults
unchanged and test the signals on fresh deals.

## Reviewed results

Root: `out/research/policy-sampling-fullgame-2026-10-09/`.
Identity: `4faad320b1f5482950b40a24c1653fc17a92ceec8c1515eab26e71ee7b805f25`.
All 2,304 games ended normally, without errors or termination limits. Independent
review checked all record/replay hashes and exact results, engine responses and
step transcripts for all 768 T1 controls against their original source replays.
Completion was acknowledged after review.

Three fixed checkpoints, two unchanged opponents, 32 shared deal blocks and four
seat/deck assignments per cell. This was an exposed development panel. Changing
only the candidate's sampling also changes trajectories and encountered states;
these totals do not measure the same set of cancellation opportunities.

| Candidate / opponent | Wins T1 / T0.5 / argmax (each /128) | Mean turns T1 / T0.5 / argmax | Cancels T1 / T0.5 / argmax |
| --- | ---: | ---: | ---: |
| seed0 cold u512 / Greedy | 103 / 100 / 96 | 9.570 / 9.656 / 8.766 | 3 / 34 / 32 |
| seed0 cold u512 / historical RL | 91 / 96 / 103 | 9.055 / 8.227 / 8.086 | 13 / 1 / 64 |
| seed0 warm u512 / Greedy | 112 / 112 / 107 | 9.023 / 9.102 / 8.539 | 6 / 29 / 192 |
| seed0 warm u512 / historical RL | 90 / 98 / 92 | 8.602 / 8.320 / 8.406 | 38 / 48 / 256 |
| seed1 cold u256 / Greedy | 94 / 98 / 87 | 10.828 / 10.438 / 10.695 | 1 / 0 / 64 |
| seed1 cold u256 / historical RL | 73 / 74 / 85 | 9.578 / 9.453 / 10.250 | 3 / 30 / 32 |

Descriptive totals: wins **563 / 578 / 570 out of 768** (73.31%, 75.26%,
74.22%); mean turns **9.443 / 9.199 / 9.124**; games exceeding 20 turns
**19 / 16 / 14**. Cancels are **64 / 142 / 640**, and games reaching a streak
of 32 cancels are **1 / 2 / 18**. No game exceeded its termination budget.

Every cell's paired win-difference interval for T0.5 or argmax versus T1 crosses
zero. Intervals resample 32 shared deal blocks within cell, condition on the
fixed checkpoints, and are unadjusted across contrasts. The total win counts
are descriptive, not independent training-seed evidence. Argmax loses wins in
all three Greedy cells and gains them in all three historical-policy cells;
this opponent dependence needs confirmation. Same-win pairs also have mixed
turn changes: for example, seed1 cold versus historical RL takes 1.019 more
turns under argmax among its 54 pairs won in both arms. Shorter overall games
alone cannot establish stronger play.

## Root-cause replay audit

Root: `out/research/policy-sampling-repeat-audit-2026-10-09/`.
All 21 games reaching 32 consecutive cancels were rerun with full policy
inference. Results, responses and complete step transcripts exactly match their
sources. Hash every actual model input tensor after `collate`, including cards,
globals, actions, events and masks; record chosen actions and all probabilities.

Within each repeated cancellation window, all input tensors and all output
probabilities are identical across 32 cancellations. The 18 argmax games contain
20 such windows (two games contain two). Consequently the deterministic choice
continues until the native guard changes the legal mask. This reproduces the
[legacy selection-history omission](selection-history-2026-10-09.md), rather
than an engine crash or a newly discovered representation bug. It explains
persistence once inside these windows, not every change in how often the policy
enters them. The learned preference and general target-selection competence
remain separate problems; the earlier opt-in history repair has not shown a
replicated strength gain.

The diagnostic collector initially used `vars` on a dictionary and failed before
completing a game. The failed script and failure record are retained; the fixed
collector reran every case. `audit-evidence.json` binds the successful source,
panel, report and analysis. No original results were replaced.

## Running fresh confirmation and bounded mitigation

Root: `out/research/policy-sampling-confirmation-2026-10-09/`.
Identity: `7f68eb7b214bf7c1084bdedcf198328de1cd8460e351eb349799654ad8e5a262`.
Fresh panel seed 202610091041, 32 shared blocks, with engine seeds disjoint from
the exposed sampling panel. The same models and deck pool remain fixed.
Four modes × 768 games = **3,072 full games**: T1, T0.5, argmax, and argmax
with the existing candidate-only cancellation budget set to 1.

Primary sampling contrast: T0.5 versus T1. Also report argmax versus T1 and
argmax-budget1 versus argmax, all per-cell paired effects and adverse outcomes.
Budget1 preserves the first cancellation and the sole encoded legal exit.
It is an overhead mitigation, not a learned repair or a guarantee that every
later target revision is unnecessary. The earlier [sampled-budget confirmation](cancel-budget-confirmation-2026-10-09.md)
showed reduced overhead without a strength gain; this arm tests the newly
observed deterministic-loop amplification on fresh deals.

All four full-game preflight modes passed cold action/response/step replay.
An additional known-loop preflight cut cancels 32→1 and decisions 318→256,
preserving the win on turn 6. This selected example is not a population result.
Every intervention records native/restricted masks and the chosen action;
review requires a nonempty subset removing only cancellation choices.

Two CPU workers, a three-hour cap, per-game artifacts and an independent watchdog
are running. Every completed game requires an exact cold engine replay. Turn
and decision limits count as non-wins in all denominators, are saved and alerted,
and mark completion nonhealthy. Other health, mask or identity failures STOP.
No automatic temperature, mask, reward or training-label promotion. Formal PPO
and its frozen runtime continue unchanged.

The [fresh four-arm confirmation](policy-sampling-confirmation-2026-10-10.md)
has completed all 3,072 healthy games. T0.5's win gain did not reproduce;
argmax-budget1 removed 1,302 repeated decisions with identical retained step
sequences and terminal outcomes on the 768 paired games. A broader matched-u512
inference comparison is running; formal defaults remain unchanged.
