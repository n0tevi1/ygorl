# Attack-history retraining: bounded result and next investigation

The fixed study finished without engine-health failures, but does **not** support
promoting the attack-history model or extending this training recipe. Across the
two training seeds and two frozen opponents, context minus control score is
**−1.37 percentage points**, paired deal-cluster 95% interval **[−4.69, +1.95] pp**.
The difference-in-differences adjusting for the initial input-schema effect is
−1.56 pp [−4.79, +1.56]. Neither establishes a strength benefit.

Protocol: [registered comparison](attack-history-retrain-2026-10-14.md).
Artifacts: `out/research/attack-history-retrain-2026-10-14/report.json`, bound to
study identity `4c750e3f73300f4e32dbe5387f4a085cba1b4a312cc7cb6d094de986f909399b`.
Runtime commit `e54e224` contains the tracing implementation merged as PR #422
(main `2c55bab`). The local full suite passed 1,946 tests with 23 skips; GitHub
reported no CI check runs for that PR. This is local validation, not remote-CI
success.

## Completion and reliability

- All 64 updates completed: 65,536 training rows, four arms from the same warm
  u128 actor/critic source, matched fresh optimizer/reference/pool.
- All 481 completed training games replayed exactly. Eight unfinished slots per
  arm at the fixed endpoint remain censored, not finished-game losses.
- All 3,584 evaluation games ended normally and passed exact cold replay. First
  game per cell additionally passed unwrapped-policy parity.
- A saved real control update reproduces the entire learner state bitwise:
  667 tensors / 29,932,939 elements, including optimizer and reference state.
  This checks the evidence path, not playing strength.
- The independent monitor verified completion and the event was acknowledged.
  The launch preflight cwd mistake was fixed before any training; its evidence
  remains in `launch-preflight-failure.json`.

## Fixed-panel results

Each row pools 512 games: 64 deal clusters × four deck/start assignments × two
opponents (Greedy and frozen source warm u128). Initial models were verified
identical across training seeds within each schema, so initial evaluations are
shared. The u512 checkpoint is an older, more extensively trained reference;
these 16-update warm restarts are not equivalent training budgets.

| Model | Wins /512 | Win rate | Mean turns | P95 turns | Games >20 turns | Candidate cancels |
|---|---:|---:|---:|---:|---:|---:|
| Control initial | 297 | 58.01% | 10.031 | 18 | 10 | 9 |
| Context initial | 298 | 58.20% | 10.092 | 19 | 11 | 9 |
| Seed 0 control | 302 | 58.98% | 9.570 | 17 | 8 | 80 |
| Seed 0 context | 311 | 60.74% | 9.459 | 17 | 9 | 14 |
| Seed 1 control | 315 | 61.52% | 10.227 | 18 | 13 | 15 |
| Seed 1 context | 292 | 57.03% | 9.951 | 18 | 8 | 15 |
| Frozen warm u512 | 398 | 77.73% | 8.811 | 15 | 0 | 155 |

Context minus control mean length is −0.193 turns [−0.448, +0.048], and the
>20-turn rate difference is −0.39 pp [−1.46, +0.68]. Shorter point estimates do
not establish stronger play. Candidate cancel totals screen for investigations;
they are not counts of loops or tactical mistakes.

The seed/opponent breakdown is heterogeneous:

| Context minus control | Greedy | Frozen warm u128 |
|---|---:|---:|
| Seed 0 | +4.30 pp [−2.34, +11.33] | −0.78 pp [−7.03, +5.47] |
| Seed 1 | −0.78 pp [−6.64, +5.08] | −8.20 pp [−15.23, −1.17] |

These subgroup intervals are nominal, without multiplicity correction. The
seed-1/RL-opponent result warrants replay review; it is not by itself proof of
a specific implementation defect. Intervals resample entire deal clusters and
are conditional on these two training seeds and opponents. They do not estimate
training-seed-population uncertainty or expert-opponent strength. The protocol
was a bounded diagnostic, not a reliable 5 pp detection study.

## Relevant training coverage

| Arm | Adjacent attack-target windows | Multiple available attackers | Finished games |
|---|---:|---:|---:|
| Seed 0 control | 192 | 70 | 128 |
| Seed 0 context | 150 | 49 | 122 |
| Seed 1 control | 158 | 48 | 124 |
| Seed 1 context | 202 | 83 | 107 |

The context arms together receive 132 multi-attacker windows in 32,768 rows
(0.40%). Context-event detection agrees with all 352 adjacent target windows in
those arms. This establishes that the input reached training; it does not prove
the model learned to use it correctly or that more samples will fix the issue.
Coverage is policy-dependent, not a matched set of supervised target labels.

## Decision and next action

Keep the input repair available and retain successful-game tracing. Do not promote
these checkpoints, change the default, reset Adam globally, or extend training
based on this result. The stronger u512 reference still has cancellation signals,
so winning more and behaving sensibly need separate checks.

A fixed read-only replay audit of **existing** games has also completed:
`out/research/attack-history-retrain-review-2026-10-14/`. It selected the two lowest-index
outcome flips in each direction for each seed/opponent, every game with at least
eight candidate cancels, and the longest game per final model. All 43
unique reconstructions match the original action transcripts and engine responses.
This adds no training updates or independent evaluation samples.

The audit found candidate no-event repeat signals in six games: two from seed-0
control versus Greedy, and four from warm-u512. Two inspected games actually
selected self-Ash (seed-0 control/Greedy game 76 and seed-1 context/frozen-u128
game 20). These are selected observations, not estimates of overall error rates;
self-Ash still needs context to judge tactical harm.

For the 16 paired outcome-flip examples, the first differing actions occur at
8 idle commands, 4 chain prompts, 2 card selections (deck/hand, not attack
targets), 1 placement and 1 yes/no prompt. All 16 diverge before the candidate
has declared any attack in that game; none first diverges at an attack-target
choice. The trained policies differ in earlier play, not only when consuming
the newly added attack context. This rules out the simple description that every selected outcome flip
starts with attacking the wrong target; it does not rule out later battle effects
or identify a causal explanation of the aggregate score difference.

See `summary.json` and the preserved public traces for exact decisions. These
remain investigation signals rather than automatic tactical-error labels.

A further inference-only probe covers **all 352 saved context-arm target rows**,
including all 132 multi-attacker rows. Remove only the latest accepted-attacker
event while keeping the board, legal actions and other history fixed. At final
checkpoints, mean total-variation distance between full/ablated action
distributions is 0.0891, with 33/352 greedy-choice changes (multi-attacker:
0.0895, 11/132). Initial checkpoints already show 0.0994 and 34/352 changes.
See `input-probe.json` and `input_probe.py` for all rows and input hashes.

This rules out complete insensitivity to event presence on these observations.
It does **not** show correct attacker identity reasoning: removing an event also
changes attention context, and the states are saved training observations, not
an independent held-out task. Similar initial sensitivity is not evidence that
the 16 updates taught the relation. No games or optimizer updates were added.

The next investigation should test correctness of attack-target rankings under
controlled attacker/target changes and inspect earlier phase/chain choices,
before another training allocation. Distinguish whether
performance differences arise at attack targets, earlier setup/phase choices,
or broader policy changes. Neither aggregate win rates nor selected divergent
replays alone identify the causal mechanism of the treatment effect.
