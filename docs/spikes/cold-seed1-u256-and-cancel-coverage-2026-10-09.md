# Seed1 cold u256 progress and missed cancellation replay

The formal continuation improves on its own seed1 cold u128 starting checkpoint,
while still exhibiting repeated selections. This is separate from the negative
[terminal policy-credit pilot](selection-terminal-policy-pilot-2026-10-09.md);
it does not change the formal training objective or promotion criteria.

## Paired milestone

Study identity: `270faf6a7cfd97e0084bb8bae3588d47a961e47ffd66f21f63a507f131bb3984`.
Checkpoint u256 SHA: `06323da975018b773957a86599441a374f566e275c7d6d3f62fcaa556efb980b`.
All four 256-game cells are healthy. Raw manifests/checkpoint hashes and common
inputs match. Audit: `out/research/behavior-immunity-2026-10-09/seed-1-cold-u256-review.py`
and its JSON report; no writes to the running formal study.

| Opponent | u128 wins /256 | u256 wins /256 | u128 mean turns | u256 mean turns |
| --- | ---: | ---: | ---: | ---: |
| Greedy | 150 | 176 | 10.5938 | 10.4531 |
| Old 256x2 | 191 | 218 | 11.4258 | 10.8555 |
| Initial 128x2 | 187 | 226 | 12.4609 | 11.7266 |
| Historical RL | 108 | 139 | 10.0859 | 9.8320 |

The primary three-opponent aggregate rises **68.7500% →80.7292%**, paired
+11.9792 percentage points, 95% deal-block interval [+7.8125, +16.0156]. Mean
turns fall **11.4935 →11.0117**, change -0.4818, interval [-0.9375, -0.0430].
Historical RL wins rise 42.1875% →54.2969%, change +12.1094 points, interval
[+4.2969, +20.3125]. Intervals resample the same64 deals jointly across opponents
20,000 times; they condition on these two trained endpoints. This is exploratory
one-seed milestone evidence, not the final three-seed gate or a warm/cold comparison.

The sealed old-task retention set is identical (81,546 rows /256 games), with
prediction hashes checked and per-game errors recomputed from saved tensors.
Q game-weighted MSE improves 0.883406 →0.821644; paired128-deal change -0.061762,
95% interval [-0.077808, -0.045577]. V MSE improves 0.882740 →0.821541. This is
retained/improved prediction on that fixed task, not proof that every tactical
skill is retained or evidence about warmup specifically (this arm is cold).

## Lower aggregate cancels concealed a repeat

Across all1,024 evaluation games, total chosen material cancels decrease112 →53,
and games above20 turns decrease48 →27. However, two u256 games have >=8 total
cancels (none at u128): initial-128x2 game147 has9 and game214 has11. Both win.

Cold reconstruction verifies original outcomes, accepted actions and engine
responses. Game214, already selected automatically, has a maximum of6 identical
choices without game events. Manual replay of game147 finds **9** repeated
cancels without a game event, triggering the existing repeat detector at8.
Evidence is in the original observer's `replays/.../214/` and
`seed-1-cold-u256-manual/replays/.../147/` directories. These remain investigation
signals, not labels that every cancel was tactically wrong.

The monitoring blind spot is selection, not a failed repeat detector: the old
selector chose only the maximum cancellation total in each cell, alongside
fixed-index and length tails. A larger total can be split across progress events,
hiding a smaller total concentrated in one repeat. The selector now includes
**every game with at least8 candidate cancellations**, retaining the largest
cancel case even below8, all fixed16 labels, and existing length tails. Total
counts select work; replay determines whether it repeats. This does not cover
all possible non-cancel loops or replace full-population trace instrumentation.

Regression tests cover the9-consecutive versus11-split counterexample, the8/7
boundary, fixed-sample labels and existing progress semantics. The behavior,
continuation-behavior and study-watch suites pass23 tests; presubmit passes.
An initial test invocation lacked native bindings in the documentation worktree;
the rerun loads its edited Python and the isolated tested native extension.

## Observer deployment and next step

An observer-only overlay uses the **unchanged formal Python/native runtime**,
with the new selector imported from a separately bound file. Original completed
replays are copied only after all stored file hashes verify; the new observer
backfills newly eligible past cells and applies the same selection to future
cells. Fixed-index frequency denominators remain separate from selected tails.

The first isolated deployment failed before observer initialization because
`colab_checkpoint` was absent from its tools import path. Its service failure,
acknowledgment and deployment record are retained at
`out/research/behavior-cancel-coverage-2026-10-09/`; the original observer stayed
active. The fresh replacement is
`out/research/behavior-cancel-coverage-restart-2026-10-09/`.
Startup preflight verifies all imports, real-game147/214 selection and unchanged
native SHA `d91ed6d81325c9e813274abd9b639823120a320c5f4c7bbdc81aca425a323eb4`.
The replacement has independent incident handoff; original observer services are
retired only after its first healthy snapshot. No STOP is cleared.

Continue the preregistered formal run to u512, retain the repeat investigation,
and await the remaining seeds/arms for final inference. The two independent
training-seed terminal-policy replications continue separately. Neither pilot
cancellation reduction nor this positive milestone authorizes default promotion.

The replacement is now verified active and the old observer/handoff services
have been retired. Its first completed backfill has677 verified replays,
zero running/queued jobs and576 fixed-index games. Sixteen newly selected
replays were audited independently, including the manually recovered game147;
15 contain candidate no-progress repetition. All new repeat incidents were
acknowledged as investigated, not resolved policy defects. Native reconstruction
and file hashes pass; no health failures occur in the replacement.

All19 warm seed0 u512 games with >=8 total cancels are now covered;18 contain
candidate repeat findings. This denominator is the flagged subset, not a global
repeat rate. For example, warm u256 Greedy game253 cancels32 times at99.2801%
probability and still wins; warm u512 Greedy game116 cancels32 times at99.9017%
and wins. The largest repeated-choice count can exceed32 because select actions
also repeat; do not confuse it with cancellation count. Winning after repetitions
reinforces why eventual win labels alone do not prove those steps useful.
`backfill-audit.json` preserves all16 cases including the non-repeat case.
