# More precise training labels improve MSE, not established choice strength

Follow-up: [Completed state-coverage comparison and budget audit](critic-state-expansion-results-2026-10-11.md). The registered expansion does not establish better choices; no model is promoted.

`critic-target-replica-fit-2026-10-11` completed all three registered arms. Independent verification reproduces all12 checkpoint predictions, retention gates and the primary bootstrap; the old16 arm exactly reproduces existing expanded128 parameters at0/8/16/32. Actor and all non-Q parameters stay unchanged. No model is promoted. Completion event `8183879b4835fdac1253f71652ee95b29f5cdd5c6b33a907c87020d03ac1e783` is acknowledged after verification.

The intervention changes only the target sample on96 training states: first16, independent second16, or pooled32 continuations. The original32 training states retain their existing pooled32 labels. Architecture, initialization, centered-MSE loss, source anchor, optimizer and32-update budget are fixed. The second16 labels now train two arms; their96-state metrics are explicitly reused training diagnostics, not independent holdout results.

| Fixed step32 target arm | Reused48-root MSE | Conditional choice wins |
|---|---:|---:|
| Unchanged initialization | .016696 | 61.458% |
| Old16 | .015190 | 61.003% |
| New16 | .012959 | 61.523% |
| Pooled32 | .013823 | 61.393% |

Primary pooled32-minus-old16 MSE is **−.001366971**, conditional shared-suffix95% interval **[−.002071010, −.000659927]**. Conditional wins differ by **+.390625 percentage points**, interval **[−.390625, +1.171875]**. All retention gates pass. The new16 arm's small favorable win point estimate is not a basis for selecting it after inspection. All48 states were already inspected; these intervals condition on fixed states and do not turn this post-hoc study into fresh confirmation. Preserve all arms/checkpoints and their less favorable prior32-panel results.

This supports target-sample sensitivity in score calibration. Together with the rank-preserving shrinkage controls, failed pairwise-objective benefit and longer-fit results, it still does not establish improved action choices or actor strength. Further parameter sweeps on the same diagnostic states are not the next step.

## Registered state-coverage expansion

Next stage: `critic-ranking-state-expansion-data-2026-10-11`, identity `47bebe615b3f6bb28e8ff57cb66886070d3de70edc16d7f47d8e59295360db27`. Collect320 independently reshuffled source games,40 per eight broad/Runick × old/mirror opponent × own/face strata. Opening validation finds320 unique deals and opening-hand pairs, zero overlap with3,136 prior unique deals. Four workers and a two-hour cap bound source collection; a separate monitor checks it. A preparation attempt using system Python lacked the project import path, so preparation was rerun in the frozen virtual environment before any study service or formal games existed; evidence is retained in `preparation-attempt.json`.

Before outcomes are inspected, register hash-based selection of the first eligible learner battle decision (2–6 supported actions) in each source game: first16 qualifying games perstratum become training, next8 validation, totaling128 new training and64 new validation states. Insufficient coverage stops rather than silently replacing games. Outcomes/errors never choose roots. None of the old diagnostic states enters training.

After source verification, branch labels use16 suffixes per supported action on new training roots and32 on new validation roots, plus192 original controls; four workers/eight-hour branch cap. Keep validation labels unaggregated until fixed fits finish. Compare existingcanonical128 training states against those same128 plusnew128, with identical32-update centered-MSE fits and exact baseline reproduction. Old training labels remain canonical (original32 pooled32 plus96 old16), so extra state coverage is the intervention. More data uses more sampling budget; this is not compute-matched.

Primary is now **conditional choice-win difference** on64 fresh validation states, expanded256-step32 versus existing128-step32, with a stratified root-plus-suffix bootstrap (50,000 draws, seed2026101126); fixed-root suffix sensitivity uses seed2026101127. Also report the unchanged initialization, MSE and retention. Do not claim improvement if the candidate is worse than unchanged or select an earlier checkpoint after inspection. The same decks/cards and frozen actors mean this is not unseen-card generalization. A positive critic result would still need actual actor-RL/full-game validation.

Evidence: each study's registration, identity, reports and verification scripts under `out/research/`. [Previous confirmation and objective diagnostics](critic-unused-root-and-ranking-objectives-2026-10-11.md). Issues #61/#83/#36 remain active; old STOP evidence is preserved. Development follows main, matched runtime stays ae09594.

## Source collection verified; branch collection launched

All **320/320** source games ended normally. Source verification checks the complete task grid, artifact identities/hashes, engine health and recorded cold-replay equality; it does not independently rerun all320 games. Deals/openings remain unique and disjoint from3,136 prior deals. Completion event `64a0947410c1e721bf949526560fb1ecf55acc8d9e92c70a6f1805daf82a74ca` is acknowledged. Behavior analysis version3 was rerun from main; legacy counts, findings and coverage match, with selection-cycle detection added.

| Frozen source panel | Games | Mean turns | Median | P90 | Maximum |
|---|---:|---:|---:|---:|---:|
| Broad | 160 | 11.069 | 9 | 16 | 47 |
| Runick | 160 | 24.688 | 23 | 46 | 65 |
| Combined | 320 | 17.878 | 12 | 39 | 65 |

Only5/320 end by turn4;93/320 exceed20 turns. The previous256-source mean was17.543, with broad12.180 and Runick22.906. These are different deals under frozen policies, **not evidence of actor improvement or regression**. Candidate self-Ash is0/117 opportunities. Other investigation signals include70 attack intervals with self-damage,32 chain intervals with self-damage, and133 exits from battle with an attack available; these are not automatic tactical error labels or proven missed lethals.

Five games have repeated-selection signals. Games26 (opponent) and75 (candidate) each repeat selection/cancellation32 times. Games95/195/259 have candidate material selection/unselection cycles of period4. All five full policy replays independently reproduce the original choices and final results. Native observation hashes repeat exactly at each cycle period, and every involved policy has `selection_history=False`. Cancel probability is.998698 and.997731 in the two cancellation loops; the32-cancel guard finally masks cancellation at decisions184/214. The material cycles exit by sampling an alternative material at decisions653/388/431. This directly establishes repeated model inputs, not merely similar public boards. It does not establish that enabling history alone will teach the right choices.

This reproduces the known frozen-actor selection-control limitation. Retain the [selection/cancel intervention evidence](policy-sampling-confirmation-2026-10-10.md) and [earlier material-cycle investigation](card-pool-training-results-2026-10-10.md). A future actor intervention must separately validate history training and cancellation controls; changing those during this critic comparison would confound state-coverage effects. Cancellation-only restrictions cannot fix the three select/unselect cycles. These finite decision loops also cannot alone explain the much longer Runick turn distribution.

All eight strata have enough eligible battle states (32–40 per40 games). The registered hash ordering selects exactly128 new training and64 new validation roots, with distinct source games. An independent selection verifier reconstructs the ordering and exact action/suffix grid without reading outcomes. No behavior finding filters a root.

`critic-ranking-state-expansion-branches-2026-10-11` is now running under identity `b075296af6ddc48c4899471c9da4da9c3d828ceb7c27a941451df71aaa66dbb5`: **14,688 tasks =192 original controls +7,136 training continuations +7,360 validation continuations**. The192 original controls run first, then collection proceeds with four workers and an eight-hour cap. Service invocation `840a8b5517804f1aba3ebfc3f70de1aa` has an independent monitor. Validation labels remain unaggregated until the two fixed fits freeze. No critic or actor has been promoted.

Audit evidence: source `verification.json`, `behavior-v3-audit/verification.json`, `encoding_audit.py`, five `encoding-*.json` replay records and `encoding-verification.json`; branch `selection-verification.json`, identity, registration, service launch and monitor. Development main is `fb0ab3f`; frozen matched runtime remains `ae09594`.
