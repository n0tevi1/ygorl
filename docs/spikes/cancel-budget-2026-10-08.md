# Cancellation-budget inference pilot

User-approved comparison for #239: restrict repeated target/material cancellation
to 32, 4, 1, or 0 attempts between the existing tracker's progress/reset boundaries.
Zero still preserves the sole legal exit. This does not remove cancel from the
engine, change checkpoints, retrain a policy, or alter running experiments.

The experiment modifies only the candidate's encoded action mask before the
existing CheckpointAgent forward/sample, retaining its RNG and all other inputs.
Use the Python session's existing selection-cancellation counter, including its
player, decision-kind, and game-event resets. Never add actions hidden by the
native mask or mask every encoded legal action. Opponents retain the usual guard.
Budget 32 returns the native observation unchanged, including the native
all-undo fallback; it must not reapply only the cancellation part of that guard.
Candidate selection cancellation covers SELECT_CARD and SELECT_UNSELECT_CARD;
it does not include chain pass, effect decline, or all other meanings of cancel.

First verify the three real target/material fixtures in both hosts, snapshot
restore, unchanged input fields, and a sole-exit case. Verify budget 32 matches
unwrapped policy actions, responses, and final result in complete games.

Then register the frozen six precision-pilot endpoints and select the three BF16
endpoints, one per seed, before opening outcomes. These checkpoints exhibit the
observed cancellation tail; conclusions are conditional on them. Use the same
three fixed opponents as the precision study. A fresh panel seed 2026100824 draws
16 shared deck/deal clusters with four seat/first variants: 64 games per cell,
3 models x 3 opponents x 4 budgets = 2,304 games. All arms share initial game and
agent seeds, not necessarily identical RNG consumption after policy divergence.
No training/GPU allocation; four single-thread CPU workers, three-hour wall cap.

Bind source, native, models, cards/scripts, environment, decks, registration and
runner identities. Save each game's complete explicit action replay, hash,
interventions and result atomically. Cold replay every game and compare actions,
responses and result. Stop on any health/replay failure, retain the failed data;
do not replace failed games. Resume only with identical bindings, validating
completed artifacts. Status is refreshed while workers run; the wall deadline
persists across restarts. Completed cells are durable checkpoints.

Report paired win differences against budget 32 with crossed seed x shared
deal-cluster bootstrap (20,000 draws, seed 2026100825), per-seed/opponent scores,
turn/decision distributions, cancellation and intervention counts, sole-exit
preservations, and selected intervention replays. These are exploratory three-arm
comparisons on only three trained seeds: no automatic promotion or equivalence
claim. Fewer cancellations alone is not success. A legal alternative need not
lead to a feasible or good continuation; winning and completing games cannot
alone prove every necessary target change remains possible. Separately examine
changed outcomes and target-changing cases before deciding on training usage.

The selection-context representation gap remains a separate cause to address;
this pilot tests a guard, not whether the actor has learned to avoid mistakes.
