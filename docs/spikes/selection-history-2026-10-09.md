# Actor-private selection history (opt-in encoding version)

Verified failures: warm/u256 repeatedly cancels materials with byte-identical input;
three different actor-chosen fusion targets produce identical material observations
and different actual summons. Missing accepted selection actions in event history
cause this information loss. More training cannot recover that omitted distinction.

Add an opt-in `NetConfig.selection_history` schema flag, default false. Keep legacy
event token IDs, shapes, masks, rewards and native guards unchanged. The new schema
adds one event type after the legacy types: `selection_choice`. Record every accepted
select/unselect/cancel/finish action, including intermediate actions not yet producing
an engine response. Only the deciding player receives this token; neither its
existence nor unknown-card count leaks to the other player. Use the already redacted
decision action, never look up a hidden identity from the omniscient core.

Token: actor-relative player/card/location; optional description-owner card; value1
is action-kind index+1, value2 is decision message type, value3 is a per-player
accepted-selection ordinal (1..65535, saturated). Existing turn/phase/LP fields and
chronological ordering remain. The ordinal resets only at duel start; it is historical
ordering, NOT an inferred currently pending target or cancel streak. A bounded suffix
of actions records target changes, cancellation and completion without claiming a
particular game's last selected card is always its fusion target. Event-window
truncation remains explicit through the configured window; a chosen target can be
forgotten after sufficiently long later history. Current fixtures require adjacent
selection memory. The finite ordinal may saturate; the existing guard is retained.

Both native HostDuel/HostPool (including forced steps) and Python PointObserver must
record actions. Recreated duels/snapshots must preserve/reset the history consistently.
Checkpoint signatures include the opt-in flag, so mixed-schema networks cannot share
observations. Old checkpoints default to legacy. Explicit warm-start into the new
schema inserts the new event-type embedding row without shifting old categorical
weights: logits on legacy observations must be identical. New observations change
logits and require separate training; this is not claimed as an immediate policy fix.

Validate native/Python parity and opponent privacy, invalid-action handling, repeated
reads/idempotent observer hooks, forced selections, reset/snapshot behavior, same-shape
schema rejection, migration and real three-target/loop fixtures. Full presubmit before
merge. Keep ongoing registered training and its shared native environment frozen;
build/test in an isolated worktree/runtime. A small separate training smoke must show
that the new inputs reach learner/checkpoint/evaluation, not prove playing strength.


## Implementation evidence

Four replay fixtures preserve all accepted actions from the original warm/u256
failures (greedy/220, old-256x2/147, initial-128x2/168, historical-rl/158).
Native and Python arrays match at every decision with both schemas. The three
fusion target identities now survive into their otherwise identical material
windows; repeated cancellation changes the acting player's history. Invalid
choices leave history untouched, opponent observations receive no private token,
and checkpoint migration preserves old-event logits exactly.

An isolated CPU integration canary migrated a tiny legacy actor, trained four
updates with selection history, resumed for one more update, and evaluated eight
complete games: eight normal wins, no retries, script errors, undecodable messages
or unknown messages. This validates the pipeline only; its random-initialized
small model is not evidence of playing strength or loop reduction. Artifacts:
`out/research/selection-history-canary-2026-10-09/report.json` and `run-fixed.log`.
The first canary launcher failed at multiprocessing import; its failed log is
retained separately, and the corrected run uses a main guard and one eval worker.

## Completed paired pilot (2026-10-09)

Implementation merged in [#249](https://github.com/n0tevi1/ygorl/pull/249),
commit `39b7dce`. Full presubmit on its exact source tree: **1716 passed,
22 skipped**; GPU-specific follow-ups: **23 passed**. The skipped suite cases
include 19 GPU cases covered by follow-ups, two opt-in network cases and one
intentional snapshot condition. Isolated native solver tests require the current
helper, rather than the old root build; see the retained validation record.

The registered pilot used one training seed (249), the same fixed warm/u256
actor **and trained critic** in both arms, fresh Adam/reference/pool, eight
critic-only updates and 24 PPO updates. Reward, legal actions and cancellation
guard were unchanged. It completed 467 legacy and 484 selection-history training
games without errors or truncation. All 128 evaluation games completed normally.
Each endpoint used the same 32 Greedy games: eight deck-pair blocks, both deck
assignments and both starting players. This is a small behavior pilot, not a
strength benchmark against top-level opponents.

| Encoding / endpoint | Wins / 32 | Mean turns | p90 turns | Cancels / opportunities | Games with >=8 cancels without public progress |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legacy initial | 23 | 9.8125 | 14.9 | 0 / 12 | 0 |
| Legacy final | 21 | 9.59375 | 15.0 | 15 / 29 | 1 |
| Selection initial | 23 | 9.875 | 14.9 | 0 / 11 | 0 |
| Selection final | 23 | 9.03125 | 12.9 | 2 / 14 | 0 |

Exploratory paired percentile bootstrap over the eight deck-pair blocks (100,000
draws, analysis seed 20261009250): final selection minus legacy win rate
**+6.25 percentage points, 95% interval [-9.375, +18.75]**; mean turns
**-0.5625, interval [-2.50, +1.25]**. The before/after difference-in-differences
for mean turns is -0.625, interval [-2.50, +1.15625]. These intervals condition
on this one training seed; they do not cover training-seed variation. All cross
zero. Selection's own win count stayed 23/32; it has not demonstrated a strength
gain. Opportunities are policy-dependent and the cancellation difference is
dominated by one game. Do not promote the schema or declare loops solved.

Exact action/engine-response replays establish:

- Legacy final game 1 repeats the same target choice and cancels 15 times at
  decisions 91, 93, ..., 119. All cancellation observations have the same SHA256,
  and cancellation probability stays 0.8873858. It then completes the choice and
  wins on turn 6. This reproduces the observation aliasing on a new pilot game.
- Selection final games 0 and 28 each cancel once, then choose a different target
  and complete selection. These are target switches, not repeated loops; tactical
  optimality is not established. The matched selection game 1 has no cancellation.
- Forced-prefix diagnostics on the four original cases are mixed: both training
  arms can reduce cancellation probability; legacy is better on some cases.
  Adding the new inputs can initially increase cancellation probability. These
  selected prefixes are not natural-policy mistake rates.

Confirmed cause: the old representation omits accepted intermediate choices,
including information needed to distinguish selected targets. Fixed by the
opt-in actor-private history. Separately, gamma=1, turn discount=1 and no
per-decision cost leave a state-preserving loop with the same winning suffix
without a terminal-return penalty. The gradient-level reason that cancellation
became strongly preferred remains unresolved; this pilot does not identify it.

Artifacts under `out/research/selection-history-pilot-2026-10-09/`:
`identity.json`, `protocol.md`, `report.json`, `paired-analysis.json`,
`analyze.py`, `audit_games.py`, `audit-summary.json`, all raw evaluation JSONL
and replay responses, `selected-case-probabilities.json`, and
`causal-scope.md`. Full validation evidence is under
`out/research/selection-history-canary-2026-10-09/VALIDATION.md`.

## Registered independent replication

`out/research/selection-history-replication-2026-10-09/` freezes three new paired
training seeds (250, 251, 252), the same source and training budget, and 128 new
Greedy games per initial/final endpoint. The 32 deck-pair blocks are shared across
seeds/arms/endpoints: 1536 game executions are **not** 1536 independent samples.
Report paired per-seed contrasts, initial-to-final changes, and seed variation;
retain loop, health, turn-tail and win checks. Audit flagged games before labeling
mistakes. No change to rewards or masks, no automatic promotion.

Preflight verified identical inherited actor/critic tensors in both schemas. The
independent training service and watchdog started on 2026-10-09, with checkpoints
every update, a pre-optimizer health gate, ten-second parent heartbeat, a four-hour
runtime limit, and persistent completion/failure handoff. Frozen identity SHA256:
`9c7d89e44d27bcbfbb5af620d51e1f8a7a6749c966048f5d179e7f55caa7dbbd`.
The original multi-seed critic-continuation study and runtime remain unchanged.

## Replication completed: no promotion

All 1536 evaluations completed normally. Over the shared 128-game panel and three
training seeds, final legacy wins 244/384 (63.542%), selection wins 253/384 (65.885%);
mean turns 9.3047 vs9.6016. Seed-wise selection-minus-legacy win deltas are +1.563,
0, +5.469pp; turn deltas +0.516, +0.711, -0.336. A crossed bootstrap preserving
the common deal panel gives win +2.344pp [-4.427,+8.854] and turns +0.2969
[-0.4036,+1.0547]. This exploratory interval has only three seeds and one source.
No replicated turn benefit or demonstrated strength advantage.

Final cancel counts are 28 vs 24, with zero vs one game containing >=8 consecutive
cancels without public progress. The new-schema seed 250/game 36 loop has distinct
inputs but 14 cancellations with probabilities 97.45%–99.03%. The input repair is
verified; learning a good selection policy remains unresolved. Keep default
legacy and do not promote this experiment based on the small pilot.

The [warm/u512 follow-up](warm-u512-repeat-audit-2026-10-09.md) records five new
formal-study replay audits, paired checkpoint progress, retention, and a bounded
credit-assignment diagnostic. The latter records actual PPO row terms but lacks
high-confidence loop coverage; it must not be presented as a resolved gradient
cause. No card-specific penalty or blanket cancellation ban was introduced.
