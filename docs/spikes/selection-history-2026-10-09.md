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

## Next behavior experiment

Compare legacy vs selection-history training from the same fixed warm/u256 actor,
with identical training seeds, critic initialization/warm-up, optimizer, deck pool,
reward and cancellation guard. Record a pre-training baseline for each schema:
new tokens alter inputs even before learning. First run a bounded pilot; do not
retune on the four regression cases or interpret those selected cases as a
population error rate. Evaluate full paired games and all selection opportunities:
cancellation rate, repeated no-progress runs, successful target switching, health,
win rate, turn distribution and its tail. A shorter game alone is not success.
The original multi-seed critic-continuation study remains frozen and continues.
