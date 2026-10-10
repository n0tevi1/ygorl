# Card-transfer audit and bounded semantic pilot, 2026-10-10

Adding semantic inputs is an experimental hypothesis, not an established strength
improvement. The current 128x2 RL checkpoint trains on 20 fixed MD decks, uses
484 distinct main/extra cards, and has text, effect text, facts and ID dropout
disabled. The corrected BC input actually contains 487 card identities: three
identities outside the deck lists appear in observations (27204312, 46647145,
52340445). Deck membership alone cannot certify interaction exposure. The full
RL observation stream has not been audited for generated/referenced cards.

Historical evidence must inform the next experiment. `docs/benchmarks.md`
records that the 2026-09-24 broader-corpus run improved held-out-type win rate
from 0.535 to 0.618, while familiar-deck performance declined. Its subsequent
three-seed semantic-view comparisons did **not** establish a benefit; some
settings harmed familiar-deck performance. Those experiments used an earlier
64x1 policy/runtime, and combined views with ID dropout in several arms. They
are motivation for controlled diagnostics, not interchangeable with the current
128x2 checkpoints or proof that text is either necessary or useless.

Also, dynamic deck ingestion already exists (`TrainConfig.deck_pool`,
`EvolvedDecks`, periodic manifest reload). It is disabled in the current study.
The missing work is validated curriculum generation and assessment; it would be
incorrect to describe all deck-pool integration as unimplemented.

## Reproduced warm-start defect

`warm_start` promised unchanged outputs when adding card views, but new card-text,
mean-effect-text and action/event effect-text projections retained their random
constructor weights. Three pre-fix tests (card only, effect only, both) fail exact
logit parity. The both-text fixture changes 44 valid logits (maximum absolute
change 0.0254). This is an initialization confound, not evidence that it explains
all earlier negative semantic results.

The fix zeros only parameters missing from the source checkpoint after validating
the added modules. Existing learned text parameters are preserved; random text
initialization for training a new network is unchanged. Tests verify exact output
parity, nonzero gradients into added views, and preservation of learned views.
Facts-only transfer and selection-history transfer remain covered.

## Registered Colab diagnostic

Evidence: `out/research/card-generalization-pilot-2026-10-10/`.
`dataset-manifest.json`, `registration.json`, source hashes and downloaded full
optimizer/RNG checkpoints bind the run. No current PPO checkpoint is used as
initial weights: all arms start from the same seed-specific random 128x2 policy.

- Three seeds, three arms: ID only, ID plus E5 card/effect text, and an identical
  text architecture with a fixed card-to-text permutation. All shared initial
  weights and initial policy logits match. ID dropout and facts stay off.
- Reserve 58 cards exclusive to the three selected deck lists (Lunalight,
  Sky Striker, Tearlaments) relative to the other 17 lists. This is an explicit
  **card subset** holdout, not proof that every related archetype card is absent.
- Exclude whole source hands/games if any reserved identity appears in card,
  action-card, effect-card, or either event-card column, including masked rows.
  The training set contains 13,339 decisions and zero reserved-ID occurrences.
- Test on 461 decisions from independently dealt held-out-deck hands; retain
  2,248 seen-deck decisions for retention. Underlying hand multisets, not just
  independently numbered hand indices, are disjoint between source train/test.
- Primary: NLL on decisions with a legal candidate naming/referencing a reserved
  card at fixed epoch 4, grouped by original source hand. Secondary: all held-out
  decisions, seen-deck retention, action agreement. Do not tune on this test.
- Few-shot phase: two epochs on 380 decisions from 12 other training hands
  (four per held-out deck), then reevaluate the unchanged test. Report retention
  and adaptation separately. Fixed text embeddings may already encode these
  cards' rules: "unseen" here means unseen in policy interaction training, not
  unseen to the language embedding model.
- One L4, two-hour/six-compute-unit ceiling, at most two allocations. Download and
  hash-verify the model, optimizer and CPU/GPU RNG after each epoch. Keep the
  zero-shot checkpoint before adaptation. A lease and local expiry timer bound
  orphaned workers. Reallocate only after server-confirmed endpoint loss;
  training failures stop and retain evidence. Independent study watchdog reports
  service failures and completion.

This is **policy imitation**, using existing corrected solver/battle labels. The
labels are useful but not guaranteed uniquely optimal. Small held-out hand and
family counts limit inference; action agreement/NLL cannot establish game strength.
Report paired seed/hand uncertainty and each family, not independent-row error
bars. True text must outperform the shuffled-text control before attributing a
benefit to semantics rather than added capacity. No automatic policy promotion.

## Follow-up gates

First audit the completed diagnostic, including actual use of reserved-card
candidates and per-family errors. Then evaluate the current PPO policy piloting
and facing unfamiliar decks separately. Preserve a fixed familiar panel and
log wins, errors, turn/decision tails and repeated selections. A new full-game
panel must be registered independently of this label test.

A matched-budget RL study can then compare fixed versus expanded deck sampling,
with and without semantic inputs. Begin with legal, functional corpus lists and
small replacements; compare dynamic mutation to a fixed expanded pool only after
coverage gains are established. Keep the current cancellation-budget canary
unchanged, and do not combine its data with a changed deck distribution.

## Setup transport incident and bounded restart

The first Colab allocation compiled successfully and identified NVIDIA L4, but
its first read-only setup-status RPC timed out after 180 seconds. No training
started. The watchdog delivered incident
`94877b403be4b8fd9fa8c42bb27f80532160519c27bfacf2bd03aaf7dd4eaa2c`,
which was acknowledged; original STOP/logs remain intact. The owned endpoint was
terminated and the direct server API confirmed no remaining assignments.

The controller now retries only idempotent status/lease queries, at most three
40-second attempts, checking endpoint existence after failure. Allocation and
worker-launch mutations are not blindly retried. Tests cover transient timeout,
confirmed preemption and persistent bounded failure. The restart at
`out/research/card-generalization-pilot-restart-2026-10-10/` retains the original
absolute deadline, initial compute balance and one already-consumed allocation.
Data, model architecture, training order and statistical protocol are unchanged.
The primary novel-action subset contains 282 decisions from 18 source hands;
no alias-equivalent reserved cards occur in training. This small diagnostic
must not be interpreted as a comprehensive unfamiliar-deck strength test.

## Second transport incident and local backend amendment

The first recovery retained three verified epochs of seed-0 ID training, then
its epoch-4 **launch** RPC timed out. This is a different path from status reads:
remote execution may have succeeded even though the acknowledgement is missing.
The previous status-only fix was incomplete. The actual execution outcome of
that epoch is unknown; no unverified epoch is counted. Endpoint cleanup proved
the owned VM was absent. Both historical STOP files and the last full checkpoint
remain intact; the original two Colab allocations are exhausted.

`card_transfer_colab.py` now launches each epoch under a durable, identity-bound
claim/receipt and separate exit/log paths. After a lost acknowledgement, query
the receipt without repeating the launch RPC. A missing, mismatched, or claimed
but unconfirmed launch fails closed. The claim is written before spawning;
repeating the same accepted request returns the prior PID, while an ambiguous
claim never spawns another worker. Tests exercise real subprocess execution,
lost acknowledgements, duplicate requests, ambiguous claims and wrong receipts.
This recovery path has fault-injection coverage; it has not yet been exercised
on another live Colab VM, and no additional allocation is authorized by this
experiment's registration.

The separate `card-generalization-local-2026-10-10` study reruns **all nine jobs
from scratch** on the same local AMD Radeon 8060S / ROCm runtime. It retains the
same data, seeds, epoch counts, arm order, initialization, adaptation and analysis
plan, but registers the backend change explicitly. Do not combine the partial
L4 metrics with the local comparison. No new Colab allocation is made; the local
controller inherits the original absolute deadline and uses an independent
watchdog, exclusive controller lock, full epoch checkpoints and digest validation.
This remains a small offline imitation diagnostic, not a gameplay result.

Local launch was verified on the real AMD GPU: seed-0 ID epochs 1 and 2 completed
and checkpoint/metric hashes passed validation. Runtime source is frozen at
`74c0e1a`; the separate documentation worktree does not modify that running tree.
The launch-recovery and checkpoint-integrity tests plus checkpoint/generalization
regressions pass (32 tests); repository format/lint checks pass (363 files).

## Completed three-seed local diagnostic

All nine runs completed six epochs on the same AMD/ROCm runtime; all final
checkpoint hashes and the predeclared analysis hash were verified. Original
Colab partial metrics were not included. Epoch 4 is the fixed zero-shot boundary;
epoch 6 follows two epochs on 12 additional training hands.

Hand-macro results on novel-action decisions (NLL lower is better):

| Arm | Epoch 4 NLL | Epoch 4 teacher agreement | Epoch 6 NLL | Epoch 6 teacher agreement |
|---|---:|---:|---:|---:|
| ID | 1.38519 | 38.91% | 1.22265 | 43.67% |
| Real text | 1.36540 | 40.02% | 1.22354 | 42.91% |
| Shuffled text | 1.42579 | 40.89% | 1.23056 | 43.80% |

The fixed primary comparison, real text minus ID, is **−0.01979 NLL**, crossed
seed/hand bootstrap 95% CI **[−0.05674, +0.01658]**. Real text minus shuffled
text is **−0.06038 [−0.11046, −0.01042]**, with the same direction in all three
seeds and all three selected deck families. This is evidence that the correct
card/text mapping helps relative to the shuffled control on this task; it is
not clear evidence of an incremental benefit over ID. Shuffled text itself has
worse NLL than ID, and action agreement ranks the arms differently. Do not
reinterpret that control comparison as demonstrated gameplay improvement.

After adaptation, text minus ID is **+0.00089 [−0.02713, +0.02666]**: no supported
text advantage in few-shot adaptation. Familiar-hand NLL differences are +0.00212
before adaptation and +0.01112 afterward (both intervals cross zero). All arms'
familiar-hand mean NLL worsens after the held-out-only adaptation; this is a
retention concern, not a measured loss of win rate.

The primary subset has 282 decisions in 18 hands. Of these, 236 teacher actions
actually refer to reserved cards; membership in the primary subset was defined
by the offered legal candidates without looking at the teacher label. The
three families contribute 51/44/187 rows (Lunalight/Sky Striker/Tearlaments).
Intervals are conditional on these three lists and a small hand sample. No
semantic switch or model promotion is made on this evidence.

Evidence: `out/research/card-generalization-local-2026-10-10/analysis.json`,
frozen `analyze.py`, reports and epoch checkpoints. Follow-up is the independent
full-game coverage panel below, not tuning this held-out label test.

## Registered full-game coverage panel

`out/research/card-coverage-fullgame-2026-10-10/` fixes 1,024 games before
inspection: three current warm-u512 training seeds plus a Greedy control,
against Greedy and the historical 256x2 policy. Four familiar targets
(Blue-Eyes, Branded, Sky Striker, Tearlaments) and four RL-pool-held-out targets
(Labrynth, Kashtira, Tenpai Dragon, Runick) each face Maliss and Ryzeal Mitsurugi
anchors, with two independently seeded deals, both deck assignments and both
first-player assignments. The same deals and agent seeds are reused across
policy/control cells; familiar/held-out target pairs also share anchor shuffles.

Held-out lists are the existing corpus medoids, with ranked/qualifier provenance
in that stored corpus. All lists are validated against the frozen environment;
no online claim about current metagame strength is made. Their unique card
counts are 33/26/38/35, of which 23/13/22/29 are absent from the audited corrected
BC inputs. Every card has a known vocabulary index. Full historical RL exposure
is not certified, so call this **RL-pool-held-out**, not strictly never seen.

Primary reporting separates the policy piloting a target from facing it, includes
paired improvement over Greedy on identical deals, and retains per-deck,
per-opponent and per-training-seed results. This control reduces—but cannot
remove—confounding from intrinsic deck strength and Greedy's own competence.
Secondary reporting includes turn/decision tails, game limits, repeated selections,
self-negation/self-damage warning signals and engine health. Warnings are not
automatic tactical-error labels. Limits stay in the denominator and are reported
separately. No training or policy promotion is part of this panel.

Two CPU workers, two-hour maximum, independent study watchdog, frozen source,
checkpoints, deck/data/runtime hashes. First run the 16 preselected unfamiliar-deck
seed0-vs-Greedy health games; any engine or cold-replay mismatch preserves the
failed case and stops. Every game stores the full trace and a verified cold replay.
