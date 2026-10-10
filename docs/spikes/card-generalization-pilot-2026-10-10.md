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
