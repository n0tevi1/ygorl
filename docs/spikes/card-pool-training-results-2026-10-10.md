# Card-pool expansion canary: no established gain after 16 updates

Latest: independent seeds1/2 did not reproduce the added-own deficit. Expansion
mostly preserves added-facing performance that fixed continuation loses; it has
not established improvement over the starting actor. See replication below.

From the same seed0 warm-u512 checkpoint, both arms trained 16 PPO updates
(262,144 rows each). Fixed retained the 20-deck pool; expanded added legal
Labrynth, Kashtira, Tenpai Dragon and Runick lists. Full state restoration was
checked before training: actor, critic, optimizer, reference, snapshot pool and
RNG were identical, with only the registered deck list changed. Text features,
reward, inference and cancellation budget stayed fixed. This tests adaptation
to those lists, **not** zero-shot generalization to unseen cards or families.

All 768 fresh paired evaluation games were healthy, with no limits and exact
cold replay matches. Frozen inputs, eight training checkpoints, all saved result
records and replay hashes were independently verified. Neither arm is promoted.

## Results

Every cell contains 64 games: four target decks, two anchors, two deals, both
seats and two fixed opponents. These are weak-opponent diagnostics. The three
candidate policies use identical evaluation deals and agent RNG seeds.

| Candidate role / target group | Start u512 | Fixed u528 | Expanded u528 |
|---|---:|---:|---:|
| Pilot familiar decks | 61/64 | 63/64 | 63/64 |
| Face familiar decks | 41/64 | 47/64 | 43/64 |
| Pilot added decks | 57/64 | 63/64 | 56/64 |
| Face added decks | 50/64 | 41/64 | 47/64 |
| Total | 209/256 | 214/256 | 209/256 |

Expanded minus fixed when piloting added decks is -10.94 percentage points;
registered paired cluster bootstrap interval [-20.31, -3.13] points. When facing
added decks it is +9.38 points, interval [-6.25, +25.00]. Familiar own-deck
performance is tied; facing familiar decks is -6.25 points, interval
[-21.88, +10.94]. These are conditional single-seed, multiple-endpoint pilot
intervals. They do not establish a general harm from diversity. Expanded is only
one win below the starting actor on added own decks; much of the relative gap
comes from the fixed arm improving. Do not call this catastrophic forgetting.

Mean turns (start / fixed / expanded): familiar-own 8.02 / 8.36 / 8.63,
familiar-facing 11.30 / 10.36 / 10.00, added-own 12.94 / 12.94 / 12.98,
added-facing 12.73 / 15.00 / 14.25. There is no broad shortening. Facing Runick,
wins are 12/16 / 7/16 / 8/16 and mean turns 17.13 / 26.13 / 26.31.
Short games alone remain an inappropriate objective for a battle-lock/deck-out
matchup; report wins, termination reason and legal action availability together.

## Root-cause checks and remaining uncertainty

New decks were actually dealt: 581/1,839 completed expanded-arm training games
involved an added deck (31.6%), versus zero of 1,869 fixed-arm games. The expected
share under uniform cross-pairs is 172/552 = 31.2%. Completed games with the learner
piloting Kashtira / Labrynth / Runick / Tenpai were 164 / 110 / 126 / 124. These
counts include correlated paired seats and are not independent sample sizes.
Engine errors and truncations were zero in both arms. Missing deck ingestion,
an invalid list, failed restoration or corrupted evaluation is not supported by
the evidence. This is still a short exposure per new deck.

The eight added-own cases that fixed won and expanded lost first diverge at a
chain/pass decision under matching recorded public inputs; there is also one
opposite outcome. The changed choice and later loss are not enough to label that
first choice tactically wrong: subsequent policy and sampled trajectories also
change. First-divergence records are retained for counterfactual investigation.
A larger per-update KL is not a sufficient explanation either: mean KL to the
moving reference was 0.0356 expanded versus 0.0493 fixed; these values are not
KL to the starting actor. The causal explanation for the role-specific gap
remains unresolved. Check replication before redesigning rewards or masking cards.

Repeated selection cancels occur in 2 / 1 / 2 evaluation games; four games overall
cross the eight-repeat warning threshold. Self-Ash selections/opportunities are
2/102, 0/78 and 1/96. Expanded's one self-Ash repeats a starting-actor case involving
its own Mulcharmy Fuwalos (probabilities 8.88% versus 8.28%). These tiny correlated
counts do not establish that either issue is fixed or that expansion caused it.
Neither loops nor self-Ash occurs in the eight added-own discordant losses, so
those warning signals do not explain that particular performance gap.

## Independent replication

Register training seeds 1 and 2 before reading their outcomes. Repeat the same
16-update fixed/expanded intervention and evaluate start/fixed/expanded on a
fresh common panel (768 games per seed, 1,536 total), disjoint from discovery
opening seeds. Report each seed, own/facing roles, familiar retention and Runick
separately; seed0 remains discovery evidence and is not pooled into confirmation.
With only two replication seeds, even a pooled interval remains exploratory.

Use full restoration checks, checkpoints every four updates, pre-optimizer
engine-health gates, exact cold replays and independent progress monitoring.
Keep failed artifacts and STOP markers. No text switch, mutation, blanket masks
or automatic policy promotion. A positive result would still require unseen-list
and unseen-family tests before claiming broader card intelligence.

Evidence: `out/research/card-pool-training-canary-2026-10-10/{analysis.json,
training-exposure.json,first-divergences.json}`. Replication registration and
frozen runtime bindings: `out/research/card-pool-training-replication-2026-10-10/`.
Related: #36, #83, #105, #218; prior coverage/lock report in PR #284.

## Independent seeds 1/2: own-deck deficit does not replicate

Both seeds completed both 16-update arms. All 1,536 evaluation games were healthy,
with no limits and exact cold replay matches. All 16 checkpoints, frozen inputs,
parent/child report hashes and saved replays verified; all three registered
analysis outputs reproduced byte-for-byte. Original seed0 discovery is excluded
from the following pooled results. Each cell has 128 games across two seeds.

| Candidate role / target group | Start u512 | Fixed u528 | Expanded u528 |
|---|---:|---:|---:|
| Pilot familiar decks | 123/128 | 125/128 | 123/128 |
| Face familiar decks | 78/128 | 74/128 | 80/128 |
| Pilot added decks | 112/128 | 113/128 | 114/128 |
| Face added decks | 89/128 | 75/128 | 88/128 |
| Total | 402/512 | 387/512 | 405/512 |

Expanded-minus-fixed added-own performance is +0.78 points, conditional interval
[-7.81, +7.81], with seed differences -1.56 / +3.13 points. The original -10.94
point own-deck deficit is not a reliable basis for rejecting expansion.

Added-facing expanded-minus-fixed is +10.16 points, interval [0.00, +21.88],
with both seeds positive (+12.50 / +7.81). But expanded-minus-start is -0.78
points, interval [-10.94, +9.38]. This distinction matters: the evidence supports
investigating **retention versus fixed-pool deterioration**, not claiming new
strength. Two-seed crossed bootstrap intervals are fragile and conditional on
these lists, opponents and openings. Familiar-facing effects vary by seed.

There is no general turn-length improvement. Added-facing means are 13.55 /
15.40 / 15.14 turns (start / fixed / expanded). Facing Runick, wins are 17/32 /
9/32 / 14/32, and mean turns 24.72 / 30.41 / 29.81. Expanded's better result than
fixed remains below the starting policy in this selected matchup.

## Repeated-selection signals remain

Games with more than one consecutive selection cancel: 1 / 7 / 10 across the
512 evaluations per candidate. Only 0 / 3 / 3 games cross the eight-identical-
choice warning threshold; small cancel counts can be legitimate target revision.
Trace audit locates those warnings in Branded material selection and Sky Striker
selection dialogs (`SelectCard` / `SelectUnselectCard`), with several reaching the
existing 32-cancel guard. This repeats the known selection-control failure class;
it does not establish a new engine bug or prove all ten expanded games are errors.
These games are piloting familiar decks, so they do not account for the separate
added-facing retention gap. Keep the cancellation rule fixed for causal comparison.
Self-Ash remains sparse: 1/241 / 1/230 / 1/236 offered windows; correlated
opportunities and tiny counts cannot certify this behavior has been repaired.

## Next: larger fresh-opening retention confirmation

Freeze all nine existing start/fixed/expanded checkpoints from training seeds
0/1/2. Run 2,304 games facing the four familiar and four added lists, two anchors,
two opponents, both seats and four new deals per target/anchor. New engine seeds
are disjoint from discovery and replication. No new training, mutation, feature
switch, mask change or policy promotion. Primary comparisons are fixed-minus-start
and expanded-minus-start facing added decks; expanded-minus-fixed is secondary.
Report per seed, opponent, deck, turn tails, deck-outs and familiar retention.

This increases opening coverage for a specific apparent forgetting pattern. It
is not a new independent training-seed replication or unseen-family test. Inputs,
all checkpoint hashes and the analysis are frozen; independent health/progress
monitoring and exact cold replays remain required. Evidence is retained under
`card-pool-training-replication-2026-10-10` (including `repeat-audit.json`) and
`card-pool-retention-confirmation-2026-10-10` in the research output directory.
