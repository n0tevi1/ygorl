# Card-pool expansion canary: no established gain after 16 updates

Latest: the larger fresh-opening confirmation has no paired interval excluding
zero. Neither a strength gain nor reliable retention benefit is established. A
separate, concrete required-material deselection no-op was found and repaired;
that repair is not evidence of a training-policy gain. See confirmation below.

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

## Fresh 2,304-game confirmation: effects smaller and uncertain

All registered games completed healthy with no limits and exact cold replays.
Nine checkpoint hashes, all saved results/replays and frozen inputs verified;
registered analysis reproduced byte-for-byte. These games are **facing** target
decks with Maliss/Ryzeal Mitsurugi, not piloting the targets. Each cell pools
384 games across three existing training seeds; this adds openings, not new
independent training runs.

| Target group | Start u512 | Fixed u528 | Expanded u528 |
|---|---:|---:|---:|
| Familiar facing | 272/384 | 267/384 | 259/384 |
| Added facing | 282/384 | 266/384 | 275/384 |

Added-facing fixed-minus-start is -4.17 points, conditional interval
[-11.72, +2.86]; expanded-minus-start -1.82 points [-10.16, +6.25];
expanded-minus-fixed +2.34 points [-5.99, +10.42]. All intervals include zero.
The earlier apparent retention advantage is not confirmed at a reliable effect
size. Fixed's point estimate is negative in all three seeds, but expansion's
advantage over fixed changes sign in seed1. No policy promotion or additional
training scale-up is justified by these results alone.

Runick accounts for 12 of the fixed arm's 16 net lost wins against added decks:
start57/96, fixed45/96, expanded50/96. Mean turns are29.58 /32.43 /32.55;
added-group deck-out losses38 /48 /44. Added-facing mean turns14.63 /15.39 /
15.49 show no shortening. Other lists and seeds differ substantially; keep those
breakdowns rather than attributing all differences to card novelty.

Self-Ash selections/opportunities across the768 games per candidate are5/527,
1/482 and0/444. These are sparse, trajectory-dependent opportunities; zero here
does not demonstrate that the behavior is eliminated. Battle exits with an attack
offered are403 /356 /415, not automatically missed lethal. There are zero
repeated-cancel games, but0 /2 /1 no-event-repeat warning games: monitoring only
cancel counts would miss these selection/unselection repetitions.

## Concrete fix: required Link material offered as removable

In expanded game2069, I:P Masquerena activates its effect and chooses Dyna Mondo.
The Link procedure starts with I:P in the required-material group. Upstream
`Link.Target` nevertheless passes the complete selected group to
`Group.SelectUnselect`, advertising I:P as removable. Its next conditional
explicitly ignores removing required cards. The agent chooses the ignored action
ten times; every prompt is identical and there are no engine retries or cancels.
This is a misleading actionable option, not evidence that the engine successfully
removed the material or that a generic ban on cancel would repair the loop.

A content-pinned in-memory override changes only the unselectable group from
`sg` to `sg-mustg`. The actual selected group and all required-material constraints
remain intact. Selectable cards, finish and cancel flags are unchanged. The
procedure reaches this prompt only when the selectable group is nonempty, so
removing required cards from the unselectable list cannot remove its only exit.
Unknown upstream versions remain untouched; no shared CardScripts files change.

Regression replays verify identical613-decision prefix, reproduce all ten no-ops
in the old script, and show the repaired game finishing with the same winner,
turn count, LP and suffix responses, ten fewer decisions. The native training
host sees the same corrected options and completes successfully. A separate real
material-selection prefix verifies optional-material deselection still changes
the group. These checks establish the narrow interface repair, not improved
win rate. Other reversible selection cycles remain unresolved.

## Next mechanism study

All28 Runick start/fixed discordant cases are included:20 fixed losses/start wins
and8 start losses/fixed wins. In each losing policy's exact prefix, force either
its original action or the other policy's action at the first divergence; retain
the losing policy thereafter. Four paired streams per action give224 branches.
All28 factual original-stream controls must exactly reproduce their source.
Report original-stream rescues separately from the three reseeded comparisons:
these roots were selected on outcomes, so control losses in the original stream
are guaranteed by selection and cannot estimate an unbiased policy gain.

The study uses frozen pre-repair scripts, preserving comparison with its source
panel; the Link fix is isolated in a separate worktree. No automatic action labels,
masking or training-example export. Evidence: `card-pool-retention-confirmation-
2026-10-10` and `card-pool-root-causality-2026-10-10` in the research output directory.

## Root intervention results and independent continuation confirmation

All 224 branches completed healthy without limits, with exact cold replays and
all 28 original controls reproducing their source. All 241 frozen file hashes
and saved result records verified. Swapping the first divergent action while
retaining the losing policy rescued 9/20 fixed-loss cases on their original RNG
streams, but 0/8 start-loss cases. Those outcome-selected controls necessarily
lose; these rescue counts are not an unbiased strength estimate.

On the three fresh continuation streams per root, fixed-loss cases won 24/60
with their original action versus 32/60 with the start policy's action. Six of
20 roots had a positive paired difference; one was negative. Start-loss cases
were 8/24 versus 8/24, with one positive and one negative root. The positive
aggregate signal is a conditional mechanism lead, not evidence of a general
action rule or reliable policy improvement.

For example, game 483's turn-10 choice favors entering Main Phase 2 (86.3%) over
activating Mystical Space Typhoon (0.285%). The alternative loses on the original
stream but wins all three new streams, compared with zero for the original
choice. Conversely, another root's original-stream rescue reverses on fresh
streams. This makes continuation replication necessary before labeling actions.

The next registered study retains **all 28 roots**, including negative/null cases,
and uses 16 new paired continuation streams (trial IDs 4–19, disjoint from the
pilot), plus 28 exact-source health canaries: 924 branches total. The same frozen
pre-repair runtime preserves the intervention comparison. Report both directions,
every root, training seed, shared RNG stream, turn lengths, deck-outs and limits;
do not treat the correlated branches as independent games. Two CPU workers,
bounded runtime, independent monitoring and cold replays are retained. No model
training, masking or promotion is performed. Evidence:
`card-pool-root-confirmation-2026-10-10` in the research output directory.

Link-interface validation: 83 focused patch/action/health tests and 111
replay/duel/native-host/pool integration tests passed; Ruff and whitespace checks
passed. The fix eliminates the ten ignored deselections in the recorded case,
with identical final outcome; no policy win-rate improvement is claimed.
