# Card-pool expansion canary: no established gain after 16 updates

Latest correction: the purported fresh-opening follow-up evaluations reused
pre-shuffled deck orders. The 2,304-game retention panel contained only **16**
distinct ordered deck pairs, repeated across nominal draws, checkpoints,
opponents and seats. Its game outcomes and exact replays remain valid, but its
fresh-opening interpretation and opening-cluster confidence intervals are
withdrawn. The critic calibration replay inherits this limitation. See
[the opening-deal audit](card-pool-opening-audit-2026-10-10.md) before interpreting
any results below. No strength gain, general forgetting mechanism, or broad
Runick calibration defect is established. Historical numbers are retained for
traceability; conditional root-action experiments remain conditional evidence.

The corrected 2,304-game panel has now completed: overall wins are
501/768, 512/768 and 507/768 (start/fixed/expanded), with no primary paired
interval excluding zero. Runick V MSE is now0.888/1.019/0.845, so the prior
all-arm overestimation conclusion does not generalize. Full corrected results
and the independent critic adaptation corpus are in the linked opening audit.

From the same seed0 warm-u512 checkpoint, both arms trained 16 PPO updates
(262,144 rows each). Fixed retained the 20-deck pool; expanded added legal
Labrynth, Kashtira, Tenpai Dragon and Runick lists. Full state restoration was
checked before training: actor, critic, optimizer, reference, snapshot pool and
RNG were identical, with only the registered deck list changed. Text features,
reward, inference and cancellation budget stayed fixed. This tests adaptation
to those lists, **not** zero-shot generalization to unseen cards or families.

All 768 paired evaluation games (reusing the earlier coverage deals) were healthy, with no limits and exact
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

## Replication registration (opening-independence claim corrected)

Register training seeds 1 and 2 before reading their outcomes. Repeat the same
16-update fixed/expanded intervention and evaluate start/fixed/expanded on a
common panel (768 games per seed, 1,536 total). Correction: core/agent seeds
were disjoint from discovery, but deck orders were reused. Report each seed, own/facing roles, familiar retention and Runick
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

## Historical registration: intended fresh-opening retention confirmation (not achieved)

Freeze all nine existing start/fixed/expanded checkpoints from training seeds
0/1/2. Run 2,304 games facing the four familiar and four added lists, two anchors,
two opponents, both seats and four intended new deals per target/anchor.
Correction: the preparation copied draw0 deck orders into all four labels. Engine seeds
are disjoint from discovery and replication. No new training, mutation, feature
switch, mask change or policy promotion. Primary comparisons are fixed-minus-start
and expanded-minus-start facing added decks; expanded-minus-fixed is secondary.
Report per seed, opponent, deck, turn tails, deck-outs and familiar retention.

The intended increase in opening coverage was not achieved. This was also
not a new independent training-seed replication or unseen-family test. Inputs,
all checkpoint hashes and the analysis are frozen; independent health/progress
monitoring and exact cold replays remain required. Evidence is retained under
`card-pool-training-replication-2026-10-10` (including `repeat-audit.json`) and
`card-pool-retention-confirmation-2026-10-10` in the research output directory.

## Historical 2,304-game panel: fixed-opening outcomes; opening intervals withdrawn

All registered games completed healthy with no limits and exact cold replays.
Nine checkpoint hashes, all saved results/replays and frozen inputs verified;
registered analysis reproduced byte-for-byte. These games are **facing** target
decks with Maliss/Ryzeal Mitsurugi, not piloting the targets. Each cell pools
384 games across three existing training seeds. Correction: nominal draws
changed continuation RNG but reused the same opening orders; they add neither
independent openings nor independent training runs.

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

## Independent root confirmation: benefits in both directions

All 924 branches completed healthy, without limits, and with exact cold replays.
The 28 original-stream canaries reproduce their sources. All 248 frozen file
hashes and saved result/replay records verified; the registered analysis completed.
The other 896 branches cover 16 new paired streams per action at all 28 roots.

| Selected source outcomes | Roots | Original action wins | Other checkpoint's action wins | Paired mean turn change |
|---|---:|---:|---:|---:|
| Fixed loses, start wins | 20 | 134/320 | 156/320 | -0.41 |
| Start loses, fixed wins | 8 | 55/128 | 67/128 | -1.30 |

The first group's conditional gain shrank from +13.33 to +6.88 points. Its
deck-out losses fell from 172 to 149, but mean duration barely moved; per-training-
seed win deltas were +1.25, +12.50 and +12.50 points. The reverse group now also
improves (+9.38 points), unlike the pilot's zero difference. Selecting actions from
the checkpoint that won a particular opening can help either checkpoint. This is
evidence for state-specific decision differences, not a general claim that fixed
continuation forgot how to play or that the start checkpoint is a superior teacher.
The original population retention comparisons remain uncertain.

Per-root effects are heterogeneous. These are wins out of 16 new streams, holding
the losing checkpoint's continuation policy fixed:

| Source game | Forced root choice | Original → alternative wins |
|---|---|---:|
| 483 | Main Phase 2 → Mystical Space Typhoon | 2 → 15 |
| 1264 | activate Wousu's optional recovery → decline | 0 → 7 |
| 1274 | Ash opponent's Runick Tip → pass | 5 → 12 |
| 2038 | one Ritual-material choice → another | 7 → 13 |
| 494 | pass → Ash opponent's Runick Golden Droplet | 11 → 6 |
| 2040 | one Ritual-material choice → another | 6 → 2 |

Exact-prefix context replay finds Grave of the Super Ancient Organism face-up
against a Special Summoned Lava Golem in game 483; the root has no legal attack.
It also finds two Dimensional Fissures, Gozen Match and Runick Fountain on the
opponent's field. Activating removal is a credible way to change the lock, but
this experiment leaves target selection and subsequent play to the policy; it
does not prove an immediate lethal or that every removal target is good.
The same root's MST probability falls from 15.44% in start to 0.285% in fixed.
Both checkpoints still prefer Main Phase 2; the source winner's sampled action
is not the winner policy's most likely action.

Game 1264 is Wousu's optional graveyard recovery, whose effect requires a discard;
it is not evidence that every optional effect should be activated. Game 1274's
Ash targets the opponent's Runick Tip, not the agent's own effect. Together with
the negative swaps, these examples rule out turning the current results into a
generic card blacklist or copying every action from a winning trajectory.
Context and probability audit artifacts are retained with the completed study.

## Next: separate root probability changes from subsequent play

A registered 498-branch factorial audit includes **all nine** roots whose absolute
paired difference was at least four wins out of 16 in the confirmation, including
three adverse swaps and one root from the reverse checkpoint direction. At each
identical shared prefix, evaluate every supported legal action under both start
and fixed continuation policies, using eight new paired streams (trial IDs 20–27).
The 18 original-policy controls must reproduce their full source games. Prefix
actions, options, masks, board, LP, player and turn agree across the two sources.

The analysis will cross the two root probability distributions with each fixed
continuation policy's estimated action values. This separates a probability shift
at the root from differences in the ability to play after that action. It remains
a conditional diagnosis on selected states, with shared correlated streams and
small per-action samples; no population-strength conclusion, automatic label,
mask, training export or promotion follows. Two CPU workers, a bounded runtime,
health/progress monitoring and exact replays remain required. The runtime remains
frozen at the source panel's pre-Link-fix version to preserve the comparison.
Evidence: `card-pool-root-confirmation-2026-10-10` and
`card-pool-action-factorial-2026-10-10` in the research output directory.

## All-action factorial: several proposed action labels fail to stabilize

All 498 branches completed healthy and without limits, with exact cold replays
and all 18 source controls reproduced. Runtime and frozen input hashes verified;
registered analysis reproduced byte-for-byte. Each action/continuation-policy cell
has only eight fresh streams. These selected states are not a strength benchmark.

At game 483's MST root, fixed continuation wins 5/8 after MST versus 4/8 after
Main Phase 2, substantially weaker than the previous 15/16 versus 2/16. Start
continuation wins 3/8 under either action. At Wousu's optional recovery root,
both continuation policies win 2/8 after activating versus 1/8 after declining;
the earlier advantage for declining does not replicate. Do not export either
state's previous binary outcome comparison as a certified training label.

Other examples retain conditional differences: at root 1274, passing instead of
Ash on Runick Tip wins 3/8 versus 1/8 under fixed continuation and 6/8 versus 2/8
under start. At root 2038, the Wousu material choice wins 8/8 under fixed and 7/8
under start, versus 4/8 and 5/8 for the originally selected Kusanagi. Small cells
and the exact opponent/hidden state still limit these conclusions.

Root 2040 exposes continuation dependence. With the same forced Kusanagi
selection, fixed wins 1/8 and start 6/8. Crossing root probabilities with each
continuation policy's action estimates gives:

| Root 2040 continuation | Start root probabilities | Fixed root probabilities |
|---|---:|---:|
| Start | 43.46% | 52.80% |
| Fixed | 25.92% | 22.80% |

These are conditional, Monte Carlo weighted estimates, not full-game evaluation
win rates. They show why a root probability change alone cannot explain this
case. A saved-replay audit of all 240 policy pairs finds 236 later divergences
and four identical games. In the Kusanagi example the first differences include
search/recovery choices, triggering Saji, attack order, normal summoning Ash,
and placement. There is no single established material-selection implementation
bug behind the continuation gap.

## Critic evolution: measure calibration separately from actor strength

A read-only inspection of both critics at the nine roots reproduces the actor
probabilities and uses privileged input only for the critic. At MST root 483,
fixed Q values for the three actions are approximately 0.802 / 0.800 / 0.810
with V=0.873, despite this small rollout panel yielding only 2–5 wins per eight
continuations. At other roots Q rankings differ from the observed action ranks.
These are diagnostic signals: fixed evaluation opponents differ from the critic's
training league, and eight samples cannot establish true action values.

The next registered audit replays **all 2,304 original population-panel games**,
with no selection by outcome. It samples the first learner main-phase prompt
per turn and compares V, chosen Q and policy-weighted Q with terminal return.
Errors are averaged within each game before aggregation, with separate arm,
training-seed, deck, opponent, turn, LP and public-card-presence breakdowns.
Original response bytes, results, action masks and actor probabilities must
match; no policy sampling, optimizer step or hidden information enters the actor.
Presence of a reviewed continuous card is not itself a tactical-error label.

The initial audit stopped after 73 completed games on an actor-parity assertion.
Game 68 reproduced a 1.6987e-6 difference: standalone PolicyNet pads action
features before its head, whereas ActorCritic scores the trimmed action width.
The repaired audit compares archived probabilities through the canonical actor
inference path and logs the combined-head difference separately. The same failed
game then reproduced all response bytes and its result. The original STOP is
retained; a new study adds it to the health canaries and restarts all games.

Critic improvements require two stages. First hold the actor fixed and compare
critic calibration and action discrimination on separate data. Depending on the
audit, test additional critic training on fresh complete trajectories, better
coverage of difficult states/actions, or a separate critic backbone. Then compare
otherwise identical RL continuations to establish whether the critic change
improves actual play without degrading familiar decks or increasing loops/turn
tails. Better critic loss alone does not justify policy promotion. The current
evaluation games remain evaluation data; do not silently turn them into training
examples or use privileged critic inputs at actor inference.

Evidence: `card-pool-action-factorial-2026-10-10` (including suffix-divergence and
critic audits), `card-pool-critic-calibration-2026-10-10` (preserved failure), and
`card-pool-critic-calibration-restart-2026-10-10` in the research output directory.
