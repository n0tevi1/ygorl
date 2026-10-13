# Policy-supported lethal search — 2026-10-12

Related: #62, #83, #105. This follows the negative
[broad combat-guidance experiment](combat-only-guidance-2026-10-12.md).

## Scope and implementation

`LethalAgent(..., policy_mask=True)` optionally restricts its own search choices
to the standard encoded policy support, including duplicate canonicalization,
undo suppression, and the encoding's action limit. Filtering happens before the
Greedy/random chooser runs; selected indices are mapped back to original engine
indices. Saved lines retain complete original prompt signatures. Cached responses
also recheck support before they are returned. Empty support rejects the rollout.

The default is `False`. No registry default, actor checkpoint, reward, or training
configuration changes. Greedy's yes/no handler accepts a filtered subset without
a `yes` row; its behavior on the ordinary complete prompt is unchanged.

The shadow still has actual hidden engine state, and the planning opponent is
passive. The existing strict event filter is not a public-information guarantee.
This option fixes compatibility with policy support; it is not a safety proof for
the line, a robust adversarial search, or permission to export teacher labels.

## Root cause and diagnostic controls

The first audit fixes ten exposed roots: six earlier missed-finish windows in five
games plus four decision points associated with the previous harmful-guidance
cases. Each root has its exact original-policy continuation and one search-guided
continuation: 20 records, 12 rollouts per search, four Greedy then random, at most
200 decisions per rollout. Only one search occurs at the selected root.

The diagnostic consumes the original policy draw even when replacing an action
and requires the applied choice to be supported by the policy. It uses existing
cached-line signatures and abandons the line on a mismatch. This is deliberately
different from the production wrapper's fallback-only policy draws. **The stock
engine-legal wrapper does not enforce this policy-mask gate by default.** The
observed partial-line failure belongs to this constrained diagnostic integration,
not an engine-illegal action or a demonstrated default-wrapper crash.

Default search found four lines; two completed as same-turn wins. Two stopped at
the execution mask gate. Direct prefix replay with matching native/Python masks
identified the exact reasons:

- cold256/greedy68, decision 575: selecting card 60764609 from deck sequence 2 is
  masked as a duplicate; its representative is the same code at sequence 16.
- cold256/historical52, decision 491: `cancel` is an undo-suppressed row. The
  diagnostic already supplied an attack at decision 490; fallback after rejection
  loses on turn 12, whereas its original-policy control wins on turn 10.

All ten original controls reproduce their full choice suffixes and final results.
All 76 internal rollouts are health-checked before snapshot restoration; all 20
outer continuations are healthy, with no cutoffs and 51 passing mask comparisons.

## Search-time constraint result

A second 20-record audit uses the same roots, seeds, and budgets. Its ten legacy
controls exactly reproduce the first audit's planned lines, applied transcripts,
abandonment records, and outcomes. The treatment filters during planning instead
of accepting a line and first discovering unsupported actions while executing it.

| Exposed window | Legacy found / actual same-turn win | Mask-constrained found / actual same-turn win |
| --- | --- | --- |
| warm512 historical72 / 466 | No / No | Yes / Yes |
| cold256 greedy65 / 402 | Yes / Yes | Yes / Yes |
| cold256 greedy65 / 499 | No / No | Yes / Yes |
| cold256 greedy68 / 534 | Yes / No | Yes / Yes |
| cold256 historical52 / 490 | Yes / No | Yes / Yes |
| cold256 historical76 / 325 | Yes / Yes | Yes / Yes |

Neither variant finds a line at the four negative-control decision points. The
masked variant has no abandoned line or eventual loss in these exposed cases.
All 138 internal rollouts and 20 outer records are healthy; 196 applied native/
Python mask checks pass. Removing candidate rows also changes rollout sampling
and subsequent trajectories, so this does not prove a universally monotonic
improvement in search coverage. The six windows are not six independent games.

The committed implementation (`1767d09`) then replays both variants at all ten
roots. All 20 full transcripts, planned lines, abandonment records, and outcomes
match the research prototype exactly. This checks implementation parity, not
generalization.

## Fresh-deal wrapper pilot

Four new deal blocks, four seat/deck assignments per block, baseline and guided
arms: 32 games plus two unwrapped baseline verification games. The candidate is
seed-0 warm512 and the opponent is historical warm128. Panel seed `2026101215`
reshuffles deck slots; ordered first-five-card openings are checked against both
earlier coverage panels. The actual wrapper is used, including its fallback-only
inner-policy RNG convention. This is not the controlled suffix-driver convention.

The wrapper has a fixed limit of eight searches per game and two per turn, 12
rollouts per search, and 200 steps per rollout. No tuning occurs after results.

| Metric | Original policy | Optional search |
| --- | ---: | ---: |
| Wins /16 | 10 | 11 |
| Mean final turn | 9.5 | 8.1875 |
| Cutoffs / engine errors | 0 / 0 | 0 / 0 |
| Summed job wall time, seconds | 32.28 | 55.03 |

One paired loss becomes a win; no win becomes a loss. Two games account for the
entire 21-turn reduction: game 10 remains a win but ends on turn 4 instead of 14;
game 12 changes from a turn-14 loss to a turn-3 win. Other final turn counts match.
The guided arm makes 109 searches and 1,211 internal rollouts, finds 11 lines,
supplies 421 cached responses, and abandons four lines. There is no shadow desync;
all 7,644 applied native/Python mask checks pass. Wall time includes setup, shared
CPU contention, and two extra baseline verification games, so it is not a clean
inference-throughput benchmark.

Only four independent deal blocks are represented. This is a small positive
privileged-search signal, not a reliable win-rate estimate or a public-information
agent improvement. No actor or critic is promoted and no larger RL run is launched.

### Abandoned-line audit

All four games with abandonment are replayed with instrumentation only. Their
complete transcripts, results, and statistics reproduce exactly. Every mismatch
has the same prompt signature but a different actual opponent response:

| Game / decision | Planned opponent response | Actual opponent response |
| --- | --- | --- |
| 3 / 80 | pass | chain card 85888377 |
| 6 / 298 | no | yes, card 8264361 |
| 7 / 344 | pass | chain card 20938824 |
| 10 / 167 | pass | chain card 55623480 |

Thus these four abandonments are active-response deviations, not mask failures
or shadow desynchronization. The wrapper correctly drops the cached line; all
four games eventually win, but that does not prove their already-applied prefixes
are harmless against other responses. It confirms that passive planning remains
a material unresolved limitation after mask compatibility is fixed.

The first instrumentation preflight stopped because live integer `by_turn` keys
were compared to JSON-loaded string keys. Its complete action trace already
matched. The restart canonicalizes the statistics before comparison; no strategy,
seed, or outcome criteria change. The original STOP is retained under
`lethal-line-abandonment-audit-2026-10-12/`; the completed restart is
`lethal-line-abandonment-restart-2026-10-12/`, identity
`1b16af155655e3a89901134ca7653df0f35f2a2aa5ee22d8ae223112ee07e243`.

## Independent same-budget replication

Panel seed `2026101216` supplies four new deal blocks (32 games plus two unwrapped
checks), with the same frozen checkpoint pair, wrapper settings, and search
budgets. Eight new ordered opening hands have no overlap with the 88 distinct
openings checked in the earlier panels. Instrumentation records abandonment
reasons without changing choices.

| Metric | Original policy | Optional search |
| --- | ---: | ---: |
| Wins /16 | 10 | 10 |
| Mean final turn | 9.25 | 9.625 |
| Cutoffs / engine errors | 0 / 0 | 0 / 0 |

There are **zero paired gains and zero paired losses**. Games 6 and 12 each take
four additional turns (8→12 and 7→11); game 13 takes two fewer (8→6). All other
final turns match. The earlier positive signal does not replicate. Pooled across
eight deal blocks, wins are 20/32 versus 21/32 and mean turns 9.375 versus 8.90625;
this pooled descriptive result must not hide the negative replication.

There are 111 searches, 1,273 internal rollouts, seven found lines, 392 cached
responses, two abandonments, and zero desynchronizations. All 7,388 applied mask
checks pass. Both abandonments again have matching prompts but active opponent
responses: game 6 decision 527 chooses `yes` for card 93854893 instead of `no`;
game 12 decision 262 chains card 58053438 instead of passing. Those are the two
longer games, but this association alone does not isolate response effects from
the preceding search actions or changed fallback-policy RNG consumption.

The replication identity is
`62596445243b955ae5a3e5fe99a34ae2cd49fa67966ab0782d3b8b4d3582915c`;
report SHA-256 is
`653c278d1596561970cfa2e86110356d53d83dd388422ac1b33b37165e71732f`.
Artifacts: `out/research/lethal-policy-support-replication-2026-10-12/`.
Do not enable the option by default or enlarge training on this evidence.

## Validation and next bounded decision

On implementation commit `1767d09`: **1,931 passed, 23 skipped, 0 failed** in
314.23 seconds. All 399 source/test/tool files pass formatting and lint. The
39 targeted lethal/combat tests include a real-engine win with the option enabled,
filtering before Greedy/random selection, original-index mapping, empty/truncated
support, and cached-response rejection. The complete test log and hash are saved
with the parity study. Documentation-only additions do not change the tested code.

## Interrupting-response causality

Six continuations of the two newly lengthened games reproduce the complete guided
controls and preserve each intervention prefix exactly. Original opponent policy
calls still occur before replacing a response. All records are healthy: 1,615
mask comparisons and 433 internal rollouts pass.

| Game | Baseline | Guided control | Reject first found line | Replace only recorded interrupting response |
| --- | ---: | ---: | ---: | ---: |
| 6 | win T8 | win T12 | win T8 | win T6 |
| 12 | win T7 | win T11 | win T13 | win T7 |

In game 6, card 93854893 is Dingirsu: its scripted `SelectEffectYesNo(...,96)`
chooses whether to detach material to replace destruction. The planner assumes
`no`; the actual opponent chooses `yes`. In game 12, card 58053438 is Songs of
the Dominators, which negates an activated monster effect on the field. This
identification uses the experiment's local card database and Lua scripts.

Removing just the recorded response restores a same-turn win in both cases. In
game 12, however, the opponent later activates that trap at decision 289, the
first line is abandoned, and a new search at decision 303 finishes the game.
Thus the result establishes the importance of response timing plus continuation,
not that the original line survives all other responses unchanged.

Rejecting only the first line is inadequate: game 12 searches again at decision
253 after rejection at 247, then encounters the same trap response at 262 and
wins only on turn 13. Game 6 also searches again on turn 8. This rejects a simple
"discard one bad proposal" fix. Cached choices skip wrapped-policy calls under
the existing fallback-only convention, so changed continuation sampling remains
a possible contributor to delays after abandonment.

Identity: `33118554bbe85554ea1dc7d3953c18394d2081d04b284609668f58f7d45d58d4`.
Report SHA-256: `2e07dd5f3e959023f39f5183338784d2860879b9de8fb8a7395f3968720574d7`.
Artifacts: `out/research/lethal-interruption-causality-2026-10-12/`.
These two outcome-selected games are causal diagnostics, not a population test.

## Fallback and re-search isolation

Eight continuations on main `d3b33e0` separate three possible workarounds. Both
control transcripts/results/counters reproduce exactly. Each treatment preserves
the prefix before its first intervention; suppressing all later searches also
reproduces each original baseline's **complete transcript and result**.

| Game | Guided control | Reject line + suppress search this turn | Reject line + suppress all later search | Consume policy calls for cached choices |
| --- | ---: | ---: | ---: | ---: |
| 6 | win T12 | win T8 | win T8 | win T12 |
| 12 | win T11 | win T7 | win T7 | **loss T12** |

The policy-call treatment consumes 64 and 11 wrapped-policy calls respectively;
it changes sampling/state progression without changing the initial accepted line.
It is not a successful fix. Suppressing same-turn re-search avoids the second
bad proposal in game 12. These interventions are selected using exposed failure
cases; this is not a general rule to disable search when a line looks risky.
All eight records are healthy; 2,016 applied mask checks and 590 internal rollouts
pass. No policy or runtime default changes.

Identity: `e0ae28708eb79cef260f219f22e46a92dd8b4c7702a13a74c591613ddb0ecfc1`.
Artifacts: `out/research/lethal-fallback-isolation-2026-10-12/`.

## Conservative gate and active planning opponent

The eight-record conservative-gate diagnostic replays proposed lines in the
shadow and rejects a line if an opponent planned `pass`/`no` has an engine-legal
active alternative, then suppresses search for that turn. Four exact controls
and treatment prefixes pass; 1,767 applied mask checks and 476 internal search
rollouts pass, in addition to saved-line verification replays.

A second eight-record pilot keeps the four cases and search budgets but replaces
the passive planning opponent with Greedy. It permits that simulated opponent's
chain events while continuing to reject own draws and coin/dice events. The real
match opponent remains the historical checkpoint. Four controls reproduce
exactly; all 1,912 mask checks and 519 internal rollouts are healthy.

| Source / game | Passive planner | Conservative gate | Active Greedy planner |
| --- | ---: | ---: | ---: |
| Replication / 6 | win T12 | win T8 | win T8 |
| Replication / 12 | win T11 | win T7 | **loss T12** |
| Fresh pilot / 10 | win T4 | win T14 | win T14 |
| Fresh pilot / 12 | win T3 | win T3 | win T3 |

**Neither proposal is accepted.** The gate loses a legitimate finish-speed gain;
the active planner additionally turns a selected win into a loss. These are
outcome-selected cases, not win-rate estimates. Both still inspect actual hidden
state, and one Greedy response policy is not an adversarial guarantee.

The active planner's two abandonments occur after activation, at opponent
selection choices. In replication game 6, decision 599 expects a selection of
card 48835607 but the real opponent selects 22850702. In game 12, decision 266
expects card 97045737 but the real opponent selects 6325660. All prompt signatures
match. This exposes reliance on an exact opponent continuation even after moving
beyond the passive assumption; it is not a mask or shadow-desync failure.

Gate identity: `68a914082e0fdabe9bcc2c773f96234ebde3766592e2e5fe9da89ce8195f3af0`.
Active-opponent identity: `25afec9635ab7f860b4ab9bbb73c9607066cba0033a6e4199d5c65080f163f27`.
Artifacts are `lethal-response-gate-diagnostic-2026-10-12/` and
`lethal-active-opponent-pilot-2026-10-12/` under `out/research/`.
## Opponent selection counterfactuals

Four continuations reproduce both active controls exactly and replace only the
first recorded opponent selection, preserving the real opponent draw and all
prior actions. Replication game 12 changes from loss T12 to win T7 when forced
to select the planned Dominus Purge instead of Dominus Spark. However, the forced
engine-legal row is **outside the actual opponent's policy support** (a duplicate
selection row); it must not be presented as a supported policy continuation. The
same card has a supported representative at row 0 (deck sequence 25), verified
by exact prefix replay in `opponent-mask-audit.json`; masked row 7 names deck
sequence 3. The card itself is available.

In replication game 6, replacing the first S:P target keeps win T8 but exposes a
second target mismatch at decision 600. The Greedy plan selects two monsters
controlled by that opponent (48835607 and 21441617), while the real opponent
selects the candidate's Chaos Angel (22850702). This is more than a permutation
of the same target set. It exposes a strategically weak opponent proxy; silently
relaxing cached-line signatures would conceal a real state difference.

Identity: `6e428f5bbbc5beab54e926e9d0f99094aa257031ae265390b7aceb022f6700e4`.
Artifacts: `out/research/lethal-selection-causality-2026-10-12/`.

## Opponent-support correction

The eight-record opponent-support audit now filters simulated Greedy choices
before sampling and maps supported rows back to the original engine indices.
All four unfiltered active controls reproduce exactly. The treatment performs
3,990 opponent-support checks; all 1,941 applied native/Python mask comparisons
and 588 internal rollouts are healthy.

All four selected games retain the same winner and final turn: win T8, loss T12,
win T14, win T3. Three complete transcripts match their unfiltered controls;
the losing case changes its intermediate path but still loses on turn 12. At its
first mismatch the simulated opponent now selects supported card 40366667 instead
of the earlier masked duplicate, while the real opponent selects 6325660.

This resolves a research-prototype sampling asymmetry, **not** the route's
strategic fragility. Stop tuning isolated Greedy rules. Next enumerate all
policy-supported first responses at the two observed selection states, retaining
actual choices as exact controls. Preserve pre-intervention prefixes and policy
draws, use the same subsequent policy/search algorithms and streams, and cap the
panel at 16 continuations. This measures sensitivity to one response; it is not
exhaustive game-tree search, a full minimax value, or a population-strength test.

Identity: `1ee8a2ee5760229e0004e8570d2a9236334763592510c5233532fb37b0162f39`.
Report SHA-256: `7fde0b42181d7e1f429fe93a9b21514f9bab16f5179d080b2fb6fb3e0ec588ee`.
Artifacts: `out/research/lethal-opponent-support-pilot-2026-10-12/`.
No production code, actor checkpoint, default, or training scale changes.

## Supported response branches and proposal reliability

The eight-record response-branch audit covers all supported first selections at
both observed mismatch states. The two actual-choice controls reproduce exactly,
and each intervention preserves the prior transcript and original opponent draw.
All 1,970 applied mask checks and 608 internal rollouts are healthy.

| Exposed state | Supported first responses | Candidate outcome under the fixed suffix |
| --- | --- | --- |
| Replication game 6, S:P target, decision 599 | rows 0–4 | all five win T8 |
| Replication game 12, Dominus search, decision 266 | rows 0, 1, 4 | rows 0/4 win T7; row 1 loses T12 |

An additional exact native-prefix audit reproduces all 375 and 139 prior
opponent policy draws, respectively. At the S:P state, the Greedy-planned row 3
has probability 3.17%, versus 85.91% for the actual row 1. At the Dominus state,
the losing Spark row 1 has probability **33.01%**; winning rows 0 and 4 have
47.40% and 19.60%. Thus the exposed harmful response is not exceptionally rare.
The probability audit is a derived follow-up, not part of the original frozen
branch registration. Scripts and input trace hashes accompany its JSON artifact.

These are selected states and one fixed suffix per response. They do not
estimate population win rate, enumerate the later S:P selection, or prove a
minimax value. In particular, a cached-line interruption need not change the
ultimate outcome. The implementation accepts the first successful simulated
trajectory; this evidence motivates checking a proposal's continuation
reliability rather than further tuning a single Greedy opponent proxy.

`lethal-proposal-continuation-2026-10-12` froze the first proposal at two
exposed roots: the failing replication game 12 (decision 247) and the retained
fresh-game-12 benefit (decision 169). Each compares executing that proposal with
clearing it and using the inner policy. Both arms disable all later search.
The completed panel has 16 continuations: two roots × two arms × four opponent suffixes
(the original stream and three independent reset seeds). Candidate RNG state,
pre-root transcripts and proposal hashes must match. The two original-stream
plan controls must reproduce their complete source transcripts and outcomes.
This is a bounded diagnostic, not a training scale-up or promotion gate.

Branch identity: `1cad7cd0500d7b630a3ee3549d9569ceca110bbb1dd692cabdb0d4564875c79c`.
Branch report SHA-256: `c9c44c58a701b6cdc0c012506836f61a87bd6078ab44123dfbe27fd772d8d309`.
Artifacts: `out/research/lethal-response-branches-2026-10-12/`, including
`opponent-probabilities.json` and its standalone audit script.

The 16 continuations are complete and healthy (2,811 applied mask comparisons,
904 internal search rollouts). Both full-transcript controls pass, and each
paired treatment has the same proposal and pre-root learner/opponent RNG states.

| Exposed proposal | Execute plan: wins / 4 | Fallback: wins / 4 | Paired outcome changes |
| --- | --- | --- | --- |
| Replication game 12, T7 | 2, both T7 | 4, all T7 | 2 harms, 0 gains |
| Fresh game 12, T3 | 4, all T3 | 2, at T5/T7 | 0 harms, 2 gains |

The risk-case line is abandoned in all four suffixes, including its two wins;
interruption alone is therefore not a useful binary failure label. The benefit
case has no abandonments. These results support measuring the value of a fixed
proposal relative to policy fallback; they do not validate a response gate or
establish policy strength. Different trajectory lengths consume RNG streams
differently after the intervention; this uses the existing wrapper fallback
convention, not a claim that subsequent draws correspond decision by decision.

Before using these outcomes to design a gate, the next bounded confirmation is
preregistered for the same two frozen proposals and three new opponent suffix
seeds: 12 new continuations plus two exact original-stream controls. Predictions
are that the exposed risk proposal can harm relative to fallback and the benefit
proposal retains same-turn finishes. No search, checkpoint or heuristic changes
are permitted between these arms. New suffixes test continuation reliability on
these states only; new states remain necessary before any promotion.

Continuation identity: `31aeb8b24329dd0f387eff40a3f45752cb862f5d01bed5c4925bac21300c5239`.
Report SHA-256: `65084115335516192dbd862f76c6ef8d8eb5d9eb42ea42fb7f27f6d5dc524817`.
Artifacts: `out/research/lethal-proposal-continuation-2026-10-12/`.

## Independent opponent-suffix confirmation

`lethal-proposal-confirmation-2026-10-12` completed all 14 records: two repeated
original-stream controls and 12 new continuations from three preregistered seeds.
Controls reproduce exactly; paired proposal hashes, pre-root RNG states and
prefixes agree. The panel has 2,474 applied mask checks and 791 healthy internal
rollouts. Repeated controls are excluded from the following counts:

| Exposed proposal | Execute: wins / 3 new streams | Fallback: wins / 3 new streams | Paired changes |
| --- | --- | --- | --- |
| Replication game 12, T7 | 2, both T7 | 3, all T7 | 1 harm, 0 gains |
| Fresh game 12, T3 | 3, all T3 | 1, at T9 | 0 harms, 2 gains |

The risk-case loss again ends T12; its line is abandoned even in winning
suffixes. Both failed benefit-case fallbacks end T8. This reproduces a difference
between the two proposals, not a reliable effect size or a validated acceptance
gate. These remain the same exposed roots against one checkpoint opponent.

The next collection fixes two fresh deal blocks and all four seat/deck
assignments (eight games). Record only the first accepted proposal per game,
then disable search and continue with the unmodified inner policy. Never execute
the proposal during collection, and retain games without a proposal. Two plain
policy controls check complete transcript equality for the first two games.
Use the existing 12-rollout/200-step/8-search limits and two CPU workers. This
provides outcome-unselected roots for a subsequent paired continuation audit;
no training, default changes, or gate promotion follows from collection alone.
If roots are available, test at most the first four by game ID, independent of
terminal outcomes; do not adapt the collection seed or budget to its results.

Identity: `dfb2be24c7edfe4f680d6ef403be52a6cff33e1712d517fb4b756f163a53b4e3`.
Report SHA-256: `17c44dcf7be50b0eceb15cf12bcf600106d8f7c44464767153d0e8c4eaeafe83`.
Artifacts: `out/research/lethal-proposal-confirmation-2026-10-12/`.

## Fresh-deal proposal collection

`lethal-fresh-proposals-2026-10-12` completed the fixed eight policy games plus
two plain-policy controls. Three games contain a first accepted proposal:

| Game ID | Proposal decision / turn | Planned decisions | Original policy outcome |
| --- | --- | --- | --- |
| 2 | 222 / T8 | 125 | win T8 |
| 3 | 173 / T7 | 61 | win T7 |
| 7 | 240 / T7 | 86 | win T9 |

Games 0, 1, 4, 5 and 6 have no proposal within the fixed search budget; this does
not prove there was no lethal. All eight games have zero executed search
overrides, abandonments and shadow desyncs. Both plain-policy controls match
complete transcripts and outcomes. Health passes across 1,991 applied mask
comparisons and 707 internal rollouts.

The roots were selected by game ID, not outcome, and all three available roots
are retained without resampling to fill the four-root cap. They share only two
fresh deal blocks; card/archetype novelty and independent root sampling are not
claimed. Although their original policy continuations all win, two already win
on the proposal turn, so merely executing a simulated lethal may add no value.

The next bounded panel fixes these three proposals and tests execution versus
policy fallback under the original opponent stream plus two fresh suffix seeds:
18 continuations total. All future search is disabled after the proposal;
pre-root transcripts, proposal hashes and learner/opponent pre-reset RNG must
match collection. Original-stream fallback controls must reproduce all three
complete collection transcripts. Report wins, harms, same-turn finishes and
abandonments separately by root; no default or training promotion.

Collection identity: `e0191a98caf8a21fde394f40714c25500c5e9c40d73689d98af56d7e2cc0bf18`.
Report SHA-256: `89221aa336e8f5fb54a9a6902dd65ec8bc42a951b2a48ca50021d2f059ff277c`.
Artifacts: `out/research/lethal-fresh-proposals-2026-10-12/`.

## Fresh-root paired continuation results

All 18 continuations in `lethal-fresh-continuation-2026-10-12` are healthy.
Three original-stream fallback controls match complete collection transcripts,
and all paired proposals, pre-root RNG states and prefixes agree. There are
3,011 applied mask comparisons and 1,362 internal rollouts.

| Root game | Execute: end turns across streams 0/1/2 | Fallback: end turns | Outcome |
| --- | --- | --- | --- |
| 2, proposal T8 | 8 / 8 / 10 | 8 / 8 / 8 | all win; one two-turn delay |
| 3, proposal T7 | 7 / 7 / 7 | 7 / 7 / 7 | all win; neutral |
| 7, proposal T7 | 7 / 7 / 7 | 9 / 11 / 9 | all win; finishes 2–4 turns earlier |

Both arms win all nine suffix continuations. These three roots share two deal
blocks and do not support a population win-rate claim. Root 2 abandons its line
on an opponent zone choice in stream 1 (decision 288), yet still wins T8. In
stream 2 it abandons on an opponent monster selection (decision 285): the plan
expects Lunalight Kaleido Chick (35618217), while the opponent chooses Lunalight
Gold Leo (8379983), and the game ends T10. These are observed divergences; this
comparison alone does not isolate a card-specific causal effect.

Next test a bounded recovery proposal instead of weakening cached signatures:
after the first line abandonment, allow at most one more search within the
original turn and eight-search game cap. Compare no further search, ordinary
main/battle-trigger replanning, and replanning at the next own decision prompt.
The two new root-2 interrupted suffixes and one previously harmful Dominus
suffix form nine records (three exact controls and six treatments). Each had
seven searches at the original proposal, so the one extra attempt stays within
the already registered total budget. Keep the per-turn search cap, mask checks,
strict own-draw/random-event filter, opponent proxy and all RNG conventions.
Require prefixes through the recorded interruption to match; report failed
replans, renewed interruptions and terminal harm. This is a diagnostic fix
candidate, not production adoption or proof of robustness to hidden responses.

Identity: `5bfb7f73650d2723e3d04aa02854f1e438653bc1f662e2b90166148e9403890d`.
Report SHA-256: `851273da093d9569bfc08aa8b2c6587efabf302b885096c43bc5a6a13ef6edc2`.
Artifacts: `out/research/lethal-fresh-continuation-2026-10-12/`.

## Bounded interruption replanning: no rescue

`lethal-interruption-replan-2026-10-12` completed all nine records healthily.
All three controls reproduce full transcripts, outcomes and search statistics;
treatment prefixes match through the observed interruption. The panel has
2,040 applied mask checks and 734 internal rollouts.

| Exposed suffix | Control | Phase replan | Next-own-prompt replan |
| --- | --- | --- | --- |
| Fresh root 2, stream 1 | win T8 | win T8; found at decision 297 | win T8; found at decision 290 |
| Fresh root 2, stream 2 | win T10 | win T10; no line found | win T10; no line found |
| Earlier Dominus case, confirmation stream 2 | loss T12 | loss T12; no line found | loss T12; no line found |

The two successful replans use four and one rollouts, respectively. Each failed
attempt exhausts its 12-rollout allowance. Neither trigger rescues a delayed or
losing continuation, so this diagnostic does not justify adopting replanning.
No-line-found within this budget is not proof that the state was irrecoverable.

Next implement an offline acceptance prototype before committing to a proposal.
Freeze this conservative rule using three already available paired probe streams:
accept only if the planned continuation wins on the proposal turn in all three
and fallback fails to do so in at least one. Reject incomplete/unhealthy probes
and require matching proposal and pre-root states. This distinguishes execution
benefit from merely finding a simulated lethal; three successes are not a safety
or minimax guarantee.

Use all five previously studied roots (two earlier and three fresh), with their
probe decisions frozen before evaluating two new opponent suffix seeds. The
bounded panel is 20 new continuations plus five exact original-stream controls.
Compare the gate-selected arm against always-plan and always-fallback. The
candidate rule is informed by these states, so only suffix sampling is held out;
this is a development diagnostic, not fresh-state validation or production
adoption. Full-game probes, actual hidden state and a known checkpoint opponent
also make this an offline prototype, not an inference-cost or public-information
solution. No actor/critic/default/training changes.

Replan identity: `91e85fc6cfaf9d8d5dd96664f57839c9dcef7f5e22a4d5b884bb334b7e0a131c`.
Report SHA-256: `6c50c9ef5f10cf415682b31713b5aced4babc72e2ac164b16b07355c2c9b023c`.
Artifacts: `out/research/lethal-interruption-replan-2026-10-12/`.

## Frozen offline gate: known-root suffix results

`lethal-paired-gate-2026-10-12` completed 25 healthy records, including five
exact original-stream controls. All frozen acceptance decisions and paired
proposal/prefix/RNG checks pass; 4,391 mask comparisons and 1,700 internal
rollouts are healthy. Four malformed/incomplete-probe checks fail closed.

On ten evaluation pairs (excluding repeated controls), the frozen gate-selected
arm wins 10/10, always-plan 9/10 and always-fallback 8/10. Compared with
always-plan, the gate avoids one loss and two four-turn delays. Compared with
always-fallback, it retains two additional wins at the earlier benefit root
and two faster wins at fresh root 7. There are no paired winner harms in these
ten evaluations. This is deterministic selection of a precommitted arm from
paired continuations, not a separate full-game autonomous gate deployment.

The rule accepts the two benefit roots, rejects both unstable roots and rejects
the neutral root. The states informed rule design; only opponent suffix seeds
were held out. These figures therefore do not validate fresh-state generalization
or production strength, and three successful probes are not a safety guarantee.

Next freeze the identical rule before collecting two new deal blocks and all
four seat/deck assignments (eight policy games plus two plain-policy controls).
Take at most the first two available first-proposal roots by game ID, ignoring
terminal outcomes and without replacement sampling. For each root, run one
exact original-stream fallback control, three new paired probe streams and two
disjoint paired evaluation streams. Gate decisions must be persisted and bound
to probe hashes before evaluation starts. The prospective pilot therefore has
at most 32 records (10 collection + 2 roots × 11 continuations), two CPU workers
and the existing 15-minute service limit. A single monitored pipeline advances
collection → probes → frozen decisions → evaluation; missing roots reduce the
panel rather than extend collection. No threshold tuning or training expansion.
This tests the rule on prospectively collected states while retaining the known
checkpoint opponent, actual hidden state and expensive offline probe limitations.

Identity: `4386a09a546750bd3923cb333d9fc8d6beaa65ba6f89aa00e93b0799b515cd9d`.
Report SHA-256: `108dbd3e60b147f89c2425ed61d437549d8e485bb74da1b7c494aea79ad2d83f`.
Artifacts: `out/research/lethal-paired-gate-2026-10-12/`.

## Prospective gate pilot: defensive evidence, no added wins

`lethal-prospective-gate-2026-10-12` completed 32 healthy records. Five of eight
collection games have proposals (IDs 1, 2, 3, 6, 7); the preregistered first-two
rule selects IDs 1 and 2. Both are from deal block 0, so evaluation did not cover
both collected blocks. Collection/gate dependency seals, all exact controls,
and decision-freeze-before-evaluation timing pass. There are 6,250 mask checks
and 1,504 internal rollouts.

The unchanged gate rejects both roots. At game 1's T8 root, plan and fallback
both finish that turn in all three probes: no sampled incremental gain. At game
2's T2 root, neither arm finishes that turn in any probe: the proposed plan is
not reliably completed against the checkpoint opponent.

| Selected root | New evaluation stream | Plan | Fallback / rejected gate |
| --- | --- | --- | --- |
| game 1, T8 | 4 / 5 | win T8 / win T8 | win T8 / win T8 |
| game 2, T2 | 4 | loss T11 | win T4 |
| game 2, T2 | 5 | win T2 | win T4 |

Thus gate and fallback each win 4/4; always-plan wins 3/4. The gate avoids one
loss while foregoing a two-turn speed gain in another suffix. **There is no
observed improvement over fallback**, and with zero accepted roots this does
not validate the acceptance branch on new states. Probe failures followed by
one successful evaluation finish also illustrate suffix variability, not a
reason to relax the frozen threshold.

One additional fixed-budget prospective run will retain byte-identical gate,
search limits, checkpoints, three probes and two disjoint evaluations, but take
at most the first proposal from each of two new deal blocks rather than the
first two globally. This selection correction is declared before the new deals
and uses no terminal outcomes. Keep the same 32-record ceiling and no replacement
sampling for blocks without proposals. After this run, summarize both pilots;
do not continue choosing seeds merely to obtain accepted candidates. Production,
training and defaults remain unchanged.

Identity: `fbc89067bc917864b6515e629bd236097005a102abfc505062920b289a7da1d0`.
Report SHA-256: `319f167697a9b99003ca5511c8b9d15df7924e1116b61e6e8a4d811d4b0b5ceb`.
Artifacts: `out/research/lethal-prospective-gate-2026-10-12/`.

## Block-balanced confirmation: stop gate replication

`lethal-balanced-gate-2026-10-12` completed 21 healthy records. Only block 0
produced proposals (games 0 and 1); the preregistered selection takes game 0,
T5, and does not replace the empty block. All three probe pairs have both plan
and fallback winning on T5, so the byte-identical gate rejects this neutral
candidate. In both independent evaluation pairs, plan, fallback and gate win
on T5. Exact controls, dependency seals, freeze-before-evaluation timing and
health checks pass (2,866 mask checks; 1,333 internal rollouts).

Across the two prospective runs: three selected roots, zero acceptances, six
evaluation pairs. Gate and fallback win 6/6; always-plan wins 5/6. These are
selected-root continuations, not a full-game win-rate estimate. The defensive
benefit against one bad proposal has replicated on a fresh state, but **there
is no demonstrated gain over the unmodified policy** or fresh-state evidence
for the gate's acceptance branch. Stop additional seed replication of this gate;
keep production, actor/critic training and defaults unchanged.

Next audit search allocation on all 16 existing collection games, with no new
deals, changed actions or added search budget. Log attempted and skipped own-turn
main/battle prompts, phase, per-turn/game budget and proposal stopping. Every
replay must match its source's full transcript, outcome, proposal and search
statistics. This can establish whether the two-per-turn/eight-per-game budget
is spent early; a skipped prompt is not evidence of a missed lethal. Only then
consider a bounded scheduling ablation. The audit retains two CPU workers,
per-record persistence and the 15-minute service limit.

Identity: `e4079d008c20c78366691ad26c6c66a3c67033ab47b6ad1b65cbea93c3744538`.
Report SHA-256: `73350ef53760a2424b036b74b7cedc6cfcadd9ea65c0c4126cebf4fb4bda146b`.
Artifacts: `out/research/lethal-balanced-gate-2026-10-12/`.

## Search allocation audit: Main1 consumes most attempts

`lethal-search-allocation-audit-2026-10-12` replayed all 16 existing collection
games with exact transcript, outcome, proposal and search-statistics parity.
All bound inputs and trace hashes pass. Of 103 searches, 93 occur in Main1 and
10 at battle-command prompts (engine phase `PHASE_BATTLE_START`). Of 49 own
turns with two searches, 39 spend both in Main1. Before proposal stopping,
140 eligible prompts are skipped by the turn cap and 61 by the game cap;
80 are battle-command prompts (60 turn cap, 20 game cap), across 15/16 games.
Seven games produce proposals; seven no-proposal games exhaust all eight
attempts. These observations identify allocation bias, not missed-lethal labels.

Next use these same 16 games for one frozen scheduling diagnostic: at most one
Main1 search and one reserved battle-command search per own turn, still at most
eight searches/game, 12 rollouts/search and 200 steps/rollout. Unused phase quota
is not transferred; Main2 is excluded. The first found proposal is captured and
immediately discarded as before; real play remains the original policy, with
no extra RNG consumption. This deliberately compares equal maximum budgets,
not guaranteed equal realized rollout counts, which must be reported.

Run 16 scheduled collection replays plus two exact legacy-schedule controls.
All 18 must match the source's complete actual game transcript and outcome;
legacy controls additionally match proposal/search statistics. Compare coverage,
first-proposal turn/index, lost versus newly found candidates, actual searches
and rollouts. A found line against the supported Greedy proxy is not a validated
win against the checkpoint opponent. No production/training change or budget
increase follows from coverage alone. Two CPU workers and the existing
15-minute service limit remain in force.

Identity: `a3a46d6716910b45e9cca9b3978f4022234f91f354924a257cbf834c01a6e864`.
Report SHA-256: `9b83f9315cb3379da038626d428f3fd15dfebe26d2e382f1981103c86bde40f5`.
Artifacts: `out/research/lethal-search-allocation-audit-2026-10-12/`.

## Phase quota pilot: lower compute, mixed discovery timing

`lethal-phase-quota-pilot-2026-10-12` completed 18 healthy records: 16 scheduled
replays plus two legacy controls. Every actual transcript/outcome matches its
source; legacy proposal/statistics, quota limits and all bound hashes pass.

| Discovery metric (16 collection games) | Legacy | Phase quota |
| --- | ---: | ---: |
| Games with first proposal | 7 | 8 |
| Searches | 103 | 96 |
| Internal rollouts | 1,165 | 1,081 |
| Main1 / battle-command searches | 93 / 10 | 59 / 37 |

Five games retain the same first-proposal index; eight produce none under either
schedule. Balanced game 0 moves three decisions earlier within T5. Balanced
game 4 gains a first proposal on T9 (decision 444), but prospective game 7 moves
from T7 (decision 198) to T9 (decision 248). Thus more candidate coverage does
not imply better early finishing. Search ordering also changes exposure to the
search RNG. Real policy outcomes are unchanged by construction, not evidence
that following the new plans is harmless. Do not promote the schedule.

Next run 18 paired continuation records on exactly the changed-turn/new roots:
balanced game 4's new T9 proposal and prospective game 7's old T7/new T9 proposals.
The same-turn index-only change is outside this follow-up. For each of three
roots, compare cached-plan execution with policy fallback under the original
opponent stream and two fixed new suffix streams (2026101301, 2026101302).
Reconstruct the root with its original schedule/search RNG, require identical
proposal and pre-root policy RNG/transcript, stop subsequent searches, and
retain exact full original-stream fallback controls. Evaluate eventual wins,
harms, same-turn finishing and abandonment. Different root boundaries mean
suffix comparisons are paired within each root, not a direct scheduler win-rate
comparison. No new deals, adaptive sampling, budget increase or production
change; two workers and 15-minute cap.

Identity: `e18e68b3a1bfca46711c07711efca25c2a7e9efa53898095c391758af70192e2`.
Report SHA-256: `18b9283d182598352596ae1bcd2868a997967e0504e0b57b910a8a202147b992`.
Artifacts: `out/research/lethal-phase-quota-pilot-2026-10-12/`.

## Quota continuation: extra coverage is neutral; investigate RNG coupling

`lethal-quota-continuation-2026-10-12` completed 18 healthy records with exact
original fallback controls, proposal/pre-root RNG equality and trace hashes.
The new balanced-game-4 T9 proposal and replacement prospective-game-7 T9
proposal each match fallback: both arms win T9 on all three streams. The old
prospective-game-7 T7 proposal wins T7 on all three streams, while fallback
wins T9. No plan is abandoned. Thus the extra discovered root adds no observed
benefit and the old early route has demonstrated value on these suffixes.
This is selected-root evidence, not a scheduler-wide strength estimate.

Crucially, both schedules **do search decision 198 on T7**, the same first
Main1 prompt and actual policy state. Legacy finds the line in its first
rollout after 72 preceding rollouts; quota fails all 12 after 60 preceding
rollouts. The lost discovery is therefore not an omitted root. Earlier
allocation changes both the rolling chooser RNG and the simulated opponent's
seed, which is based on cumulative rollout count. Quota later finds T9 at
index 248. A schedule-only causal interpretation would be unsupported.

Next run four read-only replays at this single exposed root: cross the chooser
RNG state after 60/72 prior seed draws with the opponent rollout-seed offset
60/72. Preserve the quota schedule before the root, original real-policy RNG
and full actual transcript; stop all search after the root. Keep 12 rollouts
maximum at the root. The 60/60 control must reproduce the quota root failure;
72/72 must reproduce the exact legacy winning proposal. The two crossed cells
isolate which RNG path explains the discovery change. Four records, two CPU
workers, no new deals or budget increase; no default or training changes.
This mechanism test cannot establish robust proposal performance across seeds.

Identity: `c46b782e2dedae6911b3ced46cb59e1d0561b5ad7745ac830166b33afcdd41a6`.
Report SHA-256: `933361decf73313c5e0f3fd12a9ef08996f3e991c12b4c36e0a8558a5218384f`.
Artifacts: `out/research/lethal-quota-continuation-2026-10-12/`.

## RNG crossover: chooser stream explains this missed discovery

`lethal-search-rng-crossover-2026-10-12` completed four healthy records with
complete actual-policy transcript/outcome parity. The quota RNG state rebuilt
from 60 chooser-seed draws exactly matches the live search state.

| Chooser stream | Opponent offset 60 | Opponent offset 72 |
| --- | --- | --- |
| after 60 prior draws | no proposal in 12 | no proposal in 12 |
| after 72 prior draws | same proposal on rollout 1 | same proposal on rollout 1 |

Both positive cells reproduce the legacy proposal hash
`c7a69449e23df7a76773cc6e7fed717d8fa682656e029472c1f945468a1b7730`.
At this root, changing the chooser stream alone recovers the known T7 finish;
changing the tested opponent offset alone does not. This does not show that
opponent randomness is irrelevant elsewhere, or that the selected successful
chooser seed should be deployed. The previous quota comparison conflated
scheduling with stream shifts caused by earlier search count.

Next correct the research comparison with root-keyed random streams. Derive
independent chooser/opponent seeds with SHA-256 from fixed study seed, game
identity, root decision index/turn/player, and role; opponent rollout seeds also
include the within-root rollout ordinal. Neither seed depends on previous
search count, and no seed is chosen for its result. At shared roots, both
schedules must have identical seed keys and proposal/rollout results. This is
an experimental control change; production remains unchanged.

Run both schedules on each of the existing 16 games (32 paired collection
records), plus two exact original-RNG legacy controls: 34 records total, no new
deals. Real policy actions and full source outcomes must remain identical;
proposals are captured/discarded, with no searches after the first proposal.
Retain 12 rollouts/search, 200 steps/rollout, two searches/turn and eight/game,
two workers and 15-minute cap. Report candidate coverage, timing, lost/new
roots and actual compute. The paired rerun is needed to isolate allocation;
it is not a larger strength evaluation or promotion gate.

Identity: `6cfa9b6f1acf7fcd181c8ccad472314602d280663ab100810b07a7dba959ae6f`.
Report SHA-256: `cb6f771268d9f89f7799f72c5f538c252559c5567e029f083b8b9878cd368ff8`.
Artifacts: `out/research/lethal-search-rng-crossover-2026-10-12/`.

## Root-keyed scheduling: verified comparison, remaining quota tradeoff

`lethal-root-streams-scheduling-2026-10-12` completed 34 healthy records.
All actual-policy transcripts/outcomes and both original-RNG controls pass.
Across **63 common searched roots**, seed keys, chooser/opponent seeds,
rollout counts, rejected/found flags and proposal hashes match exactly.

| Metric (16 paired collection games) | Legacy schedule | Phase quota |
| --- | ---: | ---: |
| Proposal games | 7 | 8 |
| Searches | 103 | 96 |
| Rollouts | 1,162 | 1,073 |

Six first-proposal roots match; eight games have none under either schedule.
Balanced game 4 still gains a T9 proposal. The prior prospective-game-7 T7
mismatch disappears: both schedules now find the identical T7 proposal. This
confirms that the earlier mismatch was not a skipped-root allocation effect.

A different, genuine quota exclusion appears in prospective game 6: both
schedules search Main1 decision 194 on T6 and fail. The real policy summons,
then reaches decision 200. Legacy searches this second Main1 point and finds
a proposal beginning with activation of card 1984618; quota skips it because
its Main1 allowance is spent. Policy instead enters battle at decision 200;
quota searches battle decision 203 without finding a line, and its next first
proposal is T8 decision 263. This establishes exclusion of a discovered route,
not yet exclusion of a reliable win against the checkpoint opponent.

Next evaluate the three changed/new roots under these exact root-keyed streams:
new balanced-game-4 T9, legacy prospective-game-6 T6, and quota replacement T8.
For each, execute the captured plan or fallback under the original opponent
stream and two fixed new suffixes (2026101321/2026101322): 18 records. Match
original fallback full traces, pre-root policy RNG and exact proposal; disable
later search. The new T9 plan hash differs from the earlier non-rooted plan,
so its continuation result must not be assumed from that prior study. No new
deals, adaptive seeds, budget increase or production adoption. Two workers,
15-minute cap. Report within-root gains/harms; do not infer overall strength.

Identity: `a2e717966e390a37bdc37bbb331628dd4f1abb975fefc045a685a7be4a247548`.
Report SHA-256: `d460d6fa6b46cf6779d6cc8aa833a90d64570ef21743e3d9cc4853e328103f43`.
Artifacts: `out/research/lethal-root-streams-scheduling-2026-10-12/`.

## Matched continuation rejects hard phase quotas

`lethal-root-streams-continuation-2026-10-12` completed 18 healthy records with
exact original controls, fixed proposal/pre-root RNG and source hash checks.
Legacy prospective-game-6's T6 route wins T6 on all three streams; fallback
wins T8 in each. Its quota T8 replacement and balanced-game-4's new T9 route
both match fallback timing on all streams. No plans are abandoned. The hard
one-Main1/one-battle quota excludes an empirically useful route without an
observed benefit from the newly found root. Reject promotion of this schedule.

Next test a decision-aware trigger at unchanged caps: search the first Main1
prompt; allow the second search immediately before the policy's chosen
`battle_phase` action in Main1, otherwise use the first battle-command prompt.
Do not search Main2 or transfer more than the two-per-turn/eight-per-game caps.
To inspect the pending policy choice without changing actual play, save/restore
its generator and `last_probs`; assert RNG equality after the peek and equality
between the peek and the eventual emitted policy action. This is specific to
the inspected CheckpointAgent implementation, not a generic agent contract.

Use the existing 16 games and root-keyed chooser/opponent streams, plus two
exact rooted legacy-schedule controls (18 records). Capture/discard proposals,
require all complete actual traces/outcomes unchanged, compare shared roots to
the prior rooted records, and report candidate timing, coverage and actual
compute including policy-peek calls. This trigger was motivated by exposed
states and requires later independent validation if useful. No new deals,
training, production changes or budget increase; two workers, 15-minute cap.

Identity: `13317eacebbaec7c3c8837e262fea0a664683ce4aba4481647e825f39cfdba1b`.
Report SHA-256: `3411df8ea996cf63d1e530bd539c960faca93262270517c6432cb540b96719b7`.
Artifacts: `out/research/lethal-root-streams-continuation-2026-10-12/`.

## Pre-battle trigger retains exposed proposals with fewer rollouts

`lethal-prebattle-trigger-2026-10-12` completed 18 healthy records. Actual-policy
transcripts/outcomes and rooted legacy controls match exactly. All 161 shared
root comparisons to legacy/quota references agree (this count includes both
references, not 161 unique roots). All 105 policy peeks preserve generator
state and agree with actual emitted choices.

On the 16 exposed games, the trigger retains all seven legacy proposals at the
same root/index/turn and adds balanced-game-4's T9 proposal. Eight games have no
proposal in either arm. Searches fall 103→95; rollouts 1,162→1,061 (8.7% fewer).
The extra 105 policy forwards mean this is not evidence of a wall-time speedup.
The extra root already has neutral continuation evidence; neither candidate
coverage nor unchanged actual-policy play demonstrates a playing-strength gain.

This is enough to justify a small independent discovery check, not production
adoption. Freeze the trigger, root-keyed seed scheme and caps before collecting
two new deal blocks with all four seat/deck assignments (eight games). Run both
legacy and pre-battle schedules, with two plain-policy full-trace controls:
18 records. Use panel seed 2026101340; retain empty/failed-discovery games with
no replacement or additional seed search. Verify no dealt opening-prefix overlap
with the earlier collection panels (not an unseen-card claim). Preserve exact
actual-play equality between schedules and shared-root search parity, record
lost/delayed/new candidates, actual rollouts, peek counts and peek time. Report
this one panel regardless of outcome; no adaptive tuning or training/default
changes. Two workers, unchanged search caps and 15-minute service limit.

Identity: `075eab88818de8538a1b635cf5bc1dba07dae8ff905c07b21f7d15b3fa01ce79`.
Report SHA-256: `4b349cd63b603be3b15f704e2e232525bc342493092b4a68a04e818bc1f66e7b`.
Artifacts: `out/research/lethal-prebattle-trigger-2026-10-12/`.

## Independent pre-battle panel retains all four proposals

`lethal-prebattle-fresh-2026-10-12` completed 18 healthy records on two new deal
blocks/all four assignments. All eight paired actual-policy transcripts and
outcomes, two plain-policy controls, and 34 common-root search comparisons pass.
There is no dealt opening-prefix overlap with the registered prior panels.

Both schedules find four proposals at exactly the same roots (games 0, 5, 6, 7)
and neither finds a proposal in the other four games. No candidate is lost,
delayed or added. Searches fall 50→47 and rollouts 557→521 (6.5% fewer).
The 66 additional policy peeks take 0.189 seconds in total as measured here;
this is component timing, not a controlled end-to-end speed comparison. Two
independent deal blocks are insufficient for a broad generalization claim,
and equal real play is expected because all proposals were discarded.

Next assess actual proposal value on **all four** freshly collected roots,
without outcome filtering: plan execution versus policy fallback under the
original stream and two fixed new opponent suffix streams (2026101351/1352).
There are 24 records (4 roots × 2 arms × 3 streams), no new deals. Reconstruct
the frozen pre-battle trigger and root-keyed search, require exact candidate
and pre-root policy RNG/transcript, retain full original-stream fallback
controls, and disable later search in both arms. Record eventual gains/harms,
same-turn finishes, turn changes and abandonment. Disjoint opponent suffixes
do not turn four selected roots into a full-game strength evaluation. Keep
strict line/support checks, two workers and 15-minute cap; no training/default
changes, extra search budget, replacement roots or adaptive suffix sampling.

Identity: `26853e53cef9df2f0e584d54d4761356d9bf5e759bf7929c6382f82197ca4ea6`.
Report SHA-256: `75857d5e5f6bb0c0cc7dcf18a2b3f2ee1b7009c693d8d5bd2f357bd814dcdc50`.
Artifacts: `out/research/lethal-prebattle-fresh-2026-10-12/`.

## Fresh proposal execution exposes response-dependent losses

`lethal-prebattle-value-2026-10-12` completed all 24 healthy continuations
(four collected roots × plan/fallback × three opponent suffix streams).
All original-stream fallback transcripts/results reproduce exactly; each pair
has an identical reconstructed proposal and pre-root policy RNG/transcript.

Across the 12 selected-root pairs, fallback wins 12/12 and plan wins 8/12:
four harmful win-to-loss flips and no added eventual wins. Plan finishes on
the proposal turn in 5/12, versus 3/12 for fallback. Among eight pairs where
both win, seven finish earlier with plan and one finishes two turns later.
These repeated suffixes share four roots/two deal blocks, so these fractions
are descriptive and not independent population strength estimates.

| Fresh game/root | Plan result, streams 0/1/2 | Fallback result, streams 0/1/2 |
| --- | --- | --- |
| 0, T5 | loss T6 / win T6 / win T5 | win T7 / win T7 / win T7 |
| 5, T8 | loss T9 / loss T9 / loss T9 | win T8 / win T8 / win T8 |
| 6, T2 | win T2 / win T8 / win T10 | win T10 / win T12 / win T8 |
| 7, T7 | win T7 / win T7 / win T7 | win T11 / win T11 / win T11 |

All seven abandonments are exact-signature opponent action mismatches, not
mask/signature implementation failures. Game 5 has the same destruction-target
mismatch in all three streams (actual selects 81196066; search predicts
47705572). Game 0 mismatches on pass versus chain or monster position; game 6
mismatches on pass versus chain. Thus merely searching against an active Greedy
opponent does not make a candidate safe against the checkpoint opponent.
Abandonment correctly stops the obsolete line but cannot undo its prior actions.
Do not promote plan execution or interpret shorter losing games as improvement.

Next isolate causality on **all seven** interrupted continuations, including
ones that still win: exact plan replay control versus replacing only the first
recorded opponent mismatch with the search-predicted action (14 records).
Require the complete control trace/result and pre-intervention prefix to match,
require the current signature and encoded support to permit the replacement,
and still call the real opponent policy once so its RNG consumption is unchanged.
After that single response, return to the actual policy and retain strict line
invalidation on any later mismatch. Record whether the first substitution
rescues same-turn/eventual wins or merely moves the next interruption. This is
an intentionally artificial causal diagnostic, not an executable strategy or
an opponent model evaluation. No new deals/suffixes, search-budget increase,
training or default changes; two workers, 15-minute cap.

Identity: `525d6ea137c48d5f62fca42142278b2e2e68f627293ed3a540cf945695d93382`.
Report SHA-256: `ca9ae4db5eaa5683452bd471fcd93b14dacbf9b257e1a9dffa8e7888a8eee324`.
Artifacts: `out/research/lethal-prebattle-value-2026-10-12/`.

## A single opponent response explains the fresh execution losses

`lethal-prebattle-response-2026-10-12` completed 14 healthy records. All seven
unmodified controls reproduce complete traces/results. Every intervention has
an identical pre-response prefix and exact cached signature; replacement
indices pass encoded-support checks. The real opponent still makes its ordinary
policy draw before its one action is replaced.

All seven substituted continuations win on the proposal turn, compared with
zero same-turn wins and three eventual wins among their controls. All four
original losses become wins. Four substitutions avoid all further abandonment.
The three game-5 target substitutions still abandon later at decision 426
(pass versus chain), yet policy fallback wins during T8. This establishes a
causal role for the first response under these fixed continuations, and again
shows that abandonment itself is not a failure label. It does not establish
that the real opponent will cooperate, nor validate overriding its choices.

The useful next implementation experiment is a bounded **proposal challenge**,
not another suffix sweep or a tweak to one card's Greedy rule. Reconstruct all
four fresh proposals from their original roots, then replay the entire cached
line in the shadow and require an exact supported same-turn win as a control.
At the first four opponent prompts having supported alternatives, retain up
to three alternatives in stable original action order (at most 12 challenges
per proposal); report untested alternatives explicitly. For each challenge,
restore the native snapshot at that prompt, substitute one supported response,
then continue both sides with fixed supported Greedy policies for at most 200
decisions/the remaining turn. Seeds derive from root, prompt, action and role,
not prior rollout consumption. Record same-turn wins, non-wins, strict-event
health and incomplete coverage. A failed Greedy continuation is evidence about
that continuation, **not proof that no winning reply exists**.

All proposals stay discarded in the real games; their complete policy traces,
results and RNG state must remain equal to the fresh collection. Four records,
no new roots/deals/suffix search, no actor/critic change, no production gate or
increase in discovery budget; at most 48 additional bounded shadow challenges,
two workers and 15-minute limit. This tests whether a generic branch check can
expose fragility without looking up the observed mismatch. Do not treat a pass
on this partial known-state model as a public-information safety guarantee.

Identity: `c71ff3490bfaef28dba034c5f7283ea902738060465fa22cd1c655748a955473`.
Report SHA-256: `aa5d073b37a385dfb1cd13cf9274d1c4e5473ead599c0722cc4c54f8e29324e6`.
Artifacts: `out/research/lethal-prebattle-response-2026-10-12/`.

## Bounded response challenges expose a continuation-policy mismatch

`lethal-response-challenge-2026-10-12` completed four healthy records and 13
alternate-response shadow continuations. All four exact cached-line controls
win; real full fallback traces/results and policy RNG states remain unchanged.
There are no 200-decision budget exhaustions.

| Fresh game | Tested alternatives | Same-turn Greedy wins | Untested alternatives |
| --- | --- | --- | --- |
| 0 | 5 | 0 | 0 |
| 5 | 6 | 6 | 4 |
| 6 | 2 | 0 | 0 |
| 7 | 0 | n/a | 0 |

Game 7's cached line has no opponent prompts with a supported alternative;
this says nothing about other hidden states. Game 5's failed real target choice
is explicitly among the tested branches and wins under the Greedy continuation,
while the real checkpoint continuation loses in all three observed suffixes.
Thus even a branch check can be misleading when it evaluates a continuation
policy that will not actually execute. Other failures may reflect Greedy's
limitations, not an absence of a winning reply. Reject using this modeled
success/failure as an execution gate; do not fill omitted choices by seed hunting.

Next test the concrete handoff mismatch: original plan execution versus the
same plan with supported Greedy takeover **only after line abandonment and
only through the proposal turn**. Reuse all four roots/three existing suffixes
(12 paired continuations, 24 records), including the five uninterrupted winning
lines. Require exact full original-plan controls and identical pre-abandonment
traces, candidate proposals and opponent draws. For repair, derive Greedy seeds
from the frozen root plus first mismatching response index/action, consume no
inner-policy draws for heuristic-controlled decisions, and resume the policy
next turn. Preserve legal support and engine-health checks, log every takeover
action, and permit no further search. Compare against both original plan and
the already-recorded no-plan fallback; a rescued plan is not automatically better
than fallback. This is an exposed-root heuristic pilot, not production policy
or a strength claim. Two workers, 15-minute cap, no new deals or suffix tuning.

Identity: `7eddb92897a26c2c7aada3459bca0379fce84832f5ca68e2615e35012f0959e0`.
Report SHA-256: `9b666e216e426f3fa1b8b24970f1225d709d0b0f57a3fb6592bc73891aab9eef`.
Artifacts: `out/research/lethal-response-challenge-2026-10-12/`.

## Current-turn Greedy repair introduces new harms

`lethal-turn-repair-2026-10-12` completed 24 healthy records with no cutoffs.
All 12 original-plan controls reproduce exactly. Five uninterrupted lines also
remain byte-for-byte identical under the repair treatment. All seven takeovers
preserve the pre-abandonment prefix; takeover actions stay in encoded support,
consume no inner-policy draws, and end with the proposal turn.

Repair wins 9/12 versus original plan 8/12 and no-plan fallback 12/12. It rescues
all three game-5 losses, but turns game-0/stream-1 and game-6/stream-2 wins into
losses; game-0/stream-0 still loses. Same-turn wins rise 5→8, which does not
justify the new harms. Game-6/stream-1 remains a win but is delayed T8→T10.
Reject unconditional Greedy takeover; stop tuning it on these exposed roots.
These are correlated suffixes over four roots, not population win-rate estimates.

Next test a narrower, structural execution condition: the cached route may
execute only if **every opponent prompt on the route has exactly one action
in the same encoded support used by the checkpoint opponent**, and it is the
planned action. Replay the full route in the native shadow, validate all chosen
indices, require a same-turn win, then restore the root and both actual policy
RNG states. This avoids assuming the opponent will choose one favorable branch;
it is not a public-information proof because the shadow contains real hidden
state, and it is relative to encoded support rather than all engine-legal play.
Keep normal strict line invalidation and the existing event/search constraints.

Freeze this rule on a new panel with seed 2026101360: two new deal blocks × four
seat/deck assignments (eight paired full games), gate versus discard-all fallback,
plus two plain-policy exact controls (18 records). Both arms use the same
pre-battle trigger/root-keyed search, stop searching after the first proposal,
and have unchanged discovery caps (12 rollouts/search, two searches/turn,
eight/game). Audit opening-prefix non-overlap with previous collection panels;
no replacements if no proposal passes, no seed/threshold tuning. Require paired
proposal/pre-root transcript/RNG equality; rejected/no-proposal games must have
identical complete actual traces. Record accepted coverage, same-turn wins,
eventual wins/harms, paired turns, cutoffs and actual compute. No training or
production default changes; two workers and a 15-minute cap. Sparse acceptance
is a valid outcome, not a reason to enlarge the experiment adaptively.

Identity: `1f94b9c501e889ed5fbd313b2b18db431aabcb0db2b4d9546a65b1e69091a1d9`.
Report SHA-256: `bcb295cc6caf9c574d87a4565f380102d33a40fc33b9327d66439cdb3a483740`.
Artifacts: `out/research/lethal-turn-repair-2026-10-12/`.

## Structural route condition has a small independent timing benefit

`lethal-forced-response-gate-2026-10-12` completed 18 healthy records: eight
paired games and two plain-policy controls, with no opening-prefix overlap
against prior registered panels. Both plain controls, all paired proposal/root
RNG/prefix checks, and all 50 shared search-root comparisons pass. Rejected and
no-proposal games retain complete transcripts/results exactly; no cutoffs,
line abandonments or added losses occur.

Five of eight games produce a proposal. Two pass the structural condition:
game 0 wins T5 instead of T9, and game 2 wins T8 in both arms. Both arms win
6/8 overall; mean completed-game turns fall 9.0→8.5, entirely from the one
four-turn improvement. These are two independent deal blocks, not eight
independent trials or evidence of a population win-rate improvement. Both
accepted routes also have only one **engine-legal** action at each opponent
prompt, so these two passes are not caused by the encoded mask removing a
second legal choice. This observation does not remove the hidden-state issue.

Each arm uses 50 searches/548 discovery rollouts. Checking the five proposals
replays 495 decisions and takes approximately 0.30 seconds per arm as a measured
component; this is not an end-to-end speed comparison. No scale-up or default
promotion is justified yet.

Next audit information exposure on **all five** found roots, including rejected
ones, by exact discard-all replay (five records, no new deals or suffixes).
At each proposal compare candidate native/Python observation tensors and hash
them, count opponent concealed hand/field/banished/extra/deck state, and verify
that concealed represented rows use the unknown-card token. During the cached
route replay, log private-zone occupancy at opponent decision windows and
card-revelation/draw/move event categories. Keep the gate decision, candidate
proposal, real full policy transcript/result and actual RNG states unchanged.
This audits what the known-state simulator can access beyond actor inputs;
occupancy alone does not prove a causal decision change under an alternative
hidden state. No public-information safety claim or new execution heuristic.
Two workers, 15-minute cap. Resolve this information boundary before treating
the small timing benefit as deployable playing strength.

Identity: `4aba969c81b960f049031c005412c6096e75d5165750036e791eb7d750428b39`.
Report SHA-256: `53ee3f5b807b8e676aeb05197737c0f06e54354c9fde8b4758ea78c159ecb103`.
Artifacts: `out/research/lethal-forced-response-gate-2026-10-12/`.

## Actor inputs hide private cards; planner information remains unresolved

`lethal-information-audit-2026-10-12` completed five healthy exact replays.
Native and Python actor observations agree at every audited root; all represented
concealed opponent cards use the unknown token. Proposals, gate decisions,
complete real traces/results and policy RNG states reproduce exactly. No actor
encoding leak was found by these checks.

The beneficial game-0/T5 root has an empty opposing hand and no concealed field
cards. All 31 opponent windows on its accepted route also have empty hands.
The neutral accepted game-2/T8 root has one concealed hand card at the root and
all 18 opponent windows. Rejected game 1 has two concealed hand cards; rejected
games 6 and 7 each have two concealed spell/trap-zone cards. All five states
retain concealed main/extra decks. Event-type counts alone neither assign
reveals to their owner nor prove independence from hidden deck contents.

Thus the four-turn benefit is not evidence of reading an unknown hand in that
particular state, but the route gate can still consult opponent capabilities
computed from private state elsewhere. Separate actor-input correctness from
planner information correctness; do not call either accepted route a general
public-information certificate.

Next perform a bounded compatibility test at the accepted game-2 root, using
its one concealed hand card (19434243, present at initial opponent-main indices
4, 25 and 38). Rebuild the duel with one of those initial occurrences replaced
by Effect Veiler (97268402), Ghost Ogre (59438930), or Droll (94145021): one
original control plus nine predefined replacements, no replacement search.
Only retain counterfactuals if every recorded prefix action remains legal,
actor-visible/private-own card-table history and public counts/LP/turn/phase
match at every decision, and the complete root actor tensors are identical.
Record incompatible histories instead of silently substituting different seeds.
For compatible roots, replay the **fixed original route** where legal and record
whether a previously forced opponent prompt acquires alternative supported
responses. This tests fixed-route approval, not the full planner's new search
output or counterfactual win rate. State-compatibility checks do not establish
an alternative world's likelihood. No actual policy/model/default changes;
10 bounded native replays, two workers, 15-minute cap.

Identity: `804dae7945871199740c2f69be9f157ca694d186fd2b548584575b064395c432`.
Report SHA-256: `a1a58c6af501d5aba189b86ba7e7df0014089d4d4387e80032e2113b18fd9ffd`.
Artifacts: `out/research/lethal-information-audit-2026-10-12/`.

## Hidden-hand counterfactual is inconclusive; public reveal survives in actor history

`lethal-hidden-counterfactual-2026-10-12` completed all 10 native replays.
The original control reproduces the accepted route and native win. Six of nine
replacements cannot reproduce the recorded prefix actions. The other three
(initial index 4) preserve the checked prefix and gate, but change a card still
in the deck: the root hand remains 19434243. There are **zero compatible
hand-changing counterfactuals**. This does not measure hand-dependent gate
behavior. The raw `hand_changed` flag is not meaningful for incompatible rows;
derived `analysis.json` sets it to null, preserving original records.

A subsequent exact-prefix provenance audit corrects the interpretation of this
root. Power Vice Dragon (19434243) was explicitly shown to the candidate by
`ConfirmCards` at decision 338/T7, when it was the opponent's only hand card.
There are no opposing draws or hand moves between that reveal and root 363/T8.
Its identity is therefore available from public history, despite UNKNOWN in
the current card table. Native `ShuffleHand.codes` are privileged and are not
used to establish this knowledge; public reveal, public hand count and absence
of subsequent hand changes establish it.

The actual warm512 checkpoint uses Transformer history with a 64-token window.
At root 363 the reveal remains at zero-based event row 35 (card vocabulary
index 2927). All 195 candidate decision observations in the prefix have exact
native/Python event-token and mask parity. Thus this root demonstrates neither
missing reveal input nor loss through window eviction. Input availability also
does not prove the model uses that information correctly.

The earlier four-tensor actor audit covered cards/globals/actions/action mask,
not the entire history-bearing model input. Keep its narrower parity result,
but do not infer an information set from current UNKNOWN rows. The failed
replacement experiment cannot establish hidden-hand leakage. Hidden deck/extra
state and general search information boundaries remain unresolved, so no gate
promotion, model change or scale-up follows from these results.

Next extend exact native/Python event-history parity and reveal-window auditing
to all five existing proposal roots, including rejected ones. Reuse their
original action transcripts and checkpoint; no new games, deals, policy draws,
training or replacement search. Record public hand reveals, hand counts,
subsequent moves and whether reveal tokens remain in the actual 64-token input.
This is an input/knowledge audit, not a learned-memory performance test or a
public-information certificate for the search. Five bounded CPU replays.

Identity: `73657c35e224bbf8b4818646b2debb713a9f7d7d01a025b23821687ca13ab9ac`.
Report SHA-256: `82fabc58671b8e3c1d4d9f5f61633934c5f2a9d026e9a1b86c6271e4d2a87557`.
Derived history audit SHA-256:
`eec0f90482e55e4f4d8cab23e7e9099b885dbf9dcb8b50de5370172582fff319`.
Artifacts: `out/research/lethal-hidden-counterfactual-2026-10-12/`, including
`derived-audit-manifest.json` which binds the separate post-completion scripts
and evidence without altering the frozen study identity.

## Full history check covers all five existing proposal roots

`lethal-history-coverage-restart-2026-10-12` completed five bounded CPU prefix
replays. All **581 candidate decisions** match native/Python event histories
and masks exactly; public hand-count tracking also matches engine counts at
every replayed decision. This audit uses no policy sampling or new deals.

At the accepted game-0 root the opposing hand is empty. At accepted game 2,
the sole-card reveal remains in the actual input as described above. Rejected
game 1 has two hand cards and intervening hand changes after its last reveal,
so a past reveal alone cannot certify the current hand. Rejected games 6/7
have empty hands; this does not resolve their concealed field cards. No model
memory fix is justified by these checks. They test encoding and availability,
not whether the policy reasons correctly from the tokens.

The first audit invocation failed before producing any record because the
script queried `Duel.core` instead of `DuelSession.core`. Its source, identity
and STOP remain in `lethal-history-coverage-2026-10-12`; the corrected replay
uses a separate directory and identity. These short synchronous audits were
supervised directly, with a 120-second timeout per record and persisted
per-record output, rather than a background training service.

No new win-rate or average-turn estimate was produced. The last small paired
panel remains 6/8 wins in both arms, mean turns 9.0→8.5, driven by one game.
Open work is a public-history-consistent treatment of concealed field/deck
state before any deployment gate; repeating arbitrary hand substitutions at
the now-proven revealed-card root would not answer that question. Keep the
known-state search experimental and its defaults unchanged.

Identity: `4bcd7a55f2e8ab2ad48a63ebd82d9a87a6b29ac6c89ce8c29cb0e170fba6a48e`.
Report SHA-256: `f45eb6b55ea1a7ab4d54c4bebc6835192ea9cc9849a935b1fa7a213ce44d57ff`.
Artifacts: `out/research/lethal-history-coverage-restart-2026-10-12/`.

## Matched hidden worlds establish a concealed-card dependency

Four bounded follow-ups use existing roots and fixed random streams; they add
no training or model changes. Unlike the earlier four-tensor check, they match
the candidate's complete encoded event history at every prefix boundary,
public counts/current card table, and all six actor input tensors at candidate
decisions. Native/Python input parity is checked during replay. Opponent private
card codes in chosen actions may vary; legal action structure and subsequent
candidate observations must still match. This is encoded-history consistency,
not an exhaustive proof of equivalence for every possible human observation.

1. `lethal-public-worlds-restart` has five controls and 20 deterministic swaps
   of previously unexposed card codes in the initial opposing deck. Deck
   multisets remain fixed. All 20 preserve the checked history/inputs, but only
   change root deck order. All original routes and approval decisions remain
   unchanged. This does not test changed hand/field state or fresh search.
2. `lethal-private-zones` targets every initial occurrence of the hidden hand
   or backrow cards at roots 1/6/7, swapping with the two lowest-password
   alternatives in the same broad monster versus spell/trap category. Of 20
   variants, 12 are incompatible, four change only deck order, and four change
   concealed backrow with identical checked inputs/history. Swapping Lunalight
   Fusion leaves the original route unchanged. Swapping Dominus Impulse makes
   the planned opponent chain unavailable at decisions 134/144. That alone is
   an unavailable route, **not** a demonstrated approval flip.
3. `lethal-hidden-search` reruns the original bounded search in all four valid
   changed-backrow worlds plus their two controls. Both controls reproduce the
   original line hash, rollout/rejection counts and approval. With exactly the
   same original root chooser/opponent RNG, both Dominus swaps change approval
   **false→true** and produce different later routes. The other two swaps leave
   search/approval unchanged. The first proposed candidate action stays the same
   in all comparisons. These are two root cases from the existing panel, not
   independent population trials or evidence of higher/lower win rate.

This establishes that the experimental approval decision depends on concealed
card identity beyond the actor's inputs. Do not conflate the finding with the
publicly revealed Power Vice Dragon case. Matching the actor's information
while changing hidden state is necessary when auditing a proposed helper.
The failed first `lethal-public-worlds` preflight queried a nonexistent parsed
`sequence` field; using the documented list-slot index fixes it. The original
source/STOP is preserved and the corrected run has a separate identity.

## Conservative entry guard retains the old-panel timing benefit

The research-only `concealed_zone_guard.py` reads actor tensors before search
or policy peeking. It rejects opposing concealed hand/monster/spell-trap rows,
fails closed when the card table is full or its opposing hand count disagrees
with the public global count, and otherwise leaves the existing search and
route check unchanged. Thus the demonstrated hidden-backrow identity cannot
choose between entry and fallback. Known-from-history but currently concealed
hand cards are also conservatively rejected; this does not claim they are
unknown to the player. The implementation and its hash are in the frozen study
identity; production `LethalAgent`, models, training and defaults are unchanged.

`lethal-concealed-guard` completed 18 healthy records on the **reused** eight-game
panel (two deal blocks), including two plain controls. Both arms still win 6/8,
with mean turns 9.0→8.5 from game 0 winning T5 rather than T9. Other seven games
retain the same outcomes/turns; no new losses, cutoffs or line abandonments.
All fallback full traces match their prior originals, and paired proposal,
prefix, RNG and rejected/no-proposal trace checks pass. Only game 0 now produces
and approves a proposal. Per arm, searches fall 50→4 and rollouts 548→40 versus
the previous unguarded route-gate panel; 186 entry checks reject concealed
response zones. These are compute counts, not end-to-end speed measurements.

This fixes the demonstrated entry-decision dependency in the research variant,
not information-set planning generally. Hidden deck/extra effects and hidden
cards acquired after the root remain unresolved. No production promotion.
The next bounded check freezes this guard and runs the same-size independent
panel: seed 2026101370, two deal blocks/all four assignments, eight paired games
plus two plain controls, two CPU workers and a 15-minute cap. Check opening-prefix
nonoverlap; report the whole panel with no replacement deals or tuning. Old-panel
regression success justifies this confirmation, not larger training or a search
budget increase. Score coverage, wins/turns, new losses, line abandonments,
health and compute; no-proposal coverage is itself an informative result.

Artifacts below are under `out/research/`, all suffix `-2026-10-12`.

| Study | Identity SHA-256 | Report SHA-256 |
| --- | --- | --- |
| `lethal-public-worlds-restart` | `eb9ebca20142555cf4f77bc0572bb887403e7b05dfdd76863697dc4a4c466a3e` | `5845b7e05ff91496d0454d08f5bd7ed5dcaaa064c242418a17c5bc99cf9d2eb5` |
| `lethal-private-zones` | `584a83acf88664a80485bcadd573e5915349c9fc9916cdabea8801d8082f151d` | `e93c8d53c99232dbb5ed6e47a1dc011c5d24bddbfb9a1d3fac31024ead22feca` |
| `lethal-hidden-search` | `8dc03564fda57bf8212b80d5e9083c788c41d1b9758368f0306e26dee4c87bd4` | `43e757314db1f9eb5392ce7418c7615338264f583e01da9b1d91a04d35d41370` |
| `lethal-concealed-guard` | `7a4d3f8e891f267bd80f0f3555cc156fb6a84236a48dfa715419b3c8f8be4634` | `37bcc438c2f8cfb6b4c033b206d2908da279ae72f782511dae5bd5eecc85a0a2` |

## Fresh guarded confirmation and a missed Main Phase 2 finish

`lethal-concealed-fresh` completed all 18 healthy records: eight paired games
from two new deal blocks (seed 2026101370) and two plain controls. Four opening
prefixes are new relative to the registered prior panels. Both arms win 6/8;
mean turns change 9.375→9.125. Only game 6 changes, winning T12 instead of T14;
the other seven full traces and results are identical. There are two proposals,
one approval, 12 searches/122 rollouts per arm, 197 rejected entry checks,
12 shared-root RNG parity checks, and no new losses or line abandonment. This
is a small timing benefit on two deal blocks, not a win-rate improvement or
proof of general playing strength.

The derived `route-information.json` replays both proposals with native zone
checks. Game 5's rejected route has 141 prompts and public graveyard response
choices; game 6's approved route has 39 prompts and 13 opponent prompts, each
with one legal/supported action. Neither route draws cards or encounters a new
concealed opposing hand/field response window. This does not certify all hidden
deck/extra effects. Derived files have their own manifest, preserving the
original study identity and results.

In game 6, both arms attack identically. At decision 556/T12 the opponent has
600 LP and the candidate has 6800. The sampled policy ends the turn; the plan
enters Main Phase 2, eventually sends Fiendsmith's Lacrima (46640168) to the
graveyard as Link material, then inflicts 1200 effect damage. The plan includes
Sequence's equip effect and Requiem's summon effect, but they should not be
assumed necessary merely because search included them.

`lethal-phase-exit-restart` isolates phase entry on this selected root with the
original opponent RNG and two fixed opponent suffix seeds. All nine runs have
the same six root input hashes, candidate RNG, and root probabilities. Original
fallback and full-plan streams exactly reproduce the fresh controls. Results:

| Treatment | Original suffix | Suffix 1 | Suffix 2 |
| --- | --- | --- | --- |
| Original policy | Win T14 | Win T13 | Win T14 |
| Force Main Phase 2 only | Win T14 | Win T13 | Win T14 |
| Remaining cached plan | Win T12 | Win T12 | Win T12 |

Main Phase 2 already has probability **88.369%** at the battle exit, versus
11.631% for end phase. The observed exit is a sampled minority action, not a
preference for ending there. However, all three phase-only continuations enter
Main Phase 2 and immediately end at decision 558. The cached-plan treatment
also changes candidate RNG consumption; this comparison alone cannot attribute
success to every action in its route. These suffixes share one root and are not
independent games. The first preflight's `__name__in` typo failed before writing
a record; its original source/STOP is preserved separately.

## Initiating the combination is sufficient on the selected root

`lethal-main2-sequence` uses four original-suffix continuations, with all forced
choices consuming the ordinary policy sample for that prompt. It records
probabilities without extra RNG draws and exactly reproduces phase-only's full
trace. At decision 558 the policy probabilities are:

| Action | Probability |
| --- | --- |
| End phase | 99.660921% |
| Link Summon Requiem (2463794) | 0.018839% |
| Activate Sequence (49867899) | 0.002370% |

After forcing phase entry and **only initiating Requiem**, ordinary policy
selects Lacrima as material, accepts its graveyard damage effect (yes probability
96.861%), returns Sequence, and wins T12. Additionally forcing the material,
or the material plus burn acceptance/return target, produces the exact same
full transcript. Phase-only still wins T14. All four records are healthy.

This localizes a concrete policy failure to starting a winning combination;
it does not establish whether training credit, representation, or coverage
caused the low probability. It also shows the longer searched route contains
unnecessary actions. Do not globally forbid end phase or add a card-specific
production rule based on this case. The next bounded diagnostic will test
whether the existing generic search can recover the finish at battle exit and
Main Phase 2, with the same 12-rollout budget and conservative entry guard.
No new deals, seed selection, model training, or default promotion is justified
by this selected-root result.

| Study (suffix `-2026-10-12`) | Identity SHA-256 | Report SHA-256 |
| --- | --- | --- |
| `lethal-concealed-fresh` | `dcf531e6f5b5f11f3e050f023898f8d462f57945c161442b53e3d67a3b7e1039` | `d0a07846ee55b3d227463d8944afd5259fb484f49d410c13ece9626f33762443` |
| `lethal-phase-exit-restart` | `3336dc99a6ae7a03f13a075a3b25a39878e4012b9abef38763ff0778a2f0ea95` | `cbd47a59ab8c1169a8d8b94ad3bbc2a7fc796ac791f1d27ed2fb3b4c4f9d6329` |
| `lethal-main2-sequence` | `4b650d35750ffc9851b5ed5e782f331aa451bcf999ba57155eb40fe2721da7ee` | `8925b765f9919258818db37fed1e363b5c3735b8f096e1472ffc071378d2acef` |

## Generic search recovers the finish without a card-specific override

`lethal-main2-search-restart` replays the same exact policy prefixes to decisions
556 (battle exit) and 558 (Main Phase 2 after the phase-only intervention).
Both pass the existing conservative actor-input entry guard. Existing generic
search, with the unchanged maximum of 12 rollouts and deterministic root RNG,
finds a winning line on its **first rollout at both roots**. Native replay
verifies the same-turn win and the existing forced-response gate accepts both;
all six/five opposing prompts respectively have exactly one engine-legal action.
The routes have 23/21 decisions and still include unnecessary setup. This is
proposal-generation evidence, not a claim that the search minimizes line length.
Both policy generators and the root are restored after search. The initial
preflight omitted `snapshots=True`; it failed before recording a search result.
Its original source/STOP is retained separately.

Identity: `f107a58e4cce21daf6b59057d7b72596f6e1bf482d3a6af238db933f6c4f8749`;
report: `77e57e93d0b8bec35538560d7b1475f5264b9225157fd7e0686fdc757978349c`.

The next bounded regression reuses all eight fresh-panel games, including the
seven not improved by the prior helper. Trigger search only when the sampled
policy is about to enter battle or end the turn, including Main Phase 2. Peek
and restore policy RNG; retain the concealed-zone guard, first-proposal stop,
12 rollouts/search, two searches/turn and eight/game, and forced-response gate.
Run paired discard/execute arms with identical root RNG and compare with all
prior baseline traces and guarded results. This tests search allocation and
retention on a reused panel, not independent generalization. No replacement
seeds, larger budgets, card-specific rules, production changes or promotion.

## Exit-trigger whole-panel and retention regressions

Both unchanged-budget regression studies completed 16 healthy records each,
with all eight paired games retained and every fallback trace exactly matching
its original. The exit schedule peeks/restores the ordinary policy RNG and
searches only before `battle_phase` or `end_phase`, including Main Phase 2.
It preserves the entry guard, first-proposal stop, route gate and all caps.

| Reused panel | Baseline wins / turns | Previous guarded wins / turns | Exit-trigger wins / turns | Rollouts per arm, previous→exit |
| --- | --- | --- | --- | --- |
| Fresh-panel regression | 6/8 / 9.375 | 6/8 / 9.125 | 6/8 / 9.125 | 122→110 |
| Earlier-panel retention | 6/8 / 9.000 | 6/8 / 8.500 | 6/8 / 8.500 | 40→28 |

The fresh-panel regression retains both proposals and the one approval. Game 6
now proposes at decision 556 and still wins T12; the other seven execute traces
are identical to the previous schedule. There are 17 shared-root checks and
11 searches per arm instead of 12. On the earlier panel, game 0's T9→T5 gain is
retained, every execute trace is identical to the previous schedule, and there
are five shared-root checks and three searches instead of four. Neither panel
has new losses or line abandonments. These are reduced rollout counts, not
measured end-to-end latency or additional win-rate/playing-strength gains.

| Study (suffix `-2026-10-12`) | Identity SHA-256 | Report SHA-256 |
| --- | --- | --- |
| `lethal-exit-trigger` | `9a5202d59c3743991fd6bef25b92a69d6c9d4546612203b0a59e5ef6e5da0379` | `54f853a285ce4c404bc7b032aed74e9f3bc6e9137a86e56336d8280053112223` |
| `lethal-exit-retention` | `db054e0b21b9ed9bbdf1b0a531340deffc3ff7d2439633c11ddc94d21c7dae61` | `502017f372484bb418e1ecd834482abad24587abc25455daaa2f6e62a8bf612d` |

Two-panel retention supports one same-size independent confirmation, not a larger
training run: seed 2026101390, two new deal blocks/all four assignments, eight
paired games plus two plain controls, unchanged source/model/search caps, two
CPU workers and a 15-minute cap. Check opening-prefix nonoverlap against all
prior panels including seed 2026101370; keep all outcomes, with no replacement
seeds or tuning. Report wins/turns, proposal and approval coverage, new losses,
line abandonment, health and compute. Hidden deck/extra and future information
limitations still prevent a production safety claim; no default promotion.

## Independent exit-trigger confirmation and first-action failure

`lethal-exit-fresh` (fixed seed 2026101390) completed all 18 healthy records,
with four new opening prefixes and zero registered prior-panel overlap. Both
arms win **5/8**; mean turns change **8.875→8.375**, and mean decisions
485.625→462.750. Game 2 wins T6 instead of T10. Games 3/5 remain wins on T11/T6
but use 18/21 more decisions than baseline. The other five full traces are
identical. There are three proposals/approvals, nine searches and 78 rollouts
per arm, nine shared-root checks, and no new losses or abandonment. Compare
arms within this panel; its different baseline win count is not a regression
relative to the preceding deals. This supports one faster existing win, not
an added win or a population playing-strength claim.

A native replay of all three accepted routes verifies same-turn wins and no
opposing choices with more than one engine-legal action. Games 2/3 have no
concealed opposing hand/field windows; game 5 has 24 such windows after a
public face-up monster (60764609) returns to its owner's hand at decision 284.
The recorded Move establishes its known provenance; concealed current encoding
alone does not establish unknown identity. No route draws cards. Full public
information certification, including hidden deck/extra effects, remains open.
Derived route/Move audits and analysis have a separate manifest.

The two same-turn but longer wins show that successful search routes are not
necessarily efficient training demonstrations. `lethal-first-action` therefore
covers **all three approvals**, comparing exact original baseline, exact cached
plan, and only the first planned action followed by ordinary policy. All nine
continuations are healthy and all six baseline/full-plan controls reproduce
original traces/results exactly.

| Selected game | Original policy | Complete plan | First planned action only |
| --- | --- | --- | --- |
| 2 | Win T10, 627 decisions | Win T6, 405 | **Loss T8, 511** |
| 3 | Win T11, 463 | Win T11, 481 | Win T11, 461 |
| 5 | Win T6, 287 | Win T6, 308 | Win T6, 295 |

In game 2, the first intervention activates Branded Fusion (44362883), whose
original probability is 0.760% versus battle entry 94.696%. Ordinary policy then
passes Ecclesia's chain, summons Albion the Branded Dragon (87746184), and
rejects Albion's further fusion effect with probability **99.524%**. These
observations localize divergent choices; they do not independently prove which
one causes the loss. In particular, first-action-only consumes the original
policy draw, whereas cached plan actions consume none. Do not attribute all
outcome difference to commitment length without checking this RNG distinction.
No first-action teacher labels or card-specific production rules are promoted.

Next, on these same three roots only, compare cached first-action handoff,
handoff after the initial effect transaction (next own SelectIdleCmd), and
handoff at battle entry (first own SelectBattleCmd), plus exact full-plan
controls. All cached choices use the same no-sampling convention. This tests
whether policy can complete a plan from a meaningful intermediate boundary
and whether the first-action failure persists under matched RNG semantics.
No new seeds, fullgame scale-up, training, or default changes.

| Study (suffix `-2026-10-12`) | Identity SHA-256 | Report SHA-256 |
| --- | --- | --- |
| `lethal-exit-fresh` | `bdc1b8b57ba86692ae242e25010eb6ddeb8911f2ba48b993fb99bf3f54503993` | `f34d79e1b60e52341e13c4c36e94f5add90090cd67ee565432ce2b9fc1ed7165` |
| `lethal-first-action` | `1820d37dfd6c4fcfa43e5bd557a3915d68b7cd68757fc0fab3172832e0268afe` | `65c23527557b6ad119684fceae4435c3caeb79edf451ab15ca72f94d28440e6a` |

## Matched cache convention changes the first-action conclusion

`lethal-semantic-handoff` completed all 12 healthy continuations on the same
three approved roots. Every guided prefix uses cached choices without consuming
candidate sampling RNG; intended handoff is distinct from unexpected plan
abandonment. All three complete-plan controls exactly reproduce their originals.

| Game | First action, consuming policy draw (prior study) | Cached first action | Initial transaction completed | Battle-entry handoff | Full cached plan |
| --- | --- | --- | --- | --- | --- |
| 2 | Loss T8 | Win T8 | Win T8 | Win T6 | Win T6 |
| 3 | Win T11 | Win T11 | Win T11 | Win T11 | Win T11 |
| 5 | Win T6 | Win T6 | Win T6 | Win T6 | Win T6 |

The prior loss is sensitive to candidate RNG consumption: cached-first-action
game 2 chooses a different fusion route and wins T8. It is therefore **not**
sufficient evidence that Branded Fusion's first action is itself harmful.
Neither first-action convention recovers the T6 finish. After the initial
transaction, policy at decision 342 gives battle entry 99.989% and almost never
normal summons Incredible Ecclesia (55273560), which the successful preparation
uses. These are concrete policy divergences, not a proven training-credit or
representation diagnosis.

At battle entry the original policy reproduces the **entire full-plan trace**
on all three roots, including attack order and targets. Override counts drop
67→52, 30→15 and 34→28, with no changed outcomes or abandonment. On these exact
prepared boards, combat execution works; early handoff fails to finish the
preparation. Different handoff states still imply different RNG positions and
are not interchangeable causal states. This is a three-root diagnostic, not a
population claim or readiness to train from every cached action.

Identity: `f6a5e4469d6fb7a3c4f01c3e588d37ef713afa4b5aa7cbacf73d6591c6bef93e`;
report: `d22904a2acc59c7603d530cd7334ea4b34a9f6dc404f9905fd71f3ef9b5a4e0f`.

Next audit all recorded own idle-command boundaries in the three successful
Main Phase prefixes, forcing battle entry there and letting ordinary policy
finish. Use one fixed candidate battle suffix RNG per game and one fixed
opponent suffix RNG, unchanged across boundaries; retain exact full-plan
controls. This exhausts existing preparation boundaries rather than adding
seeds or examples, and tests whether late setup actions are unnecessary and
where the first demonstrated finishing board appears. Do not interpret a
failed rollout as proof that no winning continuation exists, or promote
card-specific rules/teacher labels before checking the resulting branches.

## Preparation trimming and battle-first search allocation

`lethal-preparation-trim` completed all 17 healthy records: every recorded own
idle-command boundary on the three approved routes (14 boundaries), plus three
exact complete-plan controls. A fixed candidate battle RNG and fixed opponent
suffix RNG are reused across boundaries within each game. At each boundary,
force battle entry, then return combat execution to ordinary policy.

| Game | Earliest tested boundary that finishes this turn | Full-plan decisions | Trimmed total decisions |
| --- | --- | --- | --- |
| 2 | 372, after two preparation transactions | 405 | 401 |
| 3 | 433, original search root | 481 | 463 |
| 5 | 258, original search root | 308 | 287 |

Game 2 does not finish this turn from boundaries 313 or 342 under the fixed
suffix; both eventually win T10. Boundary 372 finishes T6, omitting the final
set. Games 3 and 5 can already win directly from the original root. All records
win eventually, without unexpected abandonment. These selected continuations
show removable preparation, not minimality or proof that failed prefixes have
no winning line. The earlier semantic-handoff result and this audit together
argue against treating every action of a successful search as a useful label.

`lethal-battle-first` changes only the first rollout chooser: prefer a legal
battle-phase action, otherwise delegate to the original Greedy chooser. The
remaining rollouts, per-root RNG, 12-rollout budget, health/event filters and
opponent-response gate remain unchanged. Six fixed-root searches cover all
three approvals and their exact original search controls; all are healthy.

| Game | Original → battle-first route prompts | Original → battle-first rollouts | Native winning route passes gate |
| --- | --- | --- | --- |
| 2 | 92 → 92 | 2 → 2 | Both; identical route hash |
| 3 | 48 → 34 | 1 → 1 | Both |
| 5 | 50 → 29 | 3 → 1 | Both |

Thus the modified chooser avoids the two unnecessary Main Phase expansions
while preserving the preparation-dependent route on this diagnostic. The
shorter game 3 route still chains Maxx “C”; shorter is not necessarily minimal
or strategically justified at every action. The first proposal still stops
search even if its response gate rejects it, so a new early proposal could
preempt an older useful one elsewhere. Hidden deck/extra information remains
unresolved; no production default, policy, critic, or teacher labels change.

Next run paired discard/execute regression on **all eight existing games** of
`lethal-exit-fresh`, not only the three approvals. Replace only the first
rollout per search; keep the concealed-zone guard, root streams and maximum
12 rollouts/search, two searches/turn, eight/game. Require exact ordinary-play
traces for discard/rejected routes and matched proposals/prefix RNG between
paired arms. Compare wins, turns, decisions, gate acceptance and abandonment
against the existing panel. This is bounded implementation regression, not
independent evidence of improved strength or grounds to scale training.

| Study (suffix `-2026-10-12`) | Identity SHA-256 | Report SHA-256 |
| --- | --- | --- |
| `lethal-preparation-trim` | `3e22826a9ad172717add940d15547d473ea99604b26fb9bc5a1dc01def0ef3b6` | `de115bd3c4db0325748d67cf005419a27429af2566d736888d2fb418ca456a20` |
| `lethal-battle-first` | `4a911e0cb6440aa668993d09d500b5558fe675c5ab18cf0c7df384746f84c5d2` | `7c719912922a4e72524747b0246f8b7920fe96a3a63f7139c74bbd5bf7a41038` |

## Battle-first full-panel regression

`lethal-battle-first-regression` completed 16 healthy records on all eight
existing exit-fresh games, with discard/execute pairing. All eight discard
traces/results exactly reproduce ordinary policy, all paired proposals and
prefix RNG match, and nine shared search roots pass parity checks.

| Metric, eight games | Ordinary policy | Previous search | Battle-first search |
| --- | --- | --- | --- |
| Wins | 5/8 | 5/8 | 5/8 |
| Mean turns | 8.875 | 8.375 | 8.375 |
| Mean decisions | 485.625 | 462.750 | 458.375 |
| Rollouts | 78 (discarded) | 78 | 76 |
| Applied overrides | 0 | 131 | 102 |

All three approved roots remain approved. The preparation-dependent T6 finish
in game 2 reproduces its previous complete trace. Game 3 drops 481→467 decisions
but remains four decisions longer than ordinary policy because the search
still chains Maxx “C”. Game 5 drops 308→287 and exactly matches ordinary policy.
The other five games never pass the concealed-zone entry guard and remain
unchanged. There are no new losses, desynchronizations, or plan abandonments.
This supports reduced unnecessary actions in the tested implementation, not
higher win rate or further reduction in turns. The panel has no rejected
proposal, so it does not cover new first-proposal rejection preempting a later
useful route.

Next reuse **all eight games of lethal-exit-trigger**, which include the Main2
finish and a rejected proposal, for the same 16-record paired regression. Keep
12 rollouts/search, two searches/turn, eight/game; compare original trace/RNG
controls and outcome/approval retention. No new deals, training, larger budgets,
or production promotion. Hidden-information certification remains a separate
unresolved requirement even if this regression passes.

Identity: `cb0fbabcd64bd570b36e01a44f59440f76c8c0c1146720457280e4e40c551ff0`;
report: `312aecd2d88f3090b9a41a4edc524a9eb2376c96ae6adca5ffca04c7e46c2033`.

## Battle-first retention: Main2 survives, rejection still stops search

`lethal-battle-first-retention` completed 16 healthy records on all eight
existing exit-trigger games. Discard traces exactly reproduce their original
policy controls and ten shared-root checks pass. There are no new losses,
desynchronizations or unexpected plan abandonments.

| Metric, eight games | Previous search | Battle-first |
| --- | --- | --- |
| Wins | 6/8 | 6/8 |
| Mean turns | 9.125 | 9.125 |
| Mean decisions | 543.375 | 543.375 |
| Searches / rollouts | 11 / 110 | 10 / 99 |
| Applied overrides | 17 | 27 |

The same Main2 effect-damage finish occurs T12 at 579 decisions. Search now
approves at decision 538 instead of 556, preserving the finish with an earlier
cached prefix. Wholegame traces differ only in two placement choices, but ten
more actions are overridden. Seven other complete traces are unchanged. Fewer
rollouts therefore do not imply fewer overrides or stronger play.

Game 5's candidate at decision 558 is still rejected: its replay wins in the
sampled continuation but includes multiple supported opponent responses. With
battle-first it appears on rollout 2 instead of 1. The research wrapper stops
at this first proposal, leaving ten of its twelve rollouts unused. This is a
confirmed search-coverage gap, **not evidence yet that a safe lethal was missed**.
The conservative response gate is not relaxed.

Next test rejection-aware search on exactly five existing roots: this rejected
root, the retained Main2 route root, and the three approvals from the preceding
panel. Ten root searches compare exact battle-first controls with continuing
after rejected proposals, retaining the same root streams and max12 rollouts.
Gate checks add replay work, which must be counted separately; they do not buy
extra rollout seeds. Require accepted-control route parity and exact rejected
first-candidate reproduction before interpreting later proposals. No new deals,
actor/critic training, or production promotion. This remains a diagnostic in the
known engine state; hidden-information certification is still unresolved.

Identity: `4411f8c98d4fbaaf46c580b9d70d664ad3d73b5a4b1ed7bf5430a0ecfe335cf6`;
report: `6e513de68f6fa5f27b8ac62f0555c61efb4b60ca2d18dae9dcc68029605cb522`.

## Rejected-proposal continuation finds a previously skipped route

`lethal-rejection-continuation` completed ten healthy fixed-root searches.
All five controls reproduce the previous proposal hash, rollout count and gate
outcome. Retry reproduces each first candidate exactly. All four previously
approved routes retain their exact hashes and rollout counts.

At the rejected root (game 5, decision 558), rollout 2 produces the original
123-prompt rejected route; rollout 3 produces another rejected 155-prompt
route; **rollout 4 produces an accepted 177-prompt same-turn winning route**.
No additional rollout seed or relaxed gate is used. The original stop therefore
skips an available candidate that satisfies the existing known-state response
gate. This proves a coverage problem for that gate, not a public-information
win guarantee or demonstrated stronger wholegame policy.

The retry performs three inner route replays, totaling 455 decisions and about
0.35 seconds in this diagnostic, plus the final verification replay. This is
extra work despite the unchanged maximum twelve rollouts. The new route is
longer than the rejected candidates and is not automatically a good teaching
trajectory. Draw/event filtering, action support and native health checks stay
unchanged; concealed deck/extra information remains unresolved.

Next run all eight existing battle-first-retention games as sixteen paired
discard/execute fullgames. Integrate the same rejection check into rollouts;
stop at the first accepted proposal, continue within the twelve-rollout cap
after rejection, and if a root exhausts its budget without acceptance, allow
later existing triggers within the unchanged two/turn and eight/game caps.
Record every gate replay separately. Require exact ordinary-policy fallback,
paired proposal/prefix RNG parity, preservation of the Main2 finish, and actual
completion of the newly accepted route without abandonment. No new deals,
training, larger search budgets, or production promotion.

Identity: `33d8404552737c62470b6a60865cedfb76f15eefe73e5d57561fa6297484f859`;
report: `b1633e398d35c852be8f1fea9d1d4b2e8c4e262b171296d6313ae1cc2dce7caa`.

## Rejection-aware fullgame execution: earlier win, longer action sequence

`lethal-rejection-fullgame` completed all sixteen healthy records. Every discard
trace/result reproduces ordinary policy exactly; ten paired roots match. The
new 177-prompt candidate executes without abandonment and wins game 5 on T8
instead of T10. All seven other executed fullgame traces remain unchanged,
including the prior Main2 finish. No new losses or desynchronizations occur.

| Metric, eight reused games | Previous battle-first | Rejection-aware |
| --- | --- | --- |
| Wins | 6/8 | 6/8 |
| Mean turns | 9.125 | 8.875 |
| Mean decisions | 543.375 | 547.000 |
| Rollouts | 99 | 101 |
| Applied overrides | 27 | 156 |

Thus the coverage fix recovers an earlier finish in actual execution, but does
not improve win count and adds 29 decisions in the changed game (706→735).
Inner gate checks replay 496 decisions across the execute arm, in addition to
final verification. Maximum search limits are unchanged; actual work increases.
These reused games are implementation evidence, not an independent strength
estimate or hidden-information certification.

The new suffix contains 177 prompts: 95 passes, 26 selections, 16 placements,
11 position choices, 11 activations, five special summons, one normal summon,
two sets, and a single final attack (plus chain/yes/option/battle-entry choices).
There are no cancel actions. This does not prove absence of every possible
loop, but the added length cannot be explained as cancel repetition. The actor
visits twenty idle-command boundaries before battle; whether these preparations
are necessary is still untested.

Next exhaust those **twenty existing boundaries**, plus one exact full-plan
control, using the established preparation-trim protocol. At each boundary,
force legal battle entry and return to ordinary policy with a fixed candidate
battle RNG and fixed opponent suffix RNG, matched across boundaries. Preserve
cached prefix semantics. This is 21 bounded continuations from one existing
route, not new deal sampling or training scale-up. Compare same-turn finish,
terminal outcome, total decisions, legality and health; a failed sampled suffix
does not prove no winning continuation exists. Do not convert the long route
into labels or promote defaults before this diagnosis and information audit.

Identity: `e5c28d70da4e4552fd2fa39fb3fbb5de791470842f7be57a46b913d3f8c35e40`;
report: `046746b0bd1d739e6771c2353745f78d07c5220929a0560195a20f7deb450392`.

## Recovered-route trimming is effective but non-monotonic

`lethal-recovered-trim` completed all 21 healthy continuations: twenty existing
idle boundaries and one exact full-plan control. All trimmed continuations use
the same candidate battle RNG state and fixed opponent suffix seed; none
abandons the intended prefix. All eventually win, while thirteen trims finish
on the original T8.

The earliest observed T8 finish is at decision **602**: total decisions fall
**735→641**, and overrides **129→33**. This is already shorter than the earlier
ordinary-policy T10 win at 706 decisions, but the suffix RNG differs from that
original baseline; do not present this as a new independent paired win-rate
result. Boundaries 558–590 do not finish T8 under the fixed suffix.

Finishing ability is non-monotonic along this route: 602, 608 and 616 finish T8;
624 and 631 instead win T10 and T9; all tested boundaries from 644 onward finish
T8 again. At 616 the recorded next setup action special summons The Bystial
Lubellion. The sampled battle options at 624 no longer include Crimson Blade
Dragon (3294539), which was available at 616. This localizes a resource change,
but exact tribute/position semantics and causality still require native board
and event inspection; it is not a general rule against that summon.

These thirteen shorter sampled-policy wins have **not yet** passed the same
singleton-opponent-support route gate as the original long plan. Next replay
all thirteen T8 trims plus the original accepted control, applying the unchanged
gate and strict event filter. Record branching opponent windows, current public
board at handoff and battle decisions, and prefix events around the non-monotonic
boundaries. This is fourteen replays of already collected winning traces, with
no new continuation seeds or search budget. A shorter line that loses gate
acceptance is not promoted as a reliable teacher or production override.

Identity: `8c79c18c5ad853405149568dc17fbd98148d106d446d4763f89ea798c1595f4f`;
report: `c983b15ca525a1b63175f06d95f0102d32225d5010e043cb04d9a50b1fff67a6`.

## Shorter winning traces do not all retain the response gate

`lethal-trim-response-audit-restart` replays all thirteen winning trims plus
the original control exactly. All fourteen records are healthy and strict-clean.
The initial audit stopped in preflight on a research-script `__name__in` typo;
its STOP and identity remain frozen dependencies of the corrected restart.

Eight trims (602 through 683 among the tested winning boundaries) fail the
singleton-opponent-support gate. The shortest observed win, 641 decisions from
602, exposes a two-choice chain window and a five-choice placement window.
Failure of this sufficient gate does **not** establish a successful opponent
refutation. Five later trims pass, beginning at **692**, which finishes T8 in
**705 total decisions**, versus 735 for the full plan. These are still results
in the known engine state, not a public-information certificate.

Native boards and Move events explain a concrete change near the non-monotonic
boundaries: at 616 Crimson Blade Dragon is face-up attack (2400 ATK); at 618 it
moves to the GY with release/cost reason, and at 620 The Bystial Lubellion arrives
in face-up defense. It is not available as an attacker at the next battle entry.
The later Synchro Summon produces an attack-position Bystial Dis Pater (3500 ATK).
This shows temporarily lost attacking resources during the sampled setup;
it does not establish that summoning Lubellion is generally wrong or that this
resource change alone explains every outcome difference.

Next enumerate supported opponent-response branches for exactly two existing
handoffs: shortest observed trim602 and earliest singleton-approved trim692.
Replay the same cached prefix and fixed candidate/opponent RNG; candidate policy
remains unchanged. At each opponent prompt with multiple supported choices,
branch over all of them. Cap each tree at16 native replays and200 suffix decisions;
report unresolved branches as incomplete, never as wins. Require an exact
recorded-trace leaf/control and count all alternative outcomes. This is a small
known-state refutation test without fresh deals or seed selection; no default,
teacher-label or public-information promotion follows merely from one tree.

Identity: `afabee770c56723bfe02e02ca9fdf0f869bccf0e4db8e43d6621f6e44c9b3743`;
report: `da18e4c58c6cb6fd405d4807ff490768c25e757383c61b78f3503cd5ea99306e`.

## Shortest trim survives the supported response tree

`lethal-trim-response-tree` exhaustively closes both preregistered trees within
the bounds. The trim692 control exactly reproduces its single T8 winning leaf.
Trim602 visits **15 replay nodes and 11 terminal leaves; every leaf wins T8**
and is engine-healthy and strict-clean. One leaf exactly reproduces the recorded
trace. Opponent choices are Lacrima the Crimson Tears' chain/pass windows and
five placement options: first activation gives five wins at641 total decisions;
passing then activating gives five at643; passing twice gives one at631.

The previous singleton-support gate rejects this shorter continuation even
though none of its supported responses refutes it. This establishes an overly
conservative sufficient condition for this known state and fixed candidate RNG,
not a general safety guarantee. Eleven tree leaves are not eleven independent
games; unsupported legal responses and alternative hidden worlds remain outside
scope. Candidate actions also adapt to each branch through the unchanged policy,
so this result cannot be substituted for one cached linear action sequence.

Next isolate the dependence on resetting candidate battle RNG: repeat both cuts
with the old reset as exact controls and with the candidate's carried RNG state
(no battle reseeding). Keep opponent suffix RNG unchanged and enumerate all its
supported branches again. Four trees, at most16 native replays/tree and200 suffix
decisions; no new deals, random seeds, training or larger search. Require exact
old controls; report losses, turn exits and incomplete trees explicitly. A carry
failure blocks treating the earlier fixed-suffix result as an ordinary-policy
handoff. Success still does not certify hidden-information safety or strength.

Identity: `183d17d47dddeec44579a26534f6255d00c8ab3feb6253e7ecb0f7d71831bbe2`;
report: `b561ef58c773ee7c9533b469df2a35c0e59d46f9ac1d0ab7f92dae600f7d281b`.

## Carried policy RNG preserves the shorter response tree

`lethal-trim-rng-carry` completes all four preregistered trees. At602 both
reset and carried candidate RNG yield11/11 T8 winning leaves; at692 both yield
1/1. The RNG state hashes differ, but every corresponding replay trace is
identical. All32 replay nodes are healthy and all terminal leaves strict-clean.
Thus this observed handoff does not require the artificial candidate battle
reseed. This is a comparison of two RNG states on one selected deal, not a
seed-robustness estimate or a public-information certificate. Opponent RNG was
held fixed, with every supported opponent response enumerated.

Next compare the carried learned-policy suffix against the existing GreedyAgent
suffix at both handoffs. Greedy gets exactly the encoded supported actions and
uses its normal base-stat heuristic; its known limitations (base rather than
current ATK/DEF, proactive chaining) remain part of the test, not assumptions of
correctness. Initialize it with the already recorded candidate seed and switch
at the first own battle prompt; keep the setup prefix and opponent handling
unchanged. Four trees, at most16 replays/tree and200 suffix decisions. Require
exact learned-policy controls and report every failed or incomplete branch.
This tests whether a simpler executable continuation suffices before building
more complex contingent-plan machinery. No fresh deals, seed selection, training,
production/default change or blanket combat mask.

Identity: `3fbacc8181673aaaa83df94997cf48467be55439f7064c3b8b5911b1cabcbe79`;
report: `0d3543e10fc18534d53eea8f84600fc0bdcbe5617bd36e496193da65bf7a8590`.

## Existing heuristic also completes every supported response

`lethal-trim-heuristic-suffix` completes all four trees with healthy native
execution. The carried-policy controls exactly match the preceding study.
The supported GreedyAgent suffix wins all11 leaves at602 and the single leaf
at692, with the same T8 outcomes and decision counts as the learned policy.
At602 its first attacker is Bystial Druiswurm rather than Crimson Blade Dragon,
so this is not merely replaying the same choices. There is no demonstrated
win-rate improvement or general superiority of the base-stat heuristic.

This supports testing a reusable bounded verifier before changing the live
handoff. Next implement DFS on native snapshots, copying the heuristic's small
mutable state at each opponent fork. Compare both existing roots against every
recorded greedy leaf, including exact actions and outcomes. Add two602 controls
with node budget1 and suffix-decision budget1: either must be incomplete and
reject certification. After each check replay the original suffix to verify
that simulation restored the caller's native state. Four jobs, no new games or
seeds; normal limits remain16 nodes and200 suffix decisions. An accepted result
is still conditional on the known state and encoded support, not a claim about
hidden worlds. This verifier remains research-only until broader execution and
information-boundary checks justify integration.

Identity: `93cd2a306f1fbac7a514dc475795f8088fed6c9ecc00e1fa5ad82a31aa230a88`;
report: `1cfc5f26a4a744a5a19de688afb5380b21ecc161ca5d6e90e3cb0f3b10cbb119`.

## Snapshot verifier preserves tree outcomes and caller state

`lethal-snapshot-tree` passes all four checks. The692 control reproduces its
single leaf using11 simulated suffix decisions. The602 verifier reproduces
all11 leaves with exact actions and outcomes in15 nodes /185 simulated decisions.
Both node-cap and decision-cap controls remain incomplete and reject
certification. After all four audits, replaying the original suffix exactly
reproduces the original outcome, checking native-state restoration beyond the
pending decision signature. This is a research component, not a production gate.

Next test actual automatic handoff on the same existing accepted plan: full-plan
control, fixed692 control, and an automatic scan in chronological plan order.
At most20 existing own idle boundaries with legal supported battle entry are
checked; each gets at most16 nodes and200 suffix decisions. Temporarily force
battle, verify the supported greedy response tree, restore state, and stop at
the first complete all-winning certificate. Execute that handoff against the
ordinary opponent with both real policy RNG streams unchanged. Require exact
full/prefix controls and an actual suffix matching a certified leaf. Incomplete
or losing checks retain the original plan. Three execution arms, no new deals,
seeds, training or production promotion; hidden-world safety remains unresolved.

Identity: `bb121fc7a47d36681b93e1e6d99710a7807aacf58bc5174ce8e458df9deb333c`;
report: `3f65087be6b54ce8ef6cd8e563e2462f95410755bcf670777bd8f8f0bd872ac4`.

## Automatic truncation succeeds in actual execution

`lethal-autotrim` exactly reproduces the full735-decision T8 win, and the fixed692
handoff wins T8 in705 decisions. The automatic arm checks six chronological
boundaries and selects602 without a hardcoded choice. It wins T8 in**641 total
decisions**, reducing cached-plan overrides **129→33**; its actual suffix exactly
matches a certified leaf with both real policy RNG streams unchanged.

Checks at558/565/571/582 are incomplete under the node bound and correctly retain
the original plan. Check590 completes but does not certify every response as a
same-turn win. Check602 closes all11 winning leaves. Thus selection excludes
both unresolved and demonstrated non-universal continuations; it does not
claim602 is globally earliest or optimal. This remains a known-state result on
one selected deal, and no production default changes.

Next run the complete existing eight-deal retention panel as paired full/auto
execution, using the previously collected accepted plan where one exists and
exact ordinary-policy fallback otherwise. This is16 executions, no fresh deals
or search rediscovery. Preserve the same20 boundary/16 node/200 decision bounds,
require exact full controls and pre-handoff prefixes, and retain the earlier
Main2 finish. Report full-panel wins/turns/decisions, affected games and verification
work; do not infer broader strength from reducing decisions in one game.

Identity: `d3845b7185859886b0ef7a2ec9355ac3f0952147a7c1ad80a18abe66100b55ed`;
report: `b0c3ca2783537da0b50ab5399825b23b07be652c7bee54da2bd619fa97932462`.

## Eight-deal retention preserves outcomes

`lethal-autotrim-retention` completes all16 paired executions with healthy native
results and exact full-plan controls. Both arms win **6/8** with **8.875 mean
turns**. Mean decisions fall **547→535.25**, entirely from game5's735→641 reduction.
Six games have exactly identical full traces. Game6 still wins T12 in579 decisions
and enters Main2 at556; its handoff at538 reduces cached overrides27→1. Two
placement choices differ, with the later chain locating the card accordingly;
the actual continuation remains a certified leaf.

This is a reduction in excess decisions on a selected panel, not fewer turns
or improved win rate. Reusing prior accepted plans also does not validate fresh
proposal discovery. Next freeze the same verifier and truncation logic on the
other complete existing eight-deal `lethal-battle-first-regression` panel, paired
full/auto (16 executions). Include its preparation-dependent win and both
previously overlong finishes, without picking only favorable roots. Require
exact full controls and prefixes, certify actual suffixes, and preserve the same
bounds. No new deals/seeds, training expansion or production promotion.

Identity: `2cd76fd55ab8364ed7d95982abbc59a49e71f5ebb17afc47ed217856e9704cd6`;
report: `8ba08e174c4f7618cec71cf39f35034acbe789b0532960fd6d79aaff6bd67e25`.

## Second panel preserves preparation-dependent wins

`lethal-autotrim-second-panel` completes16 healthy paired executions. Both arms
win **5/8**, with **8.375 mean turns**. Mean decisions change only
**458.375→457.875**. Seven full traces are identical. In preparation-dependent
game2, complete checks at313 and342 reject the handoff;372 certifies the finish,
reducing405→401 decisions while retaining the T6 win. Games3/5 hand off at433/258
with exactly unchanged execution. The algorithm therefore retains the necessary
setup in this tested case; no broad win-rate or turn-length improvement is shown.

The remaining integration gap is fresh proposal discovery: both paired panels
used saved accepted plans. Next attach the same bounded verifier to the live
research search wrapper. After an accepted line is freshly found, test its own
battle-entry boundaries and switch to the certified heuristic; do not alter
search RNG or its12-rollout/2-per-turn/8-per-game limits. Replay all eight first-panel
deals paired control/auto (16 executions). Require exact original search/proposal
controls, untouched policy RNG around verification, offline/online execution
parity, actual certified suffixes, and no shadow desynchronization. The20/16/200
truncation bounds remain unchanged. Still no new deals/seeds, training expansion,
production promotion or hidden-world safety claim.

Identity: `e79be930dbb7d6571bbfe2bc22287fcc1b39abb9086e8f8d063696da22895ba2`;
report: `120185984ddcf0adac4ae6b1cea56bae0e1d5a1831e910f304b33c197850ae9b`.

## Live search matches offline truncation but adds verification cost

`lethal-live-autotrim` completes all16 paired executions with exact offline
execution parity,20 shared search-root checks, and zero abandoned lines or
shadow desynchronizations. Search/proposal/RNG controls remain unchanged.
The6/8 wins and8.875 mean turns are retained, with mean decisions547→535.25.
This closes the first-panel live-discovery integration gap, not hidden-world safety.

Game5 adds **2.992 seconds** of verifier work in this run: whole execution is
3.697s control versus6.133s trim (descriptive, not a controlled throughput benchmark).
Its rejected checks simulate1020/892/851/396/916 decisions before the accepted
check's186. Each earlier check already contains a nonwinning leaf. The first
four also hit the node limit: previous reporting emphasized incompleteness,
but they already had concrete refutations of the fixed heuristic's universal
same-turn-win claim. Continuing their other branches cannot change rejection.

Next add optional early rejection after a nonwinning/uncertifiable leaf or an
exhausted budget. Keep skipped-prefix and failure evidence; never treat skipped
branches as winning. Approved certificates must still be complete. Compare
exhaustive versus early rejection on all eight existing live-search deals
(16 executions), requiring exact acceptance/cutoff, approved-leaf, proposal/RNG,
and actual-play parity. Report simulated work as the primary cost metric and
wall-clock time descriptively. Search limits and truncation bounds stay unchanged;
no new deals/seeds, training expansion or production promotion.

Identity: `061c23449974fd7eff40e53eacd0a1413288264362036d08f2ff7bcec064bf3b`;
report: `3be082f49458ea092768c45bb67f5bcbdbd41247c8b88ebe555215e0e01e747c`.

## Early rejection reduces work without changing play

`lethal-early-reject` completes all16 paired executions. Every one of the eight
paired traces and results is identical, with the same accepted cutoffs and
complete winning leaves. Twenty shared search roots match. Total verifier
simulation work falls **4301→845 decisions** (80.4% reduction); game5 accounts for
4261→805. Its measured check time is3.042→0.572s and whole execution6.495→3.883s;
these timings are descriptive and include preflight records, not a throughput
benchmark. Node- and decision-cap failure controls also reject and restore the
original suffix exactly. This is an efficiency improvement, not a strength gain.

All optimization so far reused known deals. Next freeze the algorithm for a
same-size held-out-deal test: seed**2026101510**, two deal blocks with all four
assignments, eight games per arm. Compare ordinary policy, live search without
truncation, and live search with early-reject truncation (24 total executions).
Audit opening-prefix nonoverlap against available earlier lethal-study panels;
overlap or engine errors stop the study rather than selecting replacement seeds.
Require unchanged policy prefixes before search intervention and control/auto
proposal-RNG parity. Retain health, certified-execution and search-budget checks.
Report all registered outcomes and zero-coverage cases, wins/turns/decisions and
verification cost. No adaptive thresholds, new training, larger sample plan,
production promotion or hidden-information safety claim follows from this pilot.

Identity: `4d64d1f8de91632eaba4ca1b9585bf94812412b43c0f057dd9c669553762cdc2`;
report: `7c34fffd19eeb95a38e0f6fb083f8fdf4c619e02d1d6f80db1db816ada733079`.

## Fresh three-arm evaluation shows no strength or length gain

`lethal-autotrim-fresh` completes all24 executions on the frozen eight-game,
two-deal-block panel. Plain policy, live search, and live search with truncation
all win4/8, average8.75 turns and371.25 decisions. **Every full actual trace is
identical across the three arms**, not merely its aggregate metrics. Ten shared
search roots match. Search finds three already-successful routes; truncation
certifies those three handoffs with82 simulated decisions and reduces cached
plan overrides46→3 without changing play. This is integration/retention evidence,
not evidence of improved strength, faster finishing or unseen-card generalization.
Do not expand replication or training on this result.

Identity: `2a14ed85a849e691afb5c87c7d2b30506cc785a1236686b4c4c924bf7a59bb90`;
report: `ef9119fdb9d846958cf807910dfafef483cae89a522e77740f8fc6c7953f6b1d`.

## Visibility guard explains missing coverage, not missed wins

`lethal-entry-coverage-restart` replays all eight control games with exact complete
trace/result/proposal/search-statistics parity. Research logging now preserves
`concealed_hand_or_field` instead of incorrectly collapsing it into `trigger_gate`.
Public observation counts are collected without exposing concealed card identities
or changing actions, RNG, search budgets or the guard.

Of58 actual battle/end-phase choices at own search-trigger prompts,48 fail the
visibility prerequisite:21 hidden-hand-only,10 hidden-field-only,17 both. All30
such choices in the four losing games fail it; these games never search. Of the
48 blocked choices,25 still have a battle-entry action. None of these counts
establishes a missed lethal or licenses removing the guard: earlier matched
hidden-world tests already demonstrated information-dependent approval.

The first diagnostic preflight stopped before any game because it copied a
runtime identity containing the old registration. Rebinding identity to the new
registration fixes the bookkeeping mismatch; original source/STOP are retained.

Next run a bounded, read-only known-state diagnostic at exactly those25 existing
blocked battle-entry roots, replaying all eight games. Force supported battle
entry only in a restored shadow and apply the unchanged fixed-heuristic verifier
with16 nodes/200 decisions and early rejection. Require exact actual trajectories,
proposal/RNG/search statistics and snapshot restoration. Report complete wins,
nonwins and budget/strict-health rejection separately. This can locate candidate
coverage opportunities, not certify public-information safety or produce teacher
labels. No new deals, rollout-budget increase, guard relaxation or training.

Identity: `8ddcfb2c46a11bd20d7c59313644bee0c5dae2e762972733b0dad862075ef890`;
report: `b55ae5b69c247d4cbc44d09e95bde1047da729cab3b6e2c134b2530ff054bf8c`.

## Blocked-root diagnosis finds one potentially useful timing repair

`lethal-blocked-battle` completes all25 registered root checks across eight
exact original-game replays, with857 simulated decisions. Two complete known-state
certificates pass. The other23 each have a concrete nonwinning branch; none
exhausts a decision/node budget. Early rejection leaves six trees intentionally
unexpanded after refutation, so these are not exhaustive impossibility proofs.

Game3 at decision342 already wins T9/355 under policy and heuristic. Game0 at
decision758 instead has a certified T15/782 win, versus actual policy loss T16/880.
The policy already enters battle: the difference is candidate chain760, where
the heuristic activates24299458 and completes its cost/target choices before
attacking. The policy passes, attacks, then activates the card in Main2 at814.
This identifies a candidate timing/selection mechanism, not its isolated causal
effect. The opponent still has one concealed hand card; do not remove the guard
or promote a known-state certificate as public-information safety.

Next run six fixed continuations: game0 original control, only force activation
at760 then return to policy, replay the certified preparation through769 then
return to policy, and the complete certified suffix; game3 original and complete
certified suffix as a neutral execution control. Preserve original policy RNG
streams, require exact pre-intervention prefixes and controls, use the live
checkpoint opponent, check every intervention against encoded support, and
require complete-suffix runs to match their certified leaf. No new deals/seeds,
training, online helper promotion or expansion of search.

Identity: `991d4d356cbab0cd4f3ad1a90112f4c23bb4a25c388fa02935f1c00ec0b4c85c`;
report: `77cafeeec819d4a1f6e4ba7b75aa6cad3ca5846f56b6324b4dda96683f672344`.

## Single activation is sufficient in the exposed timing case

`lethal-activation-timing` completes all six fixed continuations. Game0's exact
control loses T16/880. Forcing only decision760 to activate Forbidden Droplet
(24299458), then returning every subsequent choice to the original learned
policy, wins T15/781. There is exactly one overridden choice. Replaying the
certified preparation or full suffix both wins T15/782, with four overridden
choices. Game3's neutral control/full suffix remain identical T9/355 wins.
All prefixes, original controls, supported actions, identities and complete
certificate executions pass against the live checkpoint opponent.

Thus the activation-timing change alone is sufficient for this fixed state and
RNG stream; the later cost/target policy can already complete the win. It does
not establish an always-activate rule, general win-rate gain, or safety across
hidden worlds. Next observe probabilities at every candidate optional-chain
prompt in the same eight original games, retaining exact complete traces and
recording public phase/LP, supported alternatives and chosen probabilities.
Include the exposed760 and later Main2 activation814 explicitly. This separates
low-probability sampling from policy preference without changing cards, states,
weights, actions or random streams. No new deals or teacher labels.

Identity: `59e1a217a02742e37760008b20780344c04f06619a8f23ab29043bf5b11b35f4`;
report: `5c0be446836112f4130a73d0113d3e63beecd9902ff3fdbf421dcf34c6da22c7`.

## Probability audit identifies strong suppression at the useful timing window

`lethal-activation-probability` completes eight exact original-game replays with
1,021 captured prompts. At game0 decision760, Forbidden Droplet activation has
probability0.0001556338 (**0.01556%**), versus99.9844% pass. The known missed win
therefore is not chiefly an unlucky sample. The card remains supported: at
Main1 decision758 its activation probability is0.2220%, and at battle command761
it is0.0001016%, where attack receives99.9996%. At later Main2 decision814 it is
63.1920%. These states differ in board, history, phase and competing actions;
the comparison does not isolate phase or action-kind causality.

Across the fixed panel there are138 genuinely optional chain prompts (supported
chain and pass), with48 chain choices and median total chain probability18.10%.
Thus this is not evidence that the model refuses all chains. Droplet has36
supported activation opportunities across just two games; repeated prompts are
correlated and are not36 independent evaluations of card skill.

Next compare frozen warm u128/u256/u512 predictions on the same149 observations:
all optional-chain captures and all captured prompts offering Droplet. Replay
all eight original u512 trajectories unchanged, save complete encoded observations,
verify matching vocab/architecture/history settings, and require u512 probability
parity. Report checkpoint probability/ranking differences, including760/814,
without claiming causal attribution to RL or comparing full-game win rates.
No new deals, seed selection, training, phase alteration or inference changes.
This tests whether the suppression was already present before continuation.

Identity: `c42b0b6785eed6bb56fac5c0b58a70f1ba13999594c2b570e36387779fa22278`;
report: `0f9b7b0d039e19027db732207d9123dc967e5f59e0ffc52cf13a19d7f619b771`.

## Checkpoint comparison: suppression predates continuation

`lethal-activation-checkpoints` completes149 fixed-observation comparisons over
eight exact u512 trajectory replays. Vocab/config/history compatibility, observation
hashes and u512 probability controls pass. At known missed-win decision760,
activation probability is0.07721% at u128,0.02337% at u256 and0.01556% at u512.
All three strongly favor passing; continuation does not create this blind spot,
but the probability falls further on this particular observation.

On the138 optional-chain observations, median total chain probability is
67.55%/24.06%/18.10%, mean56.66%/36.10%/34.48%, and chain probability exceeds0.5
at81/48/46 prompts respectively. These are correlated states visited by u512,
without correctness labels for most choices. Do not conclude overall regression,
forgetting of a demonstrated skill, or differences in full-game checkpoint strength.

Next use the saved observations for a bounded offline input-sensitivity audit.
Evaluate u512 at all138 optional-chain states in an eight-cell factorial: retain
or replace chain action kind with activate; retain or replace decision code with
SelectIdleCmd; retain or replace phase with Main1. Other action fields, masks,
card tables and event histories stay byte-identical. Check all149 original
predictions as controls. Report activation/pass margins, probability changes and
per-game distributions, with the known760 case highlighted.

These artificial input combinations can be inconsistent with the engine protocol
and history. They diagnose numerical dependence only, not legal counterfactual
wins or a safe encoding fix. No mutated input is applied in live play or used as
a training target. No engine rollouts, new deals, model updates or guard changes.

Identity: `ff2e493abbf1f58a3a70bc4000ad209d7aba14a781df749286c4d275d46fc958`;
report: `3809d30def392f9c43d48f114fb141f4dd652404945cdb6dde21efae06cc8350`.

## Input sensitivity does not recover the missed activation

`lethal-activation-input` passes all149 original prediction controls and evaluates
138 optional-chain observations in all eight preregistered cells. Frozen source,
runtime identity, unchanged nonintervened tensors and healthy completion pass.
At the known missed-win decision760, original activation probability is0.01556%;
kind-only replacement gives0.07226%, prompt-only0.01348%, phase-only0.003125%,
kind+prompt0.14929%, kind+phase0.01248%, prompt+phase0.009924%, and all three0.12193%.
Every combination still ranks pass first. A simple input recoding is therefore
insufficient on this example; these invalid-protocol combinations remain diagnostic.

Across all138 observations, changing kind+prompt increases median chain mass
from18.10% to90.81%, and all three to96.22%. Most prompts lack correctness labels;
larger activation mass is not evidence of improved play and could promote bad
chains. No intervention is deployed or added to training.

Next replay the same eight original games and audit the three frozen actor-critic
checkpoints on the same149 genuine states. Capture each supported action's Q,
V, policy-weighted Q and activation/pass ranking, requiring actor probability,
observation, random-generator and complete trajectory parity. Privileged tensors
enter only the critic, never the actor or action selection. Save tensors for
reproducibility. The checkpoint configs record nonzero Q/V losses (0.5 each),
VRPO return mode, lambda targets and lambda0.5: Q is a trained head, but its
ranking alone is not the policy advantage or proof of a causal training fault.

Highlight decision760 against the existing single paired continuation (pass loses
T16, activation wins T15). Its single trajectory outcome is not the expected
return of either action, and u128/u256 continuations were not evaluated there.
Other149-state comparisons are descriptive, not new correctness labels. This
bounded diagnostic adds no deals, training, model selection or production changes.

Identity: `54e54d24ef1e20b9b6c9a2cc0a71f1c97dddcd6362d8c53d0d23a69a32c51506`;
report: `2aa6329d709d366120cf77ae5c221215a11f9ff573593e6bcb818229667be8e5`.

## Critic audit also favors pass at the known missed-win root

`lethal-activation-critic-restart` completes eight exact native replays and149
states across three checkpoints. Saved actor tensors, canonical actor probabilities,
agent RNG states and full trajectories match the earlier audit. Privileged tensors
are saved separately and enter only the critic. At decision760:

| Checkpoint | Q(activate) | Q(pass) | V | Policy P(activate) |
| --- | ---: | ---: | ---: | ---: |
| u128 | -0.07495 | -0.04355 | -0.01433 | 0.07721% |
| u256 | 0.02146 | 0.01618 | 0.06527 | 0.02337% |
| u512 | 0.40681 | 0.44675 | 0.48462 | 0.01556% |

The current critic also favors pass, despite the previously verified single
u512 continuation winning after activation and losing after pass. u256 slightly
favors activation while its actor strongly suppresses it, so actor preference
cannot be explained solely by the contemporaneous Q ranking. Across138 optional
chain states, Q prefers at least one chain over pass at58/81/59 states; these are
descriptive, mostly unlabeled comparisons, not critic accuracy measurements.

The first preflight stopped on a5.66e-7 probability difference. Isolating identical
actor weights and observations shows the joint model scores trimmed action rows
while the standalone actor pads features before scoring: maximum deltas across
149 states are8.35e-7/8.35e-7/2.563e-6, with no argmax changes. The restart retains
the original strict canonical probability controls and separately requires joint
path absolute delta at most1e-5 and identical argmax. Original STOP and source stay
intact. This is numerical path parity, not relaxed game-health validation.

Next audit only the two existing game0 control/activation continuations. Replay
each exactly; after760 record u512 Q, probabilities and V at every learner
decision with more than one supported action, matching snapshot-opponent and
skip-forced collection semantics. Use the actual terminal reward with the existing
Expected-SARSA/VRPO estimator, gamma1 and lambda0.5, without advantage normalization.
Report the root return, advantage, and terminal-reward contribution by recomputing
with terminal reward zero while holding all Q estimates fixed. Also show the pure
terminal-return comparison explicitly as a diagnostic, not a proposed replacement.
This is a complete-suffix calculation, not a reconstruction of a historical128-row
training segment; the forced-activation branch is counterfactual, not on-policy
training data. No optimizer steps, new RNG seeds, new deals or model promotion.

Identity: `f84d12520372860d430ef3d4f6b165693cf98fbe40330df590cec98e92016c9b`;
report: `8678d37cfcc60734f6a0caec9472a9c92e7c51be0b47d4faef4c0a8ad76d607e`.

## Return correction has the wrong sign in both known continuations

`lethal-activation-credit` exactly reproduces the two existing game0 continuations.
Using u512 estimates and learner-only nonforced decisions, unnormalized complete-
suffix VRPO return advantages at lambda0.5 have the following root values:

| Choice | Outcome | Learner rows including root | Q target | Advantage | Direct terminal contribution |
| --- | --- | ---: | ---: | ---: | ---: |
| pass | loss T16 | 34 | 0.476887 | +0.030147 | -1.164153e-10 |
| activate | win T15 | 7 | 0.425187 | -0.021553 | +0.015625 |

Root Vbar is0.446740 for both. Zeroing only terminal reward while holding Q fixed
changes root targets by exactly the listed amounts: gamma1 does not prevent the
lambda trace from attenuating distant terminal evidence. Q-based local estimates
dominate these two suffixes; the sampled correction does not recover the desired
direction here. Pure terminal-return advantages would be-1.446740/+0.553260,
but carry greater outcome variance and are not automatically better estimators.

This diagnoses a concrete credit-assignment failure in these continuations, not
an implementation error in the documented estimator or proof about historical
normalized128-row minibatches. The activation branch is forced at the root.
Single realized outcomes are not expected action values; do not automatically
label every loss-side positive advantage as an error.

Next bound a return-coverage audit to all learner decisions with multiple supported
actions in the same eight original u512 games. Preserve native trajectories and
RNG. Score frozen u512 Q and probabilities, then compare complete-game suffix
returns at lambda0.5/0.9/0.95/1 and pure terminal returns. Report per-game counts,
distance-to-terminal, direct terminal sensitivity and descriptive sign differences;
these are correlated observational data, not correctness rates. Save all rows,
check the known game0/760 result against the prior suffix audit, and isolate
terminal sensitivity analytically to avoid cancellation near machine precision.
No new deals, seeds, actor updates, model selection or training-scale increase.

Identity: `1dc7659c339c5a1cacc31f6e93a29e2c4f473925930be9907889d9798a8ab679`;
report: `6d613dc6f750f2d78535431fff9db39ebf8d8fb12e2ee43fa37c6a6fe64c684c`.

## Artifacts and reproducibility

All studies have two CPU workers, a 15-minute service cap, per-record persistence,
health-stop evidence, frozen source/model/engine/data hashes, and an independent
watcher bound to their service invocation. Artifacts are under `out/research/`:

| Study (all suffix `-2026-10-12`) | Identity SHA-256 |
| --- | --- |
| `lethal-line-response-audit` | `c2bb72b0a55419f7db1609872de147b0a19459095bbfbf8faf1e4ea9a6450d36` |
| `lethal-search-mask-audit` | `75ac699af5df52f8a79e0d6f9d379e4902fa7f3786c929385e2bacc7ff3ee812` |
| `lethal-policy-support-parity` | `b8495a095628721e0764214071bf06e18682e92cc46d1a3b7ce6297478712eb5` |
| `lethal-policy-support-fresh` | `424a2be645298656b812acb0a25c320a082c5ae919248989087a8501199ca1d0` |

The first two studies use main `589254e`; implementation parity and the fresh
pilot use `1767d09`. Each analysis records its report hash. `mask-cause.json` in
the first study records the two native-mask cause checks. Original studies and
their immutable records are preserved.
