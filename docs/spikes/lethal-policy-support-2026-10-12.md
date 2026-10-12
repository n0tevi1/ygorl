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
