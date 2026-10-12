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
