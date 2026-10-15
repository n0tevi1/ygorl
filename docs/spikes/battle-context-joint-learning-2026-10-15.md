# Joint ordinary/effect learning pilot, 2026-10-15

The frozen source checkpoint has `card_text=False`, `effect_text=False` and no card facts. It represents identities and public structured state but has no explicit text channel. The prior numeric residual failed all four effect challenge families; source was also weak. This pilot tests whether balanced native supervision plus public effect semantics can cover both ordinary and exceptional battles.

## Fixed protocol

Freeze the source trunk/head. Train five matched scalar logit residuals: numeric only; numeric plus contextual attacker/target embeddings; pair/numeric plus original public card text; pair/numeric plus static structured script text; and both text sources plus pair/numeric. All use the same 3340-input/128-hidden ReLU head with zero final layer; unused blocks are zero. Normalize from training inputs only.

Both text sources use the same frozen, locally cached E5-base-v2 revision `f52bf8ec8c7124536f0efb74aca902b2995e5bcd`. Encode token windows of 480 tokens with stride 448, passage prefix, normalized chunk mean, then renormalize; archive documents, hashes and chunk counts. Nothing is silently truncated. Raw text contains the public name/description. Structured text contains effect setter records with enclosing function and exact token expressions plus the entire comment-free canonical script. Generic setter labels/constants are expanded for readability; retain condition callbacks, target restrictions, scope, timing, clones and registration. No outcome-derived descriptions, family labels or dynamic effect flags. External helpers remain symbolic and missing scripts explicit: this is a conservative static code representation, not a complete Lua interpreter.

The protocol was expanded to these five arms at the user’s request before any text encoding, fitting or new evaluation. Script structure being easier to learn is a hypothesis, not an established result.

Reuse 109 ordinary native training prompts and 24 challenge prompts for immunity, reflection and piercing. Draw 16 ordinary and 16 effect examples per step, two seeds, 128 Adam steps at 0.001, fixed final endpoint, save every 32 steps. The entire damage-step stat-change family is excluded from fitting and normalization. No production-model or RL changes.

After all ten endpoints are frozen, collect 40 new roots/80 branches: 12 known-effect roots with changed stats, 12 new-card roots (Marshmallon, Daigusto Sphreez, Lancer Dragonute), eight withheld-family Cat’s Ear Tribe roots, eight ordinary controls. Repeat 16 branches cold. Independently specified fixture outcomes must match the native engine. Unique immediate damage/destruction Pareto labels remain conditional on the accepted attack; no strategic labels for cancel or tradeoffs. Daigusto Sphreez tests a combination of reflection and immunity.

Acceptance requires at least 75% in every effect subgroup and new ordinary panel, and at least 90% on the reused 50-prompt ordinary regression panel, in both seeds. Report newly introduced errors separately. All thresholds and inputs are fixed before fitting; no adaptive extensions or best-checkpoint selection. A failed gate is a valid result and blocks promotion. Passing only permits independent behavior confirmation.

Heldout means excluded from this supervised fit, not necessarily unseen by source pretraining or the text encoder. These constructed, mirrored roots are correlated; no frequency or win-rate inference follows. The already exposed Cat’s Ear Tribe identity is a withheld training family, not a blind discovery claim.

Evidence: `out/research/battle-context-joint-learning-2026-10-15/`.

## Execution provenance

The first launch stopped before fitting because the old challenge uses `reflect`/`pierce` category names rather than `reflection`/`piercing`. Its STOP and logs remain intact. The restart corrects only the input selector and verifies 109 ordinary training prompts, 24 effect training prompts and 50 reused ordinary regression prompts before launching. Protocol, vectors and fixture specifications are unchanged.

Restart evidence: `out/research/battle-context-joint-learning-restart-2026-10-15/`. The encoder runs locally in evaluation mode; both complete text documents and vector hashes are archived. The static-script helper has tests for retaining condition callbacks, clones and registrations, and for treating missing scripts as unknown rather than as evidence of no effects.

## Results

All ten fixed endpoints completed. The native engine verified 80 branches and 16 cold replays. The final report originally used `identity_sha256` where the monitor expects `study_sha256`; `validate_report.py` independently rechecked every input hash, every endpoint hash, native health and summary counts, and wrote `validated-report.json`. Original report/configuration and the first run’s STOP remain preserved. No results, labels or checkpoints were changed by this serialization adapter.

Correct target choices below are seed 2026101511 / seed 2026101512, except the single frozen source. These are conditional two-target decisions, not full-game win rates.

| Input arm | Effect panels (32) | New ordinary (8) | Reused ordinary regression (50) |
|---|---:|---:|---:|
| Frozen source | 13 | 6 | 30 |
| Numeric | 24 / 23 | 8 / 8 | 39 / 40 |
| Numeric + contextual pair | 17 / 13 | 5 / 4 | 33 / 35 |
| Pair/numeric + original text | 18 / 19 | 8 / 8 | 36 / 33 |
| Pair/numeric + structured script text | **28 / 28** | 7 / 8 | 38 / 35 |
| Pair/numeric + both text sources | 24 / 24 | 8 / 8 | 33 / 30 |

The script-only text arm outperforms original prose on this small fixed panel. Across the two reflection panels it gets 8/8 per seed, compared with raw text’s 4/8; on the withheld damage-step family it gets 8/8 versus raw text’s 4/8. However numeric-only also achieves 8/8 and 7/8 on the withheld family, so that result alone does not demonstrate rule comprehension. Script text still gets only 2/4 in **each** piercing panel in both seeds; concatenating both sources gets 0/4 in both piercing panels. Source-correct effect choices become wrong twice per seed for both script and combined text.

Every arm fails the preregistered joint gate. Script text’s 38/50 and 35/50 ordinary retention are below 45/50; it introduces six and eight errors on previously source-correct ordinary decisions. The three text arms all fit the 24 effect training prompts perfectly, so training accuracy does not settle generalization. Do not promote, scale or claim full-game improvement.

The next bounded investigation should distinguish text/feature interference from missing coverage: inspect the piercing failures and test a factorized representation (explicit attacker and target effect roles, applicability conditions, ordinary outcome component) with balanced counterexamples and an ordinary-retention objective. Original prose remains a useful complementary input candidate, but simple concatenation is not supported by this result. Any new fitting must use a new protocol and new evaluation roots; the present panel is now exposed development evidence.

## Post-hoc target-text role diagnostic

A separate, recorded protocol (`text-role-audit-protocol.json`) tested the now-exposed panel without fitting or changing endpoints. Hold source, numeric inputs, contextual pair and attacker text fixed; replace both candidate target-text vectors by their per-root mean. Replayed baseline logits match the original report. This artificial intervention removes target text identity differences; it is not a legal card substitution or independent evaluation. All 40 roots are archived, with interpretation focused on the eight piercing roots.

| Text arm | Piercing before (seed 1 / 2) | Equal target text (seed 1 / 2) |
|---|---:|---:|
| Original prose | 2 / 3 | 8 / 8 |
| Structured script | 4 / 4 | 8 / 8 |
| Both | 0 / 0 | 8 / 8 |

Both target cards in these piercing fixtures are ordinary monsters with missing individual scripts; their script documents have identical boilerplate and differ only in card name. Thus the script channel’s target-name difference materially affects these failures despite the unchanged attacker’s piercing semantics and unchanged stats. The diagnostic supports prioritizing an **identity-separated structured effect representation** and explicit attacker/target roles in the next bounded ablation. It does not show that equalizing target text is safe for reflection or immunity (where different target effects matter), and it does not repair ordinary retention. Original joint-gate failures stand.

## Identity preservation and non-battle transfer

User steering: card names and archetype information may be essential and must remain available. The diagnostic does **not** justify deleting identity. Keep password/alias, public name and archetype membership in an identity channel; keep effect applicability, costs, timing, target predicates and operations in a separate rule channel. A cosmetic card-name prefix may be ablated within the rule channel while the identity channel remains intact. Preserve semantic references such as `IsCode`, `IsOriginalCode`, `IsSetCard`, listed names and material requirements. Dynamic name/archetype changes require public state and engine semantics; static database membership alone is not definitive in those cases.

Concrete example: Runick Tip (`c31562086.lua`) filters `IsSetCard(SET_RUNICK) and not IsCode(id) and IsAbleToHand()`. A representation must preserve archetype membership **and** the named-card exclusion **and** the availability condition. Dropping card references or treating every Runick card as interchangeable would be wrong. Its `Effect.CreateMysteruneQPEffect` helper also illustrates why unexpanded helper calls must remain explicitly unresolved rather than treated as a complete effect specification: `cards_specific_functions.lua` supplies the alternate extra-deck summon mode, activation/count limits, banishing and the next-Battle-Phase skip. These are not recoverable from the individual card’s setter index alone. Original card prose remains a complementary source and a check against incomplete script extraction.

The repository already has optional card-fact inputs for archetype membership, referenced archetypes and action/location queries (`docs/nets.md`, `CardIdentity`, `build_card_facts.py`). Membership and reference embeddings share a table. This is useful infrastructure, not evidence that the source checkpoint uses it (it has `card_facts=False`) or that the coarse facts preserve complete predicates: full negation, named exceptions, sub-archetypes and dynamic restrictions need separate verification.

The representation can describe non-battle effects; the present fitted residual only scores accepted-attack targets. Transfer to the general policy requires wiring the representation into general card/effect/action features and training with non-battle examples. No result here establishes such behavioral transfer.

Next validation design, before any broader retraining:

1. **Search/recovery:** archetype membership, exact named-card exclusions, source/destination locations and negation; include same-archetype hard negatives and named exceptions.
2. **Summon/material selection:** explicit card references, type/level/archetype restrictions and counts; verify complete combinations with the native engine, not just individual material membership.
3. **Activation/cost/targeting:** matched public states differing in payable cost, timing or target applicability, with the same card identities. Static declarations must not stand in for active effects.

Compare raw text, structured rules, and both using matched identity channels, parameter/update budgets, and native outcomes. Separate within-archetype new-card transfer from whole-archetype holdout; also split effect templates and state conditions to prevent near-duplicate leakage. Rule applicability probes must not receive the engine’s legality label through an action mask. Conversely, improved applicability classification alone does not prove policy gains, since the live engine already supplies legal actions: subsequent behavioral validation must compare choices among legal actions and their native continuations. Retain ordinary-battle and effect-exception regressions in the joint gate.

Before a non-battle learning canary, audit shared-helper coverage and parameter binding; fail coverage validation when required callback/condition dependencies are unresolved. The existing filter IR intentionally over-approximates unknown predicates and treats some state-dependent predicates as action hints, so it must not be promoted to an exact legality oracle. Native validation must include positive and negative applicability cases.

## Executed non-battle prerequisite audits

Two bounded studies now provide native fixtures and concrete extraction requirements. Both used the unchanged official card scripts and the frozen `codex-explicit-battle-context` runtime, bound the environment/input hashes, checked engine health, completed under monitored user services, and were acknowledged after report verification. No policy was trained or promoted.

1. `out/research/effect-text-nonbattle-audit-2026-10-15/`: **24/24 cases and 24/24 cold replays pass** (10 positive, 14 negative). Runick Tip, Branded Retribution and Reinforcement of the Army cover same-name exclusion, archetype membership, source-location errors, level/race boundaries, and the Runick helper's alternate extra-deck summon. Positive cases check actual destination zones, Retribution's self-banish cost, and Runick's opponent banishment and once-per-turn suppression. Both seats are covered. Development preflight first exposed missing fixture placement responses; the original runner/results remain archived, and deterministic placement/position handling was fixed before freezing. Expected labels were unchanged.
2. `out/research/effect-text-state-gates-2026-10-15/`: **12/12 cases and 12/12 cold replays pass** (8 positive, 4 negative). Within each pair all card identities and seeds are held fixed; only the controller's Imperial Iron Wall changes between face-down and face-up. Active Iron Wall prevents Runick's search activation because the required opponent banishment is unavailable, and prevents Retribution's graveyard activation because its self-banish cost cannot be paid. ROTA remains usable in both states. The own face-down identity is known to its controller; this does not authorize revealing opponent private cards.

The coverage artifacts preserve public names, passwords/aliases/setcodes, raw card text, exact canonical scripts, and positional bindings into the Runick closure factory. The second audit archives six reachable shared definitions in the two indexed libraries: `Effect.CreateMysteruneQPEffect`, `Cost.SelfBanish`, `Auxiliary.SelectUnselectGroup`, `Auxiliary.Stringid`, `Duel.SelectEffect`, and `Auxiliary.SelectUnselectLoop`. The entire Runick factory body retains its local closures. References to native APIs and unresolved methods remain explicit.

These results demonstrate that the fixtures distinguish identity, rule and state requirements. They **do not** demonstrate that a learned representation handles them. Named dependency closure within two files is also not full semantic coverage: lexical captures, callback binding, object methods, aliases and imports still require a conservative extraction contract. Both reports explicitly leave full coverage/learning readiness false. The immediate remaining task is to validate that contract before the matched original-text/structured-text/both learning comparison; subsequent evaluation must use new cards/archetypes and retain the battle regression gates.

## Executed callback extraction and non-battle text pilot

`out/research/effect-text-transfer-pilot-2026-10-15/` completes the remaining scoped prerequisite and the four-arm learning comparison. Its identity hash is `10566c057e5266d3d41f545dbe0700bf8afaa9a6b21417f16055c29698090e44`. The report, frozen inputs, prepared artifacts and eight endpoint hashes were independently rechecked before acknowledging completion. Process health passed; **every learning arm failed its generalization gate**. These are separate outcomes.

The research extractor indexes callbacks, formal parameters and positional bindings, traverses shared dotted functions and methods, resolves aliases, and retains entire closure factories. Fourteen shared symbols are reconstructed from canonical source after clearing their originals. `Card.RegisterEffect` requires capturing the original native binding *before* utility loading, then applying the extracted wrapper once; otherwise the closure can capture a missing or already-wrapped function. `Group.Iter` is restored after its `Auxiliary.Next` referent. Native APIs are explicitly identified from the pinned native source declarations and remain runtime terminals.

All **192 official-versus-extracted cases agree**, including native decisions and positive resolutions; **24 cold replays agree**. Removing `Cost.SelfBanish` or `Duel.SelectEffect` intentionally is detected in both negative controls. This closes the executable extraction gate for seven selected source/restriction cards and these constructed cases. It does not establish arbitrary Lua equivalence or complete extraction for every card. The original prerequisite reports remain unchanged.

The learning task is binary **activation applicability**, not target-choice quality or full-game play. Four source effects (Runick Tip, Branded Retribution, Reinforcement of the Army and Fossil Dig) supply 64 training cases. Each source/candidate pair is a four-state quartet: candidate in deck/graveyard crossed with own Imperial Iron Wall face-down/face-up. Evaluation uses 64 new-candidate cases, 32 E - Emergency Call cases with the entire HERO example family excluded from fitting, and 32 Summoner's Art cases with its normal-monster rule excluded from fitting. The last split may reuse candidate identities. There are only 16 training pairs and correlated state variants, not 64 independent card combinations.

All arms retain the same source/candidate password, alias, public name, archetype membership, numeric facts and observable state. No legality mask, action list or native outcome appears in the model inputs. Raw and structured rule blocks describe the source effect and restriction card; candidate rule text is omitted consistently in all arms. Structured documents include canonical code and the dependency/callback index, rather than a verified high-level symbolic interpreter. All text uses the same frozen local E5 encoder and complete token windows. The 4751-input/64-hidden MLP, initialization, balanced minibatch draws, 128 Adam updates and fixed final endpoint are matched across four arms and two seeds. Normalization uses training rows only.

Balanced accuracy below is seed 2026101561 / 2026101562. The preregistered gate requires balanced accuracy, positive recall and specificity each at least 75% in every evaluation split in both seeds, with no balanced-accuracy regression from the identity/state baseline.

| Input arm | New candidates (64) | Unseen HERO effect (32) | Unseen normal-monster rule (32) |
|---|---:|---:|---:|
| Identity + state | 50.0 / 50.0% | 62.5 / 62.5% | 66.7 / 60.4% |
| Plus original text | 50.0 / 50.0% | 41.7 / 52.1% | 33.3 / 39.6% |
| Plus structured text | 50.0 / 50.0% | 56.3 / 62.5% | 50.0 / 43.8% |
| Plus both | 58.3 / 50.0% | 41.7 / 43.8% | 33.3 / 33.3% |

On new candidates, seven of eight endpoints predict all 64 cases unavailable: 52 true negatives and **12 false negatives**, giving a misleading ordinary accuracy of 81.25% despite zero positive recall. The combined arm's first seed recovers only 2/12 positives. Structured text does better than original prose on the two unseen-effect panels here, but does not outperform the shared baseline or meet the gate. No arm is promoted; this does not support broader retraining yet.

Training balanced accuracy itself is only 75.6–87.5%, so the failed transfer result cannot be attributed solely to representation quality or unseen-card generalization. A post-hoc, no-fit audit in `out/research/effect-text-transfer-diagnostic-2026-10-15/` reproduces all eight original logit arrays and verifies frozen hashes. Of the eight training quartets containing both positive and negative states, each endpoint gets only 0–2 quartets entirely correct. Most correctly order positives above negatives within a quartet (6–8 of eight), so state sensitivity is present but does not reliably yield the correct classification threshold across cards.

The same audit finds normalized factual/state feature mean L2 norms of 5.07 in training versus 11.41 on new candidates and 12.38 on the unseen normal rule. Binary categories absent in training become magnitude 20 when first present under the 0.05 standard-deviation floor. This is an input-distribution concern, not a proven unique cause of failure. The restriction card text is constant across this dataset and centers to zero; only its state changes. This pilot therefore cannot establish transfer to unseen restriction cards.

Before a new generalization comparison, separate optimization adequacy from representation changes. The next diagnostic is restricted to the same 64 training rows: reproduce each original 128-update endpoint, then inspect fixed 256/512/1024-update training results with no heldout inference or checkpoint selection. Any revised categorical scaling or factorized identity/rule/state architecture needs a separate protocol and fresh evaluation; the present panels are now exposed development evidence. Applicability success would still require subsequent legal-action choice and battle-retention validation.

The training-only check has now completed in `out/research/effect-text-fit-sanity-2026-10-15/`. It reproduces all eight original 128-update parameter sets before continuing the same optimizer/batch streams to fixed 1024 updates. At that final point, seven endpoints correctly classify all 64 training states; original-text seed 1 gets 62/64 (balanced accuracy 98.1%, two false positives). Both structured-text seeds fit perfectly. This establishes that the original short optimization budget was insufficient to fit the training fixtures. It does **not** repair or supersede the failed heldout gate: this diagnostic performs no heldout inference. A fresh-card, matched-budget comparison is required before judging transfer.

## Fresh-card optimization and normalization confirmation

`out/research/effect-text-transfer-confirmation-2026-10-15/` preregisters a bounded follow-up. Keep exactly the original 64 training rows; test 16 new candidate identities (64 state variants), disjoint from **all** original pilot identities. Compare the same four input arms and two seeds at fixed 1024 updates under two normalizations: original train-only standardization, or the same means/continuous/text scaling with binary feature standard deviations fixed at one. The latter isolates binary amplification; it does not remove identity or replace the rule input with hand-written applicability labels. All 16 endpoints are frozen before collecting fresh outcomes. Each fresh fixture requires official/extracted parity, with eight cold replays. Runtime cap: 900 seconds, no automatic restart.

The gate requires training balanced accuracy at least 95%, fresh balanced accuracy/positive recall/specificity each at least 75% in both seeds, and no fresh balanced-accuracy regression from the same-normalization identity/state arm. This panel tests new cards under the four already-trained effects; it makes no unseen-effect or full-game claim. Passing only permits further independent validation. Study identity: `7064f5cf5592a4061d2f7e46776a3d0caed950bb216b2bd5c0077722c5ab2c55`.
