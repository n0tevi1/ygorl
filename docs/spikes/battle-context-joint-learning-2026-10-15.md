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

Both target cards in these piercing fixtures are ordinary monsters with missing individual scripts; their script documents have identical boilerplate and differ only in card name. Thus the script channel’s target-name difference materially affects these failures despite the unchanged attacker’s piercing semantics and unchanged stats. The diagnostic supports prioritizing a **name-free structured effect representation** and explicit attacker/target roles in the next bounded ablation. It does not show that equalizing target text is safe for reflection or immunity (where different target effects matter), and it does not repair ordinary retention. Original joint-gate failures stand.
