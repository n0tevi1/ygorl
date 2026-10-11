# Critic ranking interference: Runick mechanism and controlled updates

The adverse Runick choice survives a new RNG-alignment experiment. Exact optimizer replay and loss-removal interventions then show that updates from other training situations can reverse its Q ranking. The source retention anchor and the existing Flashing Fire training example oppose that reversal. This is a demonstrated local interference mechanism, not evidence that those other examples are invalid or that a production fix is ready.

All experiments use the frozen runtime at `ae09594ec0126b5365e3c8311e9ac5c67d23f270`. Reporting starts from main `f549602` (PR #330). Actor parameters, environment rules and production defaults remain unchanged. These are privileged-critic diagnostics, not public-actor strength results. Related issues: #61, #83, #36, #105; #319 remains unresolved.

## Runick difference is not just RNG-position divergence

`critic-runick-rng-boundary-2026-10-11` completed all257 registered games: root60, two choices ×64 new trials1101–1164 × two RNG couplings, plus one original control. All games ended with a winner; identity, task coverage, controls, result/replay hashes and recorded cold equality passed verification.

| Coupling | Activate Flashing Fire wins | Attack with Geri wins | Activate-minus-attack |
|---|---:|---:|---:|
| Sequential | 18/64 | 32/64 | −21.875 pp |
| Reseed at first later-turn learner prompt | 18/64 | 30/64 | −18.750 pp |

Both modes have zero identical actor-observation pairs at that boundary and zero identical semantic suffixes; no boundary is missing. In contrast to the earlier phase-choice experiment, real input differences remain. This still does not isolate battle restrictions from resource changes, opponent responses and continuation-policy behavior. The activation can still attack in the current battle phase. This selected-root experiment supports neither a blanket Runick activation mask nor a population-wide win-rate claim.

## The training set already supplies an opposing label

Training root29 contains Flashing Fire, Smiting Storm, a TY-PHON attack and two phase transitions. Flashing Fire wins27/32 pooled suffixes versus attack32/32, giving a signed-return target margin of −.3125. Its board differs from root60's Geri board. The initial critic's Fire-minus-attack margin is −.068383; after pooled32 training it is only −.036363. The choice still favors attack on this training example, but its margin moves against its own target. On diagnostic root60, the margin moves from −.058148 to +.003205, flipping to activation.

This cannot be explained simply by absence of the card ID. It also does not establish sufficient context coverage: one example is sparse coverage, and suffix estimates remain noisy.

## Exact replay and gradient attribution

`critic-ranking-gradient-audit-2026-10-11` reproduces the original pooled32 model parameters **exactly at steps8,16,32**. It decomposes the source-anchor and32 root losses through Adam's first moments, shared realized second-moment denominator and clipping. Component gradients/updates sum to the actual update within numerical tolerances. Probe derivatives never enter the optimizer. This is a first-order attribution along the realized path, not a counterfactual removal of a loss.

| Contribution to Fire-minus-attack margin | Training root29 | Diagnostic root60 |
|---|---:|---:|
| Source retention anchor | −.074815 | −.077335 |
| Training root29 | −.084775 | −.068392 |
| Training root24 | +.135945 | +.173421 |
| Sum across all33 components | +.031993 | +.060564 |
| Actual margin change | +.032019 | +.061353 |
| Nonlinear remainder | +.000027 | +.000789 |

Root24 is a different context: Mitsurugi Prayers activation versus attacks with Kusanagi and Light Dragon @Ignister while facing Runick. The attribution does not mean its labels are wrong. The anchor is opposing the Runick reversal, so removing retention is not the supported remedy.

## Actual loss-removal interventions support interference

`critic-ranking-interference-ablation-2026-10-11` retrains four fixed arms from the same initialization for32 steps. All use pooled32 labels, original Adam settings and clipping. Omitting a root retains denominator32; remaining losses are not rescaled. Each arm recomputes its own Adam state, so the results include optimizer adaptation and need not equal path-attribution numbers. No diagnostic features or labels enter fitting.

| Arm, fixed step32 | Training root29 margin | Diagnostic root60 margin | Training centered MSE | Changed legal choices /64 vs full |
|---|---:|---:|---:|---:|
| Full | −.036363 | +.003205 | .008708 | 0 |
| Omit root24 | −.053367 | −.014125 | .008812 | 6 |
| Omit root29 | −.015502 | +.016395 | .008799 | 10 |
| Omit source anchor | −.029423 | +.004728 | .008803 | 20 |

All five existing retention gates pass for every arm. Full reproduces original parameters at8/16/32. Independent checkpoint forwards reproduce the stored Q tensors and confirm unchanged actor/non-Q parameters. Removing root24 restores this local ranking; removing root29 or the anchor worsens the activation preference. Deleting legitimate training situations is not a proposed fix, and these interventions do not establish full-panel performance gains.

**Reporting bug found and corrected:** this experiment's original `analysis.json` used unmasked argmax for its secondary `diagnostic_choices` field. Since padded Q entries are zero,14/64 choices in each of full/omit24/omit29 and13/64 in no-anchor were wrong. The sealed original remains preserved. `verify_results.py` reloads checkpoints, checks forward equality and recomputes legal argmax in `verification.json`; the table above uses those corrected choices. Training masks, Q tensors, primary legal-action margins and retention are unaffected. This is a new diagnostic-report bug, not evidence that the production actor chose padding. The subsequent experiment masks before argmax, and independent verification finds zero discrepancies.

## Parameter-path intervention: a candidate, not a finished repair

`critic-ranking-parameter-ablation-2026-10-11` keeps **all32 roots and the source anchor**. It separately updates the full Q head, its context-query MLP, candidate MLP or candidate bias. All other parameters are frozen; same initialization, optimizer settings and32-step budget.

| Updated component | Parameters | Training root29 margin | Diagnostic root60 margin | Training centered MSE | Changed legal choices /64 |
|---|---:|---:|---:|---:|---:|
| Full | 66,177 | −.036363 | +.003205 | .008708 | 0 |
| Context query only | 33,024 | −.048584 | −.020162 | .011216 | 14 |
| Candidate MLP only | 33,024 | −.033608 | +.000597 | .009360 | 16 |
| Candidate bias only | 129 | −.051573 | −.024736 | .011182 | 19 |

All existing retention gates pass. Full again exactly matches original8/16/32 parameters; independent forwards and legal argmax checks pass. A separate checkpoint audit confirms every non-selected parameter group is exactly unchanged. Context-only restores the local ranking while retaining all training examples, whereas candidate-only still reverses it slightly. However, context-only also fits the training data less strongly, and bias-only restores the ranking too. These results **do not prove** that a state-independent candidate pathway is intrinsically faulty or that more capacity is required. Reduced adaptation is a competing explanation. The64 inspected roots are now diagnostics, not an untouched test set.

## Fresh collection complete; prospective branch validation running

`critic-ranking-fresh-roots-2026-10-11` completed128 new games,16 in each of eight broad/Runick × old/mirror × own/face strata. All128 games ended with a winner and passed task/hash/recorded cold replay verification. All128 deals and opening-hand pairs are unique, with zero overlap against2,752 registered prior deals. The actors and opponents are fixed.

Six checkpoints are hashed before collection: full32, context-only32, candidate-only32, bias-only32, full16 and unchanged initialization. Full16 controls for a smaller update budget; it is not claimed to match update norm or training error. Root selection is fixed independently of outcomes or Q disagreement: hash order `ranking-root-selection-20261011:<id>` within each stratum; take eight games' first learner battle prompt with2–6 supported legal choices. Insufficient coverage must stop rather than silently replace cases.

The running `critic-ranking-fresh-confirmation-2026-10-11` panel evaluates216 supported actions at all64 roots with32 new suffix streams1201–1232 plus64 original controls: **6,976 tasks**. All six models’ predictions and features were sealed before suffix outcomes. Primary: context-only32 minus full32 centered action MSE, equal roots,50,000 shared suffix-block bootstrap draws with seed2026101117. Legal-choice conditional returns and the other models are secondary. No checkpoint tuning, model promotion or actor-strength claim. Any useful critic must later be tested through actual public-actor training/full-game evaluation.

## Artifacts and monitoring

All paths below are under `out/research/`:

- Runick boundary: `critic-runick-rng-boundary-2026-10-11/{analysis.json,training-fire-fit-audit.json}`; identity `537c6f0efc30a55931ea71ee87fc205a1a5f70d93b3513e082721e94b4e6daa0`.
- Gradient audit: `critic-ranking-gradient-audit-2026-10-11/{attribution.json,verification.json}`; identity `d3d1ed6a3a8f327e7d696487562ecddfec1f6515f4ca7a1b3c787f54bf207b85`.
- Loss interventions: `critic-ranking-interference-ablation-2026-10-11/{analysis.json,verification.json,verify_results.py}`; identity `0ca3030ef492620b39b579b23f78967bb8d4022e22d5aaabeaa03b67948d3060`. Use the verification artifact for legal choices.
- Parameter interventions: `critic-ranking-parameter-ablation-2026-10-11/{analysis.json,verification.json}`; identity `4dcd01bff12cf115493d810cafebf1e6ce0a9e51ccb147617ae5eb74a6a1a464`.
- New collection: `critic-ranking-fresh-roots-2026-10-11/{registration.json,opening-audit.json,HANDOFF.md}`; identity `e957a88498e02acebc5d14b395736980229cce56227bb485340ae6d4c03bbde9`. Service `ygorl-critic-ranking-fresh-roots-20261011.service`, invocation `1b26d92c7b8e4d1ca65b0b98ee04631d`; two workers, two-hour cap and independent watchdog. This collection is complete and verified. The branch panel is now running, with identity `5341095edb0a4f250f654e3b8ee2059f371b84a075578c2e80863b72de00768a`, service `ygorl-critic-ranking-fresh-confirmation-20261011.service`, invocation `bbeaa81fc9064869ba45bb58febf7e9b`, two workers, six-hour cap and independent watchdog. Its `HANDOFF.md` and sealed `analyze.py` define completion handling.

Completion events for the first five studies were acknowledged after verification; original STOP evidence remains untouched. No GPU is needed for these small cached-feature fits or CPU engine collection.
