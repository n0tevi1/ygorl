# Fresh validation does not establish a benefit from context-only Q fitting

The proposed context-only update does **not** pass the prospective improvement test. All6,976 registered tasks completed normally and passed verification, but its primary MSE difference has a confidence interval crossing zero. Its conditional choice win rate is also indistinguishable from the full Q head. Do not promote the model or treat the earlier repaired Runick example as a general solution.

This closes the validation registered in [the interference report](critic-ranking-interference-2026-10-11.md). Execution stayed at frozen commit `ae09594ec0126b5365e3c8311e9ac5c67d23f270`; reporting is based on main `e3ea9c7` (PR #331). Issues #61/#83/#36/#105 remain active; #319 is not resolved by this experiment.

## Verified prospective panel

The panel contains64 roots from128 independently reshuffled source games, selected by the fixed hash ordering before looking at critic errors or branch outcomes. All six checkpoints were hashed before source collection, and their features/predictions were sealed before suffix outcomes. Every supported action was evaluated:216 actions ×32 new streams1201–1232, plus64 original controls =6,976 tasks.

The sealed analyzer verified immutable inputs, model hashes, the full action/trial grid, all original controls, result/replay hashes, recorded cold replay equality and engine health. All6,976 results have terminal reason `win`; no rollout failure was excluded. This audit did not independently rerun the games beyond their recorded cold replay checks.

| Frozen model | Centered action MSE | Conditional choice wins |
|---|---:|---:|
| Full Q head,32 steps | .020775 | 62.207% |
| Context query only,32 steps | .020156 | 62.109% |
| Candidate MLP only,32 steps | .020364 | 61.475% |
| Candidate bias only,32 steps | .019977 | 62.793% |
| Full Q head,16 steps | .020183 | 62.402% |
| Unchanged initialization | .021095 | 62.354% |

Primary context-only minus full32 MSE: **−.000618685**,95% interval **[−.001329155,+.000098288]**. Context-only minus full32 conditional wins: **−.097656 pp**, interval **[−.732422,+.537109 pp]**. Only7/64 root choices differ.

Intervals use50,000 shared suffix-block bootstrap samples, seed2026101117. They condition on these roots and fixed continuation policies; they do not quantify population-wide actor strength. Legal Q argmax selects one root action, after which the frozen public actor continues. These are not full-game win rates of a newly trained actor.

The full16 control improves MSE by −.000591438, close to the context-only point estimate, with an exploratory interval [−.000889620,−.000295418]. Candidate-only and bias-only also have negative exploratory MSE intervals. None of the secondary conditional-win intervals excludes zero, and these secondary comparisons are not adjusted for multiplicity. Selecting bias-only because it has the best observed win fraction would be post-hoc selection, not a validated improvement. Full16 is a smaller-update control, not a training-error or update-norm matched model.

## What the decomposition supports

Post-hoc category counts are36 attack/phase roots,12 activation+attack roots,14 phase-only roots and2 other roots. Context-only improves attack/phase MSE (.028670 → .027109) but worsens activation+attack MSE (.014285 → .015208) and phase-only MSE (.006145 → .007066). There is no evidence here of a general activation-versus-attack repair.

Of its seven changed choices, three switch from Main2 to attack, three from End Phase to Main2, and one between activation choices. The observed net conditional outcome is two fewer wins across64×32 root continuations. These per-root/category comparisons are descriptive: phase RNG coupling and finite-suffix noise remain relevant, and empirical best actions are not tactical ground truth.

A second descriptive audit estimates the sampling variance of the32-suffix centered targets at **.008599**, retaining the covariance between paired actions by centering each trial first. The observed centered target signal is .020631; subtracting estimated sampling noise gives .012032. Full32's noise-corrected MSE is .012176, and context-only's is .011557. A predictor assigning equal values to all actions has observed centered MSE .020631—close to the learned heads. This is an error baseline, not a proposed tie-breaking policy. It suggests limited learned differentiation in this panel, but does not identify coverage as the sole cause or establish a population comparison. No new gate or significance claim is derived from this post-hoc noise estimate.

The local interference findings still stand: some training-root updates can counteract other roots and flip the held-out Runick ranking. What fails is the inference that restricting the context/candidate update path is already a generally useful repair. Reduced adaptation and small/noisy training coverage remain competing explanations.

## Next registered experiment: widen training coverage

`critic-ranking-coverage-data-2026-10-11` is running256 source games,32 in each of the same eight broad/Runick × old/mirror × own/face strata. All256 explicit deck orders and opening-hand pairs are unique, with zero overlap against2,880 registered prior deals. Actor, opponent and environment remain fixed.

Root selection is outcome-independent: fixed SHA256 order `coverage-root-selection-20261011:<id>` within each stratum; each game's first learner battle prompt with2–6 supported legal options. The first12 qualifying games are training and the next4 validation, giving96 new training roots and32 separate validation roots. Insufficient coverage stops rather than silently replacing samples. Previously inspected evaluation roots are excluded from training.

The follow-on label panel will evaluate every supported action at these128 new roots using16 suffix streams1301–1316 plus128 original controls. The widened training arm adds96 new roots to the original32; the baseline retains the original32 pooled32 targets. Equal-root weighting is used within each arm. The widened arm retains the original32 labels and uses16-suffix means for its additional roots. This adds sampling budget; it is not a compute-matched experiment.

Both arms use the same original initialization, full Q head, source anchor, centered action loss, Adam learning rate1e-4/epsilon1e-8, gradient clipping1 and32 updates. Actor, trunk, V and ridge remain frozen. Save8/16/32, with32 fixed as primary and16 secondary; no best-checkpoint selection. The original arm must reproduce the previous pooled32 parameters exactly. Namespace original/new sample IDs to prevent collisions. Freeze all fits before analyzing the new32 validation roots; validation outcomes never enter fitting.

The primary validation comparison is widened128-step32 minus original32-step32 centered action MSE, equal roots,50,000 shared suffix-block draws with seed2026101119. Report legal-choice outcomes, retention and unchanged/step16 controls too. Keep architecture and loss fixed to test the coverage intervention. If it fails, inspect fit and target uncertainty rather than continue tuning the already-inspected Runick example. Any candidate still needs actual public-actor RL/full-game validation before a playing-strength claim.

## Artifacts and handoff

Under `out/research/critic-ranking-fresh-confirmation-2026-10-11/`:

- Sealed `analyze.py`, `analysis.json`: complete prospective comparison; identity `5341095edb0a4f250f654e3b8ee2059f371b84a075578c2e80863b72de00768a`.
- Supplementary `diagnose_coverage.py`/`coverage-diagnostic.json` and `target_noise.py`/`target-noise.json`: post-hoc decompositions, explicitly not part of the primary gate.
- Completion event `7a03006eddc945653296dc7a71f2cb51893c3d7432a2232eb9944a20ade974a1` acknowledged after verification. Previous STOP evidence is retained.

Next source collection: `out/research/critic-ranking-coverage-data-2026-10-11/`, identity `c4745acf4ca05b19156755d6dc70595319359d63d70b0a00bead2fa2c52cb17c`; service `ygorl-critic-ranking-coverage-data-20261011.service`, invocation `7ecd561159854bf1b0130edf11d09891`. Two CPU workers, two-hour cap and independent watchdog. `verify_collection.py`, `prepare_branches.py` and `HANDOFF.md` specify completion handling and the planned six-hour/two-worker branch panel. No production code, defaults or model promotion in this report.
