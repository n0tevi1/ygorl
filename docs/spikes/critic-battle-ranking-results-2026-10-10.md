# Battle Q ranking and critic loss saturation

The 896-game battle-action pilot completed with healthy outcomes and passed identity, task/result, replay-hash and cold-replay checks. All 16 original-action/RNG controls exactly reproduced their source results. Every game ended with a winner; there were no limits or engine failures. Completion `794491f018582e54d12e9b377d8f0dd8879054b6449931381d5b127c96b6d195` was acknowledged after verification.

Artifacts: `out/research/critic-battle-ranking-2026-10-10/{analysis.json,analyze.py,root-predictions.json}`. Frozen runtime and earlier STOPs remain unchanged.

## Conditional action ranking, not a new policy win rate

Sixteen fixed roots cover panel × opponent × own/face role. All 55 supported legal root actions received 16 common suffix streams each (880 interventions plus 16 original controls). The learner and opponent then continued with their original policies. Root selection was outcome-independent.

Choosing the maximum-Q root action gives these **conditional** mean win fractions, equally weighting roots:

| Critic | Raw Q, unique argmax | Clipped Q, first-index tie rule | Clipped Q, uniform among tied maxima |
|---|---:|---:|---:|
| Frozen | 62.89% | 62.89% | 62.89% |
| Control residual | 62.89% | 62.89% | 63.09% |
| Terminal refresh | 65.23% | 65.23% | 65.63% |
| Anchored refresh | 63.28% | 63.28% | 63.67% |

These share roots and RNG streams and are not 256 independent games per model. Sixteen streams per action are noisy. No one of 81 action-pair comparisons survives Holm correction of paired exact sign tests across the complete panel. The terminal point estimate is encouraging but does not establish a robust improvement. Observed-best action gaps in `analysis.json` are optimistically biased by selecting the best empirical action; they are not estimates of true regret.

Removing clipping recovers score distinctions but does not improve this pilot's first-tie conditional outcome. All fitted critics retain their raw preference at root1125, where activating the attack yields14/16 wins versus main2 at16/16; this difference is too small to certify. At root551 every action wins16/16. A saturation count alone therefore cannot demonstrate harmful action choices.

Two useful examples requiring confirmation:

- Root726, broad/fixed: attack5/16, main2 10/16, end phase7/16. Terminal refresh prefers main2; frozen/control/anchored prefer attack. The attack is not automatically a blunder: the 1200-ATK attacker faces both1500-ATK and750-ATK monsters, so the following target choice and effects matter.
- Root1492, Runick-facing/fixed: effect activation11/16, the two attack choices3/16 and4/16, main2 6/16, end phase4/16. Frozen/control rank end phase highest; both refreshed critics rank main2 highest. All miss the empirically best effect action. This is an investigation lead, not a certified tactical label or proof of immediate lethal.

## Confirmation design (now complete)

`critic-battle-ranking-confirmation-2026-10-10` retains **every one of the16 roots and55 actions**, regardless of pilot results. It uses64 new common suffix streams301–364 per action plus16 original controls: **3,536 games**, two workers, four-hour cap and independent watchdog.

Primary: compare terminal versus control raw-Q argmax choices on confirmation streams alone, equally weighting the fixed roots. Resample suffix trial blocks jointly across actions and roots, because the same streams are reused. Report first-index and uniform clipped-tie alternatives separately. Per-root paired comparisons and Holm-adjusted tests are secondary. Do not pool pilot outcomes into the primary estimate or infer population playing strength. No critic trains on these evaluation outcomes.

## The loss-clamp mechanism is real; changing it is not yet an improvement

The prior head-refresh fit computes squared error after clamping the calibrated Q/V prediction. Outside[-1,1], that prediction's local gradient is zero, even when its sign is wrong. Shared-network updates from other rows can still move such predictions back into range; it is not an irreversible lock.

Read-only audit of the same511-game mixed50 training corpus (4,544 rows,9,088 Q/V values):

| Checkpoint | Wrong-sign saturated values | Game-weighted fraction | Their contribution to clipped training loss |
|---|---:|---:|---:|
| Initial control | 192 | .8275% | .03310 |
| Terminal refresh | 5 | .0373% | .00149 |
| Anchored refresh | 6 | .0535% | .00214 |

Direct autograd checks confirmed zero gradient on saturated calibrated predictions. Most wrong-saturated values already disappear during the original training, weakening the hypothesis that this is the dominant remaining problem. Audit: `critic-battle-ranking-2026-10-10/loss-saturation-audit.json`.

The subsequent `critic-clamp-loss-ablation-2026-10-10` compares original clipped terminal loss against unclipped calibrated squared error, keeping training data, initialization, actor freeze, fixed residual,32 Adam steps, learning rate, checkpoint schedule and validation gates unchanged. Calibration reporting remains bounded. It completed and was verified:

- Clipped control matches **every parameter exactly** at steps0/4/8/16/32 against the original terminal-only fit.
- All10 checkpoint validation forwards, gates and selected checkpoints independently reproduce; actor hash is unchanged. The treatment optimizer was not independently rerun.
- Both arms select step32 and pass the existing validation gates.

| Loss | Broad validation MSE | Runick validation MSE | Half/half selection score | Wrong saturated training values |
|---|---:|---:|---:|---:|
| Clipped | .563516 | .747749 | .655633 | 5 |
| Unclipped | .563787 | .749993 | .656890 | 3 |

Unclipped training reduces a few saturated errors but is slightly worse on both validation panels at this matched budget; its training errors also do not improve. **Do not replace the training default or promote the treatment.** This is a reused validation panel and an exploratory mechanism test, not independent confirmation or a general rejection of unclipped losses under all tuning.

`tools/critic_regression_probe.py` isolates the two losses for research; it detaches the fixed correction and preserves per-game weighting at the caller. Tests demonstrate recovery gradients under wrong saturation, agreement inside the interval, and rejection of unintended target broadcasting. No production PPO/actor/reward behavior changes.

The ablation completion `f3ee64d9de58c63b8352a8fde5359603b9720effdc25c18ca7e470bbf046e9ed` was acknowledged after verification. Results and checks: `critic-clamp-loss-ablation-2026-10-10/verification.json`.

Update 2026-10-11: the independent confirmation is complete. The primary improvement did not replicate; a Droplet-versus-phase-transition misranking did. See [confirmation results and the separate training-data experiment](critic-battle-ranking-confirmation-2026-10-11.md). First-idle training coverage and relative-action supervision remain unresolved mechanisms. The full declaration interface issue#319 remains open; none of these diagnostics changes its fixed128-row production limit.

Validation:30 focused critic/declaration tests passed; repository Ruff lint/format and diff checks passed. Existing models and original study artifacts remain preserved.
