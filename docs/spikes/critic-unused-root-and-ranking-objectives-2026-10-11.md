# Better critic MSE still does not establish better action choices

Follow-up: [target-sample intervention results and new-state coverage plan](critic-target-replica-results-2026-10-11.md).

The 48-root confirmation does not establish a playing-strength improvement. Its MSE point estimate improves, but both uncertainty intervals cross zero and conditional choice wins do not increase. Follow-up controls demonstrate that shrinking score differences can improve MSE without changing any action. Neither a pairwise loss nor more optimization establishes better choices on the reused diagnostic panels. No model is promoted.

## Previously unused-root confirmation

`critic-coverage-unused-root-confirmation-2026-10-11` completed all 5,360 tasks normally, including 48 matching original controls. Task/action/suffix grids, result/replay hashes, recorded cold replay equality and sealed identity `e0c5515f7cdea99e6487f849cbcada01a997d798680d8ea5eda4892e8262317c` verify. A separate audit reloads all checkpoints, reproduces predictions exactly, and recomputes metrics plus both bootstraps. This verification did not independently rerun every game. Completion event `0b3cb9690644ceddb8ae75613a130eabf6c4159d7abd216dbf4ac2fe0d3fdd1b` is acknowledged.

| Fixed model | Centered action MSE | Conditional choice wins |
|---|---:|---:|
| Unchanged initialization | .016696 | 61.458% |
| Original32, step32 | .017194 | 60.938% |
| Expanded128, step32 | .015190 | 61.003% |

Primary expanded-minus-unchanged MSE is **−.001506031**; 95% shared-suffix interval **[−.003117259, +.000112512]**, stratified root-plus-suffix interval **[−.004918754, +.001693107]**. Conditional wins differ by **−.455729 percentage points**; corresponding intervals are **[−1.497396, +.520833]** and **[−2.213542, +1.106771]**. Keep the broader state uncertainty visible rather than treating the near-zero conditional bound as almost passing a strength threshold.

These are 48 battle decisions, six per eight strata, drawn from previously unused games in an existing source panel whose aggregate behavior had already been inspected. Each supported action has 32 new continuation trials. The critic selects one action and a frozen policy continues. These are not newly trained actor full-game win rates, unseen-card generalization, or a new turn-length result.

## Shrinkage control: lower MSE without better decisions

`critic-ranking-shrinkage-audit-2026-10-11` is a post-hoc diagnostic on the already inspected data. One scalar, constrained to [0,1], is fitted using only the original128 training targets with equal-root weighting. Its value **.129747755** multiplies the unchanged critic's centered action scores and preserves their ordering. A second control matches the expanded model's centered-score norm separately at each root while keeping the original direction/order; it uses model predictions, not outcome labels.

| Readout on 48 roots | MSE | Conditional wins |
|---|---:|---:|
| Unchanged | .016696 | 61.458% |
| Training-fitted scalar | .012971 | 61.458% |
| Expanded-norm, original order | .014818 | 61.458% |
| Expanded trained head | .015190 | 61.003% |

The scalar and all panel metrics are independently recomputed, including the scalar optimum through an autograd derivative of the original centered-MSE implementation. Expanded-minus-norm-control MSE is +.000372, but its root-plus-suffix interval [−.000406, +.001219] crosses zero. Expanded-minus-scalar is +.002219, also uncertain across roots [−.000258, +.005372]. This demonstrates a metric/decision distinction; it does not identify a unique causal training mechanism or justify replacing the critic with a scalar. The scalar actually worsens point MSE on the preceding pooled32 validation panel (.024810 to .025241).

Expanded changes 12/48 choices; sampled conditional outcomes improve at two, worsen at five, and tie at five. These are descriptive paired estimates, not definitive error labels. Score magnitude calibration alone is insufficient evidence of better play.

## Pairwise objective pilot and longer fitting

`critic-pairwise-objective-pilot-2026-10-11` compares the same128-root centered MSE fit against a fixed pairwise logistic loss. Each unordered action pair contributes probabilities of paired terminal-outcome discordance times softplus of the signed Q margin at temperature .1; ties contribute zero and stay in the trial denominator. Both arms retain the same source anchor, frozen actor/trunk/V/ridge, Adam settings and32 updates. This changes objective and gradient scale, not a scale-matched intervention. Four focused tests pass: gradient direction, all-tie zero gradient, permutation/shift invariance and discordance weighting.

The MSE arm reproduces existing expanded128 parameters exactly at steps0/8/16/32. Independent verification covers every checkpoint's predictions, retention metrics, non-Q immutability and the primary bootstrap. At step32, pairwise MSE is .013753 and conditional wins61.393%, versus MSE-arm .015190 and61.003%. The win difference **+.390625 pp** has interval **[−.260417, +1.041667]**. All retention gates pass, but no strength gain is established and the unchanged61.458% point result remains higher. Centered prediction energy shrinks further, from .001292 in the MSE arm to .000416 in the pairwise arm.

`critic-pairwise-fit-continuation-2026-10-11` resumes both model and full Adam state exactly at32 and continues unchanged data/objectives to fixed256, saving32/64/128/256. Both complete and pass independent prediction/retention/bootstrap and resume-state checks. No evaluation outcomes enter fitting; the diagnostic panels have nevertheless been inspected repeatedly and are not fresh confirmation.

| Step256 model | Independent-label training MSE | 48-root MSE | 48-root conditional wins |
|---|---:|---:|---:|
| MSE objective | .027360 | .024028 | 61.263% |
| Pairwise objective | .022776 | .015602 | 61.263% |

The two arms' conditional win difference is zero, interval [−1.041667, +1.041667] pp. All retention gates pass. More fitting does not establish a better decision policy; the MSE arm's diagnostic error grows substantially. Do not select an earlier favorable checkpoint after inspection.

## Labels and learnability audit

Among432 legal action pairs at the96 new training roots, 214 have nonzero estimated return differences in both independent16-suffix batches. Only126 agree in direction (88 reverse). A descriptive subset of71 pairs across33 roots agrees in direction and has absolute signed-return margin at least .25 in both batches. This post-hoc subset is not ground truth, contains correlated pairs, and never enters the optimizer as selected training data.

On those71 pairs, the unchanged model orders36 correctly, step32 MSE45 and pairwise43, and step256 both64. Thus many stronger repeatable training preferences are learnable with more updates, but that progress does not translate into better diagnostic choice wins. There are zero exactly identical candidate-feature pairs among the432 checked pairs; this rules out exact within-root candidate collisions here, not all representation deficiencies. Noise, state coverage and interference/generalization remain relevant; neither “only noise” nor “only too few updates” adequately explains all observations.

## Next controlled target-sample intervention

`critic-target-replica-fit-2026-10-11` fixes the architecture, centered MSE objective, source anchor and32-update budget. Three arms use the first16, independent second16, or pooled32 labels at the same96 new training states; original32 training states retain their existing pooled32 targets. The first arm must exactly reproduce expanded128. The second96 label set now enters training for two arms, so its metrics are explicitly called reused training diagnostics, not independent holdout results. No other evaluation-root labels enter fitting.

This intervention is now motivated by observed label-direction sensitivity and failed objective/budget controls; it is not an automatic refit after replication. It measures target-sample sensitivity on reused32/48 diagnostic panels, without checkpoint or hyperparameter selection. A positive candidate would still require new source games, independent labels and actor-RL/full-game validation. No deployment or production training changes are made.

Evidence lives in each named study's registration, identity, scripts, reports and verification artifacts under `out/research/`. The initial failed replication STOP remains preserved. Completed pilot event `02013da59cf30a1c8849a48d509ccf651e218e966c035762c56422011d1b7a41` and continuation event `c6f702456469bfa365ff6d2a016633f18663a72122f9ad44351fadcaae640f28` are acknowledged after verification. Issues #61/#83/#36 remain active. Development follows main; matched runtime remains ae09594.
