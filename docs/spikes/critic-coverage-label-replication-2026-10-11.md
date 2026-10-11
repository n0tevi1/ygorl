# Critic training gain survives independent labels; transfer remains unresolved

The expanded critic's training-state MSE improvement survives independent continuation labels. This rules against the narrow claim that the entire training gain was merely fitting the first set of sampled outcomes. It does not establish stronger play or generalization to new states. No checkpoint is promoted.

## Verified replication

The restarted study `critic-ranking-coverage-label-replication-restart-2026-10-11` completed all 7,408 registered tasks normally, including 128 matching original controls. Full task/action/suffix grids, per-task results, replay hashes, recorded cold replay equality and identity `aaa163b26dfbc2bbcff3cdf24109f17aab1e0860d0ae5ab8c38d601c045f9298` verify. This verification did not independently rerun every game. Completion event `7e068820ce9d0caf6dc7bc8a9f43a5ea198ddde49ac4c17ce18c2f7e839cf294` is acknowledged. The failed initial launch and its STOP remain preserved.

Three models and their predictions were sealed before collecting suffixes 1401–1416. No model was fitted to these new outcomes. A separate audit reloads checkpoints, reproduces every sealed Q prediction exactly, reads per-game outcomes independently, and recomputes means and the primary 50,000-sample bootstrap by the algebraic difference of squared errors. Results are in `analysis.json` and `analysis-verification.json`; scripts are retained beside them.

## What reproduced

On the same 96 added training states, expanded128-step32 versus unchanged initialization:

| Outcome labels | Unchanged MSE | Expanded MSE | MSE difference |
|---|---:|---:|---:|
| Original 16 suffixes | .028929 | .022002 | −.006927 |
| Independent 16 suffixes | .028783 | .023098 | −.005685 |

The independent-label difference has a conditional 95% interval [−.008030, −.003425]. The registered primary test measures how this difference changes from old to new labels: **+.001241872**, interval **[−.002236857, +.004638165]**. Thus the observed attenuation is not established, while a training-state MSE benefit persists. This does not prove there is no overfitting, or identify a single remaining cause.

On independent labels, conditional choice wins are 64.323% unchanged and 65.169% expanded: **+0.846354 percentage points**, interval **[−0.520833, +2.278646]**. That win benefit is not established. These are one critic-chosen root action followed by a frozen policy, not full-game win rates of a newly trained actor.

## Validation is sensitive to continuation sampling

On the same 32 validation states, expanded-minus-unchanged MSE changes from **+.003199** on the first suffix set to **−.004730** on the independent set (new-set interval [−.008179, −.001234]). Conditional win differences change from −2.344 to +0.977 points; the new interval [−2.539, +4.297] crosses zero. Report both sets; do not select the favorable one.

Pooling both label sets gives descriptive validation MSE .024810 unchanged, .026480 original32, and .024044 expanded128. Corresponding conditional wins are 70.703%, 70.605%, and 70.020%. These pooled figures were not the preregistered confirmation test and do not justify promotion. Absolute MSE depends on label precision, so pooled32 MSE must not be compared with single16 MSE as if only the model changed.

All replication intervals resample shared suffix blocks and condition on the already inspected roots and frozen continuations. They do not cover population-wide state variation. Current evidence supports substantial continuation uncertainty plus unresolved transfer of ranking improvements. The earlier shrinkage observation remains descriptive, not proof of a capacity or anchor-loss bottleneck.

## Next: fixed models on unused states

`critic-coverage-unused-root-confirmation-2026-10-11` registers 48 previously unused battle states from the existing 256 source games: six per eight broad/Runick × old/mirror opponent × own/face strata. Exclude all 128 prior coverage states. Select by fixed SHA256 order and the first learner battle decision with 2–6 supported actions, without consulting outcomes or errors. These source games belonged to a panel whose aggregate behavior was already inspected; this is not a wholly unseen source distribution or unseen-card test.

Freeze unchanged, original32-step32 and expanded128-step32 predictions before all supported actions receive 32 new suffixes (1501–1532), with 48 original controls: 5,360 tasks. No training, model selection, or automatic promotion. Primary is expanded-minus-unchanged centered action MSE; secondary is conditional choice wins and the original32 control. Report both a fixed-root shared-suffix interval and a stratified root-plus-suffix bootstrap sensitivity, with 50,000 draws and registered seeds 2026101121/2026101122. Even a positive ranking result still needs actual actor-training and full-game evaluation before any playing-strength claim.

The confirmation launched with identity `e0c5515f7cdea99e6487f849cbcada01a997d798680d8ea5eda4892e8262317c`, service `ygorl-critic-coverage-unused-root-confirmation-20261011.service`, two workers, a six-hour cap and an independent monitor. Source-prefix, board, LP, legal-action masks, actor probabilities and selected-Q parity were verified during feature capture; sealed inputs passed hash verification before launch.

Development/reporting follows main; the matched experiments keep runtime ae09594. Issues #61/#83/#36 remain active. See [previous coverage results](critic-ranking-coverage-results-2026-10-11.md).
