# Doubling critic training states does not yet establish better choices

The registered128-versus256-state comparison is complete. On64 fresh validation roots, expanded256 has a favorable conditional choice-win point estimate, but the primary interval includes zero. It also does not establish improvement over the unchanged initialization. **Do not promote a checkpoint, enlarge the sample adaptively, or scale RL from this result.** This completes the bounded comparison already registered before collection; it is not justification for another expansion.

## Registered result

Both arms use the same initialization, centered action-return MSE, source prediction anchor coefficient1, Adam1e-4/epsilon1e-8, gradient clip1 and32 updates. Only the training-root set changes. Existing128 retains the canonical labels: original32 pooled32 suffixes plus96 old16 suffixes. Expanded256 adds128 new roots with16 suffixes each. Equal-root weighting and additional sampling make this a data-coverage intervention, not a compute-matched comparison. Actor, trunk, V head and ridge correction remain frozen.

All14,688 branch tasks ended normally, with no recorded engine-health failures. Verification checks every task/result, replay hash and recorded cold-replay equality;192 original controls reproduce the source result. This does not independently rerun every branch. The8 checkpoint files freeze before opening the64 new validation aggregates. Existing128 reproduces the earlier expanded128 parameters exactly at0/8/16/32. Independent analysis verification reproduces predictions, labels, MSE, conditional wins, retention and both bootstrap calculations. All retention gates pass, but that alone is not an improvement gate.

| Fixed step32 model | Centered action MSE | Conditional choice wins |
|---|---:|---:|
| Unchanged initialization | .015814 | 66.357% |
| Existing128 | .015970 | 65.723% |
| Expanded256 | .015480 | 66.650% |

Primary expanded256-minus-existing128 choice-win difference is **+.927734 percentage points**, with registered stratified root-plus-suffix95% interval **[−.195313, +2.490234] pp**. MSE difference is−.000490110, interval[−.001818658,+.000883014]. Relative to unchanged initialization, the choice-win difference is only+.292969 pp, interval[−.585938,+1.367188]; MSE difference−.000333956, interval[−.003143143,+.002412068].

The secondary fixed-root suffix interval for the primary win contrast is positive,[+.390625,+1.464844] pp. It conditions on these particular64 roots and omits root uncertainty. It must not replace the registered primary interval. Both use50,000 draws; root-plus-suffix seed2026101126 and fixed-root suffix seed2026101127. All intermediate checkpoints remain reported, without selecting a best step.

These are critic-selected first actions followed by frozen policy continuations. They are not actual actor full-game win rates, reductions in average turns, unseen-card performance or evidence of top-level play. The64 validation roots have now been inspected; subsequent tuned candidates cannot call them fresh confirmation or silently move them into training.

## What changed in the choices

The post-hoc descriptive audit finds12 changed choices versus existing128:3 with positive,2 with negative,7 with identical sampled return differences. All three positive cases restore the unchanged initialization's choice. Roots114/116 change from entering main2 back to attacking; root69 changes from activating an effect back to attacking. This suggests that some relative benefit is recovery from the earlier fit's regression, rather than newly learned behavior beyond initialization. The32 sampled continuations peraction are not definitive tactical labels, and this is not grounds for an always-attack heuristic. Compared with initialization,14 choices change:4 positive,3 negative and7 identical sampled differences.

## Budget audit and correction to the scale decision

The earlier experiments justified testing a coverage hypothesis, not predicting that doubling coverage would work. More precise labels and longer optimization improved some training/error metrics without establishing better held-out choices. Coverage was one remaining explanation, alongside representation and continuation-label limitations. There was **no prospective power or cost-benefit calculation supporting the exact doubling**. A four-worker/eight-hour cap bounded cost; it did not supply that missing justification.

The user challenged this before branch completion. A retrospective sensitivity audit was then run using only the previously inspected48-root panel, before opening the new64 validation labels. It projects8 roots perstratum by resampling the6 historical roots in each of8 strata, with shared32-suffix resampling. Under normal approximations, two historical model contrasts suggest an80%-power detectable effect around**1.74–2.09 pp**; a1-pp effect has only approximately**27%–36%** power. These are illustrative projections from sparse, adaptively inspected old data, not a calibrated guarantee for new256-versus128 choices. The1/2/5-pp examples are not new acceptance thresholds selected after seeing results.

Actual branch collection consumed **3h12m45s wall time,12h46m53.526s CPU and4.1 GiB peak memory**, per the systemd journal. No GPU was needed for this branch stage. Source collection and fitting are additional costs. The source cohort grew256→320 games; canonical training roots128→256; branch tasks7,408→14,688. Architecture and optimizer-update count did not grow.

The correct follow-up is to retain this result and investigate the choice regressions using existing evidence. Any new sampling proposal should state the mechanism, minimum worthwhile effect, root-versus-suffix budget and stopping rule before launch. A confidence interval crossing zero is not an instruction to collect until it excludes zero. There is no automatic enlargement or RL/model promotion from this study.

That follow-up is now complete: [full regression attribution and a bounded margin-preservation pilot](critic-regression-attribution-2026-10-11.md). Shared Q-head updates transmit both repeatable and ambiguous training preferences to other states. Preserving initial margins for ambiguous pairs restores some inspected attack choices but does not establish improvement over initialization on the reused diagnostic panel. No rollout expansion or model promotion follows.

## Evidence and tracking

- `critic-ranking-state-expansion-branches-2026-10-11`: identity `b075296af6ddc48c4899471c9da4da9c3d828ceb7c27a941451df71aaa66dbb5`, `verification.json`, `selection-verification.json`, `canary-verification.json` and original source traces.
- `critic-ranking-state-expansion-fit-2026-10-11`: identity `537002b0883acf4d24ab4e6d4c8ca8a4c0fe0cebc5f822fad5f80ccb11304eee`, registration,8 checkpoints, `fit-complete.json`, `analysis.json`, `verify_analysis.py`, `analysis-verification.json` and `changed-root-audit.json`.
- `critic-state-expansion-budget-audit-2026-10-11`: executable old-data projection and hashed inputs/report. The computation preceded new validation aggregation; it was retrospective to collection.

Branch completion event `98b0624e7b17afe5ea1589f7a48b56f6b78bf7dd627994445566bdd5a577aeec` and fit completion event `1fc68bb6afac1084c06c316ca17fa66077de00c48df4bbee527b822a4e1ee50c` are acknowledged after their respective verification. No STOP evidence is removed. Issues #61/#83/#36 remain active. Development follows main; matched runtime remains ae09594. [Protocol and source behavior](critic-target-replica-results-2026-10-11.md).
