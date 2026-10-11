# Action-conditional critic supervision: retention recovered, strength not established

The 2,312-game branch collection is complete and verified. The first fitting comparison did not improve root-action choices and failed retention. A subsequent Q-head-only comparison passes retention with a relative-action objective, but still selects the unchanged model's actions on the eight reused pilot roots. **No model is promoted.**

## Data and first registered fit

`out/research/critic-battle-branch-data-2026-10-11` contains 32 training and eight validation roots, 142 supported actions × 16 suffix streams and 40 original controls. Identity, exact task grids, recorded cold-replay checks, replay hashes, split separation and all controls passed (`verification.json`, `verify_data.py`). Completion `656edcd91b76b50d51f4d31734b17cf1dfefbc541bbce0bdd1837b0baafbbdc1` was acknowledged.

`critic-battle-supervision-fit-2026-10-11` was registered before analyzing branch validation outcomes. Both arms start from the original terminal-refresh step32 critic, freeze the actor and fixed calibration residual, and use 32 Adam steps at1e-4. The objective adds one unit of branch Q error to one unit of original mixed50 terminal Q/V error. The control supervises the originally selected action's 16-stream mean; the treatment supervises every supported action's mean. Actions are averaged within each root, then roots equally weighted. This isolates extra action coverage at a fixed optimizer budget, **not** an equal-label or equal-collection-cost comparison.

All-action features were captured by exact source-prefix replay with matching action menus, masks, public board, actor probabilities and chosen-action head parity. Both fits finished and saved every checkpoint before branch validation outcomes were analyzed. Actor hashes stayed unchanged.

| Fixed step32 model | Conditional wins | Raw action MSE | Centered action MSE | Familiar calibration MSE change | Retention |
|---|---:|---:|---:|---:|---|
| Unchanged initialization | 63.281% | .423010 | .018864 | — | Pass |
| Selected-action mean | 63.281% | .357255 | .076127 | +2.91% | Fail |
| All-action means | 63.281% | .346338 | .048578 | +3.37% | Fail |

Both trained arms choose exactly the same actions on all eight roots. The registered all-action-minus-selected benefit and suffix-bootstrap interval are therefore zero; this is **not evidence of zero population uncertainty or equivalence**. Both exceed the preregistered 2% familiar-retention tolerance. Absolute-value calibration improved while action-difference error worsened.

The equal-action MSE decomposes exactly into squared root-mean error plus centered action error. At initialization, .404146 of .423010 (95.54%) is the root-mean component. The all-action fit lowers that component while increasing centered error from .018864 to .048578. This verifies a concrete failure mode: improving the absolute prediction objective can hide worsening relative action values. It does not prove a relative objective alone fixes generalization.

The sample is not entirely uninformative: only five of32 training roots and one ofeight validation roots have identical paired outcomes for every action. Median observed within-root win-fraction spread is .125 in both splits; these maxima/minima are noisy and selection-biased, not true action advantages. See `coverage-diagnostic.json`.

## Q-head-only absolute versus relative supervision

`critic-battle-relative-fit-2026-10-11` is a subsequent **exploratory** comparison using the same32 training roots and eight now-reused validation roots. Both arms reset to the original initialization. Only the66,177 Q-head parameters train; actor, private encoder, trunk, resource path, V head and ridge stay frozen. Original-corpus raw-prediction distillation replaces the preceding terminal-MC anchor in both arms. Thus the controlled comparison here is absolute versus centered branch error; comparisons across the two studies also change trainable modules and anchoring.

Centering predicted and target Q over supported actions removes shared state offsets. Its MSE equals half the mean squared error of all ordered pair differences, with equal root weight. It is a mechanism test using paired continuation labels, not a card-specific tactical rule or hard mask.

| Fixed step32 Q-head objective | Conditional wins | Raw action MSE | Centered action MSE | All retention gates |
|---|---:|---:|---:|---|
| Absolute | 61.719% | .337834 | .038045 | Fail (familiar) |
| Relative | 63.281% | .414925 | .022515 | Pass |

Relative-minus-absolute is +1.5625 pp, with an exploratory shared-suffix bootstrap95% interval of −1.5625 to +4.6875 pp. Relative selects exactly the unchanged initialization's actions. Its centered error is still worse than initialization. The benefit is preserving retention compared with the absolute objective, **not demonstrated stronger decisions**. All V outputs and all non-Q parameters remain unchanged.

For each fitting study, independent verification reproduced nine validation forwards, all retention gates, checkpoint hashes, exact step0 parameter equality, actor hashes and the registered primary bootstrap. The second additionally checks every non-Q parameter. Optimizer training was not independently rerun. Artifacts include `registration.json`, `run.py`, `validation.json`, `verification.json` and `verify_fit.py`. Completion events `c93af862c6cfb4f5ae18d2e6cb15e50f24e568bcc13329b832a78afb28c1f937` and `92916b1a5fc7d6106b7ba2663a1a49dd5b71fb8c03d198319b7b4459bc7893b6` were acknowledged after verification.

## Conditional confirmation (now complete)

`critic-battle-relative-confirmation-2026-10-11` registers 64 distinct openings, eight per broad/Runick × old/mirror × own/face stratum, excluding all40 branch-pilot roots and the earlier16 ranking roots. Fixed hash order and the first battle prompt with2–6 supported actions determine inclusion, without consulting outcomes or critic errors. There are217 supported actions ×16 new common suffix streams601–616 plus64 original controls: **3,536 games**. Model predictions and checkpoint hashes were frozen before collecting suffix outcomes. The relative model changes13 of64 choices versus initialization. Identity SHA256: `baf59366ed7143ec14c52bf7d38fce774037578399f7225411384765eaab0bcb`.

Primary: fixed relative-step32 versus unchanged raw-Q argmax, equal weighting of64 roots;50,000 shared-suffix-block bootstrap replicates with seed2026101103. Absolute-step32 is a secondary comparison, not a candidate for promotion. Original learner/opponent policies continue after the root action. Two workers, four-hour limit, original controls before interventions, cold replay and independent watchdog.

These are new branch roots and streams, but their source validation trajectories previously contributed to calibration/retention panels. They are **not globally untouched game data**. No gradient fitting uses these openings; no new branch outcome is available when choices are frozen. Any favorable conditional result still needs fresh-deal full-game testing before a playing-strength claim.

`tools/critic_branch_probe.py` provides research losses only. Tests cover equal root weighting with different action counts, ignored padding, selected-action gradient isolation, nonzero recovery gradients beyond the return range, target detachment, pairwise equivalence and invariance to common state offsets. Fifteen focused tests passed. Production PPO, rewards, actor inference and native engine are unchanged; declaration capacity issue#319 remains open.

Update: the64-root confirmation found no demonstrated action-selection gain; see [results and the training-target replication](critic-battle-relative-confirmation-2026-10-11.md).
