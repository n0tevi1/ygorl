# Critic resource confirmation: small gains, unresolved role-dependent bias

The [resource/calibration factorial](critic-resource-factorial-2026-10-10.md)
has completed independent confirmation. All 512 games finished without limits
or engine-health failures. Cold response replay, cached critic replay, and
actor probability checks passed. All 14 checkpoint forwards and aggregate
metrics were reproduced exactly, with unchanged actor hashes. The 12 selected
checkpoints were sealed before confirmation. No critic is promoted to PPO.

## Same-game comparison

There are 128 new ordered opening-deal pairs, disjoint from the previous 448.
Eight target lists, two anchors, four draws, two opponents and both seats cover
both learner roles. Every critic scores the same frozen actor's states. Scores
average states within each game, then games equally; all 512 games have eligible
idle states. Q/V targets are terminal returns in {-1, +1}, not win probabilities.

The table averages **metrics** over three auxiliary initializations; it is not
an ensemble or three independent policy-training runs.

| Critic | Overall Q/V MSE | Familiar Q/V MSE | Facing Runick V bias | Operating Runick V bias |
|---|---:|---:|---:|---:|
| Unchanged | 0.61888 | 0.51527 | +0.27883 | -0.60109 |
| Matched ordinary fit | 0.52976 | 0.38932 | +0.30240 | -0.52760 |
| Resource inputs | 0.52940 | 0.38898 | +0.29886 | -0.52852 |
| Conditional calibration | 0.52718 | 0.38420 | +0.29802 | -0.51903 |
| Both | 0.52688 | 0.38386 | +0.29429 | -0.52018 |

Most of the improvement is ordinary fitting. Both changes together reduce MSE
by 0.00288 (0.54%) relative to the matched control. A 20,000-replicate paired
opening-deal bootstrap gives a descriptive 95% interval of [-0.00474, -0.00104].
It preserves target/anchor/role strata and keeps the four opponent/seat games
from each opening together. Initializations are averaged before bootstrapping.
These intervals are unadjusted for multiple comparisons and describe this fixed
actor, card-list and opponent panel, not all policies or decks.

Role separation changes the interpretation:

- Facing target lists: combined-minus-control MSE is **+0.00043**, interval
  [-0.00283, +0.00368]. There is no clear improvement in this role.
- Operating target lists: the difference is **-0.00619**, interval
  [-0.00807, -0.00430]. This accounts for the overall gain.
- Facing Runick specifically: bias decreases by 0.00811 versus the fitted
  control, but remains worse than the unchanged source. MSE changes by +0.00127,
  interval [-0.00948, +0.01086], on only 32 games / eight openings.

Pooling Runick roles gives combined V bias **-0.11294**, hiding opposite errors:
overestimation when facing Runick and substantial underestimation when using it.
The underlying actor won 20/32 and 28/32 games respectively. This is not evidence
that either role is generally easier; the opponent panel and samples are limited.

## Low deck plus LP advantage remains misleading

Choose the first eligible idle state with own deck count <=5 and own LP greater
than the opponent's, using public state only. Among the 15 facing-Runick games
that reach this condition, mean terminal return is -0.20:

| Critic | Mean V | V bias | V MSE |
|---|---:|---:|---:|
| Unchanged | +0.43867 | +0.63867 | 1.59512 |
| Matched ordinary fit | +0.33547 | +0.53547 | 1.46442 |
| Both | +0.31184 | +0.51184 | 1.46432 |

The additional branch/objective barely changes error beyond ordinary fitting in
this small, trajectory-conditioned slice. This supports continued investigation
of LP/resource interpretation; it does not prove LP itself causes the error or
justify a universal LP penalty. The broader first-low-deck slice has 16 games;
combined bias is +0.39090 versus control +0.41596, but V MSE is slightly worse
(1.49940 versus 1.49136). Bias alone is an insufficient acceptance criterion.

The actor's mean game length is 12.05 turns overall, 30.91 when facing Runick and
24.72 when using it. These are fixed-actor panel measurements, **not improvements
attributable to the critic changes**, and are not directly comparable to earlier
panels with different role composition.

## Next controlled experiment: role coverage

Audit confirmed the preceding critic corpus contains only `face-target` games:
512 train, 256 validation and 512 test. The missing `own-target` role is a concrete
coverage gap, not yet a proven cause of all observed biases. It motivates a
bounded data-coverage experiment before more architecture changes.

`critic-role-coverage-data-2026-10-10` is collecting **512 new own-target games**:
256 training and 256 validation, with 128 new openings and no overlap against
576 prior openings. Train and validation opening sets are disjoint. The completed
confirmation remains diagnostic/test data and is excluded from fitting/selection.

The registered subsequent factorial holds training at 512 games per arm:

- Face-only: the original 512 training games.
- Balanced: 256 original face-target games (preselected draws 0–3) plus 256 new
  own-target training games.
- Cross each with MSE only or MSE plus coefficient-0.5 calibration grouped by
  **target and role**. Group labels affect the objective only. All arms use the
  same public-resource architecture, frozen actor, 32 full-corpus updates,
  learning rate 1e-4 and three registered auxiliary initializations.

All arms select on the same balanced 512-game validation panel: original 256
face-target plus new 256 own-target validation games. A new independent
512-game confirmation follows sealed selection. The primary MSE-only contrast
isolates replacing half of face-only data with own-role coverage at equal game
count; it does not isolate adding more data. Zero-idle counts must remain explicit.
Report both roles, target lists, retention and low-deck slices. Actual policy
adoption still requires matched RL continuation and retention tests.

Artifacts: `out/research/critic-resource-confirmation-2026-10-10/` retains sealed
inputs, original report, `predictions.pt`, post-selection `analyze.py`,
`analysis.json` and `completion-verification.json`. The new data study retains
its panel, opening audit, `fit-plan.json`, runtime identity and independent
watchdog. Completion of data collection is not completion of the follow-up fit.
