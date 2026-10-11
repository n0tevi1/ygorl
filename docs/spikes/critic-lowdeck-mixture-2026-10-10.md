# Low-deck coverage improves validation; confirmation reveals tradeoffs

Update: [independent confirmation](critic-lowdeck-mixture-confirmation-2026-10-10.md)
verified all 768 games and four prediction groups. Low-deck gains replicate,
but own-Runick and familiar-retention regressions block adoption. A frozen-context
diagnostic and its fresh confirmation are the next steps.

The [opponent-coverage confirmation](critic-opponent-coverage-confirmation-2026-10-10.md)
exposed persistent low-deck overestimation despite pooled gains. The follow-up
`critic-lowdeck-coverage-data-2026-10-10` completed 512 healthy, unselected full
Runick games, split 256 training / 256 validation. Each split covers both
opponents, roles, anchors and seats. All 128 openings are disjoint across splits
and from 1,728 prior openings. Independent verification checks every replay and
critic cache hash. Training game 89 has no eligible idle sample; its outcome is
retained but it contributes no critic gradient or error.

## Coverage and the meaning of low deck size

The low-deck condition is own deck count <=5 at a sampled learner idle state.
Raw observation globals store card counts; normalized resource features divide
these by 40. Games and openings, not sampled turns, are the independent units.

| Split | Eligible games | Games reaching low deck | Unique low-deck openings | Low-deck sampled states |
|---|---:|---:|---:|---:|
| New training | 255 | 81 | 41 | 361 |
| New validation | 256 | 73 | 38 | 291 |

The prior balanced training set had only seven low-deck games. New training
includes 62 when facing Runick and 19 when operating Runick; validation has
52 and 21 respectively. Training mean terminal return among those games is
-0.67742 when facing Runick but +0.47368 when operating it. Validation shows
the same directional split (-0.69231 / +0.42857). These observational groups
are not causal effects of removing deck cards. They argue against universally
penalizing low deck size without accounting for the position and strategy.

Mean full-game length is 24.895 turns in training and 24.172 in validation.
These are unchanged-actor Runick matchups, not the broad distribution or a
critic-driven reduction in game length. Engine health is verified; health
alone does not establish tactical quality or rule out strategically pointless
play.

## Fixed mixture fit

`critic-lowdeck-mixture-fit-restart-2026-10-10` keeps the actor, base critic and
14-feature additive opponent residual architecture unchanged. It combines the
same 256 eligible broad-training games with 255 eligible new Runick games.
The two preregistered arms give the new corpus 25% or 50% of total training
mass. Within each corpus, every eligible game gets equal mass regardless of
the number of sampled turns. Normalization uses these training weights only.
This is a coverage/weight experiment with more games, not an equal-budget
comparison against the broad-only control.

`tools/critic_mixture_probe.py` enforces fixed corpus masses, sample-to-game
alignment and exclusion of zero-sample metadata. Tests cover unequal trajectory
lengths, duplicating samples, invalid masses and zero-sample games.

The six-value ridge grid is unchanged. Each arm selects the lowest mean of
broad and new-Runick validation Q/V MSE among candidates satisfying all gates:

- Broad, old-opponent and familiar Q/V MSE must each be <=1.02 times the fixed
  original broad-conditioned control.
- First-low-deck V MSE on new Runick validation must not exceed that control.

The 2% allowance is an experimental selection rule, not statistical evidence of
non-inferiority. All candidates and gate results are retained. Prior confirmation
outcomes never enter fitting or selection. Four of six candidates qualify in
each arm; both select lambda=0.0001. Independent verification refits all twelve
coefficient sets exactly, reproduces all metrics and verifies corpus mass and
selection gates.

An initial attempt stopped because the audit gate requested `familiar` instead
of the existing `exposure:familiar` metric key. Its STOP and coefficient evidence
remain intact. The separate restart corrects only this key, preserving data,
mixture weights, thresholds and selection rules.

## Validation results

Errors use terminal returns in [-1,+1]. These validation sets selected ridge
strength, so the results below do not establish independent generalization.

| Estimator | Broad Q/V MSE | New Runick Q/V MSE | Runick low-deck V MSE | Runick low-deck V bias |
|---|---:|---:|---:|---:|
| Original broad-conditioned control | 0.58013 | 0.87981 | 1.56449 | +0.53912 |
| 25% new-corpus mass | 0.55110 | 0.77088 | 1.09639 | +0.02646 |
| 50% new-corpus mass | 0.54551 | 0.74549 | 1.00361 | -0.04260 |

Low-deck metrics use the first qualifying state per game, with 73 games in the
new validation slice. Reduced average bias does not mean individual predictions
are accurate: V MSE remains around one. Coverage appears promising, but the
fixed learner, known opponents, linear residual and card-pool limits remain.

There is a role tradeoff outside the preregistered selection gates. Facing
Runick Q/V MSE falls from 1.08430 to 0.79196 / 0.75483, but operating Runick
worsens from 0.67532 to 0.74980 / 0.73615. Broad old-opponent error improves
0.44210 to 0.40301 / 0.40081, and familiar error 0.46763 to 0.45155 / 0.45128.
The own-Runick regression is therefore explicitly carried into confirmation;
passing the original gates is insufficient for adoption.

## Next: separate broad and Runick confirmation panels

`critic-lowdeck-mixture-confirmation-2026-10-10` ran with an independent
watchdog: 512 broad games plus 256 Runick games, on 192 openings disjoint from
all 1,856 prior openings. Four estimators score identical outcomes: frozen base,
original broad-conditioned control, mixed25 and mixed50. Each panel is reported
separately so the enriched Runick distribution cannot masquerade as broad play.

Primary comparisons are each mixture minus control on each panel, first-low-deck
V MSE/bias in Runick, and broad old/familiar retention. Opening-paired uncertainty,
both roles, per-target failures and zero-idle denominators remain required.
Both mixture arms are reported without choosing a winner on confirmation.
No automatic production PPO promotion; learner-policy-shift and matched RL
continuation remain subsequent gates.

Local validation: 21 mixture/opponent/residual/role helper tests pass. Evidence
is preserved in the named `out/research/` directories, including collection
verification, coverage audit, registrations, all coefficient candidates, exact
fit verification and monitor handoffs.
