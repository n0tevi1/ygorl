# Frozen resource residual: independent confirmation and resource ablations

The [registered residual pilot](critic-resource-readout-2026-10-10.md) completed
512 healthy games on 128 new openings. All six prediction groups and aggregate/
role metrics reproduce exactly from the sealed checkpoints and coefficients.
The actor remains frozen. This confirms better offline prediction on this panel,
not better playing strength or a production-ready critic.

## Coverage and paired analysis

Game 488 ended normally without an eligible learner idle decision. Its replay,
result and zero-sample cache are verified; it remains in the 512-game outcome
total, but cannot contribute to prediction metrics. Consequently, estimation
metrics cover **511 games**, including 31 facing-Runick games. No missing
prediction is assigned zero error.

The analysis uses 20,000 stratified paired opening bootstrap replicates. It
preserves target/anchor/role strata and keeps opponent/seat games from an opening
together. With unequal eligible counts, each resampled stratum uses the ratio of
summed errors to eligible-game counts, then its original eligible-game weight.
Intervals are descriptive, unadjusted 95% intervals for this fixed actor and
single auxiliary initialization; they do not capture policy-seed uncertainty.

## Results

| Correction | Overall Q/V MSE | Facing targets | Operating targets | Facing Runick V bias |
|---|---:|---:|---:|---:|
| Frozen ordinary-MSE critic | 0.46351 | 0.66104 | 0.26675 | +0.56499 |
| Intercept only | 0.46323 | 0.66133 | 0.26590 | +0.56596 |
| Resources | **0.42614** | **0.60284** | 0.25014 | +0.25693 |
| Public context | 0.46183 | 0.65249 | 0.27192 | +0.51224 |
| Context + resources | 0.43042 | 0.61350 | **0.24805** | **+0.24384** |
| Group-tau-0.25 reference | 0.48651 | 0.65045 | 0.32322 | +0.35293 |

The registered resources-versus-frozen comparison reduces overall error by
**8.1%**, with both roles improving on this panel:

| Slice | Resources minus frozen Q/V MSE | Interval |
|---|---:|---:|
| Overall | -0.03737 | [-0.05870, -0.01468] |
| Facing targets | -0.05820 | [-0.09912, -0.01430] |
| Operating targets | -0.01661 | [-0.03036, -0.00435] |
| Facing Runick | -0.42696 | [-0.75694, -0.07974] |
| Operating Runick | -0.05321 | [-0.14533, +0.03041] |
| Familiar lists | -0.00463 | [-0.01571, +0.00625] |

Against intercept-only correction, overall change is -0.03709
[-0.05832, -0.01423]. The gain is not explained by a global offset alone.
A separately retained sensitivity analysis excludes the entire opening containing
the zero-idle game: 508 games / 127 openings give -0.03837
[-0.05755, -0.01802]. The primary result is robust to that exclusion.

Improvement is not universal. Pooled Blue-Eyes error increases 0.38740 to 0.39267
and Tearlaments 0.27407 to 0.28366. Familiar-list improvement is inconclusive;
Runick contributes much of the overall gain. Overall signed V bias moves from
-0.05231 to -0.06451 even as squared prediction error falls. Better MSE therefore
does not imply every notion of calibration improves.

At the first eligible state with own deck <=5 and LP ahead, 12 facing-Runick
games have mean terminal return -0.83333. Resource correction moves mean V from
**+0.52661 to -0.30513**, and V MSE from **2.23689 to 0.69664**. It recognizes
more of the danger but still overestimates this small, trajectory-conditioned
slice. On all facing-Runick sampled states, mean bias remains +0.25693.

The frozen actor won 412/512 games with mean length 11.617 turns. All models score
those same games; these outcomes are not evidence of a critic-driven win-rate or
turn-length improvement.

## What contributes: registered refitted ablations

`critic-residual-ablation-fit-2026-10-10` refits residuals on the original 512
balanced training games and selects each view's ridge parameter using the same
512 validation games. None of the confirmation outcomes enters fitting or
selection. Both full-resource and intercept controls reproduce the previous
fit exactly; independent refits reproduce all eight selected coefficient sets,
validation metrics and registered minima.

Resource columns contain turn, both LP totals, then each player's deck, hand,
graveyard, banished and extra-deck counts. The implementation verifies LP indices
1/2 and deck-count indices 3/8. Removing a group means refitting without those
columns, not feeding out-of-distribution zeros into an existing residual.

| Residual view | Selected ridge | Validation Q/V MSE |
|---|---:|---:|
| Intercept | 0.0001 | 0.45546 |
| All resources | 0.0001 | 0.44062 |
| Without LP | 0.0001 | 0.44136 |
| Without both deck counts | 0.0001 | 0.44841 |
| LP only | 10 | 0.45552 |
| Deck counts only | 10 | 0.45574 |
| LP + deck counts | 10 | 0.45581 |
| Remaining resources | 0.0001 | 0.44749 |

Validation suggests little incremental contribution from LP compared with the
full resource vector; deck counts contribute more conditional on other resources.
Yet neither LP nor deck counts alone help. This is consistent with resources
being useful together, and does not identify a unique causal feature: the base
critic already sees information about these resources and remains unchanged.
The residual is linear, so this is not evidence of learned nonlinear interactions.
Correlated features, validation selection and outcome distributions can all
affect these differences. A universal LP penalty remains unsupported.

`critic-residual-ablation-confirmation-2026-10-10` completed all eight
selected views and two fixed references on another 512-game / 128-opening panel,
disjoint from 1,088 prior openings. Primary comparisons are all resources versus
without LP and without deck counts. An independent watchdog monitors progress,
health and completion. The next analysis must again retain zero-idle games in
outcome totals, report both-role retention and target slices, and preserve the
sealed evidence. No actor or PPO checkpoint is changed.

The [ablation confirmation and initialization replication](critic-residual-ablation-confirmation-2026-10-10.md)
report another aggregate gain, inconclusive incremental LP/deck contributions,
and failure to clearly reproduce the preceding own-role improvement.

Artifacts under `out/research/` retain identities, registrations, selections,
verification, predictions and paired analysis. The optional complete-opening
sensitivity is stored separately from the original verification used by the
next experiment. Remaining steps before any actor-facing experiment include
replication across critic initializations and checking calibration under policy
shift; current evidence is limited to this fixed actor and deck panel.
