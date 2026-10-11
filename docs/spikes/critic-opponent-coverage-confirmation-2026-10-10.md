# Opponent coverage confirms pooled gains, retention remains unresolved

The [registered coverage fit](critic-opponent-coverage-fit-2026-10-10.md) was
evaluated on 512 healthy games / 128 new openings, disjoint from 1,600 prior
openings. All games have eligible critic samples. Independent verification
checks all replay/cache hashes and exactly reproduces all four prediction
groups and reported metrics. Actor, base critic and selected residuals stayed
frozen. The completion event was acknowledged after verification.

## Independent confirmation

Errors use terminal returns in [-1, +1], with equal weight per game. Lower
Q/V MSE is better; V bias is prediction minus observed return.

| Estimator | Pooled Q/V MSE | Old Q/V MSE | Mirror Q/V MSE | Old V bias | Mirror V bias |
|---|---:|---:|---:|---:|---:|
| Frozen base | 0.62403 | 0.44076 | 0.80730 | -0.05499 | +0.30145 |
| Old-only resources | 0.65479 | 0.41297 | 0.89660 | +0.02522 | +0.43296 |
| Balanced resources | 0.58058 | 0.45133 | 0.70982 | -0.08630 | +0.23063 |
| Balanced + opponent | 0.55590 | 0.43565 | 0.67616 | +0.01039 | +0.13100 |

The conditioned estimator reduces pooled error by 10.9% relative to the frozen
base. The two primary contrasts both improve, using 20,000 stratified paired
opening bootstrap replicates that keep opponent/seat outcomes together:

| Comparison | Q/V MSE difference | Descriptive 95% interval |
|---|---:|---:|
| Balanced minus old-only | -0.07421 | [-0.09877, -0.04907] |
| Conditioned minus balanced | -0.02467 | [-0.03755, -0.01206] |
| Conditioned minus frozen | -0.06813 | [-0.09147, -0.04447] |

Intervals are unadjusted; only one frozen learner and one auxiliary base
initialization are represented. The original 255 versus 256 eligible-training
game mismatch remains. This is confirmation across openings for two known
opponents, not unseen-opponent or learner-policy-shift validation.

## Retention and rare states block adoption

Conditioned old-opponent error remains worse than old-only fitting:
+0.02268 [0.00874, 0.03706]. Against the frozen base its old-opponent difference
is -0.00512 [-0.02749, 0.01770], so an old-opponent improvement is not established.
Conditioning versus balanced improves mirror error by -0.03366
[-0.05241, -0.01532]; its old-opponent interval still crosses zero.

Both target roles improve overall against the frozen base. Tearlaments error
falls 0.52210 to 0.46114, difference -0.06096 [-0.11159, -0.01107]. However,
facing Runick remains problematic: Q/V MSE is 1.36340 frozen, 1.14029 old-only,
1.20264 balanced and 1.27268 conditioned. Conditioned minus old-only is
+0.13239 [0.06290, 0.21326], on 32 games / eight openings. This small slice
does not justify declaring the Runick problem solved despite lower signed bias.

At the first state with own deck size <=5, 21 games contribute. Frozen V bias
is +0.69665 and conditioned bias +0.69298; V MSE falls from 1.82656 to 1.39516,
but strong overestimation persists. Old-only V MSE is 1.14502 on the same slice.
Only seven balanced-training games cover low-deck states (three old, four
mirror; 23 sampled states). Old-only training has five such games. Additional
states within a game are not independent terminal labels. Sparse coverage and
the shared resource slopes both merit investigation; neither is established
as the sole cause.

The unchanged actor wins 213/256 against old and 131/256 against mirror. Mean
game lengths are 12.203 and 10.887 turns (11.545 pooled). These are outcomes
shared by every scored critic, not gameplay gains or evidence of shorter games.

## Input-matched explanation

The exploratory `critic-opponent-conditioned-input-audit-2026-10-10` selects
pairs by identical first-idle base V inputs, never outcomes. It finds 79 pairs,
all learner-first turn-one. Observed mean returns are +0.49367 against old and
-0.03797 against mirror; paired difference -0.53165 [-0.73418, -0.32911].

The unconditioned balanced estimator predicts +0.23191 for both opponents.
The conditioned estimator predicts +0.49676 for old and +0.17235 for mirror.
It can now distinguish the two inputs through the opponent indicator, but
still overestimates the mirror subset. These noisy terminal outcomes are not
exact state values. This supports an opponent-information limitation without
claiming that one identity feature resolves all state-dependent errors.

## Follow-up before RL integration

`critic-opponent-interaction-fit-2026-10-10` tested a narrower hypothesis: the
current additive identity feature gives the two opponents a common set of
resource slopes. The new diagnostic adds opponent-times-resource terms, so
each known opponent can have different slopes. It adds capacity and does not
generalize to unknown policies. It uses exactly the existing training and
validation splits, the same ridge grid and frozen actor/base critic, with all
three prior controls refit for parity. The inspected confirmation data does
not enter fitting or selection; any selected change needs fresh confirmation.

The fit and independent refit completed. All three prior controls reproduce
exactly. The interaction arm selects lambda=1 and has validation Q/V MSE
0.58421 versus additive conditioning's 0.58013. Old-opponent error is 0.44744
versus 0.44210, and mirror error 0.72098 versus 0.71815. This pilot supplies no
pooled or opponent-retention evidence for advancing the interaction model.
It does not rule out interactions with more data, but no new full-game
confirmation or production integration is justified for this candidate.

`tools/critic_opponent_interaction.py` reuses closed-vocabulary/game-alignment
checks. Seventeen interaction/opponent/residual/role tests pass, including
expressing distinct resource slopes and metadata-order invariance. Low-deck
coverage remains insufficient regardless of whether this fit improves.
No critic is promoted to production PPO. Rare-state retention and eventual
learner-policy-shift / matched RL continuation remain explicit gates.

The next study, `critic-lowdeck-coverage-data-2026-10-10`, completed 512
new full Runick games: 256 training and 256 validation, balanced across both
roles, opponents, anchors and seats. The 128 openings are disjoint from 1,728
prior openings and across splits. All outcomes are retained, without selecting
games by reaching low-deck states or by winning/losing. Exact replay/cache
checks and an independent watchdog remain enabled. This Runick-enriched corpus
is not an estimate of the broad matchup distribution. After collection,
quantify actual low-deck coverage before registering fitting weights and
selection gates. Previous confirmation data remains excluded from fitting.

Update: [coverage audit and fixed-mixture fit](critic-lowdeck-mixture-2026-10-10.md)
verified 81 new low-deck training games and 73 validation games. Both registered
mixtures pass validation retention gates; fresh broad/Runick confirmation is
running. This does not yet resolve the independent-confirmation failures above.

Evidence: `critic-opponent-coverage-confirmation-2026-10-10/{analyze.py,
analysis.json,completion-verification.json}`, the input audit, and the
interaction fit under `out/research/`. Sealed source artifacts are preserved.
