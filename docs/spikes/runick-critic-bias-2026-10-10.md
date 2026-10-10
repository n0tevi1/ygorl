# Runick critic bias: gradient conflict and misleading resource signals

The private critic refresh improved aggregate prediction error, but its Runick
mean signed error increased. The new diagnostic identifies a local training
mechanism for that increase and a concrete representation weakness. It does not
establish the entire historical origin of the original critic's bias.

## Evidence and scope

Use the frozen `card-pool-critic-refresh-recovery-2026-10-10` corpus and the
unchanged critic versus the validation-selected optimizer-shuffle 202610110202
fit. There are 64/32/64 Runick games in train/validation/test, with 63 test games
eligible for idle-state predictions. The normal zero-idle termination remains
included in the 160-game replay audit. These are one actor and two fixed
opponents, not independent actor replications or a training-league calibration.

Every Runick game was replayed with identical terminal results and response
bytes. Native observation globals and cached actor contexts were separately
checked at 218 deterministic probe states. The previously evaluated test panel
is now diagnostic data; no new held-out performance claim is made.

Artifacts are under `out/research/runick-critic-bias-audit-2026-10-10/`:
`analyze.py`, `gradient.py`, `replay_state.py`, `sensitivity.py`, `verify.py`,
`strata.json`, `rows.json`, `gradients.json`, `states.json`, `sensitivity.json`
and `verification.json`. The latter binds analysis inputs/outputs by hash.
The initial diagnostic registration write used a wrong relative path and failed
before being corrected after the gradient job had launched. The intervention
code existed before execution, but this is **not** a sealed preregistered study.

## Why mean bias grew while MSE improved

For the 63 eligible test games, the fitted V head behaves as follows:

| Realized outcome | Games | Original mean V | Fitted mean V | Original V MSE | Fitted V MSE |
|---|---:|---:|---:|---:|---:|
| Win | 41 | 0.5354 | 0.6355 | 0.3354 | 0.2130 |
| Loss | 22 | 0.5352 | 0.5409 | 2.5161 | 2.5415 |

The overall bias increase is exactly the weighted sum of these changes:
`41/63 * 0.10008 + 22/63 * 0.00566 = 0.06711`.
About 97% of this increase comes from raising estimates on realized wins;
lower underestimation on wins no longer cancels the unresolved overestimation
on losses. Outcome-conditioned residuals alone do not prove miscalibration;
the unconditional and observable-state diagnostics are needed too.

This is not mainly a few pathological outliers: 51 of 63 test games have a
higher mean V after fitting. The first idle sample of each game rises from
0.180 to 0.504 against mean realized return 0.302. Late-state estimates improve
somewhat, but remain too optimistic in the observed low-deck slices.

## Direct training intervention: gradients from other matchups oppose Runick

At the same original weights, compute the exact training-only, game-weighted
Q/V loss gradient separately for Runick and the other seven target lists.
Their cosine is **-0.523**. Runick has 1/8 of game weight. The full gradient's
cosine with the Runick gradient is **-0.321**: the initial aggregate descent
direction conflicts with reducing Runick's training loss.

Apply one fresh Adam step with the registered fit's LR `1e-4`, epsilon `1e-5`
and gradient clipping `0.5`, keeping actor/shared features frozen:

| Training subset for the step | Runick V bias before | After | Runick V MSE after |
|---|---:|---:|---:|
| All eight target lists | +0.3241 | +0.3510 | 1.0190 |
| Other seven lists only | +0.3241 | +0.3549 | 1.0247 |
| Runick only | +0.3241 | +0.2935 | 0.9747 |

The all-list step reduces overall training V MSE by 0.00847 while worsening
Runick. This is a controlled local demonstration of competing objectives,
not just a correlation with card presence. The initial derivative of Runick
mean V along SGD descent is +1.6976 for all-list training, versus -4.7466 for
Runick-only and +2.6182 for other-list training. Numerical SGD steps verify
these derivatives and the exact 1/8 versus 7/8 gradient mixture.

Removing Q loss does not remove the effect: the original all-list V-only step
raises Runick bias to +0.3509. Q contributes through the shared **critic** trunk
and privileged encoder, but is not necessary for this conflict.

At the selected fitted endpoint the other-list and Runick gradients conflict
even more strongly (cosine -0.927), but the aggregate fresh-Adam step now lowers
Runick bias slightly, from +0.3680 to +0.3603. Therefore this evidence does not
justify saying continued fitting can never help. Fresh-Adam probes are also
not a replay of the original minibatch order or accumulated optimizer moments.

## Persistent error: LP advantage overwhelms deck exhaustion signals

Seventeen of the 22 Runick test losses ended by deck-out. In those losses the
game-averaged fitted V remains about +0.553. Among the 17 losing games sampled
with at most five cards left, the conditional game-equal mean V is +0.350.
Among 14 games with a sample at zero cards, mean V is +0.425 against mean
realized return -0.857. There is a win in this slice: an empty deck is not an
automatic immediate loss, so a blanket negative label would be incorrect.

Concrete replay: game 1266, decision 884, turn 52. The learner has **8200 LP
versus 600**, but **zero cards left versus eight**. Its monsters have 600,
2000 and 0 ATK; the opponent controls Synchro Zone and Skill Drain. The learner
enters battle, and the actual subsequent `SelectBattleCmd` has **no attack
action**. It ends the turn and loses by deck-out on turn 54. Original V is
+0.9406; fitted V is still **+0.8977**. This is a factual replay of the chosen
continuation, not proof that every alternative line must lose.

The inputs are present and correctly oriented: native globals `[5:7]` contain
learner/opponent LP, `[7]` contains learner deck count, and `[12]` contains
opponent deck count. These were compared with engine counts at all 218 probes.
No swapped-seat or missing-count bug was found in these states.

Probe selection is each game's first idle sample plus its first sample with
at most five cards remaining, without selecting by outcome or prediction.
Hold all other inputs fixed and change only specific globals channels:

| Fitted V, low-deck states with LP advantage | Train (18 states) | Validation (8) | Test (16) |
|---|---:|---:|---:|
| Actual input | +0.409 | +0.463 | +0.487 |
| Equalize both LP, preserving total LP | +0.027 | +0.190 | -0.101 |
| Set own deck-count channel to 0 | +0.418 | +0.475 | +0.498 |
| Set own deck-count channel to 30 | +0.357 | +0.393 | +0.424 |

Across **all 218 probes**, setting the own-deck count channel to 0 produces
higher V than setting it to 30. The effect is smaller than the LP intervention,
but its direction is concerning for deck-exhaustion states. Card/history tokens
remain unchanged, so these deliberately isolated channel probes can be internally
inconsistent. They measure network dependence, **not legal game counterfactuals,
true state values, or a causal effect of LP on winning**. Other channels may also
encode deck composition and depletion. The combined replay/probe evidence
supports a misleading LP/resource shortcut, not a claim that the network only
reads LP or entirely ignores deck state.

In this model, own LP and deck counts reach the critic through the actor's
128-dimensional context. That entire encoder is frozen in the private-only
fit. The separately trainable privileged encoder mainly reads opponent hidden
zones/counts; deck-order features are disabled. Card/effect text and card facts
are disabled too. The fit can relearn the critic readout but cannot retrain
the shared representation of these public resource and interaction signals.
This is a concrete constraint; representation insufficiency as the sole cause
has not been established by an independent-trunk ablation.

## Explanations narrowed down and next repair test

- Incorrect winner perspective, missing own-deck input, and incorrect cached
  forward features are not supported by the replay and parity checks.
- Severe tanh saturation is not the main aggregate explanation: on test losses,
  fitted mean `1 - V^2` is 0.540 and only 2.33% of the game-weighted state mass
  has `abs(V) > 0.95`. Individual saturated errors still exist.
- A unique offending floodgate is not established. Test loss-state V is about
  +0.527 both with and without any reviewed face-up card under conditional
  game-equal averaging. Card presence alone cannot label an active lock.
- Fixed-opponent mismatch and limited training coverage can contribute to the
  original bias. They do not alone explain why adaptation raises mean values:
  the gradient conflict is reproduced on the actual training corpus.

The user's strategic point is central: LP advantage is not a sufficient measure
of advantage. The correction should learn available resources and feasible win
and loss routes, retaining LP where it matters for lethal damage and effect
costs. Do not add an unconditional LP penalty or a blanket Runick value offset.

The next useful controlled repair is a **critic-only public-resource input path**
or an independently trainable critic encoder, compared with the existing frozen
context baseline. Pair this with a separately ablated conditional-calibration
objective/coverage control so representation changes are not confused with
reweighting. Include legal attack availability, own/opponent resource counts,
and public interaction context; do not use terminal outcome categories as
inference features. Preserve useful LP information rather than zeroing it.

Select any new fit on validation, confirm on newly generated independent deals,
and retain familiar-deck and both-role evaluations. This old test set may supply
diagnostic examples but must not be advertised as a fresh test again. Critic
error reduction still requires matched RL continuation to establish policy
benefit. No diagnostic probe weights are promoted or used for new gameplay.
