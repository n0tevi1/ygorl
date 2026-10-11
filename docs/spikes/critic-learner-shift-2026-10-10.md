# Critic transfer across changed learner behavior: gains with remaining gaps

The [head confirmation](critic-head-confirmation-2026-10-10.md) is followed by a
paired actual learner-weight shift: fixed-pool versus expanded-pool seed-0 u528
actors, with the same opponents, deck orders, seats and RNG bindings. All 1,536
games completed healthily and all have eligible critic samples. There are 192
shared opening clusters, disjoint from 2,432 prior openings, not 1,536 independent
openings. Each behavior has 512 broad and 256 enriched Runick games.

Independent verification reproduces all four estimators' predictions and metrics,
checks every replay/cache hash and confirms that reference features remain frozen.
A separate behavior host verifies the changed actor's probabilities; the two
actors are not silently encoded with each other's embeddings. Within-behavior
20,000-replicate stratified paired-opening intervals and paired changes between
behaviors are both reported. These are descriptive, unadjusted intervals for one
pair of policy-training checkpoints, not inference over independent training seeds.

## Results

Q/V MSE is against terminal returns in [-1,1], sampled at the first learner idle
per turn with equal game weights. Low-deck V uses the first sampled state with
own deck <=5. Both behavior panels have 63 such Runick games.

| Behavior | Critic | Broad Q/V MSE | Familiar | Runick | Own Runick | Low-deck V MSE | Low-deck V bias |
|---|---|---:|---:|---:|---:|---:|---:|
| fixed | control | 0.58878 | 0.53643 | 0.86790 | 0.76067 | 1.82972 | +0.63968 |
| fixed | terminal_only | 0.57964 | 0.53278 | 0.79039 | 0.74962 | 1.16566 | +0.29735 |
| fixed | anchored | 0.57908 | 0.53241 | 0.79406 | 0.74924 | 1.24455 | +0.33925 |
| expanded | control | 0.56787 | 0.51641 | 0.78190 | 0.62631 | 1.62481 | +0.74496 |
| expanded | terminal_only | 0.55747 | 0.52302 | 0.72249 | 0.61500 | 1.07836 | +0.42981 |
| expanded | anchored | 0.55560 | 0.51850 | 0.72436 | 0.61463 | 1.13432 | +0.46770 |

Runick error improvements against the original conditioned control persist:

- Fixed actor: terminal-only delta -0.07751, 95% interval [-0.14401, -0.01445];
  anchored -0.07384 [-0.13164, -0.01804].
- Expanded actor: terminal-only -0.05941 [-0.12104, -0.00291]; anchored -0.05754
  [-0.11098, -0.00853].
- Expanded low-deck V MSE: terminal-only -0.54645 [-0.74713, -0.33588]; anchored
  -0.49049 [-0.66908, -0.30382]. Signed V bias still remains +0.42981/+0.46770.

Broad improvement is less certain than in the preceding confirmation. Under the
expanded actor, terminal-only broad delta is -0.01040 [-0.03225, +0.00866], and
anchored -0.01227 [-0.03110, +0.00379]. Familiar error rises slightly: +0.00661
[-0.00525, +0.01895] and +0.00209 [-0.00514, +0.00966]. These intervals do not
establish either a regression or complete retention. Expanded own-Runick deltas
are -0.01131 [-0.03050, +0.00866] and -0.01169 [-0.02632, +0.00362].

The paired change in benefit is defined as
`(refreshed-control error under expanded) - (refreshed-control error under fixed)`;
a positive number means less improvement after the policy change. On Runick this
is +0.01809 [-0.04705, +0.08133] for terminal-only and +0.01630 [-0.04037, +0.07190]
for anchored. Familiar changes are +0.01026 [-0.00863, +0.02884] and +0.00611
[-0.00651, +0.01848]. Thus a loss of benefit from this learner change is not
established, but these widths cannot prove invariance to policy shifts either.
All 18 paired difference-in-improvement point estimates independently agree with
the differences of the within-behavior estimates.

## Actual play remains unresolved

The critics only score trajectories and do not choose the actions in this study.
Any fixed/expanded outcome difference belongs to the existing actor change:

| Panel | Fixed wins | Expanded wins | Fixed mean turns | Expanded mean turns |
|---|---:|---:|---:|---:|
| Broad | 331/512 | 332/512 | 11.930 | 11.664 |
| Runick | 165/256 | 168/256 | 23.258 | 23.520 |

Expanded-minus-fixed paired win-rate changes are +0.195 percentage points
[-3.125, +3.711] and +1.172 points [-6.641, +8.984]. Mean-turn changes are -0.266
[-0.945, +0.440] and +0.262 [-1.563, +2.086]. Overall stronger play or shorter
games is not demonstrated. The familiar subgroup alone has a shorter-turn
interval, but it does not establish an overall improvement.

## Next: cover all decision types before PPO integration

The existing evidence samples idle decisions only. It cannot establish calibration
for chain responses, target/material selection or all unchosen candidate actions.
Before integrating the research critic into PPO, `critic-all-decision-audit-2026-10-10`
is running an exploratory replay audit on 144 source games selected by draw zero
and a deterministic balanced-seat rule, without looking at outcomes. Both actors,
both roles, old/mirror opponents and broad/Runick panels remain separate.

The research helper `tools/critic_action_probe.py` extends the resource critic to
all candidate Q values and V, using frozen-reference context/action features.
It detaches those features, broadcasts the closed-vocabulary opponent-conditioned
ridge correction, clips only after adding the raw outputs, and keeps illegal or
padded Q entries exactly zero. Unknown opponents are rejected. It is not a
production checkpoint adapter and does not migrate weights to a changed encoder.

Twenty relevant helper/resource/opponent/residual tests pass, including every
legal candidate versus the chosen-action path, clipping order, padded rows and
gradient isolation. A real-replay preflight covers two seats/behaviors and 197
learner decisions: maximum chosen-Q/V discrepancy is 1.79e-7. Earlier idle-cache
predictions also agree within the registered floating-point tolerance. The 144-game
run checks every learner decision, all-action finiteness/masking and chosen-action
parity; it will report equal-game errors and biases by decision type. Unchosen Q
values have no counterfactual return labels here, so numerical parity does not
prove their strategic accuracy.

This extends coverage and tests integration prerequisites; no production PPO,
actor or reward changes are made. If non-idle errors regress, investigate them
before collecting fitting data or altering the loss. Preserve this exploratory
sample as evaluation evidence rather than silently using it to train a fix.

## Evidence

`critic-head-learner-shift-2026-10-10` retains `completion-verification.json`,
`analysis.json`, `paired-shift-analysis.json` and `paired-shift-verification.json`.
The last file also corrects a copied method-description sentence: this study has
two fixed behavior actors with one frozen reference encoder, rather than one
behavior actor. The underlying within-behavior calculations and denominators
are unchanged. Completion was acknowledged after verification.

`critic-all-decision-audit-2026-10-10` contains the registered source IDs, sealed
implementation/dependencies, preflight, replay audit and independent watchdog.
