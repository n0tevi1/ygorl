# Attack credit sources and additional target roots

The previous [target audit](attack-target-credit-2026-10-13.md) found a conditional
target preference that worsened during four PPO updates. Small direct Q/V loss
gradients did not exclude indirect critic influence through policy advantages.
This follow-up verifies that path before changing training. No model fitting,
optimizer steps, production defaults or action masks change.

## Actual saved training credit

`attack-training-credit-source-2026-10-13` reads all 8,192 actual pre-update rows
from both ordered-canary arms. It reconstructs the Expected-SARSA return with an
independently written float64 recurrence and verifies invariance under renaming
the two engine seats. The Python duel result reports deck/agent indices, whereas
the native training environment reports engine seats; the existing branch win
counts and training sign conventions respect this distinction.

For the implemented VRPO return estimator, decompose the raw advantage into:

- the recorded terminal rewards with all Q estimates set to zero;
- the saved Q estimates/bootstrap with rewards set to zero.

Their sum matches the original raw advantage. With the actual batch standard
deviation held fixed, the normalized weight is `reward / std + critic / std -
mean / std`. That identity also reconstructs the saved normalized weights. This
is algebraic attribution, not an ablation that reruns training with a different
normalizer. At a known terminal distance `d`, the direct reward contribution is
exactly the acting-seat outcome times `0.5 ** d`; an unfinished segment has no
observed terminal reward contribution. Its future outcome remains unknown.

For the context arm, excluding the previously selected update 3:

| Chosen action | Rows | Terminal in segment | Of those, direct reward magnitude < 0.001 | Mean absolute reward term | Mean absolute critic term |
|---|---:|---:|---:|---:|---:|
| Attack target | 40 | 22 | 17 | 0.03837 | 0.09559 |
| End phase | 107 | 54 | 45 | 0.00297 | 0.10701 |
| Normal summon | 115 | 47 | 38 | 0.00621 | 0.03134 |
| Special summon | 78 | 16 | 14 | 0.00241 | 0.04073 |
| Enter battle phase | 193 | 105 | 74 | 0.02707 | 0.07244 |

The absolute means include unfinished rows. These are rows from three other
updates, not independent games or an untouched data set: games may cross update
boundaries. Across all 3,072 rows in those updates, 426 change sign when the batch
mean is subtracted. This is expected behavior of standard normalization, not by
itself a bug. A sampled loss with positive credit, or win with negative credit,
does not establish incorrect advantage: the relevant comparison is against other
actions in that state under the continuation policy.

## Indirect critic contribution to the selected regression

`attack-credit-gradient-source-2026-10-13` revisits the previously selected game-5
root and update 3. It holds the observed normalizer fixed and decomposes the
pre-update policy gradient by the three weight sources above. All likelihood
ratios lie inside the PPO clip interval. The component gradients reconstruct the
implemented policy gradient and the complete loss gradient within tolerance.

The derivative of the Galatea-I/Galatea log-odds margin under unit
unpreconditioned gradient descent is:

| Source | Margin derivative |
|---|---:|
| Policy weights: terminal reward component | +0.0762 |
| Policy weights: critic component | -8.2134 |
| Policy weights: batch centering component | +0.9703 |
| Reference KL | -1.5739 |
| Entropy | -0.0278 |
| Direct Q regression | -0.0178 |
| Direct V regression | -0.0177 |

Negative values favor the target whose conditional continuations won 0/8 in the
earlier study. End-phase rows contribute -5.0909 through the critic term and
+1.0187 through centering; their direct terminal term is below displayed
precision. Attack-target rows contribute -1.1044 through the critic term.

Thus the earlier small direct Q/V terms do **not** absolve the critic: its saved
estimates strongly influence the opposing policy gradient in this diagnostic.
This still does not prove those training actions have wrong advantages, assign
additive causes to finite Adam updates, or show that removing the critic,
normalization or KL improves play. The root and update were selected earlier for
mechanism diagnosis; they must not become fitting examples or an unbiased
regression-rate estimate.

## Two additional roots

`attack-target-additional-roots-2026-10-13` selects the next two qualifying game
IDs, 9 and 10, without using their outcomes. The protocol matches the preceding
branch study: every supported attack target, four new suffix seeds per action,
fixed u4 actor against Greedy, original-prefix equality and exact cold replay.
Two baselines and 20 branches complete normally.

All five targets win 4/4 in these continuations. There is no replicated losing
preference in these two roots. Game 9's least-probable target takes 10–18 turns
versus 8–10 for the other targets, despite equal sampled wins. These roots belong
to the reused panel and can share original deal clusters; they are additional
states, not independent held-out matches or evidence of general improvement.

## Fresh self-play credit checks

`attack-fresh-credit-probe-recovery-2026-10-13` collects four new full self-play
games with frozen context u4 on both seats: Clown Crew/Branded and
HERO/Tearlaments, each deal with both seat assignments. The registered seed draws
come from the same 20-deck pool. These are two paired deal clusters, not four
independent samples. The 538 recorded choice rows match their native trace
observations, chosen actions and model behavior log probabilities. All four
baselines reproduce exactly, including responses and terminal results.

Before inspecting outcomes, select the first multi-target attack choice and the
first chosen end-phase action with a supported battle-phase alternative. The
latter has no coverage in these four games; the budget is not extended to hunt
for a failure. The attack root is game 1, choice 44 (packed row 143, engine seat
0). Four suffix seeds for each of its three targets give:

| Target action index | Actor probability | Q | Conditional wins / 4 |
|---|---:|---:|---:|
| 0, recorded choice | 90.35% | -0.0224 | 3 |
| 1 | 0.29% | -0.0309 | 1 |
| 2 | 9.37% | -0.0279 | 3 |

The recorded choice's normalized weight is +0.6012. This case does not show the
positive credit rewarding a worse sampled target. Four continuations per action
cannot establish the true advantage or its magnitude. The branch actors use
public observations only; the hidden engine world remains fixed for this
conditional diagnostic. Every saved branch cold-replays exactly.

Because the battle-phase alternative criterion had no coverage, a separately
registered follow-up, `attack-endphase-credit-probe-2026-10-13`, reuses the same
four traces. It selects the first recorded end-phase choice with any supported
alternative, taking the highest-probability non-end-phase alternative. This is
explicit exploratory reuse, with no outcome filter. At game 0, choice/row 37,
end phase has probability 97.87%, Q +0.2580, normalized weight +0.1359. Its
alternative is an activation with probability 1.30% and Q +0.2435. Both win 4/4
under four new suffix seeds. There is no demonstrated erroneous end-phase credit
in this case either. The eight branches also undergo exact cold replay.

These credits use the same VRPO lambda 0.5 estimator on complete games. They are
not reconstructed historical 128-row segment credits and are not an unbiased
evaluation of self-play strength. The four baseline turn counts are 10, 9, 6 and
7 (mean 8); their small size and changed opponent protocol preclude comparison
with the earlier Greedy panel's mean turn count.

The initial collection script incorrectly called a nonexistent environment
cleanup method. Its restart saved the rollout and traces and verified all four
baselines, then failed serializing native response bytes. Recovery reused those
saved artifacts, encoded bytes as hex and retained both original STOP records.
The first standalone replay verifier compared JSON lists to native tuples;
the corrected verifier compares canonical serialized results, preserving exact
response bytes. All original failure logs remain. These were research-script
integration failures, not engine failures or evidence of a production fix.

## Evidence and next gate

The [finite-update follow-up](attack-update-factorial-2026-10-13.md) tests these
mechanisms with matched Adam updates and exactly replays historical update 3.
It distinguishes current gradients from accumulated first moments; resetting
momentum alone does not remove the historical diagnostic regression.

Each study directory contains the script, identity manifest and healthy report.
The first contains row-level decompositions; the second contains component
gradients; the third contains the 22 cold-verified games and replays. The runtime
remains pinned to `codex-rollout-order` at `af4b5d9`; this report starts from
fetched main `bd214907`. The user's main worktree and dirty native submodule are
untouched.

The unresolved question is whether the observed credit is inaccurate at the
training states themselves, or whether individually reasonable updates interfere
through shared parameters. The new fully traced examples do not establish wrong
credit. They rule out neither explanation at other states. Do not increase RL
scale, remove end-phase rows, drop the critic or switch the production estimator
from these findings. Retain complete traces in the next bounded training canary
so future regressions can be paired with their actual decision prefixes; the
old healthy-game summaries cannot supply those counterfactuals by themselves.

Validation: the three algebraic/reused-panel studies have healthy reports; both
fresh-data studies additionally have `verification.json` binding exact branch
replays and all 538 behavior-logprob checks. Production source and tests are
unchanged from the resolved full-suite validation recorded for PR #417 and
reused by #418. This documentation-only change reuses that validation and runs
presubmit plus diff checks; it does not claim new hosted CI execution.
