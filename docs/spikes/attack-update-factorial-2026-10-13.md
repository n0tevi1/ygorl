# Finite-update attack-credit and momentum controls

The [credit-source audit](attack-credit-source-2026-10-13.md) identified a large
indirect critic contribution to a local policy gradient. That derivative is not
the finite Adam update. This follow-up runs bounded, matched updates to distinguish
policy-row transfer, current auxiliary gradients and stored optimizer history.
It changes no production code or defaults and promotes no checkpoint.

## Fixed-data policy-row factorial

Use the 538 complete-game rows from the four fully traced fresh self-play games
in `attack-fresh-credit-probe-recovery-2026-10-13`. They contain five immediate
attack-target rows: 143, 148, 154, 235 and 499. Each arm restores the same context
u4 model, reference and Adam state. It uses the original two PPO epochs, one full
minibatch per epoch, and the same shuffle seed. All arms apply two Adam steps.

The two factors enable/disable the policy weights of attack-target rows and other
rows. Disabled rows receive zero weight **after** full-batch normalization; the
denominator remains 538. Q/V targets and losses, entropy, reference KL, clipping,
optimizer moments and all other hyperparameters are unchanged. The full arm uses
the unmodified production objective. The common-only arm is not a no-op: auxiliary
gradients and existing Adam moments can still move the policy.

The old game-5 root is a previously selected, off-policy diagnostic. Its margin
is the log odds of Galatea-I over Galatea, the alternatives whose earlier frozen
u4 continuations won 5/8 and 0/8 respectively. It is never a training row or label.

| Enabled policy rows | Target rows moving with their recorded credit / 5 | Mean target credit × chosen log-probability change | Old game-5 margin change |
|---|---:|---:|---:|
| None (common only) | 2 | +0.00497 | -0.20776 |
| Target only | 5 | +0.00812 | -0.21899 |
| Other only | 3 | +0.00694 | -0.19971 |
| Both (full update) | 4 | +0.00901 | -0.20827 |

For row 499, the recorded credit is negative. Target-only training decreases the
chosen log probability by 0.00135, but the full update increases it by 0.00054.
Thus other decision rows can oppose a particular target row's credit under the
actual optimizer. Yet the full update improves the aggregate recorded target
surrogate more than target-only training. Removing other rows is not supported
as a general fix, and credit-direction alignment is not an action-quality label.

At the previously probed fresh target root, the recorded choice's probability
increases from 90.35% to 91.98% under the full update. At the fresh end-phase root,
the recorded action has positive credit but falls from 97.87% to 97.01%. These are
fixed-prefix probability checks, not full-game evaluations of the new checkpoints.

The almost identical old-root drift under common-only and full updates motivates
the next control. It does not establish that policy losses are generally irrelevant.

## First moment versus current auxiliary gradients

On the same 538 rows and source u4 state, disable all policy weights. Cross current
Q/V + KL + entropy losses on/off with stored Adam `exp_avg` kept/zeroed. Preserve
`exp_avg_sq`, optimizer step counters, learning rate, epsilon and beta values.
There is no weight decay. This isolates the initial first-moment history, not a
fresh optimizer with different preconditioning or bias correction.

| Current auxiliary loss | Initial first moment | Old game-5 margin change |
|---|---|---:|
| On | Stored | -0.20776 |
| On | Zeroed | +0.03140 |
| Zero | Stored | -0.24756 |
| Zero | Zeroed | 0.00000 |

The first arm exactly reproduces the prior common-only model. Both zero-loss arms
have exactly zero gradient norm. Zero loss plus zero first moment leaves every
model tensor bit-identical; stored first moments move the model even with zero
new gradients. This is expected Adam behavior, not an implementation bug.

In this particular next-update diagnostic, stored momentum drives the old-root
decline, while the current auxiliary gradients alone improve that root. However,
zeroing the first moment makes the aggregate recorded target surrogate change
negative (-0.00128 versus +0.00497 with stored moments). Do not reset optimizer
state globally based on a benefit at one previously selected root.

## Exact historical update-3 replay

The preceding experiment starts from u4 and uses new complete games. It cannot
explain historical update 3 by itself. Restore the original u2 model, reference,
Adam state and saved global torch RNG; replay the original 1,024-row update-3
rollout with its 13 target rows. The full-loss/stored-moment control reproduces
the saved u3 model **exactly**, with zero tensor tolerance. The complete learner
state (model, reference, optimizer and update counter) is also checked.

The same two-factor control gives:

| Current loss | Initial first moment | Old game-5 margin change | Target rows moving with credit / 13 |
|---|---|---:|---:|
| Full original loss | Stored | -0.37956 | 8 |
| Full original loss | Zeroed | -0.24066 | 9 |
| Zero | Stored | -0.07888 | 6 |
| Zero | Zeroed | 0.00000 | 0 |

Historical regression persists when the old first moment is removed. Current
gradients and prior optimizer history both matter here. Their finite effects
do not add: gradient clipping, second-moment updates, bias correction and changed
second-step gradients make Adam nonlinear. The earlier unpreconditioned gradient
attribution remains a valid local identity, but must not be interpreted as an
additive explanation of the observed parameter update.

## Decision and evidence

The controlled results support two limited conclusions: updates on other decision
types can oppose individual recorded target credits, and stored optimizer history
can materially change how a new batch moves a diagnostic preference. Neither
establishes wrong true advantages, recommends excluding other actions, or proves
that resetting momentum improves playing strength. The known historical case
still regresses with its initial first moment zeroed.

Keep the model and training defaults unchanged. The remaining training decision
requires a bounded canary with complete trajectories and independent evaluation
deals, jointly measuring target-choice behavior, retention, win rate and turn
length. Do not use these already inspected roots as corrective fitting labels.
No new full-game strength metric was produced in this update-factorial study.

Evidence under `out/research/`:

- `attack-credit-update-factorial-restart-2026-10-13`: 538-row policy-mask factorial;
  each arm has an experimental checkpoint, original update stats and fixed-prefix
  probability changes.
- `attack-credit-momentum-factorial-2026-10-13`: auxiliary-loss/initial-first-moment
  factorial, exact common-model parity and exact no-loss/no-moment model identity.
- `attack-historical-momentum-replay-2026-10-13`: original 1,024-row update replay,
  exact saved-u3 control, and the matched historical first-moment interventions.

Each study binds source checkpoint/data and scripts in an identity manifest and
has a healthy report. Each uses eight diagnostic Adam steps, for 24 total; no new
rollouts, expanded training campaign or production checkpoint replacement occurs.
All new checkpoints are marked `research_only` and unsuitable for promotion.
The original factorial attempt used an inference loader's frozen parameters;
backward stopped before any optimizer step. The restart explicitly enables
gradients and preserves the original STOP and log.

The experiment runtime remains frozen at `af4b5d9`. The report worktree starts
from fetched main `ed682d1`. Production source and tests remain identical to the
resolved full-suite validation recorded for #417 and reused by #418/#419; reuse
that validation with the new exact-update controls, presubmit and diff checks.
Documentation-only changes do not run hosted CI, and no hosted test execution is
claimed.
