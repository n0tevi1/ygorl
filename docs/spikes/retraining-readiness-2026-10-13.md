# Retraining readiness and next decision

The immediate priority is an auditable training/evaluation loop for the repaired
inputs. The last studies established mechanisms at selected states, but did not
establish a better training recipe or playing-strength improvement. Stop extending
the same selected-root optimizer attribution chain without a new actionable test.

This review fetched main `863905f`, checked current code and issue histories, and
recomputed outcomes from 13,824 old primary-panel game records plus 128 recent
canary evaluation records. It adds no games, optimizer steps or promoted model.
Executable evidence: `out/research/retraining-readiness-2026-10-13/audit.py` and
`report.json`, with all input hashes. Saved-rollout coverage is reaggregated from
the prior audit, not independently decoded from tensors in this review.

## What the metrics actually establish

The older three-seed study remains the strongest population evidence of RL
progress. Each row pools 2,304 games against three fixed opponents on the same
paired panel. These opponents do not establish expert-level play.

| Model | Wins | Mean turns | P95 turns | Games above 20 turns |
|---|---:|---:|---:|---:|
| cold u128 | 69.57% | 11.362 | 21 | 129 |
| cold u512 | 84.16% | 9.325 | 17 | 41 |
| warm u128 | 77.00% | 10.510 | 19 | 81 |
| warm u512 | 86.59% | 9.001 | 17 | 27 |

The original crossed seed/deal confidence intervals support improvement in both
arms. Warm versus cold at u512 remains uncertain. See the
[full review](terminal-critic-final-review-2026-10-10.md). Recomputed mean turns
among warm wins decrease 10.414 to 8.781; among losses, 10.832 to 10.417. These
outcome-conditioned groups differ and are descriptive, not causal estimates.

The recent ordered canary starts from **warm u128**, not u512. Its different
opponent, sample, runtime and training protocol must not be compared directly
with the table above. It has one training seed and eight reused deal clusters:

| Recent arm | Wins /32 | Mean turns | P95, nearest rank |
|---|---:|---:|---:|
| Control initial | 20 | 9.250 | 15 |
| Control after four updates | 23 | 8.781 | 15 |
| Attack history initial | 20 | 9.188 | 15 |
| Attack history after four updates | 21 | 9.813 | 20 |

This does not demonstrate a strength benefit from the input repair. It also
does not establish that the repair is harmful or that retraining cannot work.
These inspected games are regression fixtures, not a fresh confirmation panel.

## Confirmed problems and limits of their explanations

1. **Some player-known context was missing.** Accepted-attacker history and
   selection history have implementations, but they are opt-in. The recent
   context arm enables only `attack_history`; both arms leave
   `selection_history` false. A merged repair does not update old checkpoints
   or demonstrate that their policies learned the new distinction. The original
   single-attacker target mistake also cannot be explained solely by ambiguity
   between multiple attackers.

2. **The repair received little relevant experience.** Of 4,096 context rows,
   53 are immediate target windows (1.29%); 21 follow a menu with multiple
   available attackers (0.51%). Control has 46 target windows and only five
   multi-attacker windows. These are correlated policy-dependent opportunities,
   not equally matched labeled examples. Four learner updates applied seven
   Adam steps in control and six in context because the unchanged KL guard
   stopped one versus two updates early. That is valid adaptive PPO behavior,
   but it is not equal realized optimizer work. Record both budgets; do not
   disable the guard merely to equalize the count.

3. **Distant terminal outcomes have weak direct influence on this estimator.**
   The canary uses 128-row segments and Expected-SARSA/VRPO lambda 0.5. At a
   terminal distance of ten learner rows, the direct reward coefficient is
   0.5^10 = 0.0009765625; these are decisions, not game turns. Of 53 context
   target rows, 28 have no terminal inside their segment, and 19 of the other
   25 have a direct terminal contribution below 0.001. Bootstrap and critic
   terms still carry learning signal. Missing a terminal in the segment is not
   a loss label and does not prove erroneous credit. Seat-sign and recurrence
   checks pass. [Decomposition and fresh probes](attack-credit-source-2026-10-13.md)
   explain the scope of this evidence.

4. **Critic calibration and action ordering remain separate problems.**
   Nonlinear terminal-head refresh improved broad Q/V MSE from 0.58255 to
   0.55679 and Runick MSE from 1.08083 to 0.90704 on fresh trajectories, but
   residual low-deck bias persisted. Those are return-scale errors, not win
   rates. [Confirmation](critic-head-confirmation-2026-10-10.md). Later ranking
   fits did not establish improved choices beyond initialization; see
   [state expansion](critic-state-expansion-results-2026-10-11.md). A critic can
   fit sampled actions better without ranking unchosen actions correctly.

5. **Update interference exists, but no general optimizer defect was found.**
   Exact historical replay matches the full saved learner state. Current
   gradients and stored Adam moments both contribute to a selected preference
   regression; removing initial momentum does not remove it. Other rows can
   oppose individual target-row credit while improving the aggregate recorded
   surrogate. This is insufficient evidence for globally resetting Adam,
   dropping other decisions, or installing a dedicated patch for that root.
   [Finite controls](attack-update-factorial-2026-10-13.md).

6. **Routine successful training games are insufficiently traceable.**
   `EncodedVecEnv(record_actions=True)` retains full action indices for errors
   and limits, but discards them for healthy wins. `Trainer.log_games` records
   terminal summaries. The recent research wrapper saves rollout tensors,
   which enabled exact optimizer replay, but routine terminal summaries cannot
   reconstruct every intervening opponent/forced action. Fresh tracing wrappers
   already demonstrated a solution on four games. Promote this capability to
   an opt-in, tested path before the next training investigation. A game can
   span several segmented updates: retain actual action history and update
   provenance; do not label its whole trajectory with its finishing checkpoint.

## Why several tempting fixes are not the next action

- **Terminal policy credit:** already replicated. The two-seed comparison gave
  -3.91 pp versus VRPO, with interval [-12.89, +4.69] pp, and no consistent loop
  improvement. Keep that negative evidence rather than rediscovering the same
  experiment. [Results](selection-terminal-policy-replication-2026-10-09.md).
  Critic-only terminal warmup is a different intervention.
- **Shorter games as the reward:** selected bad branches can lose sooner; Runick
  restrictions can legitimately prevent battle. Track wins and losses, deck-outs,
  phase availability and repeated-input cycles separately. Average four turns is
  not a strength acceptance criterion.
- **Cancellation guard as learned skill:** budget1 removed repeats at inference,
  but its training canary did not establish improvement and repeats returned
  without the guard. [Results](warm512-cancel-budget-confirmation-2026-10-10.md).
- **Text plus deck mutation plus new critic together:** the semantic pilot did
  not establish benefit over ID alone, although real text beat shuffled text.
  Coverage weaknesses are concentrated; held-out decks did not uniformly collapse.
  [Semantic diagnostic](card-generalization-pilot-2026-10-10.md),
  [full-game coverage](card-coverage-results-2026-10-10.md).
- **More selected-root optimization:** PCGrad, actual-step projection and smaller
  steps did not establish better held-out decisions. Close this chain until a
  distinct hypothesis predicts a measurable effect. [Results](critic-root-pcgrad-2026-10-12.md).

Value-based credit trades variance against approximation bias; increasing lambda
is not a free correction. This general caution is consistent with the original
[GAE paper](https://arxiv.org/abs/1506.02438), but its results do not select a
lambda for this project's different Expected-SARSA estimator. Likewise the
[PPO paper](https://arxiv.org/abs/1707.06347) proposes an aggregate surrogate;
it supplies no per-state guarantee that every desired probability increases.
The project's exact recurrence and finite-update checks are the relevant local
evidence.

## Ordered next actions and acceptance gates

**First, repair observability and lock the comparison.** Add opt-in successful
game action traces using the existing diagnostic record (actual deck/rules,
seat mapping, masks/schema provenance and engine responses). Retain every
pre-update rollout plus checkpoint/reference/optimizer/RNG in the bounded
research runner. Test tracing-on/off outcome parity, both seats, a game spanning
updates, reset cleanup, cancellation and exact cold replay. Keep failed records.
Source/runtime must be frozen after main sync, not changed beneath a running job.

**Second, finish testing the attack-history repair as one intervention.** Use
two fresh training seeds from the same warm-u128 source as the original canary;
restore actor and critic, use matched fresh optimizer/reference/pool in both
arms, and retain both initial checkpoints. Compare attack history off/on with
selection history, cancellation, rewards, estimator, precision and deck pool
held identical. This maintains interpretability and does not describe u128 as
the strongest available checkpoint. Starting from u512 is a later transfer test.

Proposed bounded pilot ceiling: 16 updates of 1,024 rows per arm and seed, 65,536
rows total, checkpoints every four updates. At the observed context rate this
would yield roughly 84 multi-attacker target windows per arm, but that is only
a coverage projection. It is not a power calculation or guaranteed count.
The mechanism and working input/replay checks justify a pilot; existing results
do not predict a strength gain. No automatic extension if coverage is inadequate.
Actual Adam steps, KL stops, visited mechanisms and game counts must be reported.

Before launching, seal fresh disjoint deal clusters and frozen opponents,
including an RL opponent, with paired seats/deck assignments. Compare fixed
initial and final endpoints, both seeds, and a frozen warm-u512 reference on the
same panel. Select no checkpoint by inspected outcomes. Size the panel for the
declared useful effect (proposed +5 pp for a pilot), accounting for deal clusters;
if the cost ceiling cannot resolve it, label the outcome inconclusive rather
than using row-level or independent-game error bars. This review is a proposal,
not a sealed execution registration; panel size and cost must be fixed first.

Track overall and per-opponent wins; mean/median/P95 turns and decisions; wins
versus losses; limits/errors; repeat cycles; self-negation opportunities; and
verified missed-win cases. Warnings are not automatic error labels. At fixed
probe states check whether attacker context affects target preferences, while
remembering that sensitivity alone is not correctness. No old inspected root
becomes a corrective training label.

**Third, act on the outcome.** If context is used but credit fails on independent
states, test a critic/credit intervention separately, with complete-trajectory
outcome calibration and action-ranking checks. If context is not used, investigate
coverage/representation with a mechanism-balanced curriculum and disjoint cards,
not another optimizer sweep. Only after a useful strength effect and retention
checks should a longer u512 continuation be proposed. Card-pool expansion and
semantic inputs remain separate later comparisons.

The current unresolved question is whether repaired observations plus ordinary
RL learn better decisions under an independent strength test. It is no longer
whether the selected historical Adam step can be explained.

## Operational state

Host-level inspection during this review found 12 running `ygorl-*` user services,
all historical watch services, and no active training service. These monitor
old stopped studies; their presence is not evidence of training progress. No new
training was launched by this review. Historical STOP evidence is untouched.
Issues #61, #62, #83 and #239 remain open. This document supplies the common
priority order; their older checklist items should not override completed
negative experiments cited above.
