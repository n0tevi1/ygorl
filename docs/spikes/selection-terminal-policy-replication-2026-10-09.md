# Terminal policy-credit replication and cancellation-budget follow-up

The pilot plus two additional seeds do not support adopting terminal policy
credit. Keep the existing policy objective. Shorter games alone are insufficient
when strength and loop improvements are not established. The next bounded
experiment tests cancellation budgets on actual high-cancellation cases and a
separate fresh evaluation panel; it does not change formal training.

## Replication evidence

Study: `out/research/selection-terminal-policy-replication-2026-10-09/`.
Identity: `ac5bd4dcf1d5cb1d82622145ffd27b683df675dbb5beeb4453d9f0d1566f3a7e`.
Independent review: `review.py` and `completion-review.json` in that root.

Training seeds266/267 share the seed250 pretrained source. Each arm runs
24 updates ×16 complete games, with identical initial actor, critic, Adam and
reference states. Only policy credit changes: normalized VRPO lambda=.5 versus
normalized terminal outcome minus the same expected-Q baseline. Critic targets,
rewards, masks and other optimization settings stay fixed. All 1,536 training
and 768 evaluation executions are healthy. Duplicate initial evaluations agree
exactly and do not count as independent evidence.

Each endpoint uses the same fresh 32 deal blocks ×4 deck/start assignments,
128 games against Greedy. Both training seeds share these deal blocks.

| Training seed | Arm | Wins /128 | Win rate | Mean turns | Evaluation cancels |
| --- | --- | ---: | ---: | ---: | ---: |
| Common initial | Initial | 86 | 67.1875% | 9.4688 | 16 |
| 266 | VRPO | 86 | 67.1875% | 10.6094 | 35 |
| 266 | Terminal | 76 | 59.3750% | 9.7500 | 26 |
| 267 | VRPO | 84 | 65.6250% | 10.3828 | 10 |
| 267 | Terminal | 84 | 65.6250% | 10.0391 | 12 |

Terminal minus VRPO win differences are -7.8125 percentage points for seed266
(95% paired interval [-18.7500, +3.1250]) and 0 for seed267
([-10.1563, +10.1563]). The two-seed average is -3.9063 points,
with shared-deal bootstrap interval [-12.8906, +4.6875]. Average turn difference
is -0.6016, interval [-1.4492, +0.3281]. The joint review resamples the same
32 blocks across both seeds, 20,000 times, seed20261009268. These intervals
condition on two trained endpoints; an exploratory crossed seed/deal bootstrap
also spans zero. Two seeds from one pretrained source do not establish a broad
training-population effect. The pilot used a different panel and is not pooled
into these intervals.

The shorter games are not exclusively faster losses: the subsets where both
arms win also shorten, by 1.47 and 0.87 turns respectively. Outcome-conditioned
subsets are descriptive and cannot isolate a causal effect. Strength remains
the adoption gate.

## Repetition and optimization

| Seed / arm | Learner rows | Training cancels | Repeated-input groups | Groups with >=8 cancels | KL early-stop updates |
| --- | ---: | ---: | ---: | ---: | ---: |
| 266 / VRPO | 57,348 | 103 | 9 | 2 | 11 |
| 266 / Terminal | 54,775 | 88 | 15 | 1 | 4 |
| 267 / VRPO | 60,075 | 115 | 13 | 2 | 8 |
| 267 / Terminal | 58,653 | 67 | 17 | 0 | 3 |

Across the pilot and replications, raw training cancellations sum to 242 in each
arm; repeated-input groups sum to 27 versus42, and >=8 groups to4 versus4.
Trajectories, opportunities and learner-row counts differ, so these totals are
not paired rates. No consistent elimination of looping is established. Identical
actor inputs can hide pending private selections; repetition is an investigation
signal, not an automatic bad-action label.

Terminal credit can reinforce both an unnecessary repetition and the subsequent
successful exit. Correcting short-return credit on selected winning exits did
not solve that distinction. Seed266 terminal gradient norms have a higher peak
(3.8268 versus0.9682), but clipping remains active, and this observation does not
establish the cause of its win-rate difference. Full-update effects include all
rows, regularizers, critic and optimizer state; individual advantage signs do
not determine probability changes.

## Next experiment: cancellation budgets with stress coverage

The earlier 2,304-game budget pilot had no baseline with more than4 cancellations.
It therefore could not validate behavior on the severe repetitions now observed.

Fresh study: `out/research/cancel-budget-stress-restart-2026-10-09/`.
Identity: `1cd40984ef6f5eadb17617690b4b76febdbb122738f7bb8eb2bffb8e1ab3064b`.

- Stress panel: all26 games with >=8 candidate material cancellations from the
  frozen completed seed0 warm u256/u512 and seed1 cold u256 cells. Include losses
  and cases without repeated-input classification. These are selected cases,
  not population win-rate evidence.
- Fresh panel: 32 new deal blocks, both deck assignments and starts, against
  Greedy and historical RL: 256 inputs. Candidate is fixed seed0 warm u512.
  Panel seed20261009269; shared-block bootstrap seed20261009270.
- Budgets32/4/1/0 for every input: 1,128 games total. Preserve sole legal exits.
  Require all26 stress budget32 controls to reproduce original outcomes, engine
  responses and semantic steps before running any intervention. Cold-replay
  every game and stop on health or replay failures. No failed-game replacement.
- Report stress and fresh results separately, including per-opponent outcomes,
  turns, decisions, cancellation counters and actual mask interventions.
  Changed sampling alignment means a changed winner is not by itself evidence
  of a particular tactical mistake caused by removing cancellation.

The known alternate-target preflight confirms budget0 blocks an available legal
route; budgets1/4/32 preserve its initial cancel. This does not prove that route
is strategically required, or that every downstream selection completes. All19
existing budget/material/target-guard tests pass. No automatic budget/model
promotion: selection-completion, tactical exceptions, retention and broader
opponents remain adoption checks.

### Preserved STOP and recording fix

The initial `cancel-budget-stress-2026-10-09/` study stopped during baseline
comparison before any intervention. Both completed controls reproduced outcomes,
responses and every step field except optional `probs`: the historical observer
recorded sampling probabilities while `cancel_budget.play` did not. This was a
recording comparison defect, not a different duel or training failure.

`transcript-mismatch-audit.json` in the failed root verifies both cases. Its STOP,
identity and artifacts remain intact. The restart keeps the identical panel,
seeds, models and budgets, excluding only optional `probs` from historical step
comparison. Responses, actions, choices, options, lengths and every other field
must still match. A separate rerun of failing game253 passed: 299 decisions,
32 cancellations, identical outcome/responses and non-probability fields.
That preflight is outside the registered 1,128-game panel.

The restart uses two workers, a three-hour cap, per-game durable records,
10-second heartbeat and an independent watchdog. Formal training runtime and
policy/mask defaults are unchanged. Completion or failure is delivered to the
research thread; delivery does not itself acknowledge or resolve an event.

## Follow-up completed

The registered 1,128-game cancellation-budget study completed and passed its
independent audit. Budget1 is a confirmation candidate, with strong cancellation
reduction but no established ordinary-game shortening or population strength
guarantee. See [results and the new three-checkpoint confirmation](cancel-budget-stress-2026-10-09.md).
