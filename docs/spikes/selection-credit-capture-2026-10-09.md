# Cancellation credit: captured evidence and replay limits

The coverage diagnostic found a real high-probability cancellation loop, but its
local policy derivatives do not explain the direction of the full model update.
The original capture omitted privileged critic inputs and intermediate optimizer
states. Re-running the same seed did not reconstruct the original rollout. Keep
that STOP and collect complete update evidence in a new, bounded experiment.

## Coverage result

`out/research/selection-credit-coverage-2026-10-09/` completed 24 PPO updates per
arm, inheriting seed-250 replication endpoint actor, critic, Adam and EMA reference.
Both arms reset environments/pool. Each processed 49,152 rows. Legacy completed
329 healthy games and chose cancel 38 times; selection completed 310 and chose
cancel 19 times. No errors or truncations. These are policy-dependent training
samples, not a paired gameplay-strength evaluation.

Legacy had 72 recorded cancel minibatch visits, 69 applied (52 positive normalized
advantage, 17 negative). Selection had 35 visits, 34 applied (9 positive, 25
negative). Rejected minibatches must not be counted as optimizer updates.

Legacy update 44 contains six identical cancellation observations in slot 2,
rows 450, 466, 482, 498, 514 and 530 (time-major layout, 16 slots), followed by a
select action at row 546 with the same input. Behavior P(cancel) is 94.5039%.
The observation SHA256 is
`b4b44d33b864b1c0517807b407b41ebdda23b17120fe31aa22fff28f2d33263a`.

Recomputing all estimator outputs from captured rollout tensors matches the
recorded advantages and Q/V targets bitwise. Raw advantage is negative for all
six cancels. Whole-rollout mean is -0.0100982692, population standard deviation
0.1004013270. Subtracting that mean makes the first three normalized advantages
positive: +0.079617, +0.066482, +0.040212; the next three are -0.012328,
-0.117408 and -0.327567. Batch centering changes individual signs; this alone
is not proof of an incorrect estimator or justification for disabling it.

During the actual update, sampled minibatch probabilities for this state rise
from about 94.97% to 95.91%. Across all ten **applied** visits of these six rows,
the sum of that row's policy, entropy and reference-KL derivatives with respect
to its chosen logit is positive: those local terms favor lowering that logit.
They are not the gradient with respect to shared model parameters. Other rows,
shared actor/critic features, Adam history and clipping still need controlled
attribution. Do not describe this as proof that positive cancel credit alone
caused the probability increase.

Evidence: coverage `audit-loop.py`, `loop-audit.json`,
`legacy/credit-inputs-44.pt` (SHA256
`9652681d267941ad01c97aa25806a8ee0abe03152eba1a3534a2d8affa6d5d3e`),
and `legacy/cancel-terms.jsonl`.

## Failed same-seed reconstruction: retained STOP

`out/research/selection-credit-replay-2026-10-09/` stopped at update 33, before
its first optimizer step, because rewards differed from the original coverage
rollout. The incident was acknowledged; STOP was not cleared. Initial learner
(model, reference and Adam), configuration and stored RNG tensors match exactly
(`initial-state-comparison.json`). A seed is not a replay transcript: collection
uses asynchronous event batches and one generator for action sampling, so event
ordering/batching can change which random draw is assigned to each decision.
This is a concrete reproducibility limitation; the exact first diverging event
was not recorded, so timing has not been isolated as the sole cause of this run.

This failure invalidates the attempted reconstruction, not the original coverage
data. No historical gradient attribution may be based on the failed rerun.

## Full capture protocol

`out/research/selection-credit-capture-2026-10-09/` is a **new** seed-262
experiment: 24 updates per arm, the same inherited sources and unchanged PPO
objective/reward/mask. Identity SHA256:
`92a9dc5b30f922efb5ff3c3ca666aab465b9cd3d052fd6e710d82f1b0ca4133c`.

Each original update now saves the complete rollout, including privileged critic
inputs and finished-game engine responses; complete before/after checkpoints,
including Adam, reference and RNG; and every evaluated minibatch index with its
accept/reject flag. Replay reads these tensors directly and launches no games.
Require the original minibatch order, KL acceptance and final learner state to
match before interpreting counterfactual updates. Capture the full fixed budget,
including updates without cancellation; do not stop at a favorable result.

The pre-optimizer rollout-health gate, 10-second heartbeat, external watchdog
and durable checkpoints remain active. The frozen formal critic-continuation
study is unchanged. Full capture completed: 325 legacy and 304 selection games, each arm 49,152
rows, zero errors/truncations. Legacy chose cancel 43 times (18 at >=90%
probability), selection 21 times (4 at >=90%). Legacy has two high-probability
identical-input groups: 13 rows at update 35 and five at update 53. Selection
has no identical-input cancel groups; its history intentionally changes after
each accepted selection, so this does not establish absence of behavioral loops.
This is new evidence, not a reproduction of the coverage update-44 loop.

## Numerical replay and first local counterfactuals

Strict bitwise replay was **not** achieved. Two initial update-33 attempts matched
all minibatch indices and acceptance decisions, but final tensors differed at
about 1.2e-7, including after matching CPU reduction thread count. Those failed
reports remain in `replay-legacy-33-attempt1.json` and `replay-legacy-33.json`.

Before running counterfactuals, `attribution-protocol.md` explicitly revised the
criterion for a separately labeled numerical diagnostic: two baseline replays,
exact index/acceptance matching, max absolute state error <=5e-6, KL error <=1e-6,
probe probability error <=1e-5. Effects within 100 times measured baseline
probability error are inconclusive. This is not a claim the original strict
criterion passed. Protocol SHA256:
`fb76949c45379a4f3bc55e78c15f531f09d61f9a0611102f9ee1cf9ad07afa5a`.

Both update-35 baselines pass that numerical gate: state error 1.1921e-7,
probability error at most 5.9605e-8, KL error at most 8.3819e-9. These are much
smaller than the measured contrasts. The original update raises cancellation
from 95.7519% to 97.0571% on the 13 identical observations. All 26 applied visits
have positive normalized advantage, yet their summed local chosen-logit
policy/entropy/reference-KL derivative favors decreasing that logit.

| Fixed-data update | Final P(cancel) | Own KL guard differs from original schedule |
| --- | ---: | ---: |
| Recorded ordinary PPO | 97.0571% | 0 |
| Set Q/V loss coefficients to zero | 97.2185% | 0 |
| Zero Adam first/second moments, retain step counters | 97.6707% | 7 minibatches |
| Scale advantages without mean subtraction | 97.1665% | 0 |

The original minibatch and acceptance schedule is imposed on every variant.
In particular the Adam variant is **not** what an ordinary live training update
would do: its own KL guard would reject steps. None of these single changes
prevents the increase on this captured batch. This does not establish that any
component is harmless everywhere, or justify changing optimizer/critic defaults.

Artifacts: `attribute-update.py`, `attribution-identity.json`, `attribution.json`
and `attribution.log`. A separately registered `row-attribution-protocol.md`
next separates the loop rows' actor losses from other rows' actor losses while
keeping critic targets, inherited state and batch denominators unchanged. That
can test local interference through shared parameters; it cannot establish a
safe long-run policy change from one batch.

## Other policy rows outweigh the loop rows' local correction

The second registered diagnostic also passes baseline validation (state error
1.1921e-7, probability error 5.9605e-8, KL error 1.3504e-8). It retains the
same complete critic losses, Adam state, minibatch order/acceptance and loss
normalization denominator in every arm:

| Actor-loss rows retained | Final P(cancel) |
| --- | ---: |
| All rows, recorded baseline | 97.0571% |
| PPO policy term on the 13 loop rows only; entropy/KL on all rows | 94.1422% |
| Policy/entropy/KL terms on loop rows only | 95.0945% |
| Policy/entropy/KL terms on all **other** rows only | 97.2207% |

None of these variants changes its own KL acceptance relative to the recorded
schedule. On this captured update, other states' policy losses, propagated
through shared policy parameters, drive the increase: excluding them reverses
its direction. Excluding the loop rows' actor losses instead makes the increase
larger. Thus these loop rows provide a net correction which is outweighed by
updates from other states. This is a parameter-space intervention, beyond the
original chosen-logit derivative observation. It is conditional on one batch,
checkpoint and inherited optimizer; nonlinear effects need not add up, and it
is not a complete historical explanation of how the high initial prior arose.

Evidence: `row-attribution-protocol.md`, `row-attribution-identity.json`,
`attribute-rows.py`, `row-attribution.json`, `row-attribution.log`. The next step
is independent captured-batch replication before designing a gameplay-validated
intervention. Do not reset Adam, remove the critic, ban cancellation or change
advantage normalization based on these one-update diagnostics.


## Second qualifying batch: direction is not universal

Update 53 is the only other >=90% identical-input cancel group in the full legacy
capture. Its five-row baseline **decreases** cancellation, from 91.5971% to
90.6790%. The same controls finish at 90.8328% (loop policy only), 91.2501%
(loop actor only), and 90.7074% (other actor rows only). Here other rows also
help reduce cancellation. The first two controls each differ from the original
KL acceptance on one minibatch; the original schedule is imposed and the
counterfactual must not be described as live PPO behavior.

Baseline maximum state error is 7.9512e-7, probe probability error 2.3842e-7,
KL error 2.2352e-8, within the prospectively declared numerical gate. Artifacts:
`row-attribution-u53-protocol.md`, `row-attribution-u53-identity.json`,
`attribute-rows-u53.py`, `row-attribution-u53.json`. This second batch limits the
claim: harmful interference is demonstrated at update 35, not on every update.

A fixed-budget independent collection-seed replication was registered at `out/research/selection-credit-replication-2026-10-09/`: seed 263,
legacy arm, 24 updates, same source checkpoint and all-state capture. Identity
SHA256 `9c45b0b41a3ce379ec7760fa70852b61b0170449027146a72806c2f9d85630e4`.
Its frozen protocol audits every qualifying high-probability repeated group;
no qualifying samples means insufficient coverage, not a negative causal result.
It includes the numerical baseline gates, same three actor-row interventions,
health gate, heartbeat, watchdog and durable checkpoints. Same source checkpoint
means this is not independent-initialization evidence. No production learning
change or schema promotion has been made.


## Independent collection-seed replication completed

Seed 263 finished all 24 legacy updates: 49,152 rows, 299 games, zero errors or
truncations, 72 chosen cancel rows (144 applied minibatch visits). All four groups
meeting the predeclared >=90% probability and identical-input-repeat rule were
analyzed. Each case has two passing numerical baseline replays and three fixed
actor-row interventions; no cases were discarded based on outcomes.

| Update / repeated rows | Before | Recorded after | Loop PPO only, all entropy/KL | Loop actor only | Other actor rows only |
| --- | ---: | ---: | ---: | ---: | ---: |
| 33 / 9 | 92.0554% | 88.6642% | 92.0303% | 94.1268% | 88.3951% |
| 37 / 7 | 95.0004% | 92.6567% | 93.1935% | 94.7082% | 93.2441% |
| 43 / 19 | 96.9788% | 99.1809% | 97.0561% | 97.1975% | 99.2153% |
| 56 / 8 | 93.9950% | 91.6987% | 94.9253% | 96.0781% | 91.8363% |

All variants retain the complete critic losses and original batch denominator,
Adam state and step schedule. The u56 loop-actor-only variant would disagree
with the original KL guard on three minibatches; all other listed variants
match original acceptance. Maximum baseline state error across this panel is
3.1665e-7, probability error 3.5763e-7. These remain numerical, not bitwise,
replays. No new trajectories are generated in the attribution comparisons.

The second seed reproduces substantial harmful interference at u43: ordinary
PPO raises cancellation 96.9788% → 99.1809%, but omitting other states' policy
loss yields only 97.0561%. The other three ordinary updates lower cancellation.
This supports a batch-dependent mechanism across two collection seeds; it does
not establish its population frequency, a full history of the original formal
model, or a tested remedy. All continuations share the same initialization.

Representation caveat: identical actor observations are **not** proof of identical
private pending choices. In seed262/u35, intervening select rows choose three
different encoded card identities (passwords 60461804, 22908820, 87758525), yet
the subsequent cancel inputs are identical. Some individual cancels therefore
switch targets; they must not all be labeled mistakes. Consecutive returns to
the same selected identity also occur. The attribution above concerns the
aliased actor input and its recorded losses, not a tactical judgment that every
recorded cancel should be prohibited. Evidence: capture `context-rows-35.json`.

Panel artifacts: `out/research/selection-credit-attribution-2026-10-09/`, identity
`ebf0b08fd3acbab614d6074b48eb60174d904222f0795a44077c8651e4d46594`,
`panel.json`, all per-update baseline/variant reports, `report.json`, `summary.json`.
Both completion events were inspected and acknowledged.

Next registered analysis: `selection-credit-kind-attribution-2026-10-09`, two
harmful cases (seed262/u35, seed263/u43) and a contrasting case (seed263/u33).
After two numerical baseline checks, separately omit PPO policy-loss rows of
every action kind present, retaining all other terms and the original schedule.
Report all categories and numerical/KL limitations, not just favorable results.
This asks which other-state learning signals produce interference and informs
a later separately registered intervention; it is not a production change.
The service has durable reports, heartbeat and an independent watchdog.
