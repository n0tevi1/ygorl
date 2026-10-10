# Same-input exit credit: correction and next experiment

The cancellation-row ablations previously called "other-state interference"
removed all non-cancel samples, including a material-selection action with the
**same actor observation**. That interpretation was too strong. The recorded
probabilities remain valid; the experiment did not isolate other states.
Identical actor input can also hide different private pending choices, so this
is an observation/credit analysis, not proof that all cancels are mistakes.

## Action-kind ablation completed

All three preselected cases completed: six passing numerical baseline replays
and 68 interventions, one for every chosen action kind present in each rollout.
All original minibatch indices and baseline acceptance decisions match; no
category intervention differs in its own KL acceptance. Source:
`out/research/selection-credit-kind-attribution-2026-10-09/` (identity
`f6ef88a5190ccf6ab61d3c04f1f851d3d1eb8a00a991f9ccf6e4b556de7a776d`).

Removing select-row PPO loss has the largest effect in all three cases:
97.0571% → 95.7666%, 99.1809% → 97.6045%, and **88.6642% → 91.6730%**.
The last contrast removes beneficial credit. Thus removing selection training
is not a remedy, and aggregate mean advantage by action kind cannot explain
its effect on a specific input. All categories, including contrary directions,
are retained in the report.

## Isolate the same-input alternative action

Each case contains one select row matching its repeated cancellation input:
seed262/u35 row1580 (normalized A=-0.677858), seed263/u43 row2032
(A=-2.522818), seed263/u33 row2014 (A=+1.024658). All differences below change
only the designated PPO policy loss, retaining full critic/entropy/reference-KL,
Adam state, original denominator and original accepted minibatch schedule.

| Case | Recorded P(cancel) after | Omit same-input select | Omit all other select rows | Only same-input policy rows |
| --- | ---: | ---: | ---: | ---: |
| seed262/u35 | 97.0571% | 96.0777% | 97.0076% | 96.5591% |
| seed263/u43 | 99.1809% | 97.9921% | 99.2353% | 99.2407% |
| seed263/u33 | 88.6642% | 91.9535% | 88.2101% | 84.9007% |

The same-input exit sample accounts for substantial movement in both harmful
cases and the beneficial case. With only same-input policy rows, u43 still
increases strongly; other states are not necessary for that increase (the
full critic/regularizers and inherited Adam are still retained). Effects are
nonlinear and do not form an additive attribution. The u43 other-select-removal
variant differs from its own KL guard once; every other variant agrees.

All six numerical baselines pass, with maximum state error 8.531e-7, probe
probability error 5.961e-8, KL error 3.540e-8. Strict bitwise replay is still
not claimed. Reports: `selection-credit-exit-attribution-2026-10-09/`, identity
`b5bdedb5ef6bfb5328fb567e42c0d5eb9d5c4a76e95e37b6c54bfed6b8f58183`.

## Credit audit: short returns and bootstrap

In seed262/u35, exit row1580 has raw A=-0.071534 and Q target 0.557428 vs
expected Q 0.628962. The same acting player wins 19 recorded learner decisions
later, within the same rollout (slot12, t98→117). This is a realized winning
continuation, not proof that the selected action was optimal.

In seed263/u43, exit row2032 is the final rollout row (slot0, t127). Its target
-0.342191 equals the bootstrap expected Q exactly; no terminal reward has yet
been observed. This actor subsequently loses at u44/t56 under a policy that
has meanwhile been updated. That later outcome is not an on-policy estimate
of its earlier counterfactual action value. In the beneficial u33 case the
exit has positive raw A=+0.072002 and the actor later wins, also across updates.
Evidence: kind-attribution `same-input-actions.json`, `exit-credit-audit.json`.

## Frozen full-game audit completed

Two fixed checkpoints each sampled exactly 64 new complete selfplay games:
128 healthy games, zero optimizer updates, model tensors verified unchanged.
All games and full rollout tensors were retained. This removes open rollout
tails and inter-update policy changes from the new credit audit.

The warm35 source produced 49 cancels over 9240 rows and five repeated-input
groups; warm43 produced five over 9544 rows and one group. Four groups have
an observed select row at the identical input. Two winning exits have negative
raw lambda=.5 credit (-0.009131, -0.043646), while terminal z minus the **same**
expected-Q baseline gives +0.872570 and +0.705720. The other two observed exits
have matching signs: a losing exit negative, a winning exit positive. This
shows short-return sign differences persist even with complete trajectories;
it does not establish the frequency in a larger population or label every
negative advantage wrong. Sampled terminal returns also have variance.

The highest-probability group repeats 32 times at 99.0630% and wins. It has no
select row at the identical encoded input: the guard can change the mask and
forced choices can be skipped from learner rows. Do not count "no matched
select row" as "never exits," or quietly drop this severe case.

VRPO lambda=1 and pure terminal returns are reported separately: VRPO retains
control variates, so lambda=1 is not pure Monte Carlo. Source:
`out/research/selection-fullgame-credit-2026-10-09/`, identity
`28f16ec0b611a767e777d8601b9c953c47bed946246fc6da8acf2e37d88d5c57`.
Its inherited `updates` metadata field is unused; the frozen protocol and runner
use four collection batches per source and assert zero optimizer updates.

Next test: a bounded paired learning pilot using complete games in both arms,
changing only policy credit from lambda=.5 VRPO to terminal z minus expected Q.
Keep critic targets, reward, mask and other PPO terms equal. Use common fresh
Greedy evaluation games and report wins, turns, cancellation/repeats, health
and variance; no promotion from reduced cancellation alone. The representation
aliasing and sample variance remain limitations, and no card-specific rule or
blanket cancel penalty is introduced.

## Registered policy-credit pilot is running

`out/research/selection-terminal-policy-pilot-2026-10-09/`, identity
`65ca92ae4e8f7e7272bd0d9c671b2995522aebd123c01a1a292a434de7130d7a`:
training seed265, one common seed250 legacy endpoint, two arms ×24 updates
×16 complete selfplay games (384 games per arm). Actor, critic, Adam and EMA
reference inherited identically; no warmup or snapshots. Both use complete
collection and lambda=.5 critic targets. The experimental clipped-PPO objective
only overrides `prepare()` with normalized `terminal_z - expected_Q`; all other
loss terms and hyperparameters remain equal. It is a research-local registration,
not a change to production defaults. Every rollout, pre/post learner state and
minibatch index/acceptance is saved for later attribution.

Preflight matches the independently saved full-game terminal advantages and
rejects incomplete/truncated input. Fresh panel seed20261009265 provides 32
common deck/deal blocks ×4 assignments/starts =128 Greedy games per endpoint.
The identical initial actor is evaluated once; both final actors get the same
panel. Accepted action/engine-response replays and health are retained. Report
paired block uncertainty, strength/turns/decisions/cancellation, differing data
row and compute budgets, advantage variance and KL stopping. Keep one-seed,
one-source and Greedy limitations explicit. No promotion without further
multi-seed, selection-completion and retention checks. The independent service
has a two-hour cap, per-update checkpoints, heartbeat and watchdog; the formal
immunity study continues unchanged.
