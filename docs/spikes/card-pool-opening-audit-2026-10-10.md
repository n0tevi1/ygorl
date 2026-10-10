# Opening-deal reuse: evaluation correction and critic follow-up

Follow-up: [Runick critic bias mechanism audit](runick-critic-bias-2026-10-10.md)
traces the private fit's remaining overestimation to competing training gradients
and misleading LP/deck-count sensitivity, with exact replay and explicit limits.

The card-pool follow-ups did not generate the fresh opening deals claimed in
previous reports. Exact result/response replay and new engine RNG seeds passed,
but neither check proves opening independence. Preserve the original artifacts;
withdraw their new-opening interpretation before adapting the critic.

## Mechanism and scope

`pairing_slots` explicitly pre-shuffles deck lists and sets `shuffle_decks=False`
so checkpoints, opponents and seat swaps receive common deals. Later one-off
preparation scripts copied those lists and changed only core/agent RNG seeds.
`Duel.loaded_decks()` correctly preserves their order in this configuration;
the core does not shuffle at duel start. The failure is in follow-up panel
construction and its missing coverage check, not a newly discovered core RNG bug.

| Panel | Games | Distinct ordered main-deck pairs | Meaning |
|---|---:|---:|---|
| Original coverage | 1,024 | 64 | Actual original two-draw, two-role panel |
| Pool-training canary | 768 | 64 | Reused coverage orders, new continuation RNG |
| Each training replication | 768 | 64 | Same coverage orders again |
| Retention confirmation | 2,304 | 16 | Only original draw0, facing role; four nominal draws reuse it |

All listed jobs explicitly disabled engine-side shuffling. Retention games
228–231 have distinct engine seeds but identical replay-loaded top five cards
for both decks. This is direct evidence in saved replays, not just a diagnosis
from generator code. Counts distinguish deck roles; seat swaps do not count as
new orders. Full file hashes and concrete examples are saved in
`out/research/card-pool-fresh-deals-2026-10-10/legacy-opening-audit.json`.

The original outcomes, engine health checks and conditional action interventions
remain observations of those specific states. The four nominal retention draws
must not be bootstrapped as independent opening clusters. Previously published
retention intervals are withdrawn for new-opening inference; the new panel is
not pooled with them. Seed-disjoint replication was not deal-disjoint replication.
This finding alone does not invalidate the separately generated training runs:
Trainer constructs the normal shuffle-enabled DuelConfig, and both Python and
native training hosts consume `Duel.loaded_decks()`. It also does not automatically
invalidate other experiments whose deal generation has not been audited here.

## What the completed critic audit establishes

The 2,304-game critic replay was mechanically verified, including response bytes,
actor probabilities, frozen identities and predictions. It inherits the same
16-opening restriction. Game-averaged V bias and squared terminal-return error
on the Runick slice (96 games per arm) are:

| Checkpoint arm | Bias | MSE |
|---|---:|---:|
| Start | +0.2866 | 1.1350 |
| Fixed pool | +0.5553 | 1.3432 |
| Expanded pool | +0.4529 | 1.2185 |

Predicting zero has terminal-return MSE 1. Familiar-deck V bias instead ranges
from -0.2976 to -0.2232. Thus a global calibration shift is not supported even
within these sampled trajectories. Reviewed-card presence correlates with larger
overestimation but does not prove that a lock is active or identify its cause.
Possible explanations include sparse coverage of alternate win conditions,
features that work in ordinary combat becoming misleading, and a fixed-opponent
value distribution different from the training league. These remain hypotheses;
the flawed opening coverage prevents a broad root-cause conclusion.

## Repair and replacement evaluation

`ygorl.eval.panel_deals.redeal_job` shuffles canonical card multisets explicitly.
Its key contains panel seed, target, anchor, role and draw; checkpoint, opponent
and engine seat are excluded. It preserves deck contents and extra decks and
binds the generated order plus engine/agent RNG in job metadata. Validation fails
on stale metadata, double shuffling, prior-order overlap, or different draw labels
sharing an order. Repeated opening hands can legitimately occur; their count is
reported without rejecting or resampling them. Engine-loaded deck orders are
checked before play. Regression tests query actual engine hand contents across
models, opponents and seat swaps and reproduce the old seed-only failure.

Generate future panels through the shared helper, rather than copying prepared
jobs and assigning RNG seeds:

```sh
python -m ygorl.eval.panel_deals --input old-panel.json --output panel.json \
  --panel-seed 202610109101 --prior another-prior-panel.json
```

For `card_generalization_games.run`, register `fresh_opening_deals=true` and
`prior_panels` (including the source panel); those files must also be identity
bound. The runner validates coverage before scheduling any games. Legacy replay
studies keep their explicit fixed-state semantics and original artifacts.

The replacement `card-pool-fresh-deals-2026-10-10` registers 2,304 games on **64
new ordered pairs and 64 opening-hand pairs**, with zero order overlap against
all five prior panel files. Four deals per target/anchor are shared across nine
existing checkpoints, two opponents and both seats. It uses the original frozen
runtime so that reshuffling, rather than a concurrent engine patch, is the
intervention. All full games receive exact cold replays and terminal games also
receive the same no-gradient critic audit. Limits remain non-wins and are reported
as censored for terminal critic calibration. No actor or critic optimization,
reward change or policy promotion occurs in this measurement stage.

Primary comparisons remain fixed-minus-start and expanded-minus-start facing
added decks, with familiar retention and deck/opponent breakdowns. Bootstrap
blocks now correspond to verified target/anchor/draw orders, shared across
checkpoints and seats. Only three existing training seeds and eight target lists
are covered; this is not proof of broad card understanding or elite strength.

If Runick overestimation persists on this corrected panel, the next critic-only
experiment will use separately generated training/validation/test deals, frozen
actor parameters and short validation-selected fits. The evaluated old and new
panels must not become critic training data. Previous terminal-critic work showed
that extended fitting could overfit, so lower training loss is not a promotion
criterion; subsequent matched RL continuation and retained gameplay performance
are required.

Validation at repair: 79 distinct focused tests passed across panel deals, arena,
agent matrix, duel and monitoring suites; lint and whitespace checks passed. Two
end-to-end frozen-runtime preflight games also passed actual-order checks, full
cold replay, critic response replay and actor-probability parity. The replacement
run passed all 18 registered health canaries and entered full evaluation with an
independent invocation-bound monitor. Completion results are verified below.


## Corrected panel completed

All 2,304 games completed without engine-health failures or limits. Every saved
result and replay hash was verified, actual replay-loaded orders matched the
64 registered new pairs, critic replay/probability checks passed, and both
registered analyses reproduced byte-for-byte under the bound interpreter.
The completion incident was acknowledged after verification. These are the same
nine existing checkpoints; new numbers do not represent newly trained models.

| Metric | Start u512 | Fixed u528 | Expanded u528 |
|---|---:|---:|---:|
| All wins / 768 | 501 (65.23%) | 512 (66.67%) | 507 (66.02%) |
| Mean turns, all | 11.61 | 12.07 | 11.29 |
| Familiar wins / 384 | 249 | 258 | 247 |
| Added-list wins / 384 | 252 | 254 | 260 |
| Runick wins / 96 | 63 | 55 | 69 |
| Runick mean turns | 21.25 | 22.05 | 21.81 |
| Added-list deck-out losses | 29 | 29 | 17 |
| Self-Ash choices / offered windows | 9/361 | 8/349 | 6/321 |
| Candidate repeat-selection games | 2 | 4 | 2 |

Added-list fixed-minus-start is +0.52 percentage points, crossed seed/deal
bootstrap interval [-9.38, +10.68]; expanded-minus-start is +2.08 points
[-6.77, +10.94]; expanded-minus-fixed is +1.56 points [-6.51, +9.64].
No paired win interval excludes zero, including familiar retention. Expanded has
promising Runick/deck-out point estimates but not established general strength
or retention benefit. Four deals per target/anchor and three existing policy
seeds still provide limited coverage. Comparing these means with the older
reused-opening panel would confound opening distribution with policy progress.

The critic conclusion changes materially:

| Runick V diagnostic | Start | Fixed | Expanded |
|---|---:|---:|---:|
| Mean signed error | +0.047 | +0.210 | -0.076 |
| MSE | 0.888 | 1.019 | 0.845 |
| Reviewed-card-present V bias | +0.356 | +0.514 | +0.234 |
| Reviewed-card-present V MSE | 1.014 | 1.163 | 0.990 |

The earlier claim that all three Runick critics were worse than zero prediction
(MSE1) does not survive new openings. Fixed-pool continuation retains a
conditional overestimation signal, especially when reviewed cards are present.
The state-conditioned samples contain 61/67/62 games respectively. Actor
trajectories differ between arms, reviewed-card presence is not a verified lock,
and fixed opponents differ from the training league; these are descriptive
calibration diagnostics rather than proof that particular critic weights caused
losses. Familiar V bias remains negative (-0.237/-0.271/-0.226), arguing against
an indiscriminate global downward value adjustment.

Behavior is not solved: 23 self-Ash choices occurred in 1,031 offered windows,
and eight games triggered repeated-selection warnings. Game388 contains a
four-action select/select/unselect/unselect material cycle with changing legal
menus, ending when a third material is selected. This differs from the ignored
required-material deselection bug; all games eventually terminated. Warnings
remain review signals, not automatic tactical-error labels. Evidence is saved
in `completion-verification.json`, `analysis.json`, `critic-analysis.json` and
per-game traces under `card-pool-fresh-deals-2026-10-10`.

## Independent critic adaptation corpus started

`card-pool-critic-refresh-data-2026-10-10` now collects 512 training, 256
validation and 512 test games with the fixed seed0-u528 actor. Its 320 explicit
new ordered pairs are mutually split-disjoint and disjoint from all prior
coverage/retention panels, including this corrected evaluation. Two full-game
preflights verified replay parity and equivalence between cached-feature critic
outputs and full model outputs. Eight registered health canaries precede the
full collection. No actor or critic parameters are updated during collection.

Only first learner idle prompts per turn are cached, together with privileged
inputs and terminal returns. Every full game has a cold replay; truncation,
health failure or missing learner samples stops corpus construction without
silently dropping a case. Validation/test outcome scores are not summarized
before model selection. The fitting plan is registered: private critic and
privileged encoder only, frozen actor/shared trunk, equal Q/V game-weighted MC
loss, learning rates1e-5 and1e-4, epochs0/1/4 and three optimizer-shuffle seeds.
Select by validation error, retaining epoch0; evaluate selected fits once on test
and compare with unchanged, train-mean and train-only affine controls. Cached
private-gradient parity must be checked before fitting. This is one behavior
actor, not three independent policy training runs. Success would establish
limited supervised calibratability, not an RL or gameplay improvement; matched
continuations and familiar retention remain subsequent gates.

A startup monitoring race was observed and investigated: the first poll ran about
four seconds before the initial heartbeat file existed and treated missing time
as epoch0. The process was already running, passed canaries, and had no failed
games. The monitor now grants at most180 seconds only for a missing first file,
anchored to the matching invocation's systemd monotonic process-start timestamp.
It never exempts existing stale heartbeats, STOP, rollout failures, dead processes
or invocation changes. Restarting the watcher cannot renew this period. Original
incident evidence is retained; 17 monitor regression tests pass.

## Critic collection stopped on a valid zero-sample game; recovery

The corpus stopped after 1,247 committed completions; 1,248 complete cached
records existed after in-flight work settled. Game1248 ended normally on turn2
before the learner received any `SelectIdleCmd`: the opponent paid a lethal LP
cost and the learner won. The learner had chain prompts, but none of the
registered idle prompts. Exact replay and engine health passed. The assertion
that every healthy game must yield an idle sample was incorrect.

The amendment preserves the registered sampling population: retain the full game
and an explicit `no_eligible_idle` record with empty predictions/features. Do not
replace its opening, invent a fallback state, or silently drop the game. Report
both total-game and eligible-game denominators and zero-sample counts per split;
critic losses remain conditional on eligible idle states. The test-game outcome
was inspected to debug this failure; no fit hyperparameters were changed from
that outcome. Claims of literal complete test-outcome blindness must therefore
include this operational exception.

`card-pool-critic-refresh-recovery-2026-10-10` binds and reuses 1,248 completed
records, replays only the unfinished critic audit for game1248 and collects the
remaining31 games. All reused job/result/replay/cache/prediction hashes are bound
in the recovery identity. Original STOP and failure evidence are untouched.
Preflight replay verified both the exact zero-sample case and an ordinary
three-sample game. Fitting must wait for complete recovery/cache verification.

The monitor recorded this real STOP but did not redeliver it: incident IDs were
keyed only by study identity and `attention` kind, so the earlier acknowledged
startup warning swallowed subsequent failures. The repair keys episodes by
cause and occurrence, retains prior incidents, gives a new STOP a fresh pending
acknowledgment, and allows the same fault to notify again after recovery.
Secondary stale/inactive symptoms of an existing STOP remain deduplicated.
Watcher restarts do not create duplicate active episodes. Regression tests cover
new STOP after an acknowledged warning, repeated polls, cause persistence and
same-cause recurrence after recovery; all19 monitor tests pass. The previously
missed STOP has now been recorded as its own acknowledged incident, and the
corrected monitor is running on recovery.

## Recovery complete; first private-critic fit improves MSE but not Runick bias

Recovery retained all1,280 games and reproduced its frozen identity/cache
verification. Eligible games are512/256/511 for train/validation/test, with
3,115/1,591/3,220 cached rows. The single zero-idle test game remains explicitly
recorded. The restored corpus is the original registered sample, not a new
independent replication.

`card-pool-critic-refresh-fit-2026-10-10` then executed the registered short-fit
plan. Actor and shared trunk stayed frozen; only private critic and privileged
encoder parameters changed. Exact replay preflight compared cached and full-model
private gradients (maximum absolute difference7.45e-8); a private optimizer step
preserved the actor hash. Three optimizer-shuffle seeds each selected learning
rate1e-4, epoch4 on validation. Selection was sealed before test scores were read;
no test-driven refit occurred. All selected actor hashes match, and baseline plus
selected test metrics were independently reproduced exactly.

| Test mean Q/V MSE | All511 eligible games | Familiar256 | Runick63 |
|---|---:|---:|---:|
| Unchanged critic | 0.7118 | 0.6506 | 1.0910 |
| Train-mean constant | 0.8356 | 0.8474 | 0.9094 |
| Train-only affine calibration | 0.6737 | 0.6069 | 1.1594 |
| Private fits (three shuffle seeds) | 0.6253–0.6312 | 0.5648–0.5684 | 1.0227–1.0300 |

These are paired evaluations on the same fixed actor's trajectories and the
same held-out deals. Optimizer shuffle seeds do not add independent test samples
or policy training runs. The overall point-estimate reduction is about11–12%,
with familiar performance also improving and all three fits outperforming the
affine control on aggregate MSE. This establishes a limited supervised-fitting
signal, not improved policy play: the actor was unchanged throughout.

The target Runick issue remains: V bias rises from+0.2337 to+0.2984–0.3008 even
though V MSE falls from1.0969 to1.0261–1.0326. For shuffle seed202610110202, error
variance falls from1.0423 to0.9356 while squared bias increases from0.0546 to0.0905;
thus an MSE improvement can coexist with worse mean overestimation. Runick MSE
also remains worse than the train-mean constant control. An indiscriminate affine
shift worsens the Runick slice further. Do not call these fits a repair of the
Runick calibration failure or deploy them to main RL training on this evidence.

The post-evaluation decomposition also finds increased Runick V bias on train
(+0.324 to+0.368) and validation (+0.164 to+0.221), not only test. Outcome
frequencies differ between splits, but held-out sampling shift alone cannot
explain the direction. Next investigation examines state-conditional error
(including reviewed-card presence and turn/LP strata) and the effect of optimizing
aggregate MSE rather than conditional calibration. It must distinguish weaker-opponent adaptation from
training-league value quality. Any further fit needs a new registered objective
and independent confirmation data; current test results must not be used for
repeated selection. Matched RL continuation remains necessary to establish a
policy benefit. Evidence: `selection.json`, `report.json`,
`completion-verification.json` and `bias-decomposition.json` under the fit study.
