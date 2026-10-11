# Resource ablation confirmation: replicated gain, unresolved attribution

The [registered resource ablations](critic-residual-confirmation-2026-10-10.md)
completed 512 healthy games / 128 new openings. Independent verification
reproduces all ten prediction groups and aggregate/role metrics exactly, with
the actor unchanged. Game 491 has no eligible learner idle state; its normal
terminal result remains in outcome totals, while prediction metrics use 511
games. No absent prediction is assigned zero error.

## Confirmation

| Residual view | Overall Q/V MSE | Facing targets | Operating targets | Facing Runick V bias |
|---|---:|---:|---:|---:|
| Frozen critic | 0.48943 | 0.61191 | 0.36743 | +0.47553 |
| Intercept | 0.48955 | 0.61229 | 0.36730 | +0.47650 |
| All resources | 0.47663 | 0.58262 | 0.37106 | +0.11096 |
| Without LP | 0.47448 | 0.58063 | 0.36874 | +0.13429 |
| Without deck counts | 0.48085 | 0.58950 | 0.37262 | +0.18739 |
| LP only | 0.48956 | 0.61234 | 0.36726 | +0.47577 |
| Deck counts only | 0.48916 | 0.61116 | 0.36763 | +0.46702 |
| LP + deck counts | 0.48915 | 0.61118 | 0.36760 | +0.46618 |
| Remaining resources | 0.47951 | 0.58768 | 0.37177 | +0.19696 |
| Group-tau-0.25 reference | 0.49199 | 0.59911 | 0.38530 | +0.22524 |

For all resources versus the frozen critic, aggregate MSE decreases by **2.6%**.
The effect is smaller than the preceding panel's 8.1%; these are separate panels
scored by the same frozen actor/critic, not evidence of training regression.

The same 20,000-replicate stratified paired opening bootstrap gives descriptive,
unadjusted 95% intervals. Opponent/seat games from one opening remain together;
strata preserve target/anchor/role and eligible-game weights, using a sum/count
ratio when a sampled opening has fewer eligible games.

| Contrast / slice | Q/V MSE difference | Interval |
|---|---:|---:|
| Resources minus frozen, all | -0.01279 | [-0.02320, -0.00214] |
| Resources minus frozen, facing | -0.02929 | [-0.04493, -0.01251] |
| Resources minus frozen, operating | +0.00364 | [-0.01000, +0.01744] |
| Resources minus frozen, facing Runick | -0.23483 | [-0.31913, -0.13500] |
| Resources minus frozen, familiar | -0.00460 | [-0.01396, +0.00468] |
| Resources minus without LP, all (primary) | +0.00216 | [-0.00152, +0.00589] |
| Resources minus without deck counts, all (primary) | -0.00421 | [-0.01189, +0.00342] |
| Without LP minus frozen, all | -0.01495 | [-0.02382, -0.00593] |

The aggregate and facing-Runick gain replicates. The previous panel's clear
own-role improvement does not: the current point estimate is a small regression
with an interval spanning both improvement and harm. This is not a demonstrated
noninferiority result. Familiar-list improvement also remains uncertain.

Tearlaments pooled MSE increases 0.35331 to 0.37185, following a smaller regression
on the preceding panel. Labrynth (0.31701 to 0.32757) and Kashtira (0.49566 to
0.50455) also regress on this panel. These retention signals remain open and
must be reported in subsequent replication, not hidden by aggregate gains.

At the first low-deck/LP-leading idle state, nine facing-Runick games have mean
terminal return -0.33333. Resource correction changes mean V +0.58330 to -0.28610
and V MSE 1.69507 to 0.90572. The slice's small mean bias does not establish good
individual-state prediction; it remains small and trajectory-conditioned.
Across all facing-Runick states, mean high bias is still +0.11096.

## What the ablations establish

The added LP columns are not necessary for this residual path's observed gain:
the version without them improves over the frozen critic. But the direct
with/without-LP comparison is inconclusive, as is the incremental deck-count
comparison. The larger deck contribution suggested by validation is therefore
not independently established. LP-only, deck-only and their combination yield
little improvement, while remaining resource features retain much of the gain.

These are refitted linear residuals conditional on an unchanged critic that
already receives resource-related information. They do not remove LP/deck
information from the whole model, intervene on a legal game state, identify a
unique causal feature, or prove nonlinear feature interactions. The evidence
does not support a universal LP penalty or the claim that LP overvaluation alone
causes the Runick error.

The actor won 390/512 games; mean length was 11.578 turns. All ten estimators
score those same outcomes. Neither number measures a gain from changing the
critic, and comparisons across different opening panels do not establish a
turn-length trend.

## Next: critic-initialization replication without new game collection

`critic-residual-initialization-replication-2026-10-10` completed three existing
balanced-MSE auxiliary critic initializations (202610120701/702/703). For each,
it freezes the actor and critic, then fits intercept, all-resource and no-LP
residuals on the original balanced training corpus. Each view independently
selects regularization on the original validation corpus before either scoring
panel is loaded. All nine selected fits and twelve outputs per panel are retained.

The two already-evaluated residual confirmation panels are reused as **diagnostic
cross-initialization replication**, not fresh independent confirmation. This
saves game collection and makes comparisons paired. Their 1,024 games are shared
across models, not multiplied by the number of critics; auxiliary initialization
variation is not independent policy-seed variation. Report each seed and paired
differences, both roles, all targets, and the repeated Tearlaments signal.

Independent verification reproduces all nine selected coefficient sets,
validation metrics/minima and 24 panel prediction groups exactly. The results
are extremely close across auxiliary initializations:

| Panel | Frozen Q/V MSE range | Resources range | Without LP range |
|---|---:|---:|---:|
| First residual confirmation | 0.463510–0.463554 | 0.426143–0.426162 | 0.431733–0.431763 |
| Ablation confirmation | 0.489404–0.489466 | 0.476634–0.476665 | 0.474476–0.474505 |

The first panel's resource-corrected own-role MSE is 0.250124–0.250169, below
its frozen 0.266755–0.266950. The second panel's corrected own-role MSE is
0.371061–0.371094, above its frozen 0.367425–0.367518. The split therefore
persists across these seeds. The no-LP variant ranks worse on the first panel
and slightly better on the second; do not select it using this reused evidence.

Only the added resource MLP's initialization varied; the original critic,
actor, data and deterministic fitting procedure were shared. This is a narrow
initialization check, not three independently trained critics or RL runs.

## Next: paired opponent shift

`critic-residual-opponent-shift-2026-10-10` is running 512 fresh games / 128
openings, disjoint from 1,216 prior openings. It replaces the previous greedy
opponent with a frozen copy of the current u528 actor and retains old-256x2 as
control: 256 games against each opponent, sharing opening deals and seats.
The learner policy remains unchanged. All three initialization variants and
their frozen/intercept/resource/no-LP predictions are scored without refitting.

Primary analysis compares resource-minus-frozen error separately by opponent
and role, then tests how that difference changes between opponents with paired
opening resampling. Target retention, especially Tearlaments, remains required.
This tests opponent-distribution transfer, not transfer after updating the
learner's policy or encoder. The mirror opponent is not assumed to be stronger.

Runtime and source/checkpoint/report hashes are sealed, and an independent
watchdog monitors the study. No actor-facing promotion or checkpoint migration
occurs. Learner-policy-shift validation and a matched RL continuation remain
necessary before claiming better learning or playing strength.

Evidence under `out/research/critic-residual-ablation-confirmation-2026-10-10`
includes `analysis.json`, `completion-verification.json`, prediction tensors,
replays, zero-idle metadata, and the registered input identities. The next study
retains its registration, all candidate fits and validation selection records.
