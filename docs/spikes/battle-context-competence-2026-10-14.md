# Explicit battle context: implementation and bounded competence probe

The accepted-attacker input repair reaches training, but the completed 16-update
study did not establish stronger play. This follow-up makes the relationship
between that attacker and each visible target directly accessible. It does not
change default models, reward, legality, cancellation, or hidden information.

## Implementation

`NetConfig.battle_context` defaults false and requires `attack_history`. Resolve
only the most recent event at an own-turn SelectCard battle prompt without a
chain; require matching turn/phase, accepted BattleCmd attack, and exactly one
visible own monster with the recorded card identity and field coordinates.
The engine reports PHASE_BATTLE_START at ordinary target prompts; accept it as
well as PHASE_BATTLE_STEP. Do not scan older events through intervening effects.
Hidden targets, stale/ambiguous context and non-select actions abstain.

The new residual connects contextual attacker and target embeddings to the target
action embedding. It is not an arithmetic oracle. Zero final projection preserves
all old logits during transfer; earlier MLP layers retain random weights so the
path has learning gradients. Saved network configuration determines architecture;
observations retain the existing attack-history schema. Old checkpoint restoration
supplies false for the absent flag. The shared critic consumes the same action
features through its existing interface. No global optimizer reset is introduced.

## Fixed diagnostic

Artifacts: `out/research/battle-context-competence-2026-10-14/`.
Identity: `8ec2117a02d6f6c5181e05ca68c1fd0ba242e5abc02a8279626bac8d7a4e52ff`.
The source is the frozen context-initial u128 model from the October 14 study;
the original actor/trunk is not trained or replaced. The working source files and
native extension are hashed because the implementation is not yet committed.

128 artificial boards each have two attackers and two ordinary, no-effect targets.
Both engine seats and face-up attack/defense targets are covered. Each of four
attacker/target branches runs through native battle resolution. All 512 damage/
destruction outcomes agree with ordinary arithmetic in this restricted domain;
16 branches also reproduce observations, actions and outcomes on cold replay.
This arithmetic equality must not be generalized to effect monsters.

Whole-board split: 64 training, 16 validation, 32 test, 16 extra identity-holdout
boards. The latter uses identities absent from probe training, not necessarily
absent from the source model's pretraining. Each board supplies four correlated
rows; report board counts rather than pretending all rows are independent deals.

Three equal-width 256→64→8 heads use (1) frozen decision context and target,
(2) frozen actual attacker and target, or (3) the other attacker and target.
All use identical initialization and sampled rows within each of two seeds,
128 Adam steps, batch64, LR.001, and training-only normalization. Supervision is
damage direction, damage amounts and destruction flags. Every32 steps saves a
checkpoint; final128 is fixed, with no validation-based endpoint selection.

## Results

On 32 held-out boards /128 branches, ranges across the two fitting seeds:

| Input | Damage direction accuracy | Damage MAE, LP | Both destruction flags correct |
|---|---:|---:|---:|
| Decision context + target |64.84–66.41%|493.82–506.38|64.84–66.41%|
| Actual attacker + target |75.78–77.34%|430.49–447.06|77.34–78.13%|
| Other attacker + target |60.16–62.50%|476.28–569.89|60.94–63.28%|

Explicit access improves direction by10.94pp in both seeds, but damage MAE improves
by9.47% and14.99%. The registered gate required at least10% in BOTH seeds and
nondecreasing direction/destruction accuracy. **Gate failed; no RL expansion.** This does not make the direction gain
worthless: it supports a bounded mechanism follow-up, while preserving the
original decision rule.
Do not relax it after observing a near-threshold result.

Training direction accuracy reaches99.61% with actual-attacker features, versus
75.78–77.34% on test. On the separate identity holdout, actual-attacker damage
MAE is515.21/587.37LP versus context467.04/463.17LP: access alone does not establish
numeric generalization. This motivates checking an explicit public numeric path,
with effect exceptions still separate. The earlier October7 numeric-bottleneck
experiments already supply related evidence and must not be described as untried.

This is a frozen-feature consequence-prediction probe, not training of the newly
integrated policy residual and not an actor strength comparison. No new full games,
policy updates, deployment or checkpoint promotion occurred. The independent
watchdog confirmed healthy completion; its event has been acknowledged.

## Queue and gates

The working queue is retained in the artifact `PLAN.md`: finish implementation
validation; investigate the numeric-generalization gap with a bounded control;
then require actor learning and broad retention before independent game evaluation
and any expanded RL. Selection-cycle and chain-ownership tasks remain separate.
Critic calibration and action ordering must both pass; LP advantage or shorter
losses are not success criteria.

## Public numeric control and frozen fresh confirmation

`battle-context-numeric-control-2026-10-14` reused the initial data, same head,
initialization, minibatches, steps and two seeds. Its attacker/target control
reproduced the original final weights bit-for-bit. Raw public attacker/target
ATK/DEF and target position replace the frozen embeddings in the numeric arm;
the relation arm additionally exposes positive/negative/equal combat margins.
This is a representation intervention, not evidence that extra compute helped.
No source actor updates or new engine branches were used. On the reused test
boards, numeric+relations achieved94.53–95.31% direction and115.16–135.81LP MAE.
Those are exploratory results on inspected data, not an independent gate.

The next protocol froze all heads and normalization before generating48 new
boards (32 same-pool,16 probe-identity-holdout), each with four native branches.
`battle-context-fresh-confirmation-2026-10-14` identity:
`158a38251f1471b18ee388a79fe20abda6b0fc800653d43bf819b497ad2b2123`.
All192 branches were healthy and16 cold replays matched. No board composition
(actor, codes, positions) duplicates the original128 boards. No fitting occurred.

| Frozen input | Fresh same-pool direction | Damage MAE, LP | Joint destruction |
|---|---:|---:|---:|
| Context + target |63.28–64.84%|462.14–497.30|60.16–62.50%|
| Attacker + target |71.88–72.66%|388.48–437.75|67.97–70.31%|
| Public numeric |80.47%|146.71–161.64|78.91–81.25%|
| Numeric + relations |91.41–95.31%|95.89–115.95|89.84–94.53%|

On the16 fresh identity-holdout boards, numeric+relations achieved92.19%
direction in both seeds and106.68–145.53LP MAE, versus attacker/target75–76.56%
and598.40–600.97LP. Both seeds and both groups pass the predeclared requirement:
at least10% lower MAE and nondecreasing direction/destruction accuracy. Ranges
are fitting-seed ranges, not confidence intervals; boards have correlated branches.

This justifies the next small actor-learning/target-choice test, not RL scaling.
Check whether the actual policy chooses an unambiguously dominating target in
ordinary battles and whether nonbattle behavior remains intact. Partial-order
tradeoffs and effect-monster exceptions must not be mislabeled as arithmetic
errors. Then require broader behavior retention and paired full games before any
promotion. Selection loops, chain ownership and critic calibration remain open.

## Bounded policy residual pilot

`battle-context-policy-pilot-2026-10-14` trains only the zero-initialized residual
feeding the source actor's frozen action head. Both arms use the same residual
architecture, parameter count, initialization and minibatches: contextual pair
embeddings versus a zero-padded12-feature public numeric/relation input. The latter
is currently a research wrapper, **not a deployed NetConfig numeric mode**.

Only109 training prompts with a unique strictly Pareto-dominating target receive
labels: no worse own damage, opponent damage, own survival and target destruction,
and strictly better in at least one. Ties/tradeoffs are omitted. This is conditional
on committing the accepted attack; cancel is excluded from the conditional loss
and accuracy, and is not labeled bad. Both seeds run128 steps, checkpoints every32.
The source trunk and head remain bit-identical. New48-board evaluation is generated
after all four endpoints are frozen;192 native branches and16 cold replays pass.

| Fresh group | Eligible prompts | Source | Pair residual | Numeric/relation residual |
|---|---:|---:|---:|---:|
| Same identity pool |50 of64|60%|70–78%|96% both seeds|
| Probe-held-out identities |29 of32|44.83%|58.62–62.07%|100% both seeds|

Both seeds/groups pass the predeclared +10pp versus source and no worse than pair
criterion. This is evidence for learnable target preference beyond prediction,
with small, correlated samples and ordinary-battle scope. Zero initial logits,
frozen parameters and unchanged logits outside the valid residual mask are
structural checks, **not full strategic retention**. Conditional target accuracy
also does not measure changes in cancellation probability. Next run a fixed small
full-game/behavior retention panel before broader RL or integrating the numeric
input into the production configuration; inspect effect-dependent target choices.

## Full-game behavior canary: promotion withheld

`battle-context-behavior-canary-2026-10-14` completed24 healthy games: four ordered
deck pairings (Blue-Eyes/Branded, Branded/Clown Crew, Clown Crew/Tearlaments,
Tearlaments/Blue-Eyes), both first-player assignments, common deals and sampling
seeds, versus greedy. Each deal runs source and both fixed numeric-residual seeds.
No weights are changed. Complete actions/responses/steps and every affected
prompt's public observation and source/candidate probabilities are retained.

Source wins4/8, both candidates3/8. Mean turns11.75 versus10.375; this is not a
strength improvement: one source win becomes a loss and one retained win shortens.
The panel is too small to estimate population win-rate change. All24 games terminate
normally with zero retry/unknown/undecodable/script-error indicators. There are23
source and19 per-candidate eligible target prompts. Recorded cancellation mass is
zero in this panel, so it does not establish cancellation retention elsewhere.

For the changed outcome, Branded versus Clown Crew (learner first), the identical
prefix first diverges at decision457/turn13. Fallen of Albaz attacks: source samples
Clown Crew Diabolo in defense (200ATK/1800DEF); both candidates sample Dimension
Shifter in attack (1200ATK). The numeric residual most prefers Clown Crew Dristy
(1000ATK), assigning it about91–94% probability. This does not itself prove the
new first action was bad: immediate damage, effect value and later sampled play
must be separated. A bounded10-game root counterfactual completed: all five
legal targets, two common continuation RNGs, original policy after the forced
choice, exact original target0/carry replay required. No numeric integration,
RL expansion or checkpoint promotion proceeds from the ordinary-battle gains.


The counterfactual reproduces the original target0/carry complete action and
response transcript. With the source policy continuing after each forced target,
target1 (Dimension Shifter) loses in both fixed suffix RNGs; targets0/2/3/4 win
in both. This establishes an outcome difference for these interventions and
continuations, not a universal tactical label or population value estimate.

Crucially, source probability of target1 is41.72%, while the two candidates assign
9.01%/5.76%. Their preferred target4 wins in both continuations. The observed
common-uniform draw sent source to target0 but candidates to target1: one realized
paired loss is compatible with a lower probability of this failing action. Thus
the eight-deal canary cannot decide whether expected play improved or worsened.
`battle-context-sampling-retention-2026-10-14` now repeats the same eight openings
with two new fixed action-RNG offsets20000/30000, all three policies (48 games).
This is bounded sampling replication, not new-deal efficacy or expanded training.
Keep effect/cancel retention and independent-deal evaluation open.


Sampling replication completed48 healthy games. For offset20000, source and both
candidates each win3/8; for30000 each wins5/8. All16 paired outcomes match source
for each candidate; complete action transcripts match10/16 and11/16 respectively.
Mean turns are9.25→8.75 and11.25→10.75; no population strength/length claim follows.
The original regression did not repeat in these two RNGs, but no win-rate gain was
established either. These remain the same eight openings, not24 independent deals.

Next: `battle-context-fresh-retention-2026-10-14`, fixed24 games on eight new
openings, four previously untested panel pairings (Blue-Eyes/Sky Striker,
Branded/Dracotail, Clown Crew/HEROs, Tearlaments/Maliss), both first-player
assignments. Frozen source and both candidates, no fitting, independent watchdog,
full traces and target/cancel audits. This budget does not exceed the initial
canary and is not powered to promote a model; investigate any effect-sensitive
regression before broader training.
