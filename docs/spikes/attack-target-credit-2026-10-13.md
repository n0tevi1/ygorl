# Attack-target continuation and update-path audit

The missing-attacker input repair is independently valid, but its four-update
canary did not establish stronger play. This follow-up tests target alternatives
before treating policy probability or training credit as a mistake label. It
changes no production code, model weights, loss, mask or training scale.

## Selection and verification

Use the frozen `codex-rollout-order` runtime at `af4b5d9`, the context-final u4
checkpoint from `attack-history-ordered-canary-2026-10-12`, and the previously
used `attack-history-ordered-heldout-2026-10-12` panel against Greedy. Select the
first multi-target attack selection in each of the first four qualifying games
by numeric game ID, without filtering on outcomes: `(1,342)`, `(5,396)`,
`(7,136)`, `(8,241)` (game, decision index).

Enumerate every supported target and give each four fixed-policy continuation
seeds. Cancel is present in the raw action list but masked in all four roots;
it is not forcibly enabled. The first study completed four exact original-policy
baselines and 40 branches. A same-size replication retained all roots/actions
and used four different suffix seeds: another four baselines and 40 branches.
All 88 games completed normally, with exact cold replay equality. Each branch
matches the original prefix, root observations/probabilities and critic scores.
Critic privileged inputs are used only for diagnostic scoring, never for choosing
an actor action. The only intervention is the registered root target choice.

These are conditional outcomes under one hidden engine state and a frozen policy
against Greedy. They are not information-set optimal-action labels, and eight
continuations per action are not a strength gate. Common suffix seeds also do not
align later stochastic events after paths diverge. The panel is reused, not a new
unseen-card or untouched-deal holdout.

## Results

Probabilities and Q values are from the final context checkpoint at the fixed root.
Q uses the terminal-return scale, not a probability scale.

| Game / decision | Target password | Policy probability | Q | Initial wins / 4 | Replication wins / 4 |
|---|---:|---:|---:|---:|---:|
| 1 / 342 | 97522863 | 86.73% | +0.108 | 4 | 4 |
| 1 / 342 | 1528054 | 13.27% | +0.079 | 4 | 4 |
| 5 / 396 | 21441617 | 0.35% | -0.165 | 1 | 1 |
| 5 / 396 | 48835607 | 24.97% | -0.144 | 3 | 2 |
| 5 / 396 | 30741503 | 74.69% | -0.152 | 0 | 0 |
| 7 / 136 | 80538047 | 46.05% | -0.053 | 2 | 3 |
| 7 / 136 | 29301450 | 53.95% | -0.104 | 4 | 4 |
| 8 / 241 | 95561146 | 60.09% | +0.407 | 0 | 0 |
| 8 / 241 | 85888377 | 21.76% | +0.398 | 0 | 0 |
| 8 / 241 | 62514770 | 18.15% | +0.416 | 0 | 2 |

Game 5 supplies a replicated investigation case: the policy strongly prefers
Galatea (30741503), whose continuations win 0/8, over Galatea-I (48835607), whose
continuations win 5/8. The critic ranks Galatea-I slightly higher, so a wrong critic
ranking at this state is not the immediate explanation of the actor preference.
The public battle trace confirms different removals: a 1,300 ATK attacker removes
either the 900 ATK Galatea or the 0 ATK Galatea-I; attacking the 1,500 DEF Cymbal
Skeleton instead inflicts 200 damage on the attacking player and removes neither.
These traces locate the intervention; they do not isolate which subsequent card
effect causes the terminal outcome difference.

Game 7 gives the opposite caution: attacking S:P Little Knight (29301450) wins
8/8 versus 5/8 for Radiant Typhoon Swen (80538047), while the critic ranks S:P lower.
Both have 1,600 ATK, and either attack causes the same immediate 100 LP damage.
Target identity and the surviving board matter beyond immediate damage. Do not
replace the actor with an unconditional Q argmax. Game 8 also suggests a conditional
calibration problem: Q is around +0.4 despite mostly losing continuations. More
states and continuation coverage are required to generalize that diagnosis.

The losing Galatea branches in game 5 end on turns 8–9, while Galatea-I branches
end on turns 10–20. Shorter games would be a misleading improvement signal here.
Report outcome and turn length jointly; do not reward ending quickly regardless
of who wins.

## Did the four updates create the preference?

`attack-target-checkpoints-2026-10-13` cold-replays the same prefixes with each
checkpoint's own native observation schema. It scores actor and critic without
sampling new games or modifying parameters. These are fixed context-final states,
not the earlier checkpoint's on-policy state distribution.

For game 5, legacy input gives Galatea 79.74% before training and 82.88% after;
attack-history input gives it 59.78% before and 74.69% after. Thus the preference
predates this canary, and neither short retraining arm removes it. All four
critic versions rank Galatea-I slightly above Galatea. Conversely, on game 7,
context training increases the probability of attacking S:P from 47.11% to 53.95%.
This is uneven generalization across states, not uniform target degradation.

`attack-target-update-path-2026-10-13` traces the context checkpoint sequence:

| Update | Probability of Galatea in game 5 |
|---:|---:|
| 0 | 59.78% |
| 1 | 56.90% |
| 2 | 58.40% |
| 3 | 67.20% |
| 4 | 74.69% |

For mechanism diagnosis, select update 3, the largest observed worsening of the
Galatea-I/Galatea log odds. Its actual margin change is -0.380; the pre-update
margin gradient dotted with the actual recorded parameter change predicts -0.397.
That local approximation tracks the observed drift, but is not an additive causal
attribution of the finite Adam update.

The implemented pre-update loss is decomposed on the original saved 1,024 training
rows. There are 13 attack-target rows in that update. The derivative of the game-5
margin under unit unpreconditioned gradient descent is:

| Loss component | Margin derivative (negative favors Galatea) |
|---|---:|
| Target-row policy loss | -1.074 |
| Other-row policy loss | -6.093 |
| Reference KL | -1.574 |
| Entropy | -0.028 |
| Q loss | -0.018 |
| V loss | -0.018 |
| Total | -8.804 |

The component gradients sum to the complete loss gradient within numerical
tolerance. Most opposing local influence comes from policy rows and reference KL;
direct critic-loss influence is small in this diagnostic. This does **not** rule
out indirect critic effects through policy advantages, establish that the sampled
training actions received wrong credit, or prove that removing KL/critic loss
improves Adam updates or full-game play. The evaluation state is not itself a
recorded training row. No optimizer was run in either attribution audit.

The follow-up `attack-target-kind-attribution-2026-10-13` further partitions the
same policy gradient by the recorded chosen action kind. The largest opposing
local terms are `end_phase` (48 rows, -4.072), `spsummon` (19, -1.872), and
`summon` (37, -1.422). Conversely, `battle_phase` (60, +2.933) helps the evaluated
target margin. Together with the 13 attack-target rows and all other kinds, this
partition covers all 1,024 rows and reconstructs the same complete gradient.

This makes the transfer mechanism more specific: updates on other decision types
can move an unseen target preference through shared policy parameters. It does not
show that ending turns or summoning in those training states was wrong. Do not
remove those rows or penalize those action kinds on the basis of their effect on
one selected evaluation state.

## Decision and next gate

Keep these roots as diagnostic regressions, separate from any new fitting data.
Do not apply a card-specific action ban, fit the evaluated targets, promote u4,
or expand RL based on this result. Input correctness, data coverage, actor
transfer and critic ranking remain separate questions.

The next gate is to verify credit quality on independent training states for the
identified decision kinds, and measure whether this transfer recurs on separate
target roots before choosing a retraining change. The current data locates transfer
through shared policy parameters; it does not yet identify a new implementation
defect to patch. Retain the unresolved
single-attacker Snow/Havnis issue and require genuinely new evaluation deals for
any subsequent model comparison.

## Evidence

All directories below are under `out/research/` with suffix `-2026-10-13`:

- `attack-target-branches`: selection registration, runtime binding, four baselines,
  40 branches, public root/post-battle boards and cold-verified replays.
- `attack-target-branches-replication`: same selection and budget, independent seeds.
- `attack-target-checkpoints`: all four paired checkpoints scored at the same roots.
- `attack-target-update-path`: intermediate checkpoints and saved-update gradient
  attribution; no new rollouts or optimizer steps.
- `attack-target-kind-attribution`: the same recorded update partitioned into every
  chosen action kind; covers all 1,024 rows without fitting.

Each has an identity manifest and healthy report. The experiment runtime remains
frozen; the report worktree starts from main `1f2bd6f`. Python/native source and
tests are unchanged from the validation recorded for PR #417. Reuse that validation
with its resolved header and solver-path setup; the new experimental checks above
supply the additional evidence for this documentation-only update.
