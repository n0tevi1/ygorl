# Combat-only continuation diagnostics — 2026-10-12

Related: #62, #83, #105; [lethal-line guard validation](lethal-line-guards-2026-10-12.md).

## Question and boundary

Can continuing attacks repair the observed repeated battle exits without borrowing
the teacher's effect activation, chains, or main-phase development? Earlier full
teacher prefixes mixed these mechanisms and sometimes included self-negation.

`tools/combat_probe.py::combat_only_action` is an opt-in research helper. After the
driver primes a fresh chooser, it proposes supported attack declarations and
attack-target responses on that own turn. Other prompts return `None` without
advancing the chooser. It respects the policy mask and remembers cancelled
attackers. It does not establish lethal, model effects, register a new agent,
change production inference, or change training.

## Matched continuation result

Artifacts: `out/research/combat-only-continuation-2026-10-12/`.
Frozen source: `5e67dd0d85c3b3f4d32b9b427c0ce3bb0756f042`, based on main
`1f40e80bad41e8dc27fb50769e12a88092cb157a`.
Identity SHA-256: `02ea08283fa350914f8e0905a3e56b877519ef7037db4c84dd06b0cb458722a3`.
Report SHA-256: `643a4a8b411ac8c5d74860f8777e39ea7b21711e0b801c0cb521569a25579fc8`.

Six previously exposed windows in five games; eight paired suffix RNG streams per
window. Both arms receive the same previously selected first attack. The treatment
then supplies only attacks/targets, with a 64 meaningful-choice ceiling including
that first action. The original policy draw is consumed even when replaced.
Public current stats are enabled; `lethal_first` is disabled. These are conditional
mechanism tests, not 48 independent test roots or a population win-rate estimate.

| Window (decision) | First-only eventual wins /8 | Combat-only eventual wins /8 | First-only same-turn wins /8 | Combat-only same-turn wins /8 | Mean final turn, first → combat |
| --- | ---: | ---: | ---: | ---: | ---: |
| seed-0 warm512 historical72 (466) | 6 | 8 | 2 | 8 | 6.875 → 5 |
| seed-1 cold256 greedy65 (402) | 8 | 8 | 8 | 8 | 8 → 8 |
| seed-1 cold256 greedy65 (499) | 8 | 8 | 8 | 8 | 10 → 10 |
| seed-1 cold256 greedy68 (534) | 8 | 8 | 0 | 0 | 15 → 15 |
| seed-1 cold256 historical52 (490) | 4 | 4 | 0 | 0 | 11.5 → 11.5 |
| seed-1 cold256 historical76 (325) | 8 | 8 | 8 | 8 | 7 → 7 |

All 102 records are healthy with no cutoffs. The 48 first-only and six original
controls reproduce archived results and same-turn action/probability traces
exactly. All 745 native/Python mask comparisons pass. The treatment supplies 40
attack and 56 target-selection actions; no supplied effects, chains, or cancels
occur in these records. This does not mean the underlying policy never self-negates.

The warm72 result localizes a failure to continued battle execution. The lack of
improvement in greedy68 and historical52 is consistent with their previously
identified development/effect needs. General battle completion cannot replace
those decisions. No checkpoint was fitted or promoted.

## Prospective public-trigger pilot

Artifacts: `out/research/combat-autonomous-trigger-2026-10-12/`.
Registration fixes eight new deal blocks, four seat/deck assignments per block,
and baseline/guided arms: 64 games, plus two unwrapped baseline verification games.
Only seed-0 warm512 versus the historical warm128 policy is included. Panel seed
is `2026101206`; decks are shuffled by slot before engine shuffling is disabled.
The ordered first five cards are checked against the prior fresh panel.

The first autonomous candidate attack on each own turn arms a fresh chooser and
is never replaced. Only later attack/target choices can be supplied, capped at 64
meaningful choices per turn. Turn changes reset the chooser. No selected root,
forced initial attack, future rollout, effect override, or chain override is used.
Two CPU workers, a 15-minute service limit, per-game saved traces, immutable input
hashes, heartbeat, and a separate watcher bound to the service invocation are used.

The predeclared primary checks are trigger/change coverage, paired eventual gains
and harms, cutoffs, and engine health. Turn length is secondary and is split by
winner. Eight deal blocks are the independence unit; the four assignments within
each block are correlated. This pilot alone cannot justify production adoption,
larger training, or a general strength claim.

### Result and rejection

Identity SHA-256: `2b900e323b8fabd61231bc5bf509ceb9b5da4fb674c96174774697d9cd681672`.
Report SHA-256: `0ae70ef3662a7925382edc53e83677cc069f5e85bb6889d23ba47c9197caf8e6`.

| Metric | Baseline | Combat guidance |
| --- | ---: | ---: |
| Eventual wins /32 | 25 | 23 |
| Mean final turn | 8.0625 | 8.34375 |
| Median final turn | 7.5 | 7.5 |
| Games with autonomous trigger | 31 | 31 |
| Games with changed actions | 0 | 13 |
| Changed actions | 0 | 26 |
| Cutoffs / engine errors | 0 / 0 | 0 / 0 |

Paired outcomes: **one gain, three harms**. All 7,800 native/Python mask
comparisons pass; two baseline games match unwrapped `Duel.run` results. All 64
traces are retained. Post-hoc trace inspection finds no stale attacker-context
mismatch in the supplied target decisions. Reject this broad rule for deployment;
trigger coverage does not establish good action quality.

Each of the three harms first diverges at a target choice, where the helper
chooses the highest-stat beatable target:

- Game 5, decision 379: 2400 ATK attacker changes from Maiden of White (0 ATK)
  to Blue-Eyes (2200 current ATK). The actual baseline response summons a different
  target and inflicts 2400 damage; the guided path inflicts 200. This is a recorded
  continuation, not proof that attacking Maiden is universally correct.
- Game 30, decision 282: Legatia changes from Fallen of the White Dragon
  (1800 ATK) to Granguignol (2500 ATK); later target choices and effect paths also
  diverge. A single coupled rollout does not identify a general expected-value
  advantage of either target.
- Game 31, decision 177: Dis Pater changes from Golden Swordsoul (1800 ATK) to
  Titaniklad (2500 ATK). In the guided trace, Titaniklad's graveyard effect is
  activated at decision 212, selects card 60303688 at 215, and the opponent wins
  on turn 6. Baseline wins on turn 9. Raw turn reduction would reward this loss.

This exposes the limitation of ranking targets by current battle stats alone:
subsequent effects and policy choices matter. It does not justify a per-card
blacklist or assert that each observed win-to-loss flip persists across RNG draws.

## Fixed-game attribution

Artifacts: `out/research/combat-trigger-attribution-2026-10-12/`.
Identity SHA-256: `fe3c594bccfbc8d4664501bdc087715aad2b0192bf13fe902b2680190ef1cdc5`.
Report SHA-256: `19742c97b88b85fa088d13f907b10260668fb345d48e3f56919b4de291b84b4c`.

Sixteen games replay the four observed outcome flips under baseline, full guidance,
declarations-only, and targets-only arms. Eight baseline/full-guidance controls
reproduce their archived full transcripts exactly. All 16 games are healthy, with
no cutoffs and 2,067 passing mask comparisons. Every actual autonomous attack
updates the chooser's public attacker context in the decomposition driver.

| Exposed game | Baseline | Full guidance | Declarations only | Targets only |
| --- | --- | --- | --- | --- |
| 5 | Win, turn 8 | Loss, turn 9 | Win, turn 8 | Loss, turn 9 |
| 28 | Loss, turn 10 | Win, turn 11 | Loss, turn 10 | Loss, turn 12 |
| 30 | Win, turn 9 | Loss, turn 13 | Win, turn 9 | Loss, turn 13 |
| 31 | Win, turn 9 | Loss, turn 6 | Loss, turn 6 | Loss, turn 6 |

Targets-only reproduces all three harms. Declarations-only is also unsafe in this
sample: at game 31 decision 192 it replaces `main2` with a 2300 ATK attack into a
2500 ATK Titaniklad; the policy completes that target selection, loses 200 LP, and
loses on the next turn. The full guidance's one observed gain requires the combined
intervention in this fixed RNG trace; neither component alone reproduces it.

These are outcome-selected cases, not another validation set. They distinguish
mechanisms but cannot rank expected action values or justify another tuned rule.

## Decision and remaining work

- Do not deploy either broad combat guidance or declarations-only guidance; do not
  distill these teacher actions into training. Keep the diagnostic opt-in.
- Retain the existing guarded lethal-search path from #345. Its engineering
  validation does not itself establish public-information safety or playing strength.
- Any next lethal helper must validate the complete attack/target continuation and
  account for visible effects and possible opponent responses. Raw ATK comparison,
  a successful privileged rollout, and shorter games are insufficient acceptance
  criteria. Do not replace this with card-specific blacklists.
- Before another policy change, use these counterexamples as development regressions
  and reserve new deal blocks for a separately registered evaluation. Report paired
  gains/harms, win-conditioned lengths, cutoffs, and both seat/deck assignments.
- Effect initiation and main-phase development remain separate unresolved failures;
  the combat-only positive window does not solve them. No critic checkpoint or
  larger RL run was justified or launched by these experiments.

## Validation

On the frozen implementation: **1,923 passed, 23 skipped, 0 failed** (333.94 s).
Source formatting and lint pass for all 399 files. New tests cover passive
unrelated prompts, other/next turns, mask support, and cancelled-attacker state.
The isolated validation interpreter explicitly imports this worktree and uses
the recorded native library; the current solver binary is selected explicitly.
The log and its hash are saved with the matched-continuation study.
