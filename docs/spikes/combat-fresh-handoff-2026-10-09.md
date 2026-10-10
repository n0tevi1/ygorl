# Fresh policy handoff: isolated first actions can help or harm

All192 branches passed independent review, including six exact original
trial0 full-game controls, paired inputs, health and no-cutoff checks. Three
windows finish immediately in16/16 suffixes after the first taught action;
three need further decisions. Two of those contain paired win-to-loss reversals.
A successful complete demonstration is insufficient justification for training
its isolated first action.

Root: `out/research/combat-fresh-handoff-2026-10-09/`.
Identity: `2b124d1a589955340d5bb3547018b186c722ee4a0732b96d436d49edea7b82e0`.
Completion was acknowledged after independent `review.py` validation.
Each row is one fixed window,16 paired suffix streams, original versus
force-first then original-policy continuation. Shared seeds and two windows
from game65 are correlated; do not pool as independent games or estimate a
population intervention effect from these selected states.

| State / decision | Same-turn wins original → forced | Eventual wins | Mean turns | Paired win → loss |
| --- | ---: | ---: | ---: | ---: |
| warm72 historical /466 | 0→5 | 12→14 | 7.250→6.563 | 2 |
| cold65 Greedy /402 | 0→16 | 16→16 | 11.313→8.000 | 0 |
| cold65 Greedy /499 | 0→16 | 16→16 | 12.313→10.000 | 0 |
| cold68 Greedy /534 | 0→0 | 16→16 | 15.000→15.000 | 0 |
| cold52 historical /490 | 2→0 | 16→11 | 9.875→11.375 | 5 |
| cold76 historical /325 | 0→16 | 15→16 | 8.813→7.000 | 0 |

Warm is seed0/u512; cold is seed1/u256. Warm72 also has four loss-to-win
reversals: its average gain does not erase the two observed harms.

## Trace findings and limits of the teacher

- Warm72: after the first taught attack, trial0 selects main2 at decision482
  with probability75.73%. The winning teacher continues with3600 into3500,
  then2600 direct. This reproduces an additional battle-exit failure.
- Cold52: trial0 passes Heavy Borger's effect at decision492 with
  probability97.47%, after attacking a3000 monster with2500. The complete
  teacher instead reveals EARTH/FIRE and deals1500 to an opponent with1000LP.
  **Direct activation is already an option at decision490.** The attack is
  not established as necessary; test direct activation instead of teaching
  this attack-first route.
- Cold68: after the attack, trial0 ends the turn at decision552 with
  probability97.12%. The teacher continues through Fiendsmith's Tract and a
  longer main2 sequence ending in1200 effect damage. This is not merely a
  missed direct attack. Its successful line includes Ash negating its own
  Tract search at decision555; do not promote every teacher action as useful
  or necessary simply because the line eventually wins.

These are observed decision disagreements, not a complete account of learned
representations or the long-game root cause. A model can choose a different
winning route: cold65/499 succeeds despite differing from the teacher later.
Correction must be evaluated by resulting outcomes, not exact imitation alone.

## Next registered controls

`combat-fresh-prefix-2026-10-09/`, identity
`3e71bb01c4fc5019fae62f8296418cbf73261ba9f52eb4e8bef5e9497807a0af`:
the three unresolved windows ×16 suffix streams ×budgets1/2/4/8/16/32/64
=336 branches. Count meaningful native-supported choices; preserve actor draw
consumption and original opponent. Budget1 must exactly reproduce the preceding
result and same-turn action/probability trace. Retain every harmful partial
prefix. Four real preflight branches passed after correcting a preflight-only
relative reference path; the original failed preflight and identity are kept.
The production run and independent watchdog are active.

`combat-direct-effect-2026-10-09/`, identity
`e3e6f2aa71b4744935ff719157667f87c46193015a13ca10306dfdd7b171b50f`:
original choice versus direct Heavy Borger activation,16 paired suffixes,
32 branches. All16 original controls must match the preceding experiment.
This distinguishes the attack-first route from its useful effect decision.
Completed and independently reviewed: all32 branches are healthy and all16
original-arm full results and same-turn action/probability traces exactly match
the preceding study. Completion was acknowledged after review.

| First choice at cold52/490 | Same-turn wins /16 | Eventual wins /16 | Mean turns |
| --- | ---: | ---: | ---: |
| Original choice | 2 | 16 | 9.875 |
| Attack then original policy | 0 | 11 | 11.375 |
| Direct effect then original policy | 16 | 16 | 9.000 |

Direct activation has no paired win-to-loss reversal in these16 suffixes. The
same learned continuation policy can complete the effect choices when started
with the direct activation. In this fixed state, prioritizing the effect is a
better supported intervention than the teacher's attack-first route. It is not
a guarantee over unseen states or hidden cards.
At this exact root the model already assigns42.89% to direct activation and
47.91% to its sampled main2 choice. Thus this is a prioritization failure in
the sampled trajectory, not proof that the model never recognizes the effect.
The burn-option choice after activation has probability1.0; successful forced
continuations do not by themselves establish broad effect-selection skill.

The earlier combat probe only tried attacks/battle entry as its initial
interventions. It could discover the burn later in a winning sequence but
could not compare direct activation at the root. Treat that search coverage
restriction as a diagnostic limitation, not evidence that attacking is the
appropriate label. Future correction-data generation must compare available
effect initiations and measure outcomes under the learned continuation policy;
whole-line teacher success alone remains insufficient. The336-branch prefix
study continues independently, including the less efficient attack-first line.

Neither experiment trains or promotes a policy. Their purpose is to identify
which interventions merit a later correction pilot while preserving adverse
examples and avoiding indiscriminate teacher imitation.
