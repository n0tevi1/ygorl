# Fresh prefix results and complete root-action coverage

All336 branches and48 exact one-choice controls passed independent review,
with no health failures or cutoffs. Short support repairs two selected
continuations; the third needs a much longer teacher sequence, and partial
support can lengthen the game. This is intervention evidence, not a trained
policy improvement or permission to imitate every successful teacher action.

Root: `out/research/combat-fresh-prefix-2026-10-09/`.
Identity: `3e71bb01c4fc5019fae62f8296418cbf73261ba9f52eb4e8bef5e9497807a0af`.
The completion event was acknowledged after reviewing all results, source
bindings, taught-choice counts and exact budget1 action/probability traces.
Each entry below counts same-turn wins out of16 paired suffix streams.

| State | Budget1 | Budget2 | Budget4 | Budget8 | Budget16 | Budget32 | Budget64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seed0 warm512 / historical72 | 5 | 5 | 16 | 16 | 16 | 16 | 16 |
| seed1 cold256 / Greedy68 | 0 | 0 | 0 | 0 | 0 | 16 | 16 |
| seed1 cold256 / historical52 | 0 | 16 | 16 | 16 | 16 | 16 | 16 |

Budgets count meaningful native-mask-supported choices, including the first
forced action. Singleton decisions advance the teacher but do not consume the
budget. All original model draws are consumed even for overridden decisions.
States and shared suffix streams are correlated; do not pool these as336
independent games or infer population safety.

## Interpretation

- Warm72: budget2 additionally fixes a target choice the actor already makes
  with99.96% probability in trial0; it does not repair the subsequent main2
  exit. Budget4 supplies the remaining attacks and gives16/16 wins on turn5.
  Budgets1/2 each eventually win14/16, including the same two losses. Full
  teacher uses four meaningful choices. Budget3 was not tested.
- Historical52: supplying the attack-triggered Heavy Borger effect as the
  second choice yields16/16 immediate wins, versus11/16 eventual wins with
  only the attack. However, the separately validated direct effect at the
  original window already gives16/16 immediate wins with one intervention.
  The longer attack-first route is therefore not the preferred correction.
- Greedy68: all budgets eventually win16/16, but budgets1/2/4 average15 turns,
  budget8 averages16.75 and budget16 averages15.375. Budget32 wins on the
  original turn13 in16/16; budget64 completes the teacher in33 meaningful
  choices. Intermediate budgets17–31 were not tested. The sequence includes
  self-Ash as its fifth meaningful choice and is not an approved supervision
  sequence merely because it finishes with1200 effect damage.

This confirms heterogeneous continuation failures. Root action selection,
repeated battle exit, effect initiation and longer development cannot all be
repaired by a blanket attack preference or fixed demonstration length. The
long-game tail remains unresolved.

## Next: all supported root actions, original-policy continuation

Completed: see [all-action results and independent confirmation](combat-root-action-sweep-2026-10-09.md).

Root: `out/research/combat-root-action-sweep-2026-10-09/`.
Identity: `dfc32df86674d86cd4e405a2f834364bf055c0d0c53b4ab4e53a8189995f4e1e`.
The previous attack/battle-entry initial-choice restriction is removed for
this separate diagnostic. Reconstruct the exact original actor prefix and
enumerate every native-mask-supported action, including effect activation and
phase transitions. Force only that action; the original model and opponent
play the rest of the game. No heuristic teacher supplies the continuation.

State selection was fixed before measuring these interventions:

- Six fresh positive windows, with16 suffix streams each.
- The first four qualifying games by game ID in each of the six fixed
  candidate/opponent cells, lasting at least16 turns and containing a recorded
  combat window. Use the earliest recorded window in each, with8 suffix
  streams:24 long games. This selection is neither random nor exhaustive.

There are30 states,97 supported root actions and992 full continuations.
The action inventory contains33 attacks,20 main2 exits,30 end-phase exits,
10 battle entries, two activations and two spell/trap sets. This small number
of activations limits conclusions about general effect-choice failures.
The original sampled action is the per-state control; it is not duplicated
as another treatment. Trial0 must reproduce the baseline full-game result.
The other suffix streams use the same paired RNG schedule as earlier studies.

Report same-turn wins, eventual wins, turns, paired win-to-loss reversals and
all negative results. Do not turn an in-sample best action into a training label
without independent confirmation. A4096-decision bound and health checks remain;
cutoffs must not be silently dropped. The direct-effect and long-game original
controls passed real preflight. Two CPU workers, a three-hour cap, per-branch
saved results and an independent watchdog completed the study. All992 branches
and30 original controls passed review. Formal training,
rewards and policy defaults remain unchanged.
