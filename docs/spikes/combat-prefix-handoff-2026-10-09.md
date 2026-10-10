# Demonstration prefix results and fresh coverage

Short demonstrations repair two selected continuations, but partial assistance
can also turn a winning continuation into a loss. Preserve that counterexample;
these results do not justify automatically applying or training on teacher
prefixes. Fresh-state coverage is now running before a correction-training pilot.

Root: `out/research/combat-prefix-handoff-2026-10-09/`.
Identity: `d87ece65b1be677286272256588541ac6ef7566dcda4f9da290e897d3d255817`.
Independent `review.py` validates all168 records, coverage, health and taught-choice
counts. All24 one-choice controls exactly match the preceding study's final
result and same-turn candidate action/probability trace. No cutoffs or engine
health errors occurred. Completion was acknowledged; no prior evidence changed.

These are three selected fixed states, eight paired suffix streams, and seven
prefix budgets. A meaningful choice has more than one native-mask-supported
row; singleton choices advance the teacher but do not consume the budget.

| State | Budget1 | Budget2 | Budget4 | Budget8 | Budget16 | Budget32 | Budget64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seed0 warm512 / Greedy /0 | 1/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |
| seed0 warm512 / Greedy /48 | 5/8 | 5/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |
| seed1 cold256 / historical /107 | 0/8 | 0/8 | 1/8 | 3/8 | 8/8 | 8/8 | 8/8 |

Entries count same-turn wins. For the last state, budgets16/32/64 actually use
11 meaningful choices before winning; warm48's complete line uses4. The schedule
does not identify a minimum budget where intermediate lengths were untested:
for example, budget3 for warm48 and9/10 for cold107 were not evaluated.
Do not interpret repeated fixed states as168 independent deals or a population
effect estimate.

## What the extra choices change

- Warm0: after the initial attack, the second meaningful choice enables Amatsu's
  attack-trigger effect. The unassisted actor's `no` probability is99.27% in the
  trial0 state. Teaching this confirmation gives8/8 same-turn wins, compared
  with1/8 from the first action alone. Later decisions return to the original
  policy, so a complete scripted takeover is unnecessary in these suffixes.
- Warm48: after the first direct attack, the third meaningful choice continues
  attacking rather than entering main2; the fourth confirms direct attack.
  Four-choice assistance gives8/8, versus5/8 for one or two choices.
- Cold107: the successful sequence develops additional monsters, trades a2500
  attacker into the opposing2500 monster, then attacks directly with2000 ATK.
  An unfavorable initial attack and an equal-ATK trade can both be useful
  inside this sequence; isolated ATK comparisons are insufficient labels.

## Contrary case: a longer partial prefix loses

Cold107, suffix trial2, budget8, job129 loses on turn20 with LP[0,1050]. Budget1
wins for this same suffix. After the taught development ends, the original actor
chooses main2 at decision401 with probability83.71%, leaving the opposing
monster alive instead of taking the trade that opens the direct attack.
The all-supported prefix supplies the later choices and wins on turn7.

This is one observed counterexample, not an estimated rate of harm. It refutes
a blanket assumption that more teacher intervention is monotonically safer.
The complete branch/result is retained in `games/129.json` and indexed in
`reviewed-completion.json`. No training-label promotion or default change follows.

## Next: fresh natural-game coverage

Root: `out/research/combat-fresh-coverage-2026-10-09/`.
Identity: `d6aaa4de26e4476e00b8655af2c5fea1d4ce70d2299ea21a2920b6f45ffed259`.
Same three fixed checkpoints and two opponents;32 shared new deal blocks ×four
seat/deck assignments =768 new baseline games. Seed20261009911 was fixed before
launch, and all engine seeds are disjoint from the original768-game panel.
The deck pool is unchanged, so this is fresh-state coverage, not unseen-deck
or independent-training-seed generalization.

Each baseline requires exact cold replay and an independent unwrapped-policy
rerun. Probe all battle-exit/skipped-entry windows with all three fixed chooser
modes, up to four initial actions and128 further decisions; candidate masks are
enforced. Preserve every negative and positive, and verify original response/
step equality after snapshot branches. Probe health now explicitly includes
unknown/undecodable messages, and terminal turn identity is recorded so a win
after the root turn changes cannot count as same-turn lethal. The six earlier
positives were already checked against terminal turns in native follow-ups.

The frozen panel/runtime identity is separate from `resolved-panel.json`, which
binds newly generated baseline records. A complete real-game preflight passed;
two CPU workers, three-hour cap and an independent watchdog are running.
Fresh positives still require original-opponent/native-mask validation before
considering teaching data. Formal PPO, rewards and policy defaults stay unchanged.
