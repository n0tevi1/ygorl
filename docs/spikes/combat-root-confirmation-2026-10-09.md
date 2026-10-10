# Independent root-action confirmation

The larger, previously unused suffix sample rejects several exploratory
improvements. Trap setting retains a positive conditional signal while making
games longer; the harmful attack reliably makes games shorter by losing more.
Do not promote the exploratory best actions as training labels.

Root: `out/research/combat-root-confirmation-2026-10-09/`.
Identity: `921fed2204470124908e620d3d1ab65de76f84ea3c62b443e6973cc94a1179b7`.
Independent review passed all902 records, six original calibration response/
step transcripts and20 replay hashes. No engine health errors or cutoffs.
Completion was acknowledged after these checks.

Each contrast below uses64 new paired suffix streams at the same fixed state.
These test suffix variability, not new deals or hidden states. Intervals are
conditional, unadjusted across contrasts; fitted models and states must not be
pooled as independent training replications.

| State / intervention | Control → treatment wins /64 | Mean turns | Paired wins lost |
| --- | ---: | ---: | ---: |
| warm72 / prior teacher attack | 43→45 | 7.094→7.109 | 15 |
| warm72 / Elf attack | 43→52 | 7.094→6.594 | 9 |
| cold Greedy5 / end phase | 34→34 | 18.375→18.594 | 15 |
| warm Greedy50 / trap set | 17→32 | 15.359→16.828 | 13 |
| cold1 Greedy38 / attack | 36→3 | 14.063→10.547 | 34 |
| cold1 historical47 / activate | 35→38 | 11.484→11.375 | 14 |
| cold1 historical47 / attack | 35→34 | 11.484→11.719 | 14 |
| cold1 historical109 / attack | 37→29 | 11.875→12.797 | 20 |

Warm72 is seed0/u512; cold Greedy5 is seed0/u512; other cold1 states are
seed1/u256. The original sampled root action is fixed in each control; it is
not a resampled root-policy mixture.

## Decisions from this evidence

- End-phase, historical47 activation/attack and historical109 attack benefits
  do not reproduce as resolved win-rate gains. Their paired intervals cross
  zero. Reject their promotion from the earlier8-stream maxima.
- Elf attack improves same-turn finishes to25/64, but its eventual win delta
  +14.06pp has interval−1.56 to+29.69pp; turn delta−0.500 has interval−1.078
  to+0.047. Nine paired wins become losses. Its earlier16/16 result was not
  a safety guarantee. The prior teacher attack has only10/64 same-turn wins.
- Trap setting gives+23.44pp wins (conditional interval+4.69 to+42.19) and
  +1.469 mean turns (interval+0.328 to+2.594). It also has13 paired harms.
  This is a nominal selected-state signal, not multiplicity-adjusted evidence
  of a broadly safe correction. Only four pairs win in both arms, so their
  mean−2-turn difference is especially fragile.
- The harmful2500-into2800 attack gives−51.56pp wins (interval−64.06 to−37.50)
  while shortening games by3.516 turns (interval−4.844 to−2.172). Only two
  pairs win in both arms, and those take two turns longer after intervention.
  A raw duration bonus would reward the wrong outcome in this example.

No selected-state label or default is promoted. The root-sweep findings also
showed some bad observed choices were already low-probability samples, so the
next diagnostic measures sampling choices across complete natural games.

## Running: full-game sampling comparison

Root: `out/research/policy-sampling-fullgame-2026-10-09/`.
Identity: `4faad320b1f5482950b40a24c1653fc17a92ceec8c1515eab26e71ee7b805f25`.
The same768 previously fixed games ×candidate T1/T0.5/argmax =2,304 runs.
Three frozen checkpoints, two unchanged opponents,32 deal blocks with four
seat/deck assignments per cell. Only candidate sampling changes, using the
existing checkpoint-agent registry. This is an exposed development panel,
not unseen-deal confirmation. Earlier argmax evidence in the October2 handoff
was a response-window study, not this full-game comparison.

All768 T1 controls must reproduce source results and complete response/step
transcripts. Save every replay/hash and report strict terminal wins, reason
counts, decision/turn distributions, cancellations and maximum consecutive
selection cancels. Compare paired outcomes and turns among pairs won in both
arms; retain long games and adverse effects. Conditional intervals resample
32 paired deal blocks within each cell.

Turn/decision-limit games remain in all denominators and count as non-wins,
even if the engine assigns an LP-based turn-limit winner. Preserve each limit
incident and alert on the first. The bounded evaluation can continue through
limits; completion is explicitly non-healthy if any occur. Engine, script,
decode, mask, identity or T1-reproduction failures STOP the study. These are
evaluation games, never training rows; no failure is silently dropped.

Three real preflight modes passed. The first example wins at T1 and T0.5 but
loses under argmax, already showing why one cannot assume deterministic play
helps. Two CPU workers, a three-hour cap, per-game results and an independent
watchdog are active. Formal PPO, reward and inference defaults are unchanged.
