# Full-game card coverage: concentrated weakness, not uniform unseen-card collapse

All 1,024 registered games completed with healthy engine results, no game limits,
and exact cold replay matches; stored replay hashes were verified by the frozen
analysis. Three warm-u512 policies contribute 768 games; Greedy controls contribute
256. Each policy role/exposure cell below pools 192 games across the two fixed
opponents and three training seeds; each control cell has 64 games.

| Target exposure | Candidate role | PPO win rate | Greedy control | Mean turns, PPO |
|---|---|---:|---:|---:|
| Familiar | Pilot target deck | 94.27% | 73.44% | 8.04 |
| Familiar | Face target deck | 70.31% | 39.06% | 10.11 |
| RL-pool-held-out | Pilot target deck | 89.06% | 67.19% | 13.08 |
| RL-pool-held-out | Face target deck | 66.67% | 45.31% | 14.48 |

These fixed, weak-opponent panel results do not establish elite play. Differences
between deck groups are not causal effects of exposure: lists differ in intrinsic
strength and the controls differ in competence. Full historical RL exposure is
not certified. Paired results, uncertainty and per-deck/opponent/seed breakdowns
are retained in `card-coverage-fullgame-2026-10-10/analysis.json`. Do not infer that
unseen-card play is uniformly poor, or that the text switch is required.

## Runick accounts for most of the long-game increase

Each target/role has 48 PPO games. Labrynth/Kashtira/Tenpai mean turns are
7.90/8.85/11.04 when piloted, and 9.65/9.56/11.21 when faced. Runick is
24.54 turns when piloted (43/48 wins), and 27.50 when faced (19/48 wins).
Of the 29 losses facing Runick, 28 are deck-outs; the maximum is 62 turns.
The chosen Runick list contains battle-blocking and summon/effect restrictions,
so a longer game is not automatically an action loop or missed lethal.

In game 966 (seed0, Ryzeal Mitsurugi against historical-policy Runick), the policy
loses by deck-out on turn 62 while the same-deal Greedy control (990) wins on turn 6.
The opponent activates Wall of Revealing Light on turn 14. The following battle
menus offer no attacks despite a 1,200-ATK Dimension Shifter on the field and the
opponent at 400 LP. Native legality, rather than voluntary attack refusal, prevents
attacking at those menus. Earlier decisions can still be responsible for reaching
this lock. Existing trace evidence alone does not prove a legal escape remains
once the lock has formed.

Across all 768 policy games, self-Ash is chosen in 12/306 offered decision windows
(3.92%). These are correlated opportunities, not independent samples; repeated
seed/opponent variants include the same starting situations. Four policy games
contain repeated selection cancels; only two cross the eight-identical-choice
warning threshold. There are 310 battle exits with an attack still offered, which
are investigation signals, not 310 proven missed wins. Self-damage warnings also
need tactical review. No reward or action mask is changed from these counts.

## Registered next investigation

`out/research/runick-lock-counterfactual-2026-10-10/` has two parts:

* Reconstruct all 48 policy Runick-facing games, adding only public face-up spell/
  trap snapshots at candidate main/battle menus, legal options and existing public
  monster information. Match original choices/results and independently replay.
* At game 966 decision 45 (first candidate main-phase command on turn 2), enumerate
  all nine policy-supported actions. Compare original-policy and Greedy suffixes
  under four paired continuation RNG streams: 72 branches. Prefix policy draws,
  original root draw and learner histories are consumed identically before changing
  that action. Trial zero preserves the original RNG; its factual policy branch
  must reproduce the 62-turn result. Other streams are paired across actions.

This is a selected-failure mechanism study, not an unbiased improvement estimate,
a training set, or a proposal to deploy Greedy handoffs. Preserve all branches,
not just wins. No new model training, Colab allocation or policy promotion. Two
CPU workers, independent monitor and two-hour bound; checkpoint, trace, script,
source and environment hashes frozen. Engine or replay failures stop the study.
