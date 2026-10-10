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

## Counterfactual results: reachable wins and a rare sampled early branch

All 120 registered cases completed healthy with no limits and exact cold replay
matches. All frozen input and saved replay hashes verified. The 48 snapshots and
the factual branch match their original trajectories. The post-hoc analysis is
`runick-lock-counterfactual-2026-10-10/analysis.json`; it preserves all branches.

At the selected root, the policy assigns 96.6394% to normal summoning Sword Ryzeal
and 3.2477% to activating Mitsurugi Ritual; the observed long loss sampled the
latter. Thus this case does **not** establish that the policy prefers the poor
route. With the identical prefix and original RNG suffix, forcing the normal
summon wins on turn 30. Across four paired suffix streams it wins 4/4 (turns
30/6/14/10), versus 3/4 for the actual ritual activation (62-turn deck-out loss,
then wins on turns 10/10/12). This is a tiny, selected-state mechanism check, not a
reliable action-value estimate or justification to replace stochastic inference.

| Root action | Policy suffix wins /4 | Greedy suffix wins /4 |
|---|---:|---:|
| Normal summon Sword Ryzeal | 4 | 4 |
| Set Sword Ryzeal | 4 | 4 |
| Set Mitsurugi Ritual | 3 | 2 |
| Set Forbidden Droplet | 3 | 3 |
| Set Mitsurugi Great Purification | 4 | 4 |
| Activate Mitsurugi Ritual (observed) | 3 | 2 |
| Activate Forbidden Droplet | 3 | 2 |
| Enter battle phase | 3 | 1 |
| End phase | 3 | 4 |

Greedy takeover after the original ritual also loses on turn 62 under the original
stream; it is not a universal repair. The all-actions sweep gives tiny-probability
actions equal experimental replication, so its aggregate win rate is not the
policy's expected win rate. Finding a winning branch does not certify a lethal
that was already available in the later locked state.

Public spell snapshots across all 48 Runick-facing games show Synchro Zone in
28 games (26 losses), Messenger of Peace in 12 (10 losses), Wall of Revealing
Light in 8 (6 losses), and Skill Drain in 13 (10 losses). These are overlapping,
survivorship-confounded associations. They establish the presence of restrictions,
not that each card caused each loss. In game 966 every recorded candidate battle
menu from turn 14 onward lacks a legal attack. Automatic long-game diagnostics
must distinguish this situation from declining an offered attack.

Next: a bounded fixed-pool versus expanded-pool RL canary, preserving the actor,
critic, optimizer, RNG, reward, inference and cancellation settings. Evaluate on
fresh deals with familiar retention and own/facing roles separated. This tests
training coverage before adding mutation or enabling text features; the current
results justify neither automatic policy promotion nor a blanket action mask.
