# Warm/u512 milestone and repeat investigation

All five repetition incidents plus the full warm/u512 milestone were acknowledged and independently inspected. Original study, native runtime, identity, and STOP evidence remain frozen. Seed-1 continuation remains under existing supervision.

## Paired formal checkpoint results

Primary three opponents, 768 identical evaluation games per checkpoint:

| Checkpoint | Wins | Win rate | Mean turns |
| --- | ---: | ---: | ---: |
| Warm u128 | 595 | 77.474% | 10.6602 |
| Warm u256 | 630 | 82.031% | 10.5130 |
| Warm u512 | 661 | 86.068% | 9.2891 |
| Cold u512 | 653 | 85.026% | 9.2565 |

Paired shared-deal bootstrap, one training seed: warm u128→u512 win +8.594 pp [4.948,12.240], turns -1.371 [-1.779,-0.964]. Warm u256→u512 win +4.036 pp [1.172,6.901], turns -1.224 [-1.602,-0.844]. Warm-vs-cold u512 win +1.042 pp [-1.693,3.776], turns +0.0326 [-0.2969,0.3802]: no established warm advantage at matched update. The fourth historical opponent is reported separately in paired-progress.json; across all four warm u512 wins 854/1024.

Independent retention recomputation verifies identical sealed data: 81546 rows/256 games. Warm Q MSE u128→u512 0.85607→0.74729 (delta -0.10877 [-0.14948,-0.06809]); V MSE 0.85663→0.74730. This shows retention/improvement on the fixed critic task, not every gameplay skill or on-policy calibration.

## Behavior regression despite stronger play

Across all four opponents (1024 games each), total chosen material cancellations: warm u128 51, warm u512 560, cold u512 72. Games with at least eight total cancellations: 0, 19, 0. These are per-game totals, not necessarily contiguous loops. Opportunities depend on the policy. Do not promote warm on aggregate win rate alone or label every cancellation wrong.

Five selected alerts were then audited individually:

| Opponent / game | Consecutive cancels | P(cancel), percent | Turns | Decisions before / after removing loops |
| --- | ---: | ---: | ---: | ---: |
| greedy / 60 | 32 | 98.7633 | 6 | 592 / 528 |
| old-256x2 / 49 | 32 | 98.3924 | 4 | 275 / 211 |
| old-256x2 / 180 | 18 | 95.8937 | 32 | 1397 / 1361 |
| initial-128x2 / 168 | 32 | 99.8563 | 5 | 250 / 186 |
| historical-rl / 123 | 28 | 92.5965 | 18 | 1179 / 1123 |

All pre-exit legacy inputs in each loop are identical. Opt-in history produces a distinct input at every retry; native/reference parity and exact engine responses pass for all five full games. The first, second and fourth cases reach the 32-cancel guard; the others exit through sampling. Removing complete cycles while preserving the eventual target and fixed suffix leaves all aligned options/actions/events, terminal LP, health and turn counts unchanged. For initial-128x2/168, the zero-loop variant explicitly selects the eventual target at the initial target window; blindly retaining the initially selected different fusion target would be a different intervention. No natural-policy or population counterfactual claim.

The 32-turn game remains 32 turns after removing its loop: within-turn loops do not explain this turn-count tail. It also has five self-damage battle signals. Three cannot be labeled using hindsight: at decision 148 the enemy field is empty before an opponent hand response; at 358/976 the target is face-down at declaration. At 436 and1187 there is public adverse combat information, warranting separate tactical investigation (including target selection and useful leave-field effects), not an automatic ban on attacking. The current analysis does not establish missed lethal or a tactical fix for that tail.

## Selection-history replication and credit diagnostic

The independent three-seed replication completed all 1536 healthy endpoint games. Final new-history vs legacy: 253/384 vs244/384 wins (65.885% vs63.542%); mean turns 9.6016 vs9.3047; cancels 24 vs 28; >=8-repeat games 1 vs 0. Crossed seed/common-deal bootstrap: win +2.344 pp [-4.427,8.854], turns +0.2969 [-0.4036,1.0547]. Original pilot direction did not establish a replicated turn or loop benefit. Three seeds, one fixed source and Greedy only; maintain opt-in, no promotion.

Seed250 new-history game36 repeats 14 times despite distinct inputs, P(cancel) 97.45%–99.03%. Critic snapshot ranks cancel above either material selection, but this is not proof of historical gradient cause. The isolated 2×4-update credit diagnostic recorded real PPO advantage, ratio, KL rejection and surrogate/entropy/reference-KL chosen-logit derivatives, with the diagnostic surrogate numerically checked against the unchanged original objective. It completed 89 healthy games, but only 7 chosen cancel rows per arm and NO >=90%-probability cancel row. This sample cannot identify high-confidence-loop credit. It does not prove those sampled cancellations are mistakes. Further diagnostic collection needs actual repeat-state/trajectory linkage and adequate coverage before changing objectives.

The credit diagnostic's startup monitor race is resolved in its fresh launcher: write a dated starting state before launching services. The original alert occurred before any heartbeat existed, and the same invocation then progressed normally. Evidence and acknowledgment are retained; no STOP cleared or active service restarted.

Evidence: this directory's paired-progress.json, retention-audit.json, cancel-paired.json and reproducible scripts; sibling warm-u512-repeat60/49/180, warm-u512-repeat-initial168, warm-u512-repeat-historical123 directories (all dated 2026-10-09); selection-history-replication-2026-10-09/paired-analysis.json; selection-credit-audit-2026-10-09/{report.json,coverage-analysis.json,startup-race.md}.
