# Terminal critic continuation: final three-seed review (2026-10-10)

The registered u128→u512 continuation is complete and independently verified. Both cold and warm policies improve; the preregistered gate to **test longer training passes**. Warm's final advantage over cold remains uncertain across training seeds. Repeated cancellation has worsened despite better wins and shorter games, so the next diagnostic holds sampling fixed and changes only cancellation budget.

Study: `out/research/terminal-critic-continuation-immunity-2026-10-09`, identity `270faf6a7cfd97e0084bb8bae3588d47a961e47ffd66f21f63a507f131bb3984`. Frozen runtime `50385d4`. Six arms completed 2,304 added optimizer updates, 37,748,736 rows and 277,664 training games, with zero errors or truncations. All 19,456 fixed-panel evaluation games and checkpoint/source hashes passed independent audit. All final notifications were acknowledged without changing the study or historical STOP evidence.

## Strength and game length

Primary panel: three fixed opponents × 256 paired games × three training seeds = 2,304 games per arm/checkpoint. The fourth historical-RL opponent is secondary. u128 is the restored starting checkpoint on the repaired environment, not initial behavior cloning. Warm/cold refer to the existing terminal-critic initialization experiment; this continuation does not add periodic supervised refresh.

| Arm / update | Wins | Win rate | Mean turns | Median | P95 | >20 turns |
|---|---:|---:|---:|---:|---:|---:|
| cold-128 | 1603/2304 | 69.57% | 11.362 | 10 | 21 | 129 |
| cold-256 | 1822/2304 | 79.08% | 10.215 | 9 | 18 | 58 |
| cold-512 | 1939/2304 | 84.16% | 9.325 | 9 | 17 | 41 |
| warm-128 | 1774/2304 | 77.00% | 10.510 | 10 | 19 | 81 |
| warm-256 | 1880/2304 | 81.60% | 10.181 | 10 | 18 | 52 |
| warm-512 | 1995/2304 | 86.59% | 9.001 | 8 | 17 | 27 |

Crossed 95% intervals resample the three training seeds and the same 64 deal blocks jointly across opponents. Warm u512−u128: **+9.59 pp [6.68, 12.54]**, positive for every seed; historical opponent **+20.31 pp [12.89, 26.82]**. Cold growth: **+14.58 pp [11.59, 17.58]**. Warm mean-turn change **−1.509 [−1.877, −1.149]**; cold **−2.037 [−2.497, −1.576]**. Turn analysis is exploratory, not the registered strength gate.

At u512, warm−cold is **+2.43 pp [−0.09, 5.03]** and **−0.324 turns [−0.899, 0.179]**. Neither interval establishes warm superiority. This is one shared supervised corpus and a small fixed opponent/deck pool, not an expert-play benchmark. Nine average turns is progress, not evidence of top-tier strength or a universal four-turn target. Cold's maximum rises to 51 despite a shorter mean; warm's maximum is 32.

Seed-2 warm endpoint alone: wins versus greedy/old/initial/historical are **206/228/236/174 of 256**. Primary wins improve 591→620→670 across u128/u256/u512. The final three-seed analysis takes precedence over interpreting this single-seed milestone.

## Retention and behavior

Independent NumPy recomputation verifies identical 81,546-row / 256-game fixed reference data, prediction/checkpoint hashes, Q/V MSE and far-horizon MSE. Warm Q MSE u128→u512: seed0 **0.8561→0.7473**, seed1 **0.8084→0.7468**, seed2 **0.8427→0.7355**. V MSE likewise improves; every paired per-seed MSE-delta interval is below zero. This supports retention/improvement on this fixed critic task, not retention of every card skill or on-policy calibration.

Legacy material-cancel totals over **3,072 games including all four opponents**:

| Arm | u128 | u256 | u512 | Games with ≥8 cancels at u512 |
|---|---:|---:|---:|---:|
| cold | 244 | 179 | 378 | 7 |
| warm | 164 | 289 | 1,201 | 36 |

Warm u512−u128 is +0.338 cancels/game, crossed95% interval [0.112, 0.609], with increases in all three seeds. These counts depend on visited states and omit ordinary SelectCard cancellations; a per-game total is not necessarily one contiguous loop. They cannot be treated as a global tactical-error rate.

The [fourteen-case audit](warm-u512-repeat-audit-2026-10-10.md) shows concrete repeated-selection cycles and worsening same-state cancellation probabilities. Fixed-suffix excision leaves the complete remaining action transcript, winner, LP and turn count unchanged. Reducing these cycles saves decisions; reducing actual game turns requires changes in gameplay too.

## Next experiment and decision

A separately registered **warm512-cancel-budget-confirmation-2026-10-10** compares T=1 sampling with budget32 versus budget1, candidate only, for all three warm u512 checkpoints. It uses 32 fresh shared deal blocks, four assignments and four opponents: **3,072 games**, exact cold replay for every game, two CPU workers, three-hour bound and separate heartbeat/watchdog notification service. The two arms share frozen Clown-fixed runtime `885e063`; the completed training runtime and metrics remain immutable. Six preflight games cover both seats and an exact severe-loop fixture. The study and independent watchdog are launched and active; sealed identity `03bcb8278128a5b4f19906254a91bb0f85c421f872af9ee36df891f46e9d3003`.

No argmax or summon-reentry guard is mixed into this comparison. Preserve sole legal exits. Predeclared canary gate: no health/replay/termination failures; primary win-delta lower95% bound >−2 pp; no opponent point loss >5 pp; ≥50% fewer games with ≥8 cancels; no increase in decision-limit or P95 decision count. Report failures and all outcomes; no automatic default promotion. This gate permits considering a guarded training canary, not declaring expert play.

Longer RL is justified by the original registered gate. First resolve whether the isolated cancellation guard preserves strength; then register an equal-budget continuation control and guarded canary with checkpoint/optimizer/pool/RNG provenance and fixed retention/behavior checks. Earlier selection-history and credit pilots were inconclusive/negative, so this report does not declare the learning problem fixed or repeat those pilots without a new hypothesis.

## Evidence

`out/research/terminal-critic-final-review-2026-10-10/`: `independent-audit.json`, `behavior-summary.py/json`, `retention.py`, `retention-seed-{0,1,2}.json`. Original completion, analysis and source logs remain under the study directory. The independent audit reproduces all raw scores and registered bootstrap intervals; behavior and retention scripts provide the additional checks above. Related issues: #83, #218, #239.
