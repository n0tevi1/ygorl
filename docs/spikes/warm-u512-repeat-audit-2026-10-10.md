# Warm u512 repeated-selection audit (2026-10-10)

Fourteen seed-2 warm u512 alerts were taken over and acknowledged. Source hashes, complete independent policy transcripts/results/responses and cold engine replay all match. These are selected alerts, not an unbiased sample. No engine errors were found.

The frozen policy gives repeated states a strong cancel preference. Native inputs at repeated same-option prompts are unchanged, so there is no new evidence for a different decision at the next retry. Several prompts include different fusion targets, which need not have identical inputs. Existing count32 protection eventually forces progress in the most extreme cases. Ordinary target selection also loops: greedy game5 repeatedly reselects its own graveyard target and cancels the second target under effect78397661; cancelling does not retract that activation.

## Same-state checkpoint probes

All three checkpoints see exactly the same native observation at each first loop prompt. u512 probabilities reproduce the recorded policy. This controls state-distribution drift, but does not identify which PPO gradients caused the drift or establish that warmup caused it.

| Case | P(cancel) u128 | u256 | u512 | Excised decisions, fixed suffix |
|---|---:|---:|---:|---:|
| 79 | 11.41% | 5.71% | 71.28% | 18 |
| 99 | 56.15% | 93.11% | 99.87% | 62 |
| 5 | 1.69% | 11.66% | 86.24% | 36 |
| 200 | 39.46% | 23.98% | 97.18% | 62 |
| 221 | 16.64% | 13.47% | 81.66% | 24 |
| old62 | 14.78% | 18.22% | 84.39% | 14 |
| old146 | 35.94% | 12.80% | 54.67% | 14 |
| old161 | 49.93% | 10.70% | 97.89% | 62 |
| initial137 | 28.89% | 1.55% | 99.57% | 62 |
| initial215 | 15.53% | 7.92% | 79.84% | 14 |
| history79 | 11.42% | 5.72% | 71.31% | 18 |
| history204 | 32.95% | 25.57% | 94.55% | 62 |
| initial233 | 29.00% | 3.72% | 98.70% | 62 |
| old180 | 9.78% | 1.36% | 93.74% | 28 |

For every case, retain one cancel cycle and excise redundant cycles, then execute the complete original suffix. Every remaining action/options transcript matches after reindexing, with identical winner, LP and turns. This tests actual engine continuation rather than assuming equal model tensors prove equal hidden engine state.

## Live-policy budget1 diagnostics

Candidate-only mask intersection; preserve all native constraints and sole exits. Both policies use original stochastic sampling. Changes in sampling consumption can alter later play.

| Case | Candidate cancels 32→1 budget | Turns | Candidate win |
|---|---:|---:|---|
| 79 | 10→2 | 9→11 | 1→1 |
| 99 | 32→1 | 7→7 | 1→1 |
| 5 | 22→2 | 17→15 | 0→0 |
| 200 | 32→1 | 5→5 | 1→1 |
| 221 | 13→1 | 6→6 | 1→1 |
| old62 | 8→1 | 12→10 | 1→1 |
| old146 | 8→1 | 11→10 | 0→1 |
| old161 | 32→1 | 20→20 | 1→1 |
| initial137 | 32→1 | 9→10 | 0→1 |
| initial215 | 8→1 | 11→15 | 1→1 |
| history79 | 10→1 | 5→11 | 1→1 |
| history204 | 32→1 | 6→6 | 0→0 |
| initial233 | 32→1 | 7→9 | 0→0 |
| old180 | 18→3 | 9→9 | 1→1 |

All 28 live-policy games and their cold replays pass health/support checks. Two selected losses become wins, but several continuations get longer. This does not estimate general strength and does not justify banning all cancellation. `DuelResult.winner` and LP already use deck order (candidate=0); trace players use engine seats. Verification explicitly avoids applying seat conversion twice.

Self-Ash and attack-self-damage findings remain separate tactical investigations: they are not implied by the loop labels, and winning does not establish that a particular attack was sensible. The fixed-suffix comparison isolates wasted decisions without assigning global tactical labels. No frozen model, runtime or formal study changed.

Evidence: `out/research/warm512-greedy-repeat-audit-2026-10-10/` contains `source-verified.json`, original-policy and budget1 games/replays, `probe.py`, `probe-results.json`, `verify.py`, `verified-results.json`. The initial probe harness assumed the alert always named select first; one alert named cancel first. Its assertion log is retained, the harness now normalizes the pair boundary, and all fourteen audits pass. See the [final study review](terminal-critic-final-review-2026-10-10.md) for population metrics and the fresh-deal follow-up.
