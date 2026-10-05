# MD collection control (2026-10-04)

## Question and preregistration

The complete-game lambda pilot improved against its frozen initialization, but
the independent third-seed confirmation did not pass its gate. Neither comparison
isolated the value of complete-game collection from additional MD-domain RL.
This experiment compares collection recipes at approximately equal learner-row
budgets. It does not change the default PPO recipe. Related: #61, #83, #177 and
[the terminal-target experiment](terminal-mc-2026-10-04.md).

Freeze this protocol before training the controls or observing the new panel:

- Reuse **all three** complete-game lambda runs, seeds 2026100410, 2026100411,
  2026100413, exactly update 64. No selection of the strongest seed/checkpoint.
- Train three ordinary fixed-segment controls with the same per-seed initial
  actor, fresh critic/optimizer, MD-2026-09 decks, PPO/VRPO lambda=.5, opponent
  pool, 32 environments, and 64 updates. Change only `complete_games=false` and
  `steps=160`: 327,680 learner rows each. Prior complete runs have 319,481,
  338,183, and 318,071 rows (within 3.2%); equal wall time, game count, and
  optimizer minibatch count are **not** asserted.
- Preserve each collector's implementation, including event ordering, handling
  of unfinished games, and truncated-game rows. This compares collection
  recipes, not a single isolated bootstrap-boundary effect. The fixed collector
  continues games across policy updates; the complete collector discards all
  learner rows from truncated games and synchronizes on game endings.
- Reuse frozen BBL05-u400 as initialization and evaluation baseline. Use the
  same three external opponents (seed1-u400, league-u400, BBL05-u1200).
- Fresh evaluation seed **2026100415**, **256** sampled deck-pairing/shuffle
  clusters, four-way mirrored games, all three opponents, seven candidates:
  baseline + three complete + three segment. Total **21,504 games**. No interim
  efficacy analysis or budget extension. One frozen update-64 checkpoint/run.
- Primary contrast: equal-weight mean complete score minus equal-weight mean
  segment score, paired on all games. Shared 10,000-resample cluster bootstrap,
  pointwise 95% interval. Report each matched training-seed contrast as well.
  These intervals cover evaluation sampling for fixed checkpoints/opponents,
  not uncertainty over the training-seed population.
- Advance complete-game collection only if its primary difference is >=2 pp
  with lower interval >0, both recipe means improve against baseline with
  lower interval >0, and no matched seed favors segments by >=2 pp. Secondary
  baseline contrasts cannot rescue a failed primary gate. A failed gate does
  not establish equivalence or prove segments are better. Reverse superiority
  and per-opponent contrasts are descriptive, not additional promotion tests.
- Engine exceptions/errors invalidate their entire cluster jointly across all
  candidates. Decision-limit outcomes retain the arena's draw convention in
  the primary score. Report all error/limit counts, jointly exclude every
  limit-affected cluster in a prespecified sensitivity, and report finite-panel
  worst-case censoring bounds (not confidence intervals). Sensitivity does not
  rescue a failed primary gate.
- Stop on nonfinite metrics, OOM, empty batch, or >2% truncated games after
  >=128 finished training games; retain failed runs and diagnose rather than
  substitute seeds. Audit environment stamps, initial weights, row/game/decision
  budgets, optimizer steps, checkpoints, code/native/database/script identities.

Artifacts: `out/research/collection-control-2026-10-04/`. The machine-readable
manifest pins source/config/checkpoint inputs; the archived preregistration
remains immutable when this report gains results. Previously observed complete
checkpoints are reused for an efficient conditional comparison; the new panel
does not make this an independent replication over newly trained complete seeds.

## Results

All three segment controls completed their fixed 64 updates. Same-seed initial actor and critic weights match the corresponding complete run exactly; configuration differences are only collection mode, step budget and equivalent absolute environment/deck paths. All recorded games carry the expected environment fingerprint.

| Recipe / seed | Learner rows | Finished games | Decisions | Discarded rows | Limits | Optimizer minibatches | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| complete-s0 | 319,481 | 2,048 | 376,537 | 15,590 | 6 | 374 | 390.1 |
| segment-s0 | 327,680 | 2,067 | 369,326 | 0 | 1 | 378 | 332.9 |
| complete-s1 | 338,183 | 2,048 | 432,262 | 52,414 | 20 | 378 | 527.9 |
| segment-s1 | 327,680 | 1,971 | 363,707 | 0 | 3 | 379 | 347.8 |
| complete-s2 | 318,071 | 2,048 | 373,361 | 17,704 | 6 | 377 | 394.7 |
| segment-s2 | 327,680 | 1,905 | 361,610 | 0 | 5 | 383 | 332.2 |

New segment training totals **983,040 learner rows, 5,943 finished games, nine decision-limit truncations and zero engine errors**. Segments retain truncated prefixes with the standard critic bootstrap; their zero discarded rows is not evidence of zero truncation. Unfinished games at the final update are not counted as completed games. Historical complete-run timings were measured at a different time; they are descriptive costs, not a randomized throughput comparison.

### Independent panel

All **21,504 games** completed: **zero engine errors**, 143 decision-limit draws, no turn limits. The primary retains all 256 clusters.

| Candidate | Panel score | Difference vs frozen baseline (pp) | 95% CI (pp) |
|---|---:|---:|---|
| base | 0.50570 | +0.00 | [+0.00, +0.00] |
| complete-s0 | 0.49495 | -1.07 | [-2.98, +0.83] |
| segment-s0 | 0.51237 | +0.67 | [-1.38, +2.70] |
| complete-s1 | 0.50618 | +0.05 | [-1.73, +1.82] |
| segment-s1 | 0.49788 | -0.78 | [-2.77, +1.24] |
| complete-s2 | 0.51351 | +0.78 | [-1.04, +2.62] |
| segment-s2 | 0.49984 | -0.59 | [-2.51, +1.38] |
| complete_mean | 0.50488 | -0.08 | [-1.51, +1.36] |
| segment_mean | 0.50336 | -0.23 | [-1.76, +1.32] |

**Primary complete-mean minus segment-mean: +0.15 pp, 95% CI [-1.03, +1.33] pp. The preregistered promotion gate fails.**

| Matched training seed | Complete minus segment (pp) | 95% CI (pp) |
|---|---:|---|
| 0 | -1.74 | [-3.92, +0.44] |
| 1 | +0.83 | [-1.22, +2.91] |
| 2 | +1.37 | [-0.65, +3.42] |

Jointly excluding all 67 limit-affected clusters leaves **189/256**. The primary difference becomes **+0.46 pp [-0.94, +1.85] pp**, still failing the gate. Finite-panel worst-case censoring bounds are **[-0.59, +0.89] pp** (not confidence intervals). Recipe means versus baseline also remain inconclusive after joint exclusion: complete +0.16 pp [-1.48,+1.76], segment -0.29 pp [-2.07,+1.48].

Limits by candidate: base 7; complete-s0/s1/s2 19/44/21; segment-s0/s1/s2 24/17/11. Errors were not recoded as draws; there were none.

### Decision and cross-panel audit

**Keep default fixed-segment PPO unchanged. Do not promote complete-game collection or extend this short collection-only pilot to chase a positive interval.** Neither three-seed recipe mean shows a stable gain over the frozen actor on this panel. This does not establish equivalence, disprove long-budget RL or settle capacity/teacher quality.

The earlier auxiliary complete-game result (+6.27 pp for two seeds) did not reproduce here. An independent raw-record sum reproduces both old and new scores. Checkpoints, opponents, source hashes, native core, cards, rules, script revisions, sampling settings, Torch/HIP and GPU identities match across panels. The independent pairing/shuffle seeds differ; no mismatched-input or scoring error was found. That audit does not prove that sampling noise is the only cause. Do not pool panels post hoc to rescue the old gain or subtract their estimates as a causal improvement/regression.

Next prioritize the BC data-integrity repairs and legal environment-bound teacher data before #88 capacity work and #92 long training. **PR #179 has already merged**: it fixes initialization seeding, extra-data identity checks, legal environment-aware collection and erroneous game-prefix handling. A legal two-deck fixture validates the repaired chain, not competitive strength; see [the repair report](bc-input-contract-2026-10-04.md). Formal coverage and larger-model experiments remain open.

### Reproducibility and validation

Training and evaluation use protocol commit `66055d7` on main base `c3a0b6e`. Protocol SHA256: `8bcbf537d6577b3443cae54cb4b3ff85ccd64b897f52714f096e9a355aec20f7`. All 1,192 frozen input files and the separately archived execution scripts validate. Same-seed initial full model weights match; all environment stamps match. `cross-panel-audit.json` records the historical-input comparison.

Resume validation reuses all **21 cells**, plays **zero new games** and preserves hashes of all **23 panel JSON files**. Run the archived `train_one.py`, `evaluate.py`, `score.py`, `audit_training.py` and `verify_resume.py` from the protocol checkout with the manifest paths; the panel manifest binds the Git commit, so later report commits require restoring that checkout for exact resume. Raw artifacts and all six final checkpoints are retained. Full local presubmit on the frozen implementation: **1,407 passed / 3 skipped** in 279.00 s, including real ROCm tests. The skips are one snapshot-build-option test and two opt-in network tests. Final branch changes are documentation only; no hosted checks were generated.

## Audit for the subsequent capacity work

The old statement “capacity is not the bottleneck” is stronger than the tested
128×2 / 400-update evidence. #88 still requires three larger BC configurations,
and #92 requires much longer training. The older `out/bc_b2s/report.json` and
`out/why/bc128/report.json` both have `environment: null`; all 600 training and
100 held-out source demonstrations also have null environment stamps. They
reference the same demonstration files, but their processed solver sample counts
are 9,866/2,602 versus 9,825/2,594 (train/held-out). The reason for that historical
processing difference has not been established. Do not describe these as
identical encoded training sets or silently relabel them as MD-2026-09 artifacts.
Legality validation against the pinned MD environment also finds violations in
**all ten** legacy demonstration decks (37 violations total; full reasons in
`legacy-demo-legality.json`). This is not fixed by adding environment metadata.

Before expanding BC on MD, generate demonstrations using legal decks under the
pinned environment and freeze one processed dataset across network sizes.
Keep the old checkpoints as historical baselines. The input inventory and hashes
are in `capacity-input-audit.json`; this audit is separate from the preregistered
collection comparison and does not change its candidates or gate.
