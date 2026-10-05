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

Pending the fixed-budget runs and new panel. No default or strength claim changes.
