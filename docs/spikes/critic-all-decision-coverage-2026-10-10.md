# Full-decision critic audit and declaration candidate coverage

The completed exploratory replay covers 144 games and 31,544 learner decisions. Source records, replay hashes, per-game prediction hashes, the sealed runtime identity and report identity were verified. No engine errors were recorded. All encoded-action Q outputs were finite; chosen Q/V matched the existing path within 3.73e-7. These checks establish numerical consistency, not the quality of unchosen actions.

Artifacts: `out/research/critic-all-decision-audit-restart-2026-10-10/{report,analysis}.json`, reproducible aggregation in `analyze.py`. The original failed audit and STOP remain preserved. Completion event `53379eed222fd8469e45ab8d430e1f82ce5be00ed40f7e50402f516bd633e126` was acknowledged after verification.

## Critic result

Equal-game mean of chosen-Q and V squared errors against terminal return (-1/+1), averaging decisions within each game first:

| Panel | Games | Control | Terminal refresh | Anchored refresh |
|---|---:|---:|---:|---:|
| Broad, fixed actor | 64 | .593684 | .604034 | .602016 |
| Broad, expanded actor | 64 | .490544 | .501608 | .497288 |
| Runick, fixed actor | 8 | .640255 | .708532 | .702933 |
| Runick, expanded actor | 8 | .364597 | .415417 | .414305 |

The refresh does not consistently improve this subset. Broad differences have opening-cluster confidence intervals crossing zero. Runick/fixed terminal delta is +.068277 (pointwise 95% interval [.000195, .128553]); anchored delta is +.062677 ([.009276, .109021]). Only eight games enter each Runick panel, and these exploratory intervals have no multiplicity correction. Earlier first-idle errors also worsen on this subset, so this does **not** identify non-idle distribution shift as the cause. It does not overturn the larger first-idle confirmation or establish universal critic transfer.

There are 21,795 chain decisions, 2,664 idle decisions, 1,893 card selections and 1,348 battle-command decisions. `analysis.json` includes decision-type slices, original first-idle vs later-idle vs non-idle slices and familiar panels. Long games do not receive extra game weight. No model was fitted to these data.

The next audit is already running: `critic-all-decision-coverage-2026-10-10`, two workers, a 12-hour service cap, independent heartbeat/error/completion monitor. It covers all 1,392 remaining games of the same 1,536-game learner-shift panel. Union results must be checked for exact nonoverlapping coverage and aggregated across all games, not by averaging the 144/1392 report means. This is expanded replay coverage, not fresh independent confirmation. Do not promote the critic to PPO based on this exploratory subset.

## Declaration capacity defect (#319)

Two decisions in game 584 (`AnnounceCard`, indices 453 and 502) expose 9,226 legal engine choices, but Python and native encoders expose only the first 128. Only these two of the 17 declaration decisions in this 144-game sample overflow. There are no other observed overflows. This observed frequency is specific to this selected subset.

`tools/declaration_probe.py` provides a research-only full-menu encoder and chunked actor scoring. It preserves engine ordering and uses only the public legal menu; no hidden opponent deck shortlist. All chunks share one public context and their logits receive a **single global softmax**. It does not alter actor weights, the environment ABI, rollout buffers, or training.

The real-menu probe (`out/research/declaration-fullmenu-probe-2026-10-10/report.json`) confirms:

- All 9,226 candidates are represented and have vocabulary entries. Vocabulary membership does not establish training exposure or strategic competence.
- The first 128 encoded rows exactly match native observation rows; dense and chunked full-menu logits agree (observed maximum delta zero). Prefix logits match the original actor.
- Both menus' full argmax is engine index 1604, card 16509007, previously unreachable. Python and native hosts accept this index and produce identical response histories through 12 subsequent decisions, with no retry or script error.
- Approximately 98.59% of full-menu probability is outside the old prefix. This is predominantly a candidate-count effect and **not** evidence of good action selection. Single-run CPU chunk scoring took 35–37 ms; this is not a throughput benchmark.

The initial probe attempted to call the finished-game API during an unfinished duel; the corrected probe reads the tracker snapshot. `run-attempt1.py` is retained. There was no observed engine failure in that attempt.

This is a validated repair building block, **not a production fix**. Next implement a shared full-candidate interface for collection, stored action indexing, PPO log-probability recomputation and evaluation; add native/Python/train/eval response parity tests. Do not switch only evaluation to the full menu and then attribute any behavior difference to training. A fixed global 9,226-row expansion for every decision would waste memory; candidate rows and offsets need explicit handling. Actor policy quality and critic coverage remain separate gates.

Validation: current-main native build succeeded; 27 declaration/critic probe tests passed; Ruff lint and format passed across 393 files. Frozen experiment runtime and the root's pre-existing dirty core checkout were not changed.
