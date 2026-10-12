# Critic regression attribution and a bounded margin-preservation pilot

The full regression audit reproduces the canonical 128-root fit exactly. Both repeatable training preferences and ambiguous sampled preferences contribute to regressions through shared Q-head parameters. One fixed repair pilot preserves initial action margins where two label batches disagree or contain a zero margin. It restores several inspected attack choices, but does **not establish improvement over initialization**. No checkpoint is promoted, no new rollouts are added, and no sample expansion or RL scaling follows from this result.

This follows the [inconclusive 128-versus-256-state comparison](critic-state-expansion-results-2026-10-11.md). That comparison identified 24 validation roots where at least one of initialization, fitted128 and fitted256 chose a different action. All 24 are audited; the analysis does not select only favorable examples. These roots and the full 64-root diagnostic panel have already been inspected.

## Exact replay and numerical recovery

The audit preserves the original initialization, canonical targets, source anchor, Adam settings and 32 updates. Original32 roots use pooled32 suffixes; prior96 roots retain their first16 suffixes. Models **and Adam states** reproduce the old expanded128 checkpoints exactly at updates 8, 16 and 32. Actor, private trunk, V head and non-Q parameters remain unchanged.

The first attempt stopped in the audit ledger, not in a rollout: one of 66,177 reconstructed parameter updates differed by 2.33762e-7, just above the unchanged absolute tolerance 2e-7. A single-step diagnostic isolated cancellation in the FP32 per-component accumulator. FP64 component accumulation reduced the maximum root-decomposition discrepancy to 2.30110e-8. The restart changes only ledger accumulation and dot products to FP64; actual training precision, optimizer and tolerances remain unchanged. The original STOP and diagnostic evidence are preserved.

The loss is decomposed in two ways: source anchor plus 128 training roots, and source anchor plus five pair-agreement groups. Centered action MSE equals the sum of squared unordered pair-margin residuals divided by the square of the number of legal actions. Algebra and gradients were checked for padded action sets of sizes 2–6 in float32 and float64.

Component first moments use the realized shared Adam denominator. This is a first-order attribution along the actual optimizer path, **not a causal experiment deleting losses**. The two cumulative decompositions agree within 1.02e-8. The largest absolute nonlinear remainder among the 24 probe margins is .003859. Independent analysis rechecks ledger sums and saved-checkpoint margins; it does not independently rerun the optimizer.

## What contributes to the regression

The 602 training action pairs are classified using two independent 16-suffix label batches. Agreement is evidence of repeatability under these samples, not proof that a preference is correct.

| Label-margin relationship | Pairs |
|---|---:|
| Same sign, both absolute return margins at least .25 | 96 |
| Same sign, weaker margin | 94 |
| Opposite signs | 116 |
| Exactly one zero margin | 172 |
| Both zero margins | 124 |

For three previously inspected regressions, the table orients every contribution toward the fitted128 choice. Positive values push that regression; negative values oppose it. Values are return-margin units, not win percentage points.

| Root | Actual margin change | Source anchor | Strong agreement | Weak agreement | Opposite | One zero | Both zero |
|---|---:|---:|---:|---:|---:|---:|---:|
| 69: attack → activate | +.07313 | −.06825 | +.07156 | −.05100 | +.05910 | +.02733 | +.03287 |
| 114: attack → main2 | +.20807 | −.18687 | +.21222 | −.09125 | +.05841 | +.05490 | +.16047 |
| 116: attack → main2 | +.17808 | −.21342 | +.27257 | −.12716 | +.03491 | +.04165 | +.16810 |

Remainders are +.00153, +.00021 and +.00143 respectively. Thus inconsistent labels are not the sole explanation: strong repeated preferences and sampled zero differences also push these margins through shared parameters. The source anchor opposes these regressions. Removing it or deleting apparently conflicting training roots is not justified.

An additional forward-only diagnostic substitutes initial/fitted endpoint parameters for the Q head's context projection, candidate projection and candidate bias in all eight combinations. Shapley allocation includes their interactions. The candidate projection accounts for .04532 of root69's .07313 margin shift, .13620 of root114's .20807, and .10324 of root116's .17808; context and bias also contribute. This locates parameter pathways, not the causal source of the training error. Context-only fitting was already tested in earlier experiments without establishing fresh-root benefit, so it is not repeated here.

## One fixed repair comparison

The pilot uses the same 128 training roots, initialization, 32 updates, source anchor coefficient1, Adam learning rate1e-4/epsilon1e-8 and gradient clip1. Only Q-head parameters train. Two arms and checkpoints 0/8/16/32 are fixed before launch, with two CPU threads and a 30-minute cap. No games are generated.

For each action pair, if the two sampled target margins have the same nonzero sign, retain the original canonical target margin. Otherwise preserve the initial model's pair margin. This changes 412 pair targets and retains 190; all 602 pairs remain in the loss with their original equal-root normalization. No magnitude cutoff, tuned weight, validation outcome or best-step selection is introduced. The second label batch now informs training and must no longer be described as a held-out confirmation set. This intervention can preserve existing mistakes and does not directly solve interference from stable labels.

The baseline reproduces the earlier fitted128 **parameters** exactly at all four checkpoints. Pairwise canonical loss and gradients equal the original centered MSE before intervention. All eight checkpoints freeze before evaluating the previously inspected 64 roots and retention sets. A separate verifier reproduces checkpoint predictions, aggregate-result labels, choice metrics, retention gates and both bootstrap calculations. All retention gates pass; actor and non-Q parameters remain unchanged.

| Fixed model | Centered action MSE | Conditional choice wins |
|---|---:|---:|
| Initialization | .015814 | 66.357% |
| Canonical128, update32 | .015970 | 65.723% |
| Margin-preserving128, update32 | .015649 | 66.455% |

The pilot-minus-canonical128 difference is **+.732422 pp**, with descriptive stratified root-plus-suffix 95% interval **[−.781250, +2.441406] pp**. Against initialization it is only **+.097656 pp**, interval **[−.488281, +.781250] pp**. These are post-hoc diagnostics on reused roots; the intervals do not account for adaptive reuse and cannot establish generalization even if they were positive. Intermediate checkpoints remain reported without selecting a best one.

Compared with canonical128, 20 choices change: four have higher sampled returns, four lower and 12 equal. Roots69/114/116 and164 return to initialization's choice; root66's regression remains. Returning to initialization also loses some fitted gains at roots45/64/94. Against initialization, eight choices differ: two positive, two negative and four equal sampled returns. Each action has only 32 continuation samples, so these are not definitive tactical labels.

## Decision and remaining work

The intervention reduces some observed regressions, but mostly preserves earlier behavior. It provides no demonstrated gain beyond initialization. **Do not deploy it, tune it repeatedly on these 64 roots, enlarge the sample until an interval excludes zero, or start a larger RL run.** Conditional critic choices under frozen continuations are not actual actor full-game wins, shorter games or unseen-card transfer.

The remaining mechanism question is how to learn useful action preferences without exporting them to unrelated states through the shared candidate projection. Repeated target directions are insufficient protection. A next representation or update-isolation proposal must distinguish learning from merely freezing the initial ranking, retain counterexamples where the fit genuinely helped, and state an independent evaluation and resource budget before running. The follow-up below supplies an important competing explanation: a simple output-movement-matched shrinkage control has at least as favorable diagnostic point estimates as the proposed margin repair. Issues #61/#83/#36 remain open; the broader critic and long-game problems are unresolved.

## Follow-up: context dependence and a matched shrinkage control

`critic-context-path-audit-2026-10-11` is a forward-only audit of the same initialization, canonical128-step32 and margin-preserving128-step32. No optimizer steps or new games are performed. All64 reused roots, every legal action pair and all128 training-state queries are included.

The Q head scores an action with `dot(query(state), candidate(action)) / sqrt(hidden) + bias(action)`, followed by tanh. Candidate embeddings for card actions already contain contextual board-transformer features. Phase actions have no card token and depend on the separate query path for changing state information. Across this panel each phase kind has two bitwise-distinct candidate vectors, but their maximum coordinate differences are only7.15e-7 for main2 and3.58e-7 for end-phase. The implementation has no board-dependent term for these identical no-card action rows; the small floating differences should not be interpreted as useful phase-state encoding.

Holding each root's candidate vectors fixed and replacing only its query with the mean of the128 training queries changes3/64 initial choices,11/64 canonical128 choices and3/64 repaired choices. Replacing it with each of the128 individual training queries leaves the original choice unchanged for every replacement at39/64 roots initially,0/64 after canonical fitting and1/64 after repair. Thus the fitted head is not simply ignoring its query. Candidate vectors still contain the original board context, and mismatched query/candidate combinations may be off the data manifold: this does not measure tactical sensitivity under legal game-state changes.

An exact symmetric decomposition of the **pre-tanh** score change separates query updates, candidate updates against the training-mean query, candidate updates against each root's query deviation, and candidate-bias updates. For two roots, main2-minus-attack shifts are:

| Root | Total raw margin shift | Query update | Candidate × common query | Candidate × query deviation | Candidate bias |
|---|---:|---:|---:|---:|---:|
| 64: fitted choice has higher sampled return | +.37270 | +.09054 | +.21022 | −.00206 | +.07400 |
| 116: fitted choice has lower sampled return | +.36341 | +.08712 | +.19683 | +.01221 | +.06724 |

The common-query pathway contributes strongly to both examples. Similar contributions appear in roots69/114, but simply suppressing this pathway can also remove useful adaptation. These raw-score shifts are not the bounded return-margin units in the earlier attribution table. The mean-query component is not wholly state-independent because card-action candidate vectors are themselves contextual. Independent matrix multiplication verifies all scores, swapped choices and pair decompositions; sums agree within1e-12.

To check whether the prior repair merely moves less from initialization, a separate fixed comparator uses `Q_control = Q_initial + alpha * (Q_canonical - Q_initial)`. Choose alpha **only on the128 training roots**, matching the repaired model's equal-root RMS change in centered Q scores. The resulting **alpha=.5909121643** is sealed before opening the reused diagnostic predictions/outcomes. There is no alpha sweep or selection by win rate. This matches output movement magnitude, not its direction, parameter norm, optimizer history or retention. It is a functional diagnostic comparator, not a deployable model or a recommendation to change the learning rate by this factor.

| Model on reused64 roots | Centered action MSE | Conditional choice wins |
|---|---:|---:|
| Initialization | .015814 | 66.357% |
| Canonical128, update32 | .015970 | 65.723% |
| Margin-preserving128, update32 | .015649 | 66.455% |
| Movement-matched shrinkage | .015201 | 66.602% |

Repair and shrinkage differ on eight choices: repair has one higher, two lower and five equal sampled returns. The point estimates therefore do not distinguish a special benefit of handling ambiguous labels from ordinary reduced adaptation. They do not establish superiority of shrinkage either: the roots are reused, the contrast is post-hoc and only32 continuations per action are available. Independent verification recomputes training movement by explicit action pairs, the sealed alpha, individual game-result labels, all four models' choices and metrics.

The first shrinkage verifier had an advanced-indexing mistake: two column vectors were subtracted instead of forming an outer difference, producing zero pair energy. It stopped **before alpha was sealed or diagnostic outcomes were opened**. The original script, design and failure record remain; a corrected explicit outer-difference calculation and a separate scalar-loop verifier pass at the unchanged tolerance. No training run or model checkpoint was affected.

**Updated decision:** do not treat the margin-preservation pilot as a validated mechanism-specific repair, and do not change architecture or scale data on this evidence. Any next proposed repair needs a movement-matched reduced-adaptation control, retained examples of beneficial updates and independent validation. No additional training, collection or hyperparameter sweep is queued by this audit.

## Evidence

All paths below are under `out/research/` and retained locally:

- `critic-regression-attribution-2026-10-11`: original identity `7533d0575a210a1fe208ed3b4e486fbbbaf89e3e5234b3fb26318d803413e2bd`, STOP and `numerical-diagnostic/diagnosis.json`.
- `critic-regression-attribution-restart-2026-10-11`: identity `2ba2a78ca35fa5910285037aed1e4a26c180955292b8a07f41b30e83d2358e06`; exact-replay report, `attribution.json`, `analysis.json`, `verification.json`, `parameter_audit.py` and `parameter-audit.json`.
- `critic-pair-margin-anchor-pilot-2026-10-11`: identity `aa6296c072161598b0e064589837a8f18670e54777c57a69b712132ce1aa0277`; registration, `pair-targets.json`, eight checkpoints, `fit-complete.json`, `report.json`, `analysis.json`, `analysis-verification.json` and `choice-audit.json`. Analysis SHA256 `504249b98b204839c700f15f0f9a0df8755ec2159e527bfc601200f5af4a532e`.
- `critic-context-path-audit-2026-10-11`: identity `009d6c3bb88fa1656e344400e9e8948340325a1652f2bd896e49c60bc5e01593`; `report.json`, `arrays.npz` and independent `verification.json`. Supplemental `shrink-control-corrected-design.json`, `shrink-alpha.json`, `shrink-control.json`, `verify_shrink.py` and `shrink-verification.json` pin the comparator separately. Original verifier failure is retained in `shrink-control-check-failure.json` with its original script/design. Completion event `9610d85089880c06bc49eb59fccc81f0d67d5f6dcda219b23f78065a6ad4f269` is acknowledged. Reporting follows main9c95958; matched runtime remains ae09594.

Attribution completion event `048b83ca283d33b3f532f87f0405afb6122f39ec910b59eb9d8ebe34cb2ac011` and pilot completion event `a109248eb6dcc9553e0fba42a47d788d073f9f9ff2f6d3d3720d8b5650312faa` are acknowledged with verification scope recorded. The runtime remains frozen at ae09594 for comparability; the reporting branch follows main, fetched at 88062b3 before publication. No original STOP is cleared.
