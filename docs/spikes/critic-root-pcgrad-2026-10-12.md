# Root-level gradient projection does not establish a critic repair

The bounded root-level PCGrad pilot is complete and rejected as a replacement critic. It does not improve the reused-panel point estimate beyond initialization, and its action-ranking MSE worsens. The optimizer replay identifies remaining conflict both before and after Adam. No production training default, checkpoint promotion or data scale changes.

This follows the [context-path and movement-matched controls](critic-regression-attribution-2026-10-11.md). The implementation is a research adaptation of [Yu et al., Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782): individual training roots serve as tasks. Their targets may be noisy or legitimately opposed, so simultaneously improving every root is not assumed to be desirable or possible.

## Fixed experiment and implementation

Both arms use the same128 training roots, canonical targets, initialization, source retention anchor, Adam1e-4/epsilon1e-8, clipping1 and32 steps. Actor, trunk, V and non-Q parameters remain frozen. The baseline reproduces earlier model parameters exactly at0/8/16/32. Original32 roots retain pooled32 suffixes; the additional96 retain their original16. No new games are generated.

The candidate projects each equal-root-weighted gradient against conflicting **original** gradients in a seeded independent permutation, sums the projected rows, then adds the unchanged source-anchor gradient before clipping and Adam. It uses one fixed projection seed2026101201, with no order sweep. Projection calculations use float64 and are cast back to the original optimizer precision. Sequential projections need not remove every final conflict; subsequent source anchoring and Adam can change the direction further.

`tools/critic_gradient_projection.py` computes projections in the gradient span through its Gram matrix and returns diagnostics without modifying inputs. It is not enabled in production training. Tests compare against an independent sequential vector implementation and cover zero/orthogonal/aligned/conflicting gradients, input immutability, disabled second-order autograd, malformed permutations and invalid inputs. The frozen experiment uses a hashed local helper; the repository utility adds validation and an explicit no-grad boundary around the same projection algorithm.

All eight checkpoints freeze before evaluation. The service automatically chains analysis, independent verification, a movement-matched score control and independent control verification; its monitor waits for `completion.json`, not merely the fit report. Two CPU threads,20-minute cap; no GPU or additional rollout budget.

## Results on the reused64-root panel

| Fixed model | Centered action MSE | Conditional choice wins |
|---|---:|---:|
| Initialization | .015814 | 66.357% |
| Canonical128, step32 | .015970 | 65.723% |
| Root-PCGrad128, step32 | .018105 | 65.674% |
| Movement-matched canonical score control | .016781 | 64.795% |

PCGrad-minus-canonical conditional wins differ by−.048828 pp, with descriptive stratified root-plus-suffix95% interval[−1.416016,+1.416016] pp. Retention gates pass, but they are not an improvement gate. The movement-matched coefficient is1.2184759605, determined solely from centered-Q RMS changes on the128 training roots; this is **extrapolation**, not shrinkage. Its lower choice-win point estimate does not rescue PCGrad's lack of improvement over initialization or worse ranking error. The score control is not a deployable model and does not match optimizer history or retention.

These64 roots have already been inspected. Results and intervals are diagnostic, do not account for adaptive reuse, and are not independent actor full-game validation. All intermediate checkpoints remain reported without best-step selection. Independent checks recompute predictions, labels, metrics, retention and bootstrap calculations; the control verifier separately recomputes training movement, coefficient, individual game outcomes and choices.

## Why gradient projection was insufficient

A separate replay starts from candidate checkpoints0/8/16 with their saved optimizer and projection RNG states, reproducing updates1/9/17. Projection counts and realized update norms match the original logs. All comparisons below are at the **same parameters** within each row, not separately trained SGD models.

| Replayed update | Raw root aggregate: roots harmed | Projected root aggregate | Plus source anchor | Actual Adam direction | Actual losses increasing |
|---|---:|---:|---:|---:|---:|
| 1 | 44 | 30 | 30 | 36 | 36 |
| 9 | 49 | 30 | 49 | 49 | 51 |
| 17 | 53 | 35 | 46 | 49 | 51 |

The first four counts use positive gradient dot update as a first-order loss-increase signal among128 roots. The last column evaluates the actual finite update. Directions have different norms, so their mean directional losses are not directly comparable as learning rates.

PCGrad reduces the number of conflicts in its root aggregate, but leaves30–35. Source retention introduces a competing objective. Adam also rotates the anchored direction: its cosine with the pre-Adam descent direction is .310/.295/.333. At update9, Adam introduces36 root increases and removes36 others; equal aggregate counts hide a different affected set. At updates9/17 the source loss derivative changes from negative before Adam to positive under the realized update. This is local evidence that gradient-space surgery is not a guarantee on the actual optimizer step. It does not prove Adam alone causes the overall generalization failure, and it does not justify removing retention.

## Evidence and decision

- `out/research/critic-root-pcgrad-pilot-2026-10-12`: identity `3685d885d0779613c19e2136c4b6555317f0ac7a4824af3d9e57a1a527220047`; registration, implementation/reference tests, checkpoints, `training-diagnostics.json`, `completion.json`, analysis and two independent verification artifacts. Completion event `a80b8744bb7f0daeb9c8f82e2bd953b91bddb03119a2e90d9ccbf5f532a2a8ec` acknowledged after verification.
- `out/research/critic-pcgrad-optimizer-audit-2026-10-12`: identity `18d490d2c3896ac935e91f0fc4a4522dfd6381fd1b2d66c023e438f08dd1307d`; three-step replay, finite-loss checks and `report.json`. No new fit or evaluation games.

Reject the PCGrad candidate and retain its negative evidence. Issues #61/#83/#36 remain open.

## Actual-step projection and finite-step limits

The follow-up projects the realized Adam displacement onto the intersection of the 128 root-gradient and source-anchor halfspaces. It minimizes distance to the proposed step, subject to nonpositive first-order loss changes. This is a diagnostic of local interference, not an assumption that every noisy training root deserves protection.

The original coordinate-descent solver did **not converge** after 10,000 sweeps at any of the three saved states. Its output was never applied. A separate restart solves the same problem through an orthogonal thin-QR transformation followed by nonnegative least squares, without rank truncation or relaxed tolerances. The normalized gradient matrices have condition numbers about 1,730–2,352; forming their Gram matrices squares those condition numbers. This explains a plausible numerical difficulty, not a proof that conditioning alone caused the first solver's slow convergence.

All three restarted solves satisfy full-space feasibility and complementarity below 1e-8. A separate verifier checks saved gradient/step/dual witnesses without reusing QR or NNLS: stationarity is below 1e-16 and absolute primal–dual gaps below 1e-16. Zero-step forward passes reproduce the original losses bitwise.

| Replayed update | Displacement retained | Roots with finite loss increase: cone / equal-norm Adam | Sum of positive loss changes: cone / equal-norm Adam |
|---|---:|---:|---:|
| 1 | 96.74% | 17 / 36 | 0.0000293 / 0.0236730 |
| 9 | 96.91% | 18 / 51 | 0.0001120 / 0.0115353 |
| 17 | 97.78% | 16 / 51 | 0.0000322 / 0.0114874 |

Finite-increase counts use a fixed 1e-8 threshold on each unweighted root loss. This is **not** a game error rate or win-rate metric. At full displacement, positive training-loss changes shrink by 99.0–99.9%, but average fitting progress also slows. None of these checks establishes better action selection on unseen games.

A fixed scale grid of 1, 1/2, 1/4 and 1/8 reduces the residual finite increases. At 1/8, cone updates still increase 7/9/7 root losses above 1e-8; their maximum increases are 4.05e-7 / 6.71e-7 / 1.89e-7. Equal-norm Adam increases 36/49/48 roots. The remaining effect is consistent with finite-step nonlinear behavior, including possible changes in piecewise activation regions; it is not evidence that the solved linear constraints failed.

Retention has a separate blind spot: at initialization the squared source-anchor loss and its gradient are both zero. The cone cannot constrain its second-order increase. Its actual increase is 6.99e-6 at full displacement, decreasing to 1.75e-6, 4.38e-7 and 1.10e-7 on the smaller scales. At updates 9/17 the full cone step instead decreases source loss by 5.17e-6 / 1.11e-5. Linear no-increase constraints therefore do not replace actual retention tests.

Evidence:

- `critic-step-cone-audit-2026-10-12`: original nonconverged attempts, preserved; identity `f52848c57f2379fa3c2cd4a01bdf4c19ecea1895f09140d1dc02996e1aeddd1d`.
- `critic-step-cone-qr-audit-2026-10-12`: converged same-state restart; identity `f497e33898399441c4d8d217a60eb50fdc50346bbf3d9d915abd4ddb939e4d6f`.
- `critic-step-finite-audit-2026-10-12`: fixed scale grid, actual source losses and independent KKT verification; identity `b0b61e1b702050022d4467cead8177b49fde2c8c74dea02efb50146752a85eb2`. Verification report SHA256 `ad2174c1830e1a85a04fcd6d78376c03019aad6deb90e65cbd4b4509e0292379` identifies the input report, not the verifier itself.

These local results justify one bounded accumulation test, not more rollout data. `critic-step-cone-pilot-2026-10-12` uses the unchanged scalar objective and canonical Adam gradients, then projects the proposed **parameter displacement**. It does not stack PCGrad preprocessing. Adam moments follow the unmodified objective; only parameters are projected. The original 128 roots, 32 updates, checkpoints 0/8/16/32 and retention gates remain fixed. Solver nonconvergence stops the study. A 20-minute CPU cap and independent monitor cover the full fit → analysis → verification → training-movement-matched control → control-verification chain. No checkpoint is promoted from this reused diagnostic panel. Identity: `fda5eb7677983817c5b6b5467a06860bd073bddc3d0d7029245c84dd3ad51aa7`.

## Accumulation test: no advantage over simple score shrinkage

All 32 updates and the main independent verification completed. The analysis chain then stopped because its launch preparation omitted `shrink-control-corrected-design.json`. This was an orchestration error, not lost training progress. Original checkpoints and STOP remain intact. `critic-step-cone-control-recovery-2026-10-12` supplied the same predeclared control formula in a separate hashed study and completed the control plus independent verification without retraining. Recovery identity: `3e2768d112465b78ce34d3dfc43247544c1a81a41a4dcd2c9408208b4d840de8`.

| Fixed model | Centered action MSE | Conditional choice wins |
|---|---:|---:|
| Initialization | .015814 | 66.357% |
| Canonical Adam, step 32 | .015970 | 65.723% |
| Actual-step cone, step 32 | .015520 | 66.455% |
| Training-movement-matched score shrinkage | .015273 | 66.455% |

The cone passes all retention gates. Versus initialization it changes only 4/64 choices, with +0.097656 pp conditional wins and descriptive stratified root-plus-suffix interval [−0.292969, +0.634766] pp. Versus canonical Adam it changes 20/64 choices, with +0.732422 pp and interval [−0.732422, +2.441406] pp. Neither comparison establishes improvement. Adaptive panel reuse further limits these intervals.

The matched coefficient is **0.2768236137675729**, sealed from training-root centered-Q movement before the control reads diagnostic outcomes. The control ties cone choice wins and has lower MSE; their only differing choice has identical sampled win frequency. Consequently the accumulated result does not justify the more complex projection method. Local conflict reduction was real, but was insufficient evidence of a better critic.

The next bounded test is a realizable small-learning-rate model, not another projection sweep. `critic-small-step-pilot-2026-10-12` fixes Adam's learning rate to `1e-4 × 0.2768236137675729` for the candidate, alongside the canonical baseline, with the same 128 roots, 32 updates and all retention checks. This train-derived factor is not claimed to match final output movement exactly. A preflight now checks that the control-design file exists and hashes its executable. The complete analysis and verification chain remains automatic; no extra rollout data or promotion. Identity: `0c746b66350d1252a2d3c5bf7f3b73ccf8e2d7f28df34c7538295e3773412fb2`.

That test and both independent verifiers subsequently completed. Its fixed step-32 MSE is **.015890**, with **65.430%** conditional wins. All retention gates pass, but it improves neither initialization nor canonical conditional wins. Against initialization the descriptive root-plus-suffix interval for the −0.927734 pp difference is [−2.832031, +0.830078] pp. Step 8's higher point estimate is not selected as a replacement endpoint.

The trained small-step model's centered-Q movement is **88.8177%** of the canonical model's, despite using 27.6824% of its learning rate. Its own movement-matched score control yields .015663 MSE and 65.723% conditional wins. Smaller learning rate changes the optimization trajectory; it is not interchangeable with shrinking fixed model outputs. This closes the simple-learning-rate follow-up without a promotion or another rate sweep. The local conflict mechanism has evidence, but neither optimizer intervention has established improved playing strength. Further model changes need a new, independently motivated hypothesis; repeated tuning on these 64 roots is not justified.
