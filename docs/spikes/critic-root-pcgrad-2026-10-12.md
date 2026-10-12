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

Reject the candidate and retain its negative evidence. The next mechanism check should operate on the realized optimizer direction and include retention, while measuring whether avoiding conflicts merely collapses the update toward zero. It must precede another training experiment; changing projection strength or order until the reused panel looks better is not a valid follow-up. Issues #61/#83/#36 remain open.
