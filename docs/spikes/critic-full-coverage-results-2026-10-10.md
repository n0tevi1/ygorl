# Critic full-decision coverage: gains, battle regression and clipped action ties

The 1,392-game complementary replay is complete. Together with the preceding 144 games it covers exactly the original 1,536-game learner-shift panel, with no duplicate or missing IDs: 365,656 learner decisions. Both sealed runtime identities, report identities, all per-game prediction hashes and source record/replay hashes pass. No engine errors; maximum chosen-Q/V path difference 3.88e-7. Sixteen original first-idle point estimates reproduce the earlier complete learner-shift analysis within 4.20e-9.

Completion `e7c6fda2ef6cb77acae3b630432818fbf82c9bbbb49e0d8fdb18230dc718aba4` was acknowledged after verification. Earlier STOP evidence remains preserved. Reproducible scripts/results are under `out/research/critic-all-decision-coverage-2026-10-10`: `analyze-union.py`, `union-analysis.json`, `decompose.py`, `decomposition.json`, `verify_slices.py`, `slice-verification.json`.

## Full-panel results supersede the 144-game subset

Equal-game mean of chosen-Q and V squared error against terminal return, averaging states within each game first:

| Panel | Games | Control | Terminal refresh | Anchored refresh |
|---|---:|---:|---:|---:|
| Broad/fixed actor | 512 | .611149 | .602120 | .603323 |
| Broad/expanded actor | 512 | .589942 | .579101 | .579331 |
| Runick/fixed actor | 256 | .863645 | .822181 | .826269 |
| Runick/expanded actor | 256 | .784590 | .749552 | .752636 |

Runick terminal-refresh differences are -.041464 (95% interval [-.07959, -.00714]) and -.035038 ([-.06729, -.00393]), about 4.8% and 4.5% reductions. Anchored improves both Runick panels as well. Broad terminal-refresh intervals cross zero; anchored broad improvements are small, with pointwise intervals just below zero. Familiar overall intervals cross zero.

These are 4,000 opening-cluster bootstrap replicates, preserving correlated opponent/seat observations within a panel. Intervals are exploratory, unadjusted for multiple slices. This expands coverage of the same evaluation games; it is not another independent game confirmation or evidence of playing-strength improvement. The earlier 144-game subset's adverse aggregate results do not persist across the complete panel.

## A specific battle-decision regression survives decomposition

There are 249,458 chain and 15,960 battle-command decisions. Restricting to more than one represented action removes forced prompts: 15,957 battle decisions remain. General choice-only Runick Q and V errors improve, so aggregate gains do not arise solely from forced-pass decisions.

However, broad battle-choice Q error worsens:

| Panel | Control Q MSE | Terminal Q MSE | Anchored Q MSE | Anchored paired delta, 95% interval |
|---|---:|---:|---:|---|
| Broad/fixed | .620667 | .626871 | .635298 | +.014632 [.002847, .026648] |
| Broad/expanded | .562117 | .575620 | .582440 | +.020323 [.008586, .032641] |

The anchored relative increases are approximately 2.4% and 3.6%; V also worsens in these slices. Terminal Q worsens for broad/expanded (+.013503, interval [.000689, .027112]); its fixed-actor interval crosses zero. Runick battle Q and V instead improve. Thus the observed problem is context-dependent retention, not a universal battle-head failure. The refresh training's first-idle coverage is a plausible contributor, but these evaluations alone do not establish its causal contribution.

## State-only calibration can erase action distinctions

For multiple-choice battle prompts, the resource/opponent residual plus clipping produces fully saturated, tied Q menus:

| Estimator | All actions tied at +1 or -1 | Eligible battle decisions | Fraction |
|---|---:|---:|---:|
| Control | 3,353 | 15,957 | 21.01% |
| Terminal refresh | 2,922 | 15,957 | 18.31% |
| Anchored refresh | 3,182 | 15,957 | 19.94% |

The frozen raw critic has zero flat menus in these slices. These are descriptive decision counts, not equal-game inference. A state-only additive residual mathematically preserves within-state Q differences **before clipping** and cannot improve their strict ranking; clipping can turn different scores into ties. It may still improve value calibration. This mechanism is separate from the battle-MSE regression and does not prove that the preserved raw ordering is strategically correct.

`all_action_values(..., clip=False)` now exposes calibrated unbounded values for ranking diagnostics, retaining zero output on masked rows. The default `clip=True` preserves prior calibration results. Tests check exact pairwise differences before clipping, saturated ties under the default, and masked-row behavior. This is a research tool, not a production critic or PPO change. An unbounded score must not be misreported as a calibrated probability.

## Next experiment is running

`critic-battle-ranking-2026-10-10`: 16 source roots, one per panel × opponent × own/face role; deterministic hash ordering and first battle menu with 2–6 supported choices. Selection never consults game outcome or critic error. Every supported legal action gets 16 new shared suffix RNG streams (201–216), with unchanged source learner/opponent policies. Sixteen original-action, original-RNG controls run first: total **896 games**. A control and a changed-stream branch passed cold replay preflight; the live study passed all 16 controls before starting interventions.

Each root's frozen/control/terminal/anchored Q and V, raw Q, original behavior probabilities and public options were scored and sealed before the continuation trials. Compare predicted pairwise ordering/margins to paired terminal-return estimates; report clipped ties separately. The same hidden root state is held fixed across actions; only the private critic sees privileged inputs. This is a conditional diagnostic at 16 roots, not population policy evaluation. Sixteen trials can be inconclusive; do not export automatic labels from noisy differences.

Two workers, four-hour service cap, independent systemd watchdog and completion/error delivery are active. No model is trained or promoted by this experiment. Investigation of action discrimination must precede using this calibrated critic as a tactical chooser.

## Declaration coverage remains an implementation blocker (#319)

All **77** capacity overflows are 9,226-choice `AnnounceCard` menus, in **38/1,536 games** (2.47%). They are **77/246 declaration decisions** (31.30%). No other decision type overflowed in this panel. These rates describe this deck/policy panel only.

The full-menu prototype from #320 solves research scoring and demonstrates legal tail response mapping. Production encoded collectors, rollout action storage, PPO probability recomputation and evaluation still need one consistent variable-candidate interface. Do not close #319 or silently change only evaluation. Frozen historical studies retain their original menu semantics.

Validation: 28 declaration/critic probe tests passed; full Ruff lint/format and diff checks passed. Experiment source/runtime and existing model weights remain unchanged.
