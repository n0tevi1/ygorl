# Battle critic confirmation: calibration gains did not establish better action selection

The independent 3,536-game confirmation does **not** establish an aggregate action-ranking improvement. Terminal refresh beats the control by only **0.293 percentage points**, with a paired 95% interval of **−1.270 to +1.855 points**. Do not promote the refreshed critic as a better action selector or infer an improvement in full-game playing strength.

## Evidence and primary analysis

Artifact: `out/research/critic-battle-ranking-confirmation-2026-10-10/` (`registration.json`, `root-predictions.json`, `analysis.json`, `analyze.py`). Runtime remains frozen at `ae09594ec0126b5365e3c8311e9ac5c67d23f270`; study identity SHA256 is `afb1cdac0e30dd676b3453eb0dafe3e4cb1259a7b820feb50a162ae14a929813`.

All 16 previously registered roots and all 55 supported legal actions were retained, independent of pilot outcomes. Each action received 64 new common suffix streams (301–364): 3,520 interventions plus 16 original-action/original-RNG controls. The original actor and opponent continue after the root action. All 3,536 games ended normally with a winner. The audit verified sealed inputs/runtime identity, complete task grids, task/result agreement, replay hashes and recorded cold-replay equality. All 16 controls matched the original games. This audit did not independently rerun every intervention.

The preregistered primary compares terminal refresh against control, selecting each root's raw-Q argmax and weighting the 16 roots equally. Bootstrap resampling uses the same suffix-trial indices jointly across actions **and roots**; 50,000 replicates, seed 2026101101. Pilot outcomes are excluded. Roots themselves are fixed, not a random sample of all game situations.

| Critic | Conditional win fraction, raw-Q argmax | Difference from control | Paired 95% interval |
|---|---:|---:|---:|
| Frozen / control | 62.500% | — | — |
| Terminal refresh (primary) | 62.793% | +0.293 pp | −1.270 to +1.855 pp |
| Anchored refresh (secondary) | 62.500% | 0.000 pp | −1.367 to +1.465 pp |

Clipped first-index tie choices yield the same outcomes here. Uniform clipped-tie results are 62.402%, 62.663%, and 62.370% for control, terminal, and anchored, respectively. Two roots have clipped top ties for each fitted critic; raw-Q maxima are unique. Observed-best action gaps remain selection-biased descriptive quantities, not true regret estimates.

The pilot's terminal advantage was +2.344 pp. Its failure to repeat is a reason to retain uncertainty, not to pool the streams until significance appears. No actor, reward, production critic or inference default changes.

Completion event `2ba25235cd4f24fb1780cf88f7accef65f5d0a262781281aa1f06f8403fbeda5` was acknowledged after verification; prior STOP evidence remains intact.

## A persistent tactical misranking

Six of 81 action-pair comparisons survive Holm correction of exact two-sided paired sign tests: two at root325 and four at root1492. Intervals for individual roots are descriptive and unadjusted; the significance statement uses the corrected tests.

Root1492 is a Runick-facing, fixed-learner game against mirror-u528, decision34 on turn2. Both players have 8,000 LP. The learner has Forbidden Droplet available, Kusanagi (1,600 ATK), and Ice Ryzeal (1,700 ATK); the opposing face-up Inspector Boarder has 2,000 ATK. Results with the original policies continuing:

| Root action | Wins / 64 | Control calibrated Q | Terminal-refresh calibrated Q |
|---|---:|---:|---:|
| Activate Forbidden Droplet | 50 | .45523 | .36361 |
| Attack with Kusanagi | 23 | .45722 | .32703 |
| Attack with Ice Ryzeal | 23 | .38857 | .27024 |
| Enter Main Phase 2 | 22 | .46270 | .38862 |
| End phase | 23 | .47578 | .37417 |

The control prefers ending the turn, while both refreshed heads prefer Main Phase 2. All miss the successful effect activation. Droplet versus Main Phase 2 has 30 paired wins in its favor and 2 against, with Holm-adjusted p≈0.00001995; all four Droplet comparisons survive correction.

The local card database says Droplet can halve opposing effect monsters' ATK and negate their effects, with a sending cost. Boarder restricts monster-effect activations. These effects offer a plausible tactical explanation, but the experiment establishes the benefit of this root action **under the subsequent original policies**, not an immediate lethal or a universally correct activation rule. Later cost/target choices and opponent responses still matter.

This root's Q values are inside the clamp range, so clipping cannot explain its ranking error. Equal LP also rules out an LP-difference shortcut as a direct explanation of this example; it does not show the critic has learned LP's value generally. Broader representation, coverage and training-target explanations remain hypotheses. Prior calibration improvements alone did not resolve relative-action value learning.

## Next experiment is running: separate action-conditional data

`out/research/critic-battle-branch-data-2026-10-11` collects **2,312 games**: 40 original controls plus 142 root actions × 16 new paired suffix streams (501–516). Two workers, four-hour cap, fail-closed engine health checks, exact original controls before interventions, cold replay and independent watchdog.

The 40 roots comprise **32 training roots and 8 validation roots**, balanced across broad/Runick corpus × old/mirror opponent × own/face role. They come from the existing `critic-opponent-coverage-data` and `critic-lowdeck-coverage-data` corpora. Broad training remains restricted to the original balanced `draw < 2` subset. A fixed SHA256 order chooses four training and one validation root per stratum, then the first learner battle decision with 2–6 supported actions. Neither outcomes nor critic errors enter selection.

Opening hashes enforce train/validation separation and exclude every reserved opening from the 13 existing critic collection panels from training, including the entire learner-shift evaluation panel and the 16 current ranking roots. Source jobs, traces, replays, results, actor checkpoints and split audit are sealed in the identity. This is separation from held-out data for the proposed critic fit, not a claim that the existing actor never saw related cards or positions.

Identity: `05dbda23192973d22663d71415071c28df2684cb71379890bddffe228641b076`. Service: `ygorl-critic-battle-branch-data-20261011.service`; invocation: `0591d3c3b7d149babd973beabd8909b4`. All 40 original controls passed before the intervention phase began. A separate watchdog tracks invocation, progress, failures and final report identity.

Collection alone is not a trained improvement. Before examining validation outcomes, register the fit protocol comparing selected-action terminal supervision against all-action branch-return / paired relative-action supervision, with the actor frozen and original-corpus anchoring. Fit only on the 32 training roots. Calibration retention, action-ranking uncertainty, and later fresh full-game evaluation are required before promotion. No card-specific hard mask or private information enters the actor.

The production declaration-candidate limit in issue #319 remains unresolved and is outside these supported battle menus. Original runtime, native library, checkpoints and earlier STOP records are preserved.
