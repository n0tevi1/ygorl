# Training infrastructure audit — 2026-10-08

## Scope and observed bottleneck

Tracking: [#60](https://github.com/n0tevi1/ygorl/issues/60). Base commit
`9fecfe686c4503b7115efbea77a3a95416d5fdbf`, environment `md-2026-09`, AMD Radeon
8060S, PyTorch `2.14.0+rocm7.2`. The ongoing three-seed experiment stays bound to
its frozen source/native library and FP32 configuration. All probes and edits use
the separate `codex-training-infra` worktree.

Read-only timing snapshot (50 completed updates per branch; these windows are
different policies/progress, not a controlled cold-versus-warm comparison):

| Branch/window | Mean collection | Mean PPO update | Mean step |
|---|---:|---:|---:|
| seed 0 cold, 463–512 | 10.56 s | 18.45 s | 29.61 s |
| seed 0 warm, 211–260 | 13.12 s | 19.05 s | 32.83 s |

PPO accounts for roughly 58–62% of step time. Each update has 16,384 rows and
normally 16 accepted minibatches (two epochs of 2,048). Some collection times
increase near concurrent evaluation/retention; contention is plausible, but these
observations do not distinguish it from native long-tail work. One GPU-use
snapshot was 83%; it is not a utilization time series or proof of free capacity.

## Initial implementation scope: eliminate redundant padding scans

`ActorCritic.forward` already calls `trim_padding` before evaluating the actor and
critic heads. Its private `_forward` then calls `PolicyNet.features`, which scans
the same masks again. Each scan reads up to three device scalars into Python;
shared-backbone inference therefore performs six prefix reductions instead of
three, and an independent critic trunk adds another three. Actual timing gains
must be measured on the current hardware rather than inferred from this count.

The private actor-critic path will use `PolicyNet._features` on the batch that its
caller already trimmed. The public `PolicyNet.features` and actor-only inference
remain unchanged. The outer actor-critic call still restores the original action
width, including masked logits and zero Q padding. No model parameters, checkpoint
format, observations, precision, random draws, PPO budgets or environment behavior
change. Both shared and separate critic backbones must preserve logits, Q, V and
parameter gradients, including when padding trimming is disabled for debugging.

Validation compares the optimized path with the previous nested-trimming path on
identical inputs and weights, across history encoders and backbone configurations.
Existing padding and full PPO update equivalence tests remain applicable. Work is
isolated from the frozen source and native library used by the ongoing experiment.

## Benchmark timing contract

`bench_train.py` currently enables `PPOLearner.timing` unconditionally. That mode
synchronizes the GPU at every profiling boundary and can obscure the benefit of
removing synchronization. Make detailed profiling explicitly opt-in with
`--profile`; default runs measure throughput without those extra boundaries.
Record `profile` in JSON so instrumented results are identifiable, and report
evaluated/accepted minibatches, early-stop rate and approximate KL so less learning
cannot masquerade as faster execution. Existing section timings remain available
when requested. This changes benchmark
instrumentation only, not training configuration or PPO work.

## Sampling transfer reuse

Learner collection previously copied the same probability tensor from the device
to the CPU twice: once for CPU multinomial sampling, then again for rollout
storage. Retain the first copy for both uses. Sampling stays on the CPU with the
same generator and operation ordering; logits, Q and V follow the existing path.
A regression compares the previous sampler's actions, generator state and stored
training targets against the optimized collector. This is independent of the
padding optimization and should be measured separately.

## Fixed-input GPU verification

`benchmark.py` compares the previous and optimized private forward paths with
identical actual training observations and cold-u512 weights. Six alternating
A/B blocks, whole-call synchronization boundaries, no per-stage profiling. The
GPU is shared with the ongoing experiment, so timings are exploratory.

| Work | Previous median | Single-trim median | Prefix reductions |
|---|---:|---:|---:|
| 128-row inference | 22.16 ms | 22.18 ms | 6 → 3 |
| 2,048-row forward + backward | 1.631 s | 1.617 s | 6 → 3 |

Outputs match exactly; maximum gradient absolute difference is `1.91e-6`, within
the recorded all-parameter tolerances. The reduction in scans is confirmed, but
there is **no established end-to-end throughput gain**. Probability-copy reuse is
covered by sampling equivalence tests, not included in this forward microbenchmark.

## Precision probe

Historical BF16 timing cannot settle the question: the previous autocast scope
regression left PPO updates in FP32 ([prior correction](ppo-autocast-2026-10-05.md)).
`precision.py` explicitly uses autocast with parameter caching disabled, two warmup
calls and six alternating timing pairs on the same 2,048-row training input.

FP32 forward/backward median **2.060 s**, BF16 **1.045 s** (about 1.97×), with finite
gradients. BF16 actor/V outputs are actually BF16 (Q output is FP32); the maximum
legal-logit difference is 0.2198 and 12/2,048 argmax decisions differ. There are no
optimizer steps in this microbenchmark. Different shared-load timing windows must
not be compared against the previous table as another optimization effect.

This supports a controlled learner-precision experiment, not switching the formal
run or claiming equal playing strength. Full PPO work counts and subsequent
independent strength/behavior measurements are required before adoption.

`ppo_precision.py` then collected one fresh FP32 rollout from the same checkpoint
in the isolated worktree: 16,384 rows, 58 completed healthy games, zero truncations
or errors. It restored the same learner/reference/Adam state and CPU/GPU RNG for
each FP32/BF16/BF16/FP32 update. Collection is shared, so this probes **learner-only
BF16**, not BF16 acting or end-to-end self-play. The private probe uses the latest
native cancellation guard; it is not another result from the frozen formal study.

| Learner precision | Update times | Mean | Accepted/evaluated minibatches | Approx. KL |
|---|---|---:|---:|---:|
| FP32 | 27.17, 25.98 s | 26.58 s | 16 / 16, both runs | 0.003893 |
| BF16 | 23.89, 19.22 s | 21.55 s | 16 / 16, both runs | 0.003948–0.003977 |

All parameters remain finite; none of the four updates early-stops. Final CPU/GPU
RNG hashes match across runs. This removes fewer-minibatches as the explanation
for the observed difference: learner elapsed time is about 19% lower (1.23× speed).
Only two measurements per precision on a shared GPU are available, with noticeable
variation. There is no confidence interval or established learning-quality result.
No resulting probe weights are promoted into the running experiment.

One additional synchronized stage profile per precision (`learner-profile.json`)
locates about 62–63% of profiled time in backward, 19–23% in forward, and 13–16%
in reference scoring; targets/optimizer/reporting are much smaller. The profile
runs have different shared load and add synchronization, so their total times
(37.52 versus 12.94 s) **must not replace the uninstrumented ABBA comparison**.
This supports prioritizing network compute over checkpoint I/O or scalar-reporting
micro-optimizations.

## Remaining priorities

1. Recheck BF16 at equal PPO work and then equal training budget. Keep autocast
   parameter caching disabled across Adam updates; retain the pre-step KL guard.
   The current `TrainConfig.bf16` affects both acting and learning. To test the
   learner-only result above, first provide a separately registered learner
   precision option with FP32 collection, then compare matched seeds/rows against
   FP32 using independent strength and behavior panels. A global `--bf16` run
   would test a different intervention.
2. Profile minibatch gathering, forward/backward, metric extraction and transfers.
   PPO extracts several scalar diagnostics separately each minibatch; batching
   reporting-only reads may help. The KL scalar needed to reject Adam must remain
   immediately available. CPU-derived padding lengths could remove more scans,
   but require an explicit batch metadata/lifetime contract.
3. Record learner/opponent inference batch sizes: eight snapshot policies fragment
   ready events into small batches. Sweep `min_batch` and native worker count at
   fixed rows and report opponent composition, errors and truncations. Native
   batched receive/step APIs may remove array copies and per-event dispatch, but
   must keep retained observations alive and independent of recycled buffers.
4. Improve checkpoint cadence for future preemptible jobs. The completed cold
   continuation has about 4.07 s between total elapsed time and summed rounded
   step timers: checkpoint I/O is not its major bottleneck. However, endpoint-only
   saves leave roughly 84- and 136-minute gaps. Use the existing immutable Colab
   checkpoint generations with verified off-machine copies at a shorter interval.
   Unfinished native duels are not checkpointed, so recovery is not exact trajectory
   continuation.
5. Evaluation already consumes immutable checkpoints in a separate four-worker
   process pool. Formal `eval_every=0`; moving it out of Trainer is already done.
   Additional GPU workers should be justified by aggregate throughput, not a
   single-job timing. `overlap_collect` changes policy staleness; earlier apparent
   speedups partly came from fewer accepted PPO minibatches and are not free gains.

## Evidence and validation

Local artifacts: `out/research/training-infra-audit-2026-10-08/` contains frozen
timing windows, previous source, reproducible probe scripts, input/checkpoint
SHA256s and JSON results. The probe uses a training split, not evaluation labels.
Tests cover legacy-versus-new output/padding/gradient/RNG behavior across shared
and separate trunks, three history types and trimming switches, plus collector
sampling/generator/row parity. Full-suite and CLI validation are recorded with the
final result below.

Validation completed: 13 targeted new regression cases; formatting/lint and CLI
smokes with/without profiling pass. The full run reports 1,578 passed, three solver
failures and four skips. Those failures used the stale default solver under the
root build directory; one extra skip came from the new worktree's empty LFList
directory. After binding the validated declaration-list solver and copying the
nine pinned LFList fixtures, all 64 solver/LFList/header tests pass. Together these
runs verify **1,590 applicable cases**, with only three expected skips (two opt-in
network tests and one unsupported snapshot case). No product changes were needed
for these fixture corrections. An earlier sandbox attempt was interrupted because
Numba could not write its external cache; the host run uses `/tmp/ygorl-numba`.

Logs: `full-tests-host.log`, `fixture-recheck.log`, `test-environment.json`;
validated native SHA `d91ed6d81325c9e813274abd9b639823120a320c5f4c7bbdc81aca425a323eb4`,
solver SHA `8ef61ea05119e0b2be0c6ea2644e3d8337eff700cf08ed7ed538f9d5d53c6ad0`.
