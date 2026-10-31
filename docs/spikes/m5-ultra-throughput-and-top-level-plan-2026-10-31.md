# Apple M5 Ultra Throughput & Top-Level Agent Training Plan (2026-10-31)

## 1. Scope & Context

This document synthesizes:
1. The complete opencode research and engineering history on the **Apple M5 Ultra** machine (30 logical CPU cores: 10 Super + 20 Performance, 96 GB Unified Memory, Apple GPU via PyTorch 2.14 MPS, Oct 29–31, 2026, session `ses_e779f0a8affeLNEtw1qAQ59wWq` and its 13 subagent investigations: `first-seat`, `fs2/A–D`, `fs2/R1–R5`, `fs2/GIL`, and intervention postmortems `fs-bc`, `fs-pun`, `fs-aux`, `fs-aux-branch`).
2. New micro-benchmarks and end-to-end `bench_train.py` experiments conducted on 2026-10-31 isolating the exact kernel-launch, batch-sizing, snapshot-fragmentation, and window-history bottlenecks on M5 Ultra MPS.
3. A concrete 4-pillar training pipeline plan to reach top human-level Master Duel play (`md-2026-09` 20-deck meta): **first-player (`seat 0`) win rate exceeding second-player (`seat 1`) win rate (>60% vs ~42.9% today)** and **faster game closure (~3.8–4.5 turns vs ~5.6 turns today)**.

---

## 2. Audit of Opencode History & Currently Running Jobs (Oct 29–31, 2026)

### 2.1 Completed Platform & Throughput Work in Opencode
- **macOS / MPS Port**:
  - Removed GNU-only linker flags (`--whole-archive`) on AppleClang in `CMakeLists.txt`.
  - Switched accumulator statistics dtype from `float64` to `float32` on MPS (`src/ygorl/train/ppo.py`).
  - Added `torch.mps.empty_cache(); torch.mps.synchronize()` after every `learner.update()` in `src/ygorl/train/trainer.py` (~0.4 ms overhead) to work around the PyTorch 2.14 Metal caching allocator deadlock (`pytorch/pytorch#201378`).
  - Forced `workers = 1` on `mps` in `Trainer._evaluate` to avoid Metal dual-process fork/spawn hangs.
- **Learner Update De-Synchronization (`src/ygorl/train/ppo.py`)**:
  - Converted `isfinite(loss)` and `approx_kl > 1.5 * target_kl` from inline host reads to on-device predicates read once after the `clip_grad_norm_` sync, speeding up `128×128` updates by **+24%** (`2.47s -> 1.57s`, `3,450 -> 4,279 rows/s` on `d64×1` bench net).
- **LineLibrary Warm Cache (`src/ygorl/train/lines.py`)**:
  - Added `LineLibrary.load_cached` keyed by `(mtime, env_stamp, vocab, event_length)` under `out/linecache/`, reducing startup line validation from **224s to 2.5s**.
- **Collection Loop Profiling & Single-Thread Opponent Offload (`--opp-cpu-threads 1`)**:
  - `cProfile` on a production `d128×2` arm (`128×128`, `min_batch=96`) showed C++ `EncodedVecEnv` stepping takes only **2%** of collection time, while small-batch snapshot opponent forwards (`~394 calls/update`, avg `~48 rows/call` across 8 snapshots) plus `collate` (`2,390 calls × 2.7 ms`) took **58%** (and 95% together with learner forwards).
  - Moving snapshot opponent inference to 1 background CPU thread (`--opp-cpu-threads 1`, with CPU module copies built on the main thread and `torch.multinomial` kept on the main thread for bitwise RNG equivalence) reduced collection wall time from `18.1s -> 11.4s` and lifted single-arm throughput from **~500 to ~1,080 rows/s (2.16×)**.
- **Why Multi-Thread CPU / C++ `ygorl._infer` / Sharding Did Not Help in Production**:
  - `--opp-cpu-threads 2/3/6` degraded collection time to `13.6s / 16.4s / 27.7s`.
  - `ygorl._infer` (`csrc/infer.cpp`, libtorch C++ traced `ActPath` runner with `py::gil_scoped_release`) achieved **14,170 rows/s (3.7× over eager)** in isolated 4-thread CPU benchmarks (`out/research/fs2/GIL/report.md`), but in the real loop alongside main-thread MPS learner forwards took `12.0s / 15.8s / 16.9s` at 1/2/3 threads.
  - **Root cause**: Concurrent multi-thread CPU ATen dispatch (even with the Python GIL completely released in C++) contends with the main thread's MPS command encoding across PyTorch's global dispatcher, `c10` refcounts, and Unified Memory SLC fabric, slowing main-thread MPS forward passes by **2.84×** (`4.9 ms -> 14.0 ms`).

### 2.2 Critical Bugs Found in the Live `fs-aux` and `fs-aux-branch` Runs
Auditing the two 300-minute runs launched at the end of the opencode session (`out/train/fs-aux` and `out/train/fs-aux-branch`) uncovered three issues:
1. **`--aux-coef` was omitted (defaults to `0.0`)**:
   - Both processes were launched with `--critic-aux 4` (adding the 4th `first_hand` target in `src/ygorl/train/aux.py`) **without `--aux-coef`**.
   - In `PPOLearner.update` (`src/ygorl/train/ppo.py:597`), `aux_targets` and `aux_loss` only execute when `cfg.aux_coef > 0`. Both runs therefore log `'aux_loss': 0.0` on every update—`fs-aux` is running as a pure no-op duplicate of `combined-ctrl`.
2. **Critic-only aux heads cannot move the actor under `--separate-critic`**:
   - With `--separate-critic` (`shared_backbone=False`), `ActorCritic.critic_trunk` is an independent `PolicyNet` copy (`src/ygorl/nets/actor_critic.py:101`). Auxiliary heads on `Critic` shape `critic_trunk` features without altering the scalar TD return targets `q_targets` / `v_targets` or `actor` gradients.
   - Furthermore, predicting next-turn *hand size* (`first_hand`) does not distinguish between holding live handtraps after building a 2-negate board vs prematurely passing Turn 1 after 3 actions with 3.85 cards in hand (`fs2/D`).
3. **`--branch-windows` disables `--opp-cpu-threads 1` (`src/ygorl/train/rollout.py:1210`)**:
   - `threaded = self._opp_executor is not None and not branch_groups and not look_groups` forces every collection round with a branched window back onto synchronous main-thread opponent evaluation, cutting `fs-aux-branch` throughput from ~1,000 rows/s to ~597 rows/s while `branch_ce_margin = 0.5` leaves only `2–4` active `branch_ce_rows` per 16,384-row update.

---

## 3. Part 1: Squeezing Maximum Throughput on Mac M5 Ultra

### 3.1 Micro-Benchmark Diagnosis (`2026-10-31`)

#### A. Operator Dispatch Count per Forward (`d128×2`)
Instrumenting `TorchFunctionMode` on `ActorCritic(d_model=128, board_layers=2, history_layers=2, shared_backbone=False)` at mid-game trimmed widths (`n_cards=28, n_actions=14, n_events=36`):
- **`ActorCritic.policy_logits(obs)`**: **477 ATen function calls** per forward pass:
  - `history` (`TransformerHistory` + `EventEmbedding`): **241 ops** (50.5% of all ops; **11.16 ms / 55.4% of CPU time** at `B=64` vs only `3.36 ms` for the 2-layer `BoardEncoder` Transformer!).
  - `board` (`BoardEncoder` + `CardEncoder` + `GlobalEncoder`): **134 ops** (`4.77 ms` at `B=64`).
  - `actions` (`ActionEncoder`): **52 ops** (`1.63 ms` at `B=64`).
- **`ActorCritic.forward(obs, priv)`**: **1,078 ATen function calls** per forward pass (`privileged` adds **80 ops** due to a 5-iteration Python loop over `P_WIDTHS` zones).

#### B. Why `TransformerHistory._run` Takes 55% of Forward Time (and the Exact Fix)
In `src/ygorl/nets/history.py` (`lines 103–107` and `176–217`), every forward call—even in **window mode (`state=None`)**, which is 100% of PPO collection, opponent acting, and learner updates:
1. Allocates 5 zero tensors on device in `initial_state(b, x.device)`.
2. Recomputes a dynamic 4D `[B, H, T, T]` causal + padding ALiBi tensor (`cumsum`, `unsqueeze`, subtraction, `masked_fill`).
3. Calls `_next_state(...)`, which runs `argsort(dim=1, stable=True)` and per-layer `torch.gather` on `[B, T, d]` to build a streaming `TransformerState` that is **immediately discarded**.
4. Runs `self.norm(raw)` on all `[B, T, d]` history tokens even though `PolicyNet._features` only uses `summary` (`[B, d]`).

**Mathematical Identity for Window Mode (`state=None`)**:
Because valid event tokens always occupy a contiguous prefix `0 .. n_b - 1` (`docs/encoding.md`) and causal attention restricts keys to `k <= q`, **every causal key `k <= q` of any valid query `q < n_b` is automatically valid (`k < n_b`) with exact relative distance `q - k`**. Since `_last` only reads `h[b, n_b - 1]` (a valid query), the 4D per-batch mask `[B, H, T, T]` can be replaced by slicing a **single precomputed `[1, H, 64, 64]` static causal ALiBi buffer**, and `_next_state` + `self.norm(raw)` can be skipped entirely when `state is None`.
- **Verified**: `max |Δsummary| = 0.0` (bitwise identical on both CPU and MPS).

#### C. M5 Ultra MPS Batch-Size Scaling Curve (`ActorCritic` `d128×2` with `collate`)
Measured on M5 Ultra MPS (`25` synchronized iterations after warmup):

| Batch Size `B` | MPS Latency (`ms`) | Throughput (`rows/s`) | Scaling vs `B=8` |
|---:|---:|---:|---:|
| `8` (typical fragmented snapshot batch) | `16.51 ms` | `485 rows/s` | `1.0×` |
| `16` | `16.18 ms` | `989 rows/s` | `2.0×` |
| `48` (typical learner batch at `128×128`) | `21.64 ms` | `2,218 rows/s` | `4.6×` |
| `96` | `19.47 ms` | `4,931 rows/s` | `10.2×` |
| `192` | `24.66 ms` | `7,785 rows/s` | `16.1×` |
| `384` | `28.55 ms` | **`13,452 rows/s`** | **`27.8×`** |

**Key Insight**: Going from `B=8` (`16.51 ms`) to `B=384` (`28.55 ms`) processes **48× more decisions in 1.73× the time** because M5 Ultra MPS is entirely command-buffer/kernel-launch latency bound at `B <= 96`.

### 3.2 End-to-End `bench_train.py` Ablation on M5 Ultra MPS (`2026-10-31`)
Tested with `tools/bench_train.py` (`d128×2`, `--separate-critic`, `--selfplay-fraction 0.5`, `--opp-cpu-threads 1`, fixed `T × B = 4,096` rows/update while 2 background jobs were active on MPS):

| Configuration (`T × B = 4,096`) | `collect_s` | `update_s` | `step_s` | `decisions/s` | `rows/s` | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_ratio_pool8` (`64×64`, `min_batch=48`, 8 active snapshots) | `8.21 s` | `2.83 s` | `11.29 s` | `825` | `363` | `1.00×` |
| `wide_256x16_pool8` (`256×16`, `min_batch=192`, 8 active snapshots) | `4.64 s` | `3.35 s` | `8.24 s` | `1,387` | `497` | `1.37×` (`1.77×` collect) |
| **`wide_256x16_pool2` (`256×16`, `min_batch=192`, 2 active snapshots)** | **`3.02 s`** | **`2.31 s`** | **`5.55 s`** | **`2,145`** | **`738`** | **`2.03×` (`2.72×` collect)** |
| `wide_256x16_bf16` (`256×16`, `min_batch=192`, 2 active + `--learner-precision bf16`) | `3.02 s` | `3.95 s` | `7.21 s` | `2,197` | `568` | `1.56×` (BF16 slower) |

### 3.3 Prioritized Throughput Engineering Plan
1. **Use Wide Rollout Geometry (`--envs 512 --steps 32 --min-batch 384`)**:
   - Keeps `T × B = 16,384` rows/update identical while cutting collection rounds per update by **4×** (`~220 -> ~55`) and increasing learner/opponent batch sizes by **4×** (+1.77× faster collection with zero code change; ~1.2 GB C++ env RAM out of 96 GB).
2. **Active Snapshot Sub-Cohort (`--pool-active-k 2` in `SelfPlaySchedule` / `RolloutCollector`)**:
   - Retain a 16-snapshot `OpponentPool` for league diversity, but draw $K=2$ active snapshots per rollout segment so all pool-opponent slots share 2 batched forwards per round instead of 8 tiny forwards (+1.54× additional collection speedup; **2.72× combined collection speedup**).
3. **Bitwise-Identical Network Fast Paths**:
   - **Window-mode `TransformerHistory` fast path** (`src/ygorl/nets/history.py`): Precompute static `[1, H, 64, 64]` causal ALiBi buffer and skip `_next_state` + `self.norm(raw)` when `state is None`.
   - **Trim-before-head in `ActorCritic.policy_logits`** (`src/ygorl/nets/actor_critic.py:112`): Call `self.actor._features(trim_padding(obs))` and `_pad_to(self.actor.logits(f), width, value=MASKED_LOGIT)` to skip 4 `_pad_to` tensor allocations and score only `n_actions` (~12) instead of `128` (**1.40× CPU speedup** on small opponent batches: `13.61 ms -> 9.71 ms`).
   - **Zone-concatenated `PrivilegedEncoder.forward`** (`src/ygorl/nets/actor_critic.py:64`): Concatenate `P_WIDTHS` zones along dim 1 before embedding lookup (`80 ops -> 12 ops`).
4. **Restore Async CPU Opponent Threading Under `--branch-windows` and `--lookahead-share`**:
   - In `RolloutCollector._forward_opponents` (`src/ygorl/train/rollout.py:1210`), submit combined `[*main, *extra, *look]` CPU batches to `self._opp_executor` and split the returned probabilities in `_step_opponents`, recovering ~35% throughput on branch-enabled runs.
5. **Keep FP32 on MPS (`--learner-precision fp32`, no `--bf16`)**:
   - PyTorch 2.14 MPS `bfloat16` autocast slows `PPOLearner.update` by 1.71× (`2.31s -> 3.95s`) due to cast kernel insertion.

---

## 4. Part 2: Tuning the Pipeline for Top-Level Play (Seat 0 Win Rate & Turn Count)

### 4.1 Root-Cause Synthesis Across Audits `A–D` & `R1–R5` (`md-2026-09`)

| Diagnostic Audit | Key Quantitative Finding | Mechanistic Takeaway |
|---|---|---|
| **`fs2/D` & `fs2/R4`** (Turn 1 Board & Interrupt Quality) | Going first, the policy summons **1.90 Extra Deck monsters/T1** (vs `0.62` ref), **but 91% are non-interruption bosses/intermediates** (*Accesscode*, *Kagari*, *Dingirsu*, *Galatea-I*, *Kitkallos*), sets **0.46 dead normal spells/T1** (`adv/sset = -0.215`), burns hand to **2.12–2.44 cards** (`0.33` handtraps vs `0.69` ref), and ends T1 with **0.79 total interruptions** (vs solver **2.33–3.15**). On 160 uncontested solver hands, full target board completion is **6.2% (10/160)** (`@szone:fd = 0%`). | The policy either stops after ~3 actions (102/160 hands) or over-extends into zero-threat cards that consume handtraps and combo starters without producing opponent-turn disruptions. |
| **`fs2/R1` & `fs2/A`** (Turn 2 Board Depreciation) | Only **34% of Seat 0's T1 field cards survive Turn 2** (vs 47% for Seat 1's T2 board); Seat 1 runs a +16-step longer combo and deals **2,140–3,127 combat damage on T2** (17% T2 OTK in Maliss/Orcust). By end of T2, Seat 0 is at **−2.42 card advantage** and **−2,033 LP deficit** (trailing in 73–79% of games). | Because the T1 board lacks negates/interruptions (0.79 vs 2.33+), Seat 1's 6-card hand + Battle Phase access easily wipes Seat 0's over-extended board. |
| **`fs2/R5`** (GY / Banish Resource Engine) | Modern `md-2026-09` decks rely on GY/banish recursion, yet the policy triggers GY special summons and banish recovery at only **31%–47% of reference rate** (failing to recur *Sky Striker Ace - Raye* and *Maliss* banish traps/monsters). | Cards sent to GY on T1/T2 become permanently dead instead of generating Turn 3 follow-up. |
| **`fs2/C` & `fs2/R3`** (Chain Window Counterfactuals) | On the first opponent chain window (`k=1`), the policy holds *Maxx "C"* / *Mulcharmy Fuwalos* **52%–55% of the time** when immediate activation yields **+19 pp** counterfactual win value. | Multi-card response prompts suffer from dilute terminal credit and passive `pass` bias (`rows/pass` is 2.3× `rows/chain`). |
| **`fs2/B` & Plan §13.2** (Turn 3+ Combat & Lethal) | Clean empty-board lethal is converted 100% of the time, but on **contested boards** the policy refuses to attack with attack-position monsters **30%–38% of the time** (vs `0%–6%` for strong anchor) and loses **20%–25% of games after holding a ≥4000 LP lead**. Best-of-8 lookahead on attacking turns (`t=2,3`) gives **+10.2 to +11.8 pp WR** and **doubles Turn-3 kills (`12.5% -> 24.5%`)**. | Combat hesitation against non-empty boards is the primary driver of inflated turn counts (`5.6` turns instead of `~4.0` turns) and thrown winning positions. |

### 4.2 Why Previous Interventions (`fs-bc`, `fs-pun`, `LA`) Underperformed
1. **`fs-bc` (`--kl-prior 0.05 --kl-prior-turns 1`)**:
   - `PPOLearner.kl` averages over all rollout rows, and Turn 1 is only 16.6% of rows; combined with `target_kl` early stopping, the effective gradient weight of `kl_prior` was diluted **15× (to 0.3%)**.
   - Worse, **37/160 solver trajectories in `lines_train.jsonl` fired handtraps/effects during empty Turn 1 Draw Phase windows**, teaching the BC prior to waste handtraps on Turn 1.
2. **`fs-pun` (`--line-first-share 0.75`)**:
   - When the learner interrupts a scripted opponent line at a window without a pre-computed recovery line, the opponent falls back to a weak snapshot that bricks—rewarding indiscriminate early firing while punishing Turn 1 boards whenever a broken-line opponent high-rolls.
3. **`LA` (`--lookahead-share 1.0` in Stage 4)**:
   - Lookahead opponents punished weak Turn 1 boards (−5 pp training WR), but without positive supervision toward interruption-producing Turn 1 lines or turn-boundary potential shaping, terminal ±1 PPO could not discover 25-step negate combos from scratch.

### 4.3 Four-Pillar Pipeline Tuning Plan

1. **Pillar 1 — Clean Turn-1 Solver Line Supervision (`--line-ce-coef` + `--line-bonus-coef`)**:
   - Strip the 37/160 Draw-Phase waste prefixes from `lines_train.jsonl`.
   - Replace diluted `--kl-prior` with `LineLabels` (`--line-ce-coef 0.15 --line-bonus-coef 0.30`), which normalizes cross-entropy strictly over matched solver-line rows (`per_row.sum() / has.sum().clamp(min=1)` in `PPOLearner.line_ce`) and directly supervises `@atk` and `@szone:fd` end-board completion (targeting `>60%` completion vs `6.2%` today).
2. **Pillar 2 — Turn-Boundary Survival Potential Shaping (`--critic-phi` + `--phi-shape-coef`)**:
   - Enable the implemented potential-based shaping head (`--critic-phi --phi-coef 0.2 --phi-shape-coef 0.15 --phi-shape-first-seat-only`) in `src/ygorl/train/aux.py:phi_shaping`, which adds $r'_t = r_t + \gamma \Phi(s_{\text{next turn}}) - \Phi(s_{\text{turn}})$ directly to rollout rewards at turn boundaries.
   - Pass `--aux-coef 0.2` whenever `--critic-aux 4` is used so `aux_loss` actually trains.
   - *(Optional experimental extension, requires design C3 update before default adoption)*: Add a Turn-1 End-Phase live-interruption count term (counting active Extra Deck quick-effect/negate monsters + live set traps/quick-plays + retained handtraps, minus face-down normal spells) to eliminate the 91% dead Extra Deck summon rate and `0.46/game` dead spell sets.
3. **Pillar 3 — High-Signal Chain Window Counterfactuals (`--branch-windows` + `--response-weight`)**:
   - Lower `--branch-ce-margin` from `0.5` to `0.15` (in board points) so clear counterfactual wins are not discarded as ties (`branch_tie_share` was 50% in `fs-aux-branch`).
   - Run `--branch-windows --branch-k 4 --branch-rollouts 4 --branch-fraction 1.0 --branch-mode replace --branch-ce-coef 0.10 --branch-ce-margin 0.15 --branch-horizon board` alongside `--response-weight 2.0 --first-seat-weight 1.4`.
4. **Pillar 4 — Turn 2/3 Combat & Lethal Lookahead Distillation (Shortening Game Length)**:
   - Use rollout lookahead (`Forks` in `src/ygorl/train/rollout.py`) on the learner's attacking turns (`t=2,3`, `K=4`) to evaluate entering Battle Phase + attacking vs passing on contested boards, writing the best candidate into `_branch_ce` targets. This directly captures the **+10.2 to +11.8 pp** combat lookahead headroom (`first-seat-interruption-plan §13.2`), doubles Turn-3 lethal conversion (`12.5% -> 24.5%`), and brings mean game length down from `~5.6` turns to `~4.0` turns.

---

## 5. Status (2026-11-01)

Implemented and measured (docs/benchmarks.md "M5 Ultra 训练管线修复与宽几何实测"):
- §2.2.1 fixed by validation: `critic_aux > 0` without `--aux-coef > 0` (and `--phi-shape-coef` without a trained
  `--critic-phi`) now fail at configuration time. The `first_hand` target also read the wrong hand when the next-turn row
  was from the opponent's view; fixed.
- §2.2.3 / §3.3.4: threaded CPU opponents now also serve branch and look-ahead decisions (bit-identical on a CPU learner).
  Branch arm: ~600 -> 1,869 rows/s.
- §3.3.1 + §3.3.2: `512×32, min_batch 384` + `--pool-active-k 2`: 1,117 -> 1,959 rows/s on the production config
  (with `lam = 0.5` the 32-row segments barely change the advantages). This is the `tools/hero_ppo.sh` configuration.
- §3.3.3: window-mode history fast path (bit-identical logits; lazily built streaming state) and trim-before-head
  `policy_logits`. The privileged-encoder concatenation was not done (critic-only, small).

Corrections to §4:
- Pillar 1 describes `--line-ce-coef` / `--line-bonus-coef` as supervising the learner's turn-1 end board. They do not:
  both act only on the learner's *response* rows at labelled Ash Blossom windows of the scripted opponent line
  (docs/training.md "标签的监督项"). Turn-1 board quality has no direct supervised signal in the current pipeline.
- Pillar 3: with the margin lowered, branch-CE still sees only 2–11 target rows per 16,384-row update on the 20-deck
  meta (few windows have recovery lines). It cannot be a main 先手 lever until windows are far denser
  (e.g. `--branch-cards` on covered hand traps, or `--no-branch-require-recovery` with a fraction).
