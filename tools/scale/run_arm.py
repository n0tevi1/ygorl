"""Resume a PPO checkpoint with config overrides and train in chunks, saving an immutable checkpoint per chunk.

  uv run --no-sync python out/claude-scale/run_arm.py --ckpt CKPT --out DIR --updates 512 --chunk 64 \
      [--adv-keep 0.75] [--bf16] [--freeze-id-embedding] [--envs 1024 --steps 16] [--overlap]
"""

import argparse, json, os, sys, time
from dataclasses import replace
from pathlib import Path

os.chdir(os.environ.get("YGORL_ROOT", "/home/ya0guang/Code/ygorl/.claude/worktrees/claude-rl-review"))
from ygorl.train.trainer import Trainer  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--out", required=True)
p.add_argument("--updates", type=int, default=512)
p.add_argument("--chunk", type=int, default=64)
p.add_argument("--adv-keep", type=float, default=None)
p.add_argument("--bf16", action="store_true")
p.add_argument("--freeze-id-embedding", action="store_true")
p.add_argument("--overlap", action="store_true")
p.add_argument("--envs", type=int, default=None)
p.add_argument("--steps", type=int, default=None)
p.add_argument("--env-threads", type=int, default=None)
p.add_argument("--seed", type=int, default=None)
p.add_argument("--minibatch", type=int, default=None)
p.add_argument("--adv-keep-mode", default=None)
p.add_argument("--entropy", type=float, default=None)
p.add_argument("--response-weight", type=float, default=None)
p.add_argument(
    "--branch-replay",
    default=None,
    help="branch-target dir: after every PPO update, one critic-only MSE step on a batch of branched windows",
)
p.add_argument("--branch-batch", type=int, default=32)
p.add_argument("--branch-lr", type=float, default=1e-4)
p.add_argument("--branch-steps", type=int, default=1)
p.add_argument(
    "--fresh", action="store_true", help="new run: actor warm-started from --ckpt, fresh critic/optimizer/pool"
)
p.add_argument("--separate-critic", action="store_true")
p.add_argument("--critic-hidden", type=int, default=None)
p.add_argument("--critic-warmup", type=int, default=None)
p.add_argument("--critic-lam", type=float, default=None)
p.add_argument("--critic-aux", type=int, default=None)
p.add_argument("--aux-coef", type=float, default=None)
p.add_argument(
    "--set",
    action="append",
    default=[],
    help='TrainConfig override key=value (JSON value), e.g. --set bc_prior="x.pt" --set pin_opponents=["y.pt"]',
)
p.add_argument(
    "--ppo-set",
    action="append",
    default=[],
    help="PPOConfig override key=value (JSON value), e.g. --ppo-set kl_prior_coef=0.5",
)
p.add_argument(
    "--critic-from-ckpt",
    action="store_true",
    help="fresh separate-critic run: critic trunk <- actor trunk, critic head and privileged encoder <- checkpoint",
)
a = p.parse_args()
if os.environ.get("YGORL_GPU_FRACTION"):  # share one GPU between arms: cap the caching allocator per process
    import torch

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(os.environ["YGORL_GPU_FRACTION"]))
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
latest = out / "checkpoints" / "latest.pt"
ov = dict(eval_every=0, checkpoint_every=16, snapshot_every=10, log_games=True)
if a.envs:
    ov["num_envs"] = a.envs
if a.steps:
    ov["steps"] = a.steps
if a.env_threads:
    ov["env_threads"] = a.env_threads
if a.bf16:
    ov["bf16"] = True
if a.freeze_id_embedding:
    ov["freeze_id_embedding"] = True
if a.overlap:
    ov["overlap_collect"] = True
if a.seed is not None:
    ov["seed"] = a.seed
if a.separate_critic:
    ov["shared_backbone"] = False
if a.critic_hidden:
    ov["critic_hidden"] = a.critic_hidden
if a.critic_warmup is not None:
    ov["critic_warmup"] = a.critic_warmup
if a.critic_aux is not None:
    ov["critic_aux"] = a.critic_aux
for kv in a.set:
    k, v = kv.split("=", 1)
    ov[k] = json.loads(v)
log = open(out / "run.log", "a")


def say(*x):
    print(time.strftime("%H:%M:%S"), *x, file=log, flush=True)
    print(*x, flush=True)


src = str(latest) if latest.exists() else a.ckpt
if a.fresh and not latest.exists():
    from ygorl.train.checkpoint import load_checkpoint
    from ygorl.train.trainer import TrainConfig

    cfg = replace(TrainConfig.from_dict(load_checkpoint(a.ckpt)["config"]), init_from=a.ckpt, **ov)
    t = Trainer(cfg, out, log=say)
    if a.critic_from_ckpt:
        import torch

        sd = load_checkpoint(a.ckpt)["learner"]["model"]
        new = dict(sd)
        for k, v in sd.items():
            if k.startswith("actor."):
                new["critic_trunk." + k[len("actor.") :]] = v
        missing, unexpected = t.model.load_state_dict(new, strict=False)
        say(f"critic initialized from the checkpoint: missing {missing}, unexpected {unexpected}")
        t.learner.reference.load_state_dict(t.model.state_dict())
        if t.cfg.overlap_collect:
            t.acting.load_state_dict(t.model.state_dict())
else:
    t = Trainer.resume(src, out, log=say, **ov)
ppo = {}
if a.adv_keep is not None:
    ppo["adv_keep"] = a.adv_keep
if a.minibatch is not None:
    ppo["minibatch_size"] = a.minibatch
if a.adv_keep_mode is not None:
    ppo["adv_keep_mode"] = a.adv_keep_mode
if a.entropy is not None:
    ppo["entropy_coef"] = a.entropy
if a.response_weight is not None:
    ppo["response_weight"] = a.response_weight
if a.critic_lam is not None:
    ppo["critic_lam"] = a.critic_lam
if a.aux_coef is not None:
    ppo["aux_coef"] = a.aux_coef
for kv in a.ppo_set:
    k, v = kv.split("=", 1)
    ppo[k] = json.loads(v)
if ppo:
    t.cfg = replace(t.cfg, ppo=replace(t.cfg.ppo, **ppo))
    t.learner.cfg = t.cfg.ppo
if not (out / "arm.json").exists():
    (out / "arm.json").write_text(
        json.dumps(
            {
                "source": a.ckpt,
                "overrides": ov,
                "ppo_overrides": ppo,
                "start_updates": t.learner.updates,
                "argv": sys.argv[1:],
            },
            indent=1,
        )
    )
say(f"resumed from {src} at update {t.learner.updates}; adv_keep={t.cfg.ppo.adv_keep} overrides={ov}")
target = json.loads((out / "arm.json").read_text())["start_updates"] + a.updates
if a.branch_replay:
    import glob, pickle, random, numpy as np, torch
    from ygorl.nets import collate
    from ygorl.nets.actor_critic import collate_privileged

    rows = []
    for f in sorted(glob.glob(os.path.join(a.branch_replay, "game-*.pkl"))):
        for r in pickle.load(open(f, "rb"))["records"]:
            if "obs" in r and "priv" in r:
                opts = sorted(r["z"].keys())
                rows.append((r["obs"], r["priv"], opts, [float(np.mean(r["z"][o])) for o in opts]))
    cparams = [p_ for n, p_ in t.model.named_parameters() if n.startswith(("critic_trunk.", "privileged.", "critic."))]
    bopt = torch.optim.Adam(cparams, lr=a.branch_lr)
    dev = next(t.model.parameters()).device
    say(
        f"branch replay: {len(rows)} windows, {sum(p_.numel() for p_ in cparams):,} critic params, {a.branch_steps} step(s)/update"
    )
    _orig_update = t.learner.update

    def _update(ro, policy=True):
        stats = _orig_update(ro, policy)
        if policy and rows:
            t.model.train()
            for _ in range(a.branch_steps):
                b = random.sample(rows, min(a.branch_batch, len(rows)))
                obs = collate([x[0] for x in b], device=dev)
                priv = collate_privileged([x[1] for x in b], device=dev)
                with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=bool(t.cfg.bf16)):
                    q = t.model(obs, priv).q.float()
                loss = sum((q[i, o] - tt) ** 2 for i, x in enumerate(b) for o, tt in zip(x[2], x[3])) / sum(
                    len(x[2]) for x in b
                )
                bopt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(cparams, 0.5)
                bopt.step()
            stats["branch_loss"] = float(loss)
        return stats

    t.learner.update = _update
while t.learner.updates < target:
    n = min(a.chunk, target - t.learner.updates)
    t0 = time.time()
    last = t.train(max_updates=n)
    dt = time.time() - t0
    u = t.learner.updates
    t.save(f"update_{u:06d}.pt")
    t.save("latest.pt")
    say(
        f"update {u}: {n} updates in {dt:.0f}s ({last.get('rows', 0) * n / dt:.0f} rows/s), "
        f"kept {last.get('kept_rows')}/{last.get('rows')} rows, early_stop {last.get('early_stop')}, "
        f"entropy {last.get('entropy', 0):.3f}, approx_kl {last.get('approx_kl', 0):.4f}"
    )
say("done")
