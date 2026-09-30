"""Continue a PPO run from a checkpoint with changed hyperparameters (plateau experiments, #83).

  tools/continue_run.py CKPT OUT_DIR UPDATES [--steps N] [--minibatch N] [--lr X] [--epochs N]

Unlike ``tools/train_ppo.py --resume`` (which keeps the checkpoint's configuration), this replaces the given
fields. The optimizer state is restored with the checkpoint, learning rate included, so ``--lr`` is applied to the
optimizer after the restore.
"""

import argparse
import dataclasses
import os

os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

from ygorl.train.checkpoint import load_checkpoint  # noqa: E402
from ygorl.train.trainer import TrainConfig, Trainer  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpt")
    p.add_argument("out")
    p.add_argument("updates", type=int)
    p.add_argument("--steps", type=int, help="rollout rows per environment slot")
    p.add_argument("--minibatch", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--epochs", type=int)
    a = p.parse_args()
    cfg = TrainConfig.from_dict(load_checkpoint(a.ckpt)["config"])
    ppo = {k: v for k, v in (("minibatch_size", a.minibatch), ("lr", a.lr), ("epochs", a.epochs)) if v is not None}
    over = {"ppo": dataclasses.replace(cfg.ppo, **ppo)} if ppo else {}
    if a.steps:
        over["steps"] = a.steps
    trainer = Trainer.resume(a.ckpt, a.out, **over)
    if a.lr is not None:
        for g in trainer.learner.optimizer.param_groups:
            g["lr"] = a.lr
    print(f"continuing from update {trainer.counters['updates']} with {over or 'the same config'}", flush=True)
    trainer.train(max_updates=a.updates)


if __name__ == "__main__":
    main()
