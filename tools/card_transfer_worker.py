"""Bounded, resumable offline card-transfer diagnostic (not a win-rate experiment)."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ygorl.cards.cdb import CardVocab
from ygorl.nets import NetConfig, PolicyNet, TextFeatures
from ygorl.train.checkpoint import warm_start
from ygorl.train.generalization import grouped_metrics, reserved_action_rows
from ygorl.train.heuristic_demos import load_data


def sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def atomic(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def evaluate(net, data, device):
    net.eval()
    losses, correct = [], []
    with torch.no_grad():
        for start in range(0, len(data), 128):
            obs, actions = data.batch(np.arange(start, min(start + 128, len(data))), device)
            out = net(obs)
            losses.extend((-out.log_probs().gather(1, actions[:, None]).squeeze(1)).cpu().tolist())
            correct.extend((out.greedy() == actions).cpu().tolist())
    return {
        "rows": len(data),
        "nll": float(np.mean(losses)),
        "accuracy": float(np.mean(correct)),
        "groups": grouped_metrics(losses, correct, data.meta),
    }


def run(root, seed, arm, device, stop_after=None):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    registration = json.loads((root / "registration.json").read_text())
    for key, path in [
        ("worker_sha256", Path(__file__)),
        ("generalization_sha256", Path(__file__).resolve().parents[1] / "src/ygorl/train/generalization.py"),
        ("warmstart_sha256", Path(__file__).resolve().parents[1] / "src/ygorl/train/checkpoint.py"),
    ]:
        if key in registration:
            assert sha(path) == registration[key], "registered code changed: " + key
    manifest = json.loads((root / "dataset-manifest.json").read_text())
    assert sha(root / "dataset-manifest.json") == registration["dataset_manifest_sha256"]
    for name, digest in manifest["files"].items():
        assert sha(root / name) == digest, name
    assert not manifest["train_reserved_overlap"] and manifest["hand_overlap"] == 0
    identity = {"registration_sha256": sha(root / "registration.json"), "seed": seed, "arm": arm}
    outdir = root / "jobs" / f"seed-{seed}-{arm}"
    outdir.mkdir(parents=True, exist_ok=True)
    data = {name: load_data(root / "data" / (name + ".npz"))[0] for name in ["train", "seen", "unseen", "adapt"]}
    data["novel_actions"] = data["unseen"].subset(
        reserved_action_rows(data["unseen"].obs, manifest["reserved_vocab_indices"])
    )
    assert len(data["novel_actions"]) > 0
    vocab = CardVocab.load(root / "data/vocab.json")
    torch.manual_seed(seed)
    cfg = NetConfig(vocab_size=len(vocab), **registration["net"])
    baseline = PolicyNet(cfg)
    if arm == "id":
        net = baseline
    else:
        text = TextFeatures.load(root / "features", vocab)
        if arm == "shuffled":
            # A fixed permutation preserves feature size and distribution while breaking card meaning.
            perm = np.arange(len(vocab))
            perm[2:] = np.random.default_rng(registration["permutation_seed"]).permutation(perm[2:])
            text.card = text.card[perm].copy()
            text.effect_lookup = text.effect_lookup[perm].copy()
        from dataclasses import replace

        cfg = replace(cfg, card_text=True, effect_text=True).with_text(text)
        net = PolicyNet(cfg, text)
        warm_start(net, SimpleNamespace(net=baseline, net_config=baseline.cfg, path="paired-initialization"))
        obs, _ = data["seen"].batch(np.arange(min(16, len(data["seen"]))))
        baseline.eval()
        net.eval()
        with torch.no_grad():
            torch.testing.assert_close(net(obs).logits, baseline(obs).logits, rtol=0, atol=0)
    net.to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=registration["lr"], weight_decay=1e-4)
    latest = outdir / "latest.pt"
    completed = 0
    history = []
    if latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=True)
        assert state["identity"] == identity
        net.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        if device.startswith("cuda"):
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        completed, history = state["epoch"], state["history"]
    total = registration["epochs"] + registration["adapt_epochs"]
    for epoch in range(completed + 1, total + 1):
        if os.environ.get("YGORL_LEASE"):
            assert float(Path(os.environ["YGORL_LEASE"]).read_text()) > time.time(), "controller lease expired"
        train = data["train" if epoch <= registration["epochs"] else "adapt"]
        order = np.random.default_rng(registration["shuffle_seed"] + seed * 1000 + epoch).permutation(len(train))
        net.train()
        losses = []
        for start in range(0, len(train), registration["batch_size"]):
            if os.environ.get("YGORL_LEASE"):
                assert float(Path(os.environ["YGORL_LEASE"]).read_text()) > time.time(), "controller lease expired"
            obs, actions = train.batch(order[start : start + registration["batch_size"]], device)
            optimizer.zero_grad(set_to_none=True)
            loss = -net(obs).log_probs().gather(1, actions[:, None]).mean()
            assert torch.isfinite(loss)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            assert torch.isfinite(norm)
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics = {
            "epoch": epoch,
            "phase": "zero_shot" if epoch <= registration["epochs"] else "few_shot",
            "train_rows": len(train),
            "train_nll": float(np.mean(losses)),
            "seen": evaluate(net, data["seen"], device),
            "unseen": evaluate(net, data["unseen"], device),
            "novel_actions": evaluate(net, data["novel_actions"], device),
        }
        history.append(metrics)
        state = {
            "identity": identity,
            "epoch": epoch,
            "model": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.startswith("cuda") else [],
            "history": history,
            "net_config": cfg.to_dict(),
        }
        tmp = outdir / "checkpoint.tmp"
        torch.save(state, tmp)
        tmp.replace(latest)
        atomic(
            outdir / "progress.json",
            {**identity, "epoch": epoch, "updated_unix": time.time(), "checkpoint_sha256": sha(latest)},
        )
        atomic(outdir / "metrics.json", history)
        print(
            json.dumps(
                {
                    "job": outdir.name,
                    "epoch": epoch,
                    "train_nll": metrics["train_nll"],
                    "seen_accuracy": metrics["seen"]["accuracy"],
                    "unseen_accuracy": metrics["unseen"]["accuracy"],
                }
            ),
            flush=True,
        )
        if stop_after is not None and epoch >= stop_after and epoch < total:
            return
    atomic(
        outdir / "report.json",
        {
            **identity,
            "healthy": True,
            "epochs": total,
            "history": history,
            "checkpoint_sha256": sha(latest),
            "device": device,
            "gpu": torch.cuda.get_device_name() if device.startswith("cuda") else None,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--arm", choices=["id", "text", "shuffled"], required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    run(args.root, args.seed, args.arm, args.device, args.stop_after)
