"""Frozen card-text and effect-text vectors for the policy network (T5.2; layout in ygorl.nets.text, docs/nets.md).

  uv pip install "sentence-transformers>=3"      # the optional ``text`` extra (README「快速开始」)
  uv run --no-sync python tools/build_text_embeddings.py md-2026-09 [--model intfloat/e5-base-v2] [--device cuda]

Encodes every card of the card database (not only the environment's pool: the vocab covers the whole database):
``card_text`` = "passage: <name>. <card text>", and every non-empty effect string (cdb ``str1..16``) on its own.
Vectors are L2-normalized. Cards without text get no row (the network then uses a zero vector). Writes
``environments/<v>/artifacts/text/`` (not in git; rebuild with this tool) with a ``meta.json`` naming the model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("env", help="environment version or directory (output goes to its artifacts/text)")
    ap.add_argument("--model", default="intfloat/e5-base-v2")
    ap.add_argument("--prefix", default="passage: ", help="text prefix the model expects (E5: 'passage: ')")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max-length", type=int, default=512, help="truncate inputs to this many tokens")
    ap.add_argument("--trust-remote-code", action="store_true", help="for models that ship their own code")
    ap.add_argument("--out", type=Path, default=None, help="output directory (default: <env>/artifacts/text)")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    from ygorl.data.environment import load_environment
    from ygorl.engine.duel import default_cards
    from ygorl.nets.text import CARD_FILES, EFFECT_FILES

    cards = default_cards()
    env = load_environment(args.env, cards=cards)
    out = args.out or env.artifact_path("text", "meta.json").parent
    out.mkdir(parents=True, exist_ok=True)
    card_pw, card_txt, eff_keys, eff_txt = [], [], [], []
    for pw in sorted(cards.keys()):
        c = cards[pw]
        text = " ".join(c.desc.split())
        if text or c.name:
            card_pw.append(pw)
            card_txt.append(f"{args.prefix}{c.name}. {text}")
        for i, s in enumerate(c.strings):
            s = " ".join((s or "").split())
            if s:
                eff_keys.append((pw, i))
                eff_txt.append(f"{args.prefix}{s}")
    model = SentenceTransformer(args.model, device=args.device, trust_remote_code=args.trust_remote_code)
    model.max_seq_length = min(args.max_length, model.max_seq_length or args.max_length)
    t0 = time.time()
    enc = dict(batch_size=args.batch, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    card_vec = model.encode(card_txt, **enc).astype(np.float32)
    eff_vec = model.encode(eff_txt, **enc).astype(np.float32)
    np.save(out / CARD_FILES[0], card_vec)
    np.save(out / CARD_FILES[1], np.asarray(card_pw, dtype=np.int64))
    np.save(out / EFFECT_FILES[0], eff_vec)
    np.save(out / EFFECT_FILES[1], np.asarray(eff_keys, dtype=np.int64).reshape(-1, 2))
    meta = {"model": args.model, "prefix": args.prefix, "normalized": True, "cards": len(card_pw),
            "card_dim": int(card_vec.shape[1]), "effect_strings": len(eff_keys), "effect_dim": int(eff_vec.shape[1]),
            "card_input": "<prefix><name>. <card text>", "database_cards": len(cards),
            "max_length": model.max_seq_length, "seconds": round(time.time() - t0, 1)}  # fmt: skip
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
