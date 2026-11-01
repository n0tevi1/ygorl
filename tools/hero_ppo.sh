#!/bin/zsh
# Long ("hero") PPO run on the md-2026-09 meta with the measured Apple M5 Ultra configuration
# (docs/benchmarks.md "M5 Ultra rollout geometry"). Extra arguments are passed to tools/train_ppo.py.
#
#   OUT=out/train/hero INIT=out/train/combined-ctrl/checkpoints/latest.pt MINUTES=1440 tools/hero_ppo.sh [args...]
#   OUT=out/train/hero RESUME=1 MINUTES=600 tools/hero_ppo.sh      # continue OUT/checkpoints/latest.pt
#
# Resuming forwards only the overrides given here; --eval-every 0 is always passed (a resumed run would
# otherwise restore the checkpoint's eval schedule).
set -eu
cd "$(dirname "$0")/.."
OUT=${OUT:-out/train/hero}
MINUTES=${MINUTES:-1440}
HT=out/claude-scale/handtrap
if [[ -n "${RESUME:-}" ]]; then
  exec env PYTHONUNBUFFERED=1 uv run --no-sync python tools/train_ppo.py --resume "$OUT/checkpoints/latest.pt" \
    --minutes "$MINUTES" --eval-every 0 "$@"
fi
INIT=${INIT:-out/train/combined-ctrl/checkpoints/latest.pt}
DECKS=(environments/md-2026-09/meta/*.ydk)
exec env PYTHONUNBUFFERED=1 uv run --no-sync python tools/train_ppo.py $DECKS --env md-2026-09 \
  --init-from "$INIT" \
  --d-model 128 --layers 2 --separate-critic --critic-init actor --critic-hidden 128 --critic-warmup 8 \
  --line-demos $HT/lines_train.jsonl \
  --line-recovery $HT/recovery_full.jsonl --line-recovery $HT/recovery_multi_97268402.pruned.jsonl \
  --line-share 0.5 --line-first-share 0.5 --selfplay-fraction 0.5 \
  --envs 512 --steps 32 --min-batch 384 --pool-active-k 2 \
  --collect-threads 8 --env-threads 8 --opp-cpu-threads 1 --device mps \
  --eval-every 0 --checkpoint-every 16 --seed 21 \
  --minutes "$MINUTES" --out "$OUT" "$@"
