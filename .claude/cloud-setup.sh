#!/bin/bash
# Setup script for the Claude Code cloud environment (paste into the environment
# dialog's "Setup script" field at claude.ai/code). It runs as root before Claude
# Code starts, and its result is snapshotted and reused by later sessions for
# about 7 days, so the heavy downloads and the first C++ build happen once
# instead of in every session. .claude/hooks/session-start.sh still runs each
# session and then only relinks from the warm caches.
# Constraints: must exit 0 and finish in about 5 minutes.
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive

REPO=/home/user/ygorl
TORCH_FALLBACK=2.14.0   # used only when the repo isn't cloned yet; keep in step with uv.lock

log() { echo "[cloud-setup] $*" >&2; }

# ccache speeds up rebuilds of ygorl._core; libsqlite3-dev is needed by
# tools/build_combo_solver.sh. Failures here are not fatal.
log "apt: ccache libsqlite3-dev"
{ timeout 120 apt-get update -qq && timeout 120 apt-get install -y -qq --no-install-recommends ccache libsqlite3-dev; } \
  >/dev/null 2>&1 || log "apt-get failed; continuing"

if [ -f "$REPO/uv.lock" ]; then
  # Repo present: build once so both uv's cache (torch, ~3 GB) and ccache are warm.
  log "uv sync --extra train in $REPO"
  (cd "$REPO" \
    && git submodule update --init --recursive --depth 1 \
    && uv sync --extra train) || log "uv sync failed; the SessionStart hook will retry"
else
  # No repo yet: at least put Python 3.11 and torch in uv's cache.
  log "warming uv cache: python 3.11, torch $TORCH_FALLBACK"
  warm=$(mktemp -d)
  (uv python install 3.11 \
    && uv venv -q -p 3.11 "$warm/venv" \
    && uv pip install -q -p "$warm/venv/bin/python" "torch==$TORCH_FALLBACK") || log "uv cache warm-up failed; continuing"
  rm -rf "$warm"
fi

exit 0
