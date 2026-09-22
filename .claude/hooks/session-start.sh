#!/bin/bash
# SessionStart hook for Claude Code on the web: fetch submodules and build
# the ygorl._core extension so `uv run pytest` works as soon as the session
# starts. Idempotent. Runs in every session and is not part of the cloud
# environment's snapshot; .claude/cloud-setup.sh (the environment's setup
# script) warms uv's cache and ccache so this step stays short.
# Can also be run by hand with CLAUDE_CODE_REMOTE=true.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel)}"

log() { echo "[session-start] $*" >&2; }

# 1. Toolchain. uv, cmake, ninja and g++ are preinstalled in the default
#    cloud image; install anything missing. ccache is optional (faster rebuilds).
if ! command -v uv >/dev/null 2>&1; then
  log "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  echo "export PATH=\"$HOME/.local/bin:\$PATH\"" >> "${CLAUDE_ENV_FILE:-/dev/null}"
fi
missing=()
for tool in cmake ninja g++ ccache; do
  command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
done
if [ ${#missing[@]} -gt 0 ] && command -v apt-get >/dev/null 2>&1; then
  pkgs=()
  for tool in "${missing[@]}"; do
    case "$tool" in
      ninja) pkgs+=(ninja-build) ;;
      *) pkgs+=("$tool") ;;
    esac
  done
  log "installing ${pkgs[*]}"
  SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
  { $SUDO apt-get update -qq >/dev/null 2>&1; $SUDO apt-get install -y -qq --no-install-recommends "${pkgs[@]}" >/dev/null; } \
    || log "apt-get failed; continuing (ccache is optional)"
fi

# 2. Submodules: ygopro-core (+ nested Lua), CardScripts, BabelCDB, LFLists.
log "updating submodules"
git submodule sync --recursive >/dev/null
git submodule update --init --recursive --depth 1

# 3. Python env + C++ extension (uv rebuilds only when native sources change).
log "uv sync --extra train"
uv sync --extra train   # torch for M4 (policy training); first install ~2 min / ~5 GB

uv run --no-sync python -c "import ygorl._core as c; print('[session-start] ygorl._core OK, ocgcore', c.ocg_version())" >&2

# 4. Optional combo solver (tests/test_solver.py). Only when .claude/cloud-setup.sh
#    prebuilt it into the snapshot: then this is a no-op, or a ccache-warm rebuild
#    if the core or our patches changed. Without it the solver tests stay skipped
#    rather than every cold session paying ~45 s for them.
solver_dir="$HOME/.cache/ygorl/combo-solver"   # keep in step with .claude/cloud-setup.sh
if [ -d "$solver_dir" ]; then
  if bin="$(YGORL_SOLVER_BUILD_DIR="$solver_dir" tools/build_combo_solver.sh 2>/dev/null | tail -1)" && [ -x "$bin" ]; then
    log "combo solver: $bin"
    echo "export YGORL_COMBO_SOLVER=\"$bin\"" >> "${CLAUDE_ENV_FILE:-/dev/null}"
  else
    log "combo solver build failed; its tests will be skipped"
  fi
fi
