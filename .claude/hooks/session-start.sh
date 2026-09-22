#!/bin/bash
# SessionStart hook for Claude Code on the web: fetch submodules and build
# the ygorl._core extension so `uv run pytest` works as soon as the session
# starts. Idempotent; the container is cached after this completes.
# Can also be run by hand (or as a cloud environment setup script) with
# CLAUDE_CODE_REMOTE=true.
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
#    torch (M4 policy training) is ~5 GB and ~2 min on first install; if it
#    can't be fetched, fall back to the base env so engine/eval/deck tests still
#    run (training tests skip without torch).
log "uv sync --extra train"
if ! uv sync --extra train; then
  log "uv sync --extra train failed; falling back to uv sync (no torch)"
  uv sync
fi

uv run --no-sync python -c "import ygorl._core as c; print('[session-start] ygorl._core OK, ocgcore', c.ocg_version())" >&2
