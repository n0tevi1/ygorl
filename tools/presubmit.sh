#!/usr/bin/env bash
# Presubmit: the checks CI runs, before a commit or a push.
#
#   tools/presubmit.sh           format src/tests/tools in place (ruff format), then lint (ruff check)
#   tools/presubmit.sh --check   change nothing: fail if anything is unformatted or lint fails (CI, git hook)
#   tools/presubmit.sh --test    format + lint, then the test suite (pytest -q)
#
# As a git pre-commit hook (check only, so it never rewrites what you staged):
#   printf '#!/bin/sh\nexec tools/presubmit.sh --check\n' > "$(git rev-parse --git-path hooks)/pre-commit"
#   chmod +x "$(git rev-parse --git-path hooks)/pre-commit"
# ruff comes from the dev group (`uv sync`), pinned by uv.lock, so local and CI agree.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PATHS=(src tests tools)
RUFF=(uv run --no-sync ruff)
case "${1:-}" in
  --check)
    "${RUFF[@]}" format --check "${PATHS[@]}"
    "${RUFF[@]}" check "${PATHS[@]}"
    ;;
  ""|--test)
    "${RUFF[@]}" format "${PATHS[@]}"
    "${RUFF[@]}" check "${PATHS[@]}"
    if [ "${1:-}" = --test ]; then uv run --no-sync pytest -q; fi
    ;;
  *)
    echo "usage: tools/presubmit.sh [--check | --test]" >&2
    exit 2
    ;;
esac
