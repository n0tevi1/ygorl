#!/bin/bash
# ThreadSanitizer check of the DuelPool (T2.1): build an instrumented ygorl._core in a
# separate directory, run pooled games with libtsan preloaded, fail on any report.
# Usage: tools/tsan/check.sh [games=12]
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
build="${TSAN_BUILD_DIR:-$root/build/tsan}"
games="${1:-12}"
pybind="$(cd "$root" && uv run --no-sync --with pybind11 python -c 'import pybind11; print(pybind11.get_cmake_dir())')"
cmake -S "$root" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_C_FLAGS=-fsanitize=thread -DCMAKE_CXX_FLAGS=-fsanitize=thread \
  -DCMAKE_SHARED_LINKER_FLAGS=-fsanitize=thread -Dpybind11_DIR="$pybind" \
  -DPython_EXECUTABLE="$root/.venv/bin/python" -DYGORL_ARENA=OFF > "$build.cmake.log"
cmake --build "$build" -j "$(nproc)" > "$build.build.log"
rm -f "$build"/tsan_report*
# setarch -R: TSAN cannot map its shadow memory with full ASLR entropy on recent kernels.
setarch "$(uname -m)" -R env LD_PRELOAD="$(gcc -print-file-name=libtsan.so)" \
  TSAN_OPTIONS="halt_on_error=0 log_path=$build/tsan_report" \
  "$root/.venv/bin/python" "$root/tools/tsan/run_pool.py" "$build" "$games"
if compgen -G "$build/tsan_report*" > /dev/null; then
  grep -h "WARNING: ThreadSanitizer" "$build"/tsan_report* | sort | uniq -c
  echo "ThreadSanitizer reported problems; see $build/tsan_report*" >&2
  exit 1
fi
echo "ThreadSanitizer: no reports"
