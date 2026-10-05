#!/usr/bin/env bash
# Build ygo-combo-solver (T4a.1) against this repository's pinned, patched ygopro-core.
#
# The solver is AGPL-3.0 and is never vendored: its sources are fetched at build time
# into the build directory (default build/combo-solver/, git-ignored), together with the
# LZMA sources of EDOPro (gframe/lzma). The core is a copy of third_party/ygopro-core
# with our patches 0001 (deterministic iteration order) and 0002 (constant Lua string
# hash seed) applied -- 0003 (our Lua allocator hook) is left out because the solver
# brings its own -- followed by the solver's five patches, taken from the pinned
# tools/fetch_solver_deps.sh. See docs/solver.md and docs/spikes/combo-solver.md.
#
# Idempotent: when nothing changed since the last build the binary is reused.
# The last line on stdout is always the binary path.
#
# Usage: tools/build_combo_solver.sh            (env: YGORL_SOLVER_BUILD_DIR, JOBS, CXX, CC)
# Needs: git, python3, patch, g++/gcc (C++17), sqlite3 headers + library (libsqlite3-dev).
set -euo pipefail

SOLVER_REPO=https://github.com/96jonesa/ygo-combo-solver
SOLVER_COMMIT=e0c7221802a23657d5a9a0b020025e0ceca04675
EDOPRO_REPO=https://github.com/edo9300/edopro
EDOPRO_COMMIT=c250b6ab9bebb6eca9fdd07ee0c5bd2278426e81
OUR_PATCHES=(0001-deterministic-iteration-order.patch 0002-constant-lua-string-hash-seed.patch)
SOLVER_PATCHES=(0001-reject-unresolved-chain-goals.patch 0002-enumerate-exact-card-declarations.patch)

root="$(cd "$(dirname "$0")/.." && pwd)"
out="${YGORL_SOLVER_BUILD_DIR:-$root/build/combo-solver}"
jobs="${JOBS:-$(nproc)}"
CXX="${CXX:-g++}"
CC="${CC:-gcc}"
core_src="$root/third_party/ygopro-core"
bin="$out/bin/combosolver"

log() { echo "$@" >&2; }
die() { echo "build_combo_solver: error: $*" >&2; exit 1; }

[ -f "$core_src/ocgapi.h" ] && [ -f "$core_src/lua/src/lua.h" ] \
  || die "ygopro-core sources not found; run: git submodule update --init --recursive"
for tool in git python3 patch "$CXX" "$CC"; do
  command -v "$tool" > /dev/null || die "$tool not found"
done
printf '#include <sqlite3.h>\nint main(){return 0;}\n' | "$CC" -x c - -o /dev/null -lsqlite3 2> /dev/null \
  || die "sqlite3 development files not found (Debian/Ubuntu: apt install libsqlite3-dev)"

mkdir -p "$out"

# fetch_pinned <dir> <repo> <commit> [sparse path...]: shallow checkout of exactly <commit>.
fetch_pinned() {
  local dir="$1" repo="$2" commit="$3"
  shift 3
  if [ "$(git -C "$dir" rev-parse HEAD 2> /dev/null || true)" = "$commit" ]; then
    return
  fi
  log "fetching $repo @ ${commit:0:7}"
  rm -rf "$dir"
  git init -q "$dir"
  git -C "$dir" remote add origin "$repo"
  if [ $# -gt 0 ]; then
    git -C "$dir" fetch -q --depth 1 --filter=blob:none origin "$commit"
    git -C "$dir" sparse-checkout set --no-cone "$@"
  else
    git -C "$dir" fetch -q --depth 1 origin "$commit"
  fi
  git -C "$dir" -c advice.detachedHead=false checkout -q FETCH_HEAD
}

fetch_pinned "$out/src/solver" "$SOLVER_REPO" "$SOLVER_COMMIT"
fetch_pinned "$out/src/edopro" "$EDOPRO_REPO" "$EDOPRO_COMMIT" gframe/lzma
solver_src="$out/src/solver"
lzma_src="$out/src/edopro/gframe/lzma"

core_commit="$(git -C "$core_src" rev-parse HEAD 2> /dev/null || echo unknown)"
lua_commit="$(git -C "$core_src/lua/src" rev-parse HEAD 2> /dev/null || echo unknown)"
stamp="$(
  {
    echo "$SOLVER_COMMIT $EDOPRO_COMMIT $core_commit $lua_commit"
    "$CXX" --version | head -1
    cat "$0"
    for p in "${OUR_PATCHES[@]}"; do cat "$root/patches/ygopro-core/$p"; done
    for p in "${SOLVER_PATCHES[@]}"; do cat "$root/patches/combo-solver/$p"; done
    # Uncommitted edits of the core submodule count as well.
    git -C "$core_src" diff HEAD 2> /dev/null || true
  } | sha256sum | cut -d' ' -f1
)"
if [ -x "$bin" ] && [ "$(cat "$out/stamp" 2> /dev/null || true)" = "$stamp" ]; then
  log "combo solver up to date"
  echo "$bin"
  exit 0
fi

start=$(date +%s)
rm -f "$out/stamp"

# Apply local solver fixes to a copy, preserving the pinned upstream checkout.
solver_src="$out/solver-patched"
rm -rf "$solver_src"
mkdir -p "$solver_src"
(cd "$out/src/solver" && tar --exclude=.git -cf - .) | tar -xf - -C "$solver_src"
for p in "${SOLVER_PATCHES[@]}"; do
  patch -d "$solver_src" -p1 --forward --batch --quiet -i "$root/patches/combo-solver/$p" > /dev/null \
    || die "failed to apply patches/combo-solver/$p"
done

# --- core: our copy, our patches 0001-0002, then the solver's patches -------------------
core="$out/ocgcore"
rm -rf "$core"
mkdir -p "$core"
(cd "$core_src" && tar --exclude=.git -cf - .) | tar -xf - -C "$core"
for p in "${OUR_PATCHES[@]}"; do
  patch -d "$core" -p1 --forward --batch --quiet -i "$root/patches/ygopro-core/$p" > /dev/null \
    || die "failed to apply patches/ygopro-core/$p"
done
# The solver's patch block is the Python heredoc of its fetch script (anchor-based edits).
sed -n "/^python3 - \"\$dst\"/,/^PYEOF\$/p" "$solver_src/tools/fetch_solver_deps.sh" | sed '1d;$d' > "$out/solver_patches.py"
[ -s "$out/solver_patches.py" ] || die "could not extract the patch block from the solver's fetch_solver_deps.sh"
python3 "$out/solver_patches.py" "$core" "$core_commit" "$lua_commit" >&2

# --- compile ----------------------------------------------------------------------------
launcher=()
command -v ccache > /dev/null && launcher=(ccache)
obj="$out/obj"
rm -rf "$obj"
mkdir -p "$obj"/{lua,core,lzma,solver} "$out/bin"
opt=(-O2 -DNDEBUG -ffp-contract=off -g0)
lua_skip=" lbitlib.c lcorolib.c ldblib.c linit.c loadlib.c loslib.c ltests.c lua.c luac.c lutf8lib.c onelua.c "
cmds="$out/compile.cmds"
: > "$cmds"
for f in "$core"/lua/src/*.c; do
  name="$(basename "$f")"
  case "$lua_skip" in *" $name "*) continue ;; esac
  echo "${launcher[*]} $CXX ${opt[*]} -std=c++17 -w -x c++ -include luaconf-customize.h -I$core/lua -I$core/lua/src -I$solver_src -c $f -o $obj/lua/${name%.c}.o" >> "$cmds"
done
for f in "$core"/*.cpp; do
  name="$(basename "$f")"
  echo "${launcher[*]} $CXX ${opt[*]} -std=c++17 -w -fno-rtti -I$core -I$core/lua/src -c $f -o $obj/core/${name%.cpp}.o" >> "$cmds"
done
for name in Alloc LzFind LzmaDec LzmaEnc LzmaLib; do
  echo "${launcher[*]} $CC ${opt[*]} -w -D_7ZIP_ST -I$lzma_src -c $lzma_src/$name.c -o $obj/lzma/$name.o" >> "$cmds"
done
# GCC fix-ups: several solver sources rely on headers MSVC / libc++ pull in transitively.
for f in "$solver_src"/*.cpp; do
  name="$(basename "$f")"
  echo "${launcher[*]} $CXX ${opt[*]} -std=c++17 -w -include cmath -include cstring -include algorithm -include cstdint -I$solver_src -I$core -I$core/lua -I$core/lua/src -I$lzma_src -c $f -o $obj/solver/${name%.cpp}.o" >> "$cmds"
done
log "compiling $(wc -l < "$cmds") sources with $jobs jobs"
tr '\n' '\0' < "$cmds" | xargs -0 -P "$jobs" -I{} sh -c '{}' || die "compilation failed (commands in $cmds)"
"$CXX" -o "$bin.tmp" "$obj"/solver/*.o "$obj"/core/*.o "$obj"/lua/*.o "$obj"/lzma/*.o -lsqlite3 -lpthread \
  || die "link failed"
mv "$bin.tmp" "$bin"
"$bin" --help > /dev/null 2>&1 || die "the built binary does not run: $bin --help failed"
echo "$stamp" > "$out/stamp"
log "built combo solver in $(($(date +%s) - start)) s"
echo "$bin"
