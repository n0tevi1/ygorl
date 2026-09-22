cmake_minimum_required(VERSION 3.20)
project(ygorl LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Use ccache when available (CI caches it; also speeds up local rebuilds).
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

include(cmake/ocgcore.cmake)

pybind11_add_module(_core MODULE
    csrc/arena.cpp
    csrc/binding.cpp
    csrc/core_backend.cpp
    csrc/duel_pool.cpp
    csrc/host.cpp
    csrc/obs_encoder.cpp
    csrc/host_pool.cpp
    csrc/privileged.cpp
    csrc/event_encoder.cpp
    csrc/event_binding.cpp
)
target_include_directories(_core PRIVATE csrc)
target_link_libraries(_core PRIVATE ocgcore)

# Duel snapshots (T2.8): the extension replaces operator new/delete to route the
# core's allocations into per-duel arenas (csrc/arena.h). libstdc++ is linked
# statically so its out-of-line code uses the same operators, and only the module
# init symbol is exported so nothing leaks into the rest of the process.
option(YGORL_ARENA "Per-duel arenas for snapshot/restore" ON)
if(YGORL_ARENA)
    target_compile_definitions(_core PRIVATE YGORL_ARENA)
    target_link_options(_core PRIVATE -static-libstdc++ -Wl,--exclude-libs,ALL
                        -Wl,--version-script=${CMAKE_SOURCE_DIR}/csrc/exports.map)
endif()

install(TARGETS _core LIBRARY DESTINATION ygorl)
