# Builds edo9300/ygopro-core (and its bundled Lua) as static libraries.
# Mirrors third_party/ygopro-core/meson.build and lua/meson.build.

set(OCGCORE_DIR ${CMAKE_SOURCE_DIR}/third_party/ygopro-core)
set(OCGCORE_LUA_DIR ${OCGCORE_DIR}/lua)

if(NOT EXISTS ${OCGCORE_DIR}/ocgapi.h OR NOT EXISTS ${OCGCORE_LUA_DIR}/src/lua.h)
    message(FATAL_ERROR
        "ygopro-core sources not found. Run: git submodule update --init --recursive")
endif()

# Lua is compiled as C++ so that Lua errors unwind with C++ exceptions,
# and every source force-includes the core's luaconf customization header.
set(OCGCORE_LUA_SOURCES
    lapi.c lauxlib.c lbaselib.c lcode.c lctype.c ldebug.c ldo.c ldump.c
    lfunc.c lgc.c liolib.c llex.c lmathlib.c lmem.c lobject.c lopcodes.c
    lparser.c lstate.c lstring.c lstrlib.c ltable.c ltablib.c ltm.c
    lundump.c lvm.c lzio.c)
list(TRANSFORM OCGCORE_LUA_SOURCES PREPEND ${OCGCORE_LUA_DIR}/src/)
set_source_files_properties(${OCGCORE_LUA_SOURCES} PROPERTIES LANGUAGE CXX)

add_library(ocgcore_lua STATIC ${OCGCORE_LUA_SOURCES})
target_include_directories(ocgcore_lua SYSTEM PUBLIC ${OCGCORE_LUA_DIR}/src)
target_compile_options(ocgcore_lua PRIVATE
    -w -include ${OCGCORE_LUA_DIR}/luaconf-customize.h)

set(OCGCORE_SOURCES
    card.cpp duel.cpp effect.cpp field.cpp interpreter.cpp libcard.cpp
    libdebug.cpp libduel.cpp libeffect.cpp libgroup.cpp ocgapi.cpp
    operations.cpp playerop.cpp processor.cpp processor_visit.cpp scriptlib.cpp)
list(TRANSFORM OCGCORE_SOURCES PREPEND ${OCGCORE_DIR}/)

add_library(ocgcore STATIC ${OCGCORE_SOURCES})
target_include_directories(ocgcore SYSTEM PUBLIC ${OCGCORE_DIR})
target_link_libraries(ocgcore PUBLIC ocgcore_lua)
target_compile_options(ocgcore PRIVATE -w)
