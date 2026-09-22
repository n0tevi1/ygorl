# Builds edo9300/ygopro-core (and its bundled Lua) as static libraries.
# Mirrors third_party/ygopro-core/meson.build and lua/meson.build.

# The submodule stays pristine: its sources are copied into the build tree and
# the patches in patches/ygopro-core/ are applied there, in file-name order.
set(OCGCORE_UPSTREAM_DIR ${CMAKE_SOURCE_DIR}/third_party/ygopro-core)
set(OCGCORE_DIR ${CMAKE_BINARY_DIR}/ygopro-core-patched)
set(OCGCORE_LUA_DIR ${OCGCORE_DIR}/lua)

if(NOT EXISTS ${OCGCORE_UPSTREAM_DIR}/ocgapi.h OR NOT EXISTS ${OCGCORE_UPSTREAM_DIR}/lua/src/lua.h)
    message(FATAL_ERROR
        "ygopro-core sources not found. Run: git submodule update --init --recursive")
endif()

find_program(PATCH_PROGRAM patch REQUIRED)
file(GLOB OCGCORE_PATCHES ${CMAKE_SOURCE_DIR}/patches/ygopro-core/*.patch)
list(SORT OCGCORE_PATCHES)
file(GLOB OCGCORE_UPSTREAM_FILES ${OCGCORE_UPSTREAM_DIR}/*.cpp ${OCGCORE_UPSTREAM_DIR}/*.h)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${OCGCORE_PATCHES} ${OCGCORE_UPSTREAM_FILES})

file(REMOVE_RECURSE ${OCGCORE_DIR})
file(COPY ${OCGCORE_UPSTREAM_DIR}/ DESTINATION ${OCGCORE_DIR} PATTERN ".git" EXCLUDE)
foreach(patch_file IN LISTS OCGCORE_PATCHES)
    execute_process(
        COMMAND ${PATCH_PROGRAM} -p1 --forward --batch -i ${patch_file}
        WORKING_DIRECTORY ${OCGCORE_DIR}
        RESULT_VARIABLE patch_result
        OUTPUT_VARIABLE patch_output ERROR_VARIABLE patch_output)
    if(NOT patch_result EQUAL 0)
        message(FATAL_ERROR "Failed to apply ${patch_file}:\n${patch_output}")
    endif()
    message(STATUS "ygopro-core: applied ${patch_file}")
endforeach()

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
