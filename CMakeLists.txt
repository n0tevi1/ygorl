cmake_minimum_required(VERSION 3.20)
project(ygorl LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core MODULE
    csrc/binding.cpp
)
target_include_directories(_core PRIVATE csrc)

install(TARGETS _core LIBRARY DESTINATION ygorl)
