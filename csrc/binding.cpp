// pybind11 entry point for ygorl._core.
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "ygorl native core (C++17 + pybind11)";
    m.def("hello", []() { return "ygorl._core"; }, "Smoke-test function.");
}
