// pybind11 entry point for ygorl._core.
#include <pybind11/pybind11.h>

#include "ocgapi.h"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "ygorl native core (C++17 + pybind11)";
    m.def("hello", []() { return "ygorl._core"; }, "Smoke-test function.");
    m.def(
        "ocg_version",
        []() {
            int major = 0, minor = 0;
            OCG_GetVersion(&major, &minor);
            return py::make_tuple(major, minor);
        },
        "Return (major, minor) of the linked ygopro-core (OCG_GetVersion).");
}
