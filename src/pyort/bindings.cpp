#include <pybind11/pybind11.h>

namespace py = pybind11;

#ifndef ONNXRUNTIME_PY_VERSION
#define ONNXRUNTIME_PY_VERSION "0.1.0"
#endif

PYBIND11_MODULE(_pyort, m) {
    m.doc() = "Minimal pybind11 extension linking to ONNX Runtime";
    m.attr("__version__") = ONNXRUNTIME_PY_VERSION;

    // Placeholder: we'll add real ONNX Runtime bindings later.
    m.def("hello", [](){ return std::string("onnxruntime-py ready"); });
}
