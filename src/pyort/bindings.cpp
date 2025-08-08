#include <pybind11/pybind11.h>
#include <onnxruntime_c_api.h>

namespace py = pybind11;

#ifndef PYORT_VERSION
#define PYORT_VERSION "0.1.0"
#endif

PYBIND11_MODULE(_pyort, m) {
    m.doc() = "onnxruntime binding build upon C API.";
    m.attr("__version__") = PYORT_VERSION;

    m.attr("ORT_API_VERSION") = ORT_API_VERSION;

    // Enum mappings following Python naming conventions
    py::enum_<ONNXTensorElementDataType>(m, "ONNXTensorElementDataType")
        .value("UNDEFINED", ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED)
        .value("FLOAT", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        .value("UINT8", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8)
        .value("INT8", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8)
        .value("UINT16", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16)
        .value("INT16", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16)
        .value("INT32", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)
        .value("INT64", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
        .value("STRING", ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING)
        .value("BOOL", ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL)
        .value("FLOAT16", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
        .value("DOUBLE", ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE)
        .value("UINT32", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32)
        .value("UINT64", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64)
        .value("COMPLEX64", ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64)
        .value("COMPLEX128", ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128)
        .value("BFLOAT16", ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16)
        .value("FLOAT8E4M3FN", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN)
        .value("FLOAT8E4M3FNUZ", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ)
        .value("FLOAT8E5M2", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2)
        .value("FLOAT8E5M2FNUZ", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ)
        .value("UINT4", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4)
        .value("INT4", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4)
        .export_values();

    py::enum_<ONNXType>(m, "ONNXType")
        .value("UNKNOWN", ONNX_TYPE_UNKNOWN)
        .value("TENSOR", ONNX_TYPE_TENSOR)
        .value("SEQUENCE", ONNX_TYPE_SEQUENCE)
        .value("MAP", ONNX_TYPE_MAP)
        .value("OPAQUE", ONNX_TYPE_OPAQUE)
        .value("SPARSE_TENSOR", ONNX_TYPE_SPARSETENSOR)
        .value("OPTIONAL", ONNX_TYPE_OPTIONAL)
        .export_values();

    py::enum_<OrtSparseFormat>(m, "OrtSparseFormat")
        .value("UNDEFINED", ORT_SPARSE_UNDEFINED)
        .value("COO", ORT_SPARSE_COO)
        .value("CSRC", ORT_SPARSE_CSRC)
        .value("BLOCK_SPARSE", ORT_SPARSE_BLOCK_SPARSE)
        .export_values();

    py::enum_<OrtSparseIndicesFormat>(m, "OrtSparseIndicesFormat", py::module_local())
        .value("COO_INDICES", ORT_SPARSE_COO_INDICES)
        .value("CSR_INNER_INDICES", ORT_SPARSE_CSR_INNER_INDICES)
        .value("CSR_OUTER_INDICES", ORT_SPARSE_CSR_OUTER_INDICES)
        .value("BLOCK_SPARSE_INDICES", ORT_SPARSE_BLOCK_SPARSE_INDICES)
        .export_values();

    py::enum_<OrtLoggingLevel>(m, "OrtLoggingLevel")
        .value("VERBOSE", ORT_LOGGING_LEVEL_VERBOSE)
        .value("INFO", ORT_LOGGING_LEVEL_INFO)
        .value("WARNING", ORT_LOGGING_LEVEL_WARNING)
        .value("ERROR", ORT_LOGGING_LEVEL_ERROR)
        .value("FATAL", ORT_LOGGING_LEVEL_FATAL)
        .export_values();

    py::enum_<OrtErrorCode>(m, "OrtErrorCode")
        .value("OK", ORT_OK)
        .value("FAIL", ORT_FAIL)
        .value("INVALID_ARGUMENT", ORT_INVALID_ARGUMENT)
        .value("NO_SUCH_FILE", ORT_NO_SUCHFILE)
        .value("NO_MODEL", ORT_NO_MODEL)
        .value("ENGINE_ERROR", ORT_ENGINE_ERROR)
        .value("RUNTIME_EXCEPTION", ORT_RUNTIME_EXCEPTION)
        .value("INVALID_PROTOBUF", ORT_INVALID_PROTOBUF)
        .value("MODEL_LOADED", ORT_MODEL_LOADED)
        .value("NOT_IMPLEMENTED", ORT_NOT_IMPLEMENTED)
        .value("INVALID_GRAPH", ORT_INVALID_GRAPH)
        .value("EP_FAIL", ORT_EP_FAIL)
        .value("MODEL_LOAD_CANCELED", ORT_MODEL_LOAD_CANCELED)
        .value("MODEL_REQUIRES_COMPILATION", ORT_MODEL_REQUIRES_COMPILATION)
        .export_values();

    py::enum_<OrtOpAttrType>(m, "OrtOpAttrType")
        .value("UNDEFINED", ORT_OP_ATTR_UNDEFINED)
        .value("INT", ORT_OP_ATTR_INT)
        .value("INTS", ORT_OP_ATTR_INTS)
        .value("FLOAT", ORT_OP_ATTR_FLOAT)
        .value("FLOATS", ORT_OP_ATTR_FLOATS)
        .value("STRING", ORT_OP_ATTR_STRING)
        .value("STRINGS", ORT_OP_ATTR_STRINGS)
        .export_values();
}
