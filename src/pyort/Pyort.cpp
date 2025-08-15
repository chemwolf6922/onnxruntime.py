#include "Pyort.h"
#include <cstring>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif /** _WIN32 */

#ifdef _WIN32
static std::wstring StringToWString(const std::string& str)
{
    if (str.empty()) {
        return std::wstring();
    }
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), static_cast<int>(str.length()), nullptr, 0);
    if (size_needed <= 0) {
        throw std::runtime_error("Failed to convert string to wide string");
    }
    std::wstring wstr(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), static_cast<int>(str.length()), &wstr[0], size_needed);
    return wstr;
}

#define StringToOrtString(str) StringToWString(str)
#else
#define StringToOrtString(str) (str)
#endif /** _WIN32 */


/** Global */

const OrtApi* Pyort::GetApi()
{
    static const OrtApi* api = nullptr;
    if (api == nullptr)
    {
        api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        if (api == nullptr)
        {
            throw std::runtime_error("Failed to get ONNX Runtime API");
        }
    }
    return api;
}

OrtAllocator* Pyort::GetAllocator()
{
    static OrtAllocator* allocator = nullptr;
    if (allocator == nullptr)
    {
        Pyort::Status status = GetApi()->GetAllocatorWithDefaultOptions(&allocator);
        status.Check();
    }
    return allocator;
}

/** Status */

OrtErrorCode Pyort::Status::GetErrorCode() const
{
    if (_ptr == nullptr)
    {
        return ORT_OK;
    }
    return GetApi()->GetErrorCode(_ptr);
}

std::string Pyort::Status::GetErrorMessage() const
{
    if (_ptr == nullptr)
    {
        return "";
    }
    return GetApi()->GetErrorMessage(_ptr);
}

void Pyort::Status::Check() const
{
    OrtErrorCode code = GetErrorCode();
    if (code != ORT_OK)
    {
        throw std::runtime_error(GetErrorMessage());
    }
}

void Pyort::Status::ReleaseOrtType(OrtStatus* ptr)
{
    GetApi()->ReleaseStatus(ptr);
}

/** Env */

std::shared_ptr<Pyort::Env> Pyort::Env::_instance = nullptr;

std::shared_ptr<Pyort::Env> Pyort::Env::GetSingleton()
{
    if (!_instance) 
    {
        _instance = std::shared_ptr<Pyort::Env>(new Pyort::Env());
    }
    return _instance;
}

Pyort::Env::Env()
    : OrtTypeWrapper<OrtEnv, Env>(nullptr)
{
    Pyort::Status status = GetApi()->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "Pyort", &_ptr);
    status.Check();
}

void Pyort::Env::ReleaseOrtType(OrtEnv* ptr)
{
    GetApi()->ReleaseEnv(ptr);
}

/** SessionOptions */

Pyort::SessionOptions::SessionOptions()
    : OrtTypeWrapper<OrtSessionOptions, SessionOptions>(nullptr)
{
    Pyort::Status status = GetApi()->CreateSessionOptions(&_ptr);
    status.Check();
}

void Pyort::SessionOptions::ReleaseOrtType(OrtSessionOptions* ptr)
{
    GetApi()->ReleaseSessionOptions(ptr);
}

/** Session */

Pyort::Session::Session(const std::string& modelPath, const SessionOptions& options)
    : OrtTypeWrapper<OrtSession, Session>(nullptr)
{
    OrtSession* session = nullptr;
    Pyort::Status status = GetApi()->CreateSession(
        *Pyort::Env::GetSingleton(),
        StringToOrtString(modelPath).c_str(),
        options,
        &session);
    status.Check();
    _ptr = session;
}

void Pyort::Session::ReleaseOrtType(OrtSession* ptr)
{
    GetApi()->ReleaseSession(ptr);
}

std::unordered_map<std::string, pybind11::array> Pyort::Session::Run(
    const std::unordered_map<std::string, pybind11::array>& inputs)
{
    /** TODO */

    return inputs;
}

/** Value */

Pyort::Value::Value(const pybind11::array& npArray)
    : OrtTypeWrapper<OrtValue, Value>(nullptr), _npArray(npArray)
{
    auto type = NpTypeToOrtType(npArray.dtype());
    std::vector<int64_t> shape;
    auto ndim = _npArray->ndim();
    shape.reserve(ndim);
    for (int i = 0; i < ndim; i++)
    {
        shape.push_back(_npArray->shape(i));
    }
    Pyort::MemoryInfo memInfo{};
    Pyort::Status status = GetApi()->CreateTensorWithDataAsOrtValue(
        memInfo,
        _npArray->mutable_data(),
        _npArray->nbytes(),
        shape.data(),
        shape.size(),
        type,
        &_ptr);
    status.Check();
}

Pyort::Value::Value(const std::vector<int64_t>& ortShape, ONNXTensorElementDataType ortType)
    : OrtTypeWrapper<OrtValue, Value>(nullptr), _npArray(std::nullopt)
{
    auto npType = OrtTypeToNpType(ortType);
    pybind11::array npArray(npType, ortShape);
    _npArray = npArray;
    Pyort::MemoryInfo memInfo{};
    Pyort::Status status = GetApi()->CreateTensorWithDataAsOrtValue(
        memInfo,
        _npArray->mutable_data(),
        _npArray->nbytes(),
        ortShape.data(),
        ortShape.size(),
        ortType,
        &_ptr);
    status.Check();
}

void Pyort::Value::ReleaseOrtType(OrtValue* ptr)
{
    GetApi()->ReleaseValue(ptr);
}

Pyort::Value::operator pybind11::array() const
{
    if (!_npArray.has_value())
    {
        throw std::runtime_error("Value is empty");
    }
    return *_npArray;
}

ONNXTensorElementDataType Pyort::Value::GetType() const
{
    if (_ptr == nullptr)
    {
        throw std::runtime_error("Value is empty");
    }
    OrtTensorTypeAndShapeInfo *info = nullptr;
    Pyort::Status status = GetApi()->GetTensorTypeAndShape(_ptr, &info);
    status.Check();
    ONNXTensorElementDataType type;
    status = GetApi()->GetTensorElementType(info, &type);
    status.Check();
    return type;
}

std::vector<int64_t> Pyort::Value::GetShape() const
{
    if (_ptr == nullptr)
    {
        throw std::runtime_error("Value is empty");
    }
    OrtTensorTypeAndShapeInfo *info = nullptr;
    Pyort::Status status = GetApi()->GetTensorTypeAndShape(_ptr, &info);
    status.Check();
    size_t dimCount = 0;
    status = GetApi()->GetDimensionsCount(info, &dimCount);
    status.Check();
    std::vector<int64_t> shape(dimCount);
    status = GetApi()->GetDimensions(info, shape.data(), dimCount);
    status.Check();
    return shape;
}

size_t Pyort::Value::GetSize() const
{
    if (_ptr == nullptr)
    {
        throw std::runtime_error("Value is empty");
    }
    auto shape = GetShape();
    auto type = GetType();
    size_t typeSize = 0;
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            typeSize = sizeof(float);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            typeSize = sizeof(uint8_t);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            typeSize = sizeof(int8_t);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            typeSize = sizeof(uint16_t);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            typeSize = sizeof(int16_t);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            typeSize = sizeof(int32_t);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            typeSize = sizeof(int64_t);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            typeSize = sizeof(bool);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            typeSize = 2; // float16 is 2 bytes
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            typeSize = sizeof(double);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            typeSize = sizeof(uint32_t);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            typeSize = sizeof(uint64_t);
            break;
        default:
            throw std::runtime_error("Unsupported ONNX tensor element data type: " + std::to_string(type));
    }
    size_t size = typeSize;
    for (const auto& dim : shape)
    {
        if (dim < 0)
        {
            throw std::runtime_error("Invalid dimension size: " + std::to_string(dim));
        }
        size *= static_cast<size_t>(dim);
    }
    return size;
}

void* Pyort::Value::GetData() const
{
    if (_ptr == nullptr)
    {
        throw std::runtime_error("Value is empty");
    }
    void* data = nullptr;
    Pyort::Status status = GetApi()->GetTensorMutableData(_ptr, &data);
    if (data == nullptr)
    {
        throw std::runtime_error("Failed to get data from OrtValue.");
    }
    return data;
}

ONNXTensorElementDataType Pyort::Value::NpTypeToOrtType(const pybind11::dtype& npType)
{
    ONNXTensorElementDataType type;
    switch (npType.num()) {
        case pybind11::detail::npy_api::constants::NPY_BOOL_:
            type = ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
            break;
        case pybind11::detail::npy_api::constants::NPY_BYTE_:
            type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
            break;
        case pybind11::detail::npy_api::constants::NPY_UBYTE_:
            type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
            break;
        case pybind11::detail::npy_api::constants::NPY_SHORT_:
            type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
            break;
        case pybind11::detail::npy_api::constants::NPY_USHORT_:
            type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
            break;
        case pybind11::detail::npy_api::constants::NPY_INT_:
        case pybind11::detail::npy_api::constants::NPY_LONG_:
            type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
            break;
        case pybind11::detail::npy_api::constants::NPY_UINT_:
        case pybind11::detail::npy_api::constants::NPY_ULONG_:
            type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
            break;
        case pybind11::detail::npy_api::constants::NPY_LONGLONG_:
            type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
            break;
        case pybind11::detail::npy_api::constants::NPY_ULONGLONG_:
            type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
            break;
        case pybind11::detail::npy_api::constants::NPY_FLOAT_:
            type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
            break;
        case pybind11::detail::npy_api::constants::NPY_DOUBLE_:
            type = ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
            break;
        case pybind11::detail::npy_api::constants::NPY_CFLOAT_:
            type = ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64;
            break;
        case pybind11::detail::npy_api::constants::NPY_CDOUBLE_:
            type = ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128;
            break;
        case 23: /** float16. Not defined in pybind11's included numpy header. */
            type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
            break;
        default:
            throw std::runtime_error("Unsupported NumPy data type: " + std::to_string(npType.num()));
    }
    return type;
}

pybind11::dtype Pyort::Value::OrtTypeToNpType(ONNXTensorElementDataType ortType)
{
    pybind11::dtype npType;
    switch (ortType) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            npType = pybind11::dtype::of<float>();
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            npType = pybind11::dtype::of<uint8_t>();
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            npType = pybind11::dtype::of<int8_t>();
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            npType = pybind11::dtype::of<uint16_t>();
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            npType = pybind11::dtype::of<int16_t>();
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            npType = pybind11::dtype::of<int32_t>();
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            npType = pybind11::dtype::of<int64_t>();
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            npType = pybind11::dtype::of<bool>();
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            /** 23: numpy float 16 */
            npType = pybind11::dtype(23);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            npType = pybind11::dtype::of<double>();
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            npType = pybind11::dtype::of<uint32_t>();
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            npType = pybind11::dtype::of<uint64_t>();
            break;
        default:
            throw std::runtime_error("Unsupported ONNX tensor element data type: " + std::to_string(ortType));
    }
    return npType;
}

/** MemoryInfo */

Pyort::MemoryInfo::MemoryInfo()
    : OrtTypeWrapper<OrtMemoryInfo, MemoryInfo>(nullptr)
{
    Pyort::Status status = GetApi()->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &_ptr);
    status.Check();
}

void Pyort::MemoryInfo::ReleaseOrtType(OrtMemoryInfo* ptr)
{
    GetApi()->ReleaseMemoryInfo(ptr);
}
