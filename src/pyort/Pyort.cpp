#include "Pyort.h"

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

Pyort::Status::Status(OrtStatus* status)
    : _status(status)
{
}

Pyort::Status::~Status()
{
    if (_status) {
        GetApi()->ReleaseStatus(_status);
        _status = nullptr;
    }
}

OrtErrorCode Pyort::Status::GetErrorCode() const
{
    if (_status == nullptr)
    {
        return ORT_OK;
    }
    return GetApi()->GetErrorCode(_status);
}

std::string Pyort::Status::GetErrorMessage() const
{
    if (_status == nullptr)
    {
        return "";
    }
    return GetApi()->GetErrorMessage(_status);
}

void Pyort::Status::Check() const
{
    OrtErrorCode code = GetErrorCode();
    if (code != ORT_OK)
    {
        throw std::runtime_error(GetErrorMessage());
    }
}

Pyort::Status::operator OrtStatus*() const
{
    return _status;
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
{
    OrtEnv* env = nullptr;
    Pyort::Status status = GetApi()->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "Pyort", &env);
    status.Check();
    _env = env;
}

Pyort::Env::~Env()
{
    if (_env) {
        GetApi()->ReleaseEnv(_env);
        _env = nullptr;
    }
}

Pyort::Env::operator OrtEnv*() const
{
    return _env;
}

/** SessionOptions */

Pyort::SessionOptions::SessionOptions()
{
    OrtSessionOptions* options = nullptr;
    Pyort::Status status = GetApi()->CreateSessionOptions(&options);
    status.Check();
    _options = options;
}

Pyort::SessionOptions::~SessionOptions()
{
    if (_options) {
        GetApi()->ReleaseSessionOptions(_options);
        _options = nullptr;
    }
}

Pyort::SessionOptions::operator OrtSessionOptions*() const
{
    return _options;
}

/** Session */

Pyort::Session::Session(const std::string& modelPath, const SessionOptions& options)
{
    OrtSession* session = nullptr;
    Pyort::Status status = GetApi()->CreateSession(
        *Pyort::Env::GetSingleton(),
        StringToOrtString(modelPath).c_str(),
        options,
        &session);
    status.Check();
    _session = session;
}

Pyort::Session::~Session()
{
    if (_session) {
        GetApi()->ReleaseSession(_session);
        _session = nullptr;
    }
}

Pyort::Session::operator OrtSession*() const
{
    return _session;
}

std::unordered_map<std::string, pybind11::array> Pyort::Session::Run(
    const std::unordered_map<std::string, pybind11::array>& inputs)
{
    /** TODO */

    return inputs;
}

/** Value */

Pyort::Value::Value(const pybind11::array& source)
    : _source(source)
{
    auto sourceType = _source->dtype();
    ONNXTensorElementDataType type;
    switch (sourceType.num()) {
        case pybind11::detail::npy_api::constants::NPY_BOOL_:
            type = ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
            break;
    }
}

Pyort::Value::Value(OrtValue* value)
    : _value(value)
{
}

Pyort::Value::~Value()
{
    if (_value) {
        GetApi()->ReleaseValue(_value);
        _value = nullptr;
    }
}

Pyort::Value::operator pybind11::array() const
{
    if (_source.has_value())
    {
        /** OK, but why? */
        return _source.value();
    }
    /** 
     * Create a pybind11::array by copying the data. 
     * As there is no safe way to detach the buffer from OrtValue*. (Stupid!)
     */
    if (_value == nullptr)
    {
        throw std::runtime_error("Cannot convert OrtValue to pybind11::array, value is null.");
    }

    /** TODO */
}

Pyort::Value Pyort::Value::Create(std::vector<int64_t> shape, ONNXTensorElementDataType type)
{
    OrtValue* value = nullptr;
    Pyort::Status status = GetApi()->CreateTensorAsOrtValue(
        GetAllocator(),
        shape.data(),
        shape.size(),
        type,
        &value);
    status.Check();
    return Value(value);
}

Pyort::Value Pyort::Value::CreateRef(
    std::vector<int64_t> shape, ONNXTensorElementDataType type,
    void* data, size_t dataSize)
{
    OrtValue* value = nullptr;
    Pyort::MemoryInfo memInfo{};
    Pyort::Status status = GetApi()->CreateTensorWithDataAsOrtValue(
        memInfo,
        data,
        dataSize,
        shape.data(),
        shape.size(),
        type,
        &value);
    status.Check();
    return Value(value);
}

/** MemoryInfo */

Pyort::MemoryInfo::MemoryInfo()
{
    OrtMemoryInfo* memoryInfo = nullptr;
    Pyort::Status status = GetApi()->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memoryInfo);
    status.Check();
    _memoryInfo = memoryInfo;
}

Pyort::MemoryInfo::~MemoryInfo()
{
    if (_memoryInfo) {
        GetApi()->ReleaseMemoryInfo(_memoryInfo);
        _memoryInfo = nullptr;
    }
}

Pyort::MemoryInfo::operator OrtMemoryInfo*() const
{
    return _memoryInfo;
}
