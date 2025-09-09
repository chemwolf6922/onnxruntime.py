#include "Pyort.h"
#include <cstring>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif /** _WIN32 */

#if defined(__linux__) || defined(__APPLE__)
#include <dlfcn.h>
#endif /** __linux__ || __APPLE__ */

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

std::unordered_map<std::string, std::string> Pyort::KeyValuePairsToMap(const OrtKeyValuePairs* pairs)
{
    std::unordered_map<std::string, std::string> map{};
    if (pairs == nullptr)
    {
        return map;
    }
    const char* const* keys = nullptr;
    const char* const* values = nullptr;
    size_t count = 0;
    GetApi()->GetKeyValuePairs(pairs, &keys, &values, &count);
    for (size_t i = 0; i < count; ++i)
    {
        if (keys[i] && values[i])
        {
            map[keys[i]] = values[i];
        }
    }
    return map;
}

/** HardwareDevice */

Pyort::HardwareDevice::HardwareDevice(const OrtHardwareDevice* device)
{
    if (device == nullptr)
    {
        throw std::runtime_error("HardwareDevice cannot be null");
    }
    type = GetApi()->HardwareDevice_Type(device);
    vendorId = GetApi()->HardwareDevice_VendorId(device);
    auto vendorRaw = GetApi()->HardwareDevice_Vendor(device);
    vendor = vendorRaw ? vendorRaw : "";
    deviceId = GetApi()->HardwareDevice_DeviceId(device);
    auto metadataRaw = GetApi()->HardwareDevice_Metadata(device);
    metadata = KeyValuePairsToMap(metadataRaw);
}

/** EpDevice */

Pyort::EpDevice::EpDevice(const OrtEpDevice* epDevice)
{
    if (epDevice == nullptr)
    {
        throw std::runtime_error("EpDevice cannot be null");
    }
    _ptr = epDevice;
    epName = GetApi()->EpDevice_EpName(epDevice);
    epVendor = GetApi()->EpDevice_EpVendor(epDevice);
    auto metadataRaw = GetApi()->EpDevice_EpMetadata(epDevice);
    epMetadata = KeyValuePairsToMap(metadataRaw);
    auto optionsRaw = GetApi()->EpDevice_EpOptions(epDevice);
    epOptions = KeyValuePairsToMap(optionsRaw);
    device = GetApi()->EpDevice_Device(epDevice);
}

Pyort::EpDevice::operator const OrtEpDevice*() const
{
    return _ptr;
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
    /** Return value ignored */
    GetApi()->DisableTelemetryEvents(_ptr);
}

void Pyort::Env::ReleaseOrtType(OrtEnv* ptr)
{
    GetApi()->ReleaseEnv(ptr);
}

void Pyort::Env::RegisterExecutionProviderLibrary(const std::string& name, const std::string& path)
{
    Pyort::Status status = GetApi()->RegisterExecutionProviderLibrary(_ptr, name.c_str(), StringToOrtString(path).c_str());
    status.Check();
}

void Pyort::Env::UnregisterExecutionProviderLibrary(const std::string& name)
{
    Pyort::Status status = GetApi()->UnregisterExecutionProviderLibrary(_ptr, name.c_str());
    status.Check();
}

std::vector<Pyort::EpDevice> Pyort::Env::GetEpDevices() const
{
    /** DO NOT free this. This is owned by onnxruntime. */
    const OrtEpDevice* const* devicesRaw = nullptr;
    size_t deviceCount = 0;
    Pyort::Status status = GetApi()->GetEpDevices(_ptr, &devicesRaw, &deviceCount);
    status.Check();
    std::vector<Pyort::EpDevice> devices;
    devices.reserve(deviceCount);
    for (size_t i = 0; i < deviceCount; ++i)
    {
        devices.emplace_back(devicesRaw[i]);
    }
    return devices;
}

/** ModelCompilationOptions */

void Pyort::ModelCompilationOptions::ReleaseOrtType(OrtModelCompilationOptions* ptr)
{
    GetApi()->GetCompileApi()->ReleaseModelCompilationOptions(ptr);
}

void Pyort::ModelCompilationOptions::SetInputModelPath(const std::string& path)
{
    Pyort::Status status = GetApi()->GetCompileApi()->ModelCompilationOptions_SetInputModelPath(
        _ptr, StringToOrtString(path).c_str());
    status.Check();
}

void Pyort::ModelCompilationOptions::SetInputModelFromBuffer(const pybind11::bytes& modelBytes)
{
    auto bufferView = static_cast<std::string_view>(modelBytes);
    Pyort::Status status = GetApi()->GetCompileApi()->ModelCompilationOptions_SetInputModelFromBuffer(
        _ptr,
        reinterpret_cast<const void*>(bufferView.data()),
        bufferView.size());
    status.Check();
}

void Pyort::ModelCompilationOptions::SetOutputModelExternalInitializersFile(
    const std::string& path, size_t externalInitializerSizeThreshold)
{
    Pyort::Status status = GetApi()->GetCompileApi()->ModelCompilationOptions_SetOutputModelExternalInitializersFile(
        _ptr,
        StringToOrtString(path).c_str(),
        externalInitializerSizeThreshold);
    status.Check();
}

void Pyort::ModelCompilationOptions::SetEpContextEmbedMode(bool embedContext)
{
    Pyort::Status status = GetApi()->GetCompileApi()->ModelCompilationOptions_SetEpContextEmbedMode(_ptr, embedContext);
    status.Check();
}

void Pyort::ModelCompilationOptions::CompileModelToFile(const std::string& path)
{
    Pyort::Status status = GetApi()->GetCompileApi()->ModelCompilationOptions_SetOutputModelPath(
        _ptr, StringToOrtString(path).c_str());
    status.Check();
    status = GetApi()->GetCompileApi()->CompileModel(*Env::GetSingleton(), _ptr);
    status.Check();
}

pybind11::bytes Pyort::ModelCompilationOptions::CompileModelToBuffer()
{
    void* buffer = nullptr;
    size_t bufferSize = 0;
    Pyort::Status status = GetApi()->GetCompileApi()->ModelCompilationOptions_SetOutputModelBuffer(
        _ptr, GetAllocator(), &buffer, &bufferSize); 
    status.Check();
    status = GetApi()->GetCompileApi()->CompileModel(*Env::GetSingleton(), _ptr);
    status.Check();
    pybind11::bytes result{ static_cast<const char*>(buffer), bufferSize };
    GetAllocator()->Free(GetAllocator(), buffer);
    return result;
}

/** LibraryHandle */

void Pyort::LibraryHandle::ReleaseOrtType(void* ptr)
{
    /**
     * For whatever reason, onnxruntime lefts the handler to the user to free
     *     w/o a unified API.
     * So we have to do it ourselves here.
     */
#ifdef _WIN32
    HMODULE hModule = static_cast<HMODULE>(ptr);
    if (hModule)
    {
        FreeLibrary(hModule);
    }
#endif /** _WIN32 */
#if defined(__linux__) || defined(__APPLE__)
    if (ptr)
    {
        dlclose(ptr);
    }
#endif /** __linux__ || __APPLE__ */
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

void Pyort::SessionOptions::SetOptimizedModelFilePath(const std::string& path)
{
    Pyort::Status status = GetApi()->SetOptimizedModelFilePath(
        _ptr, StringToOrtString(path).c_str());
    status.Check();
}

void Pyort::SessionOptions::SetSessionExecutionMode(ExecutionMode mode)
{
    Pyort::Status status = GetApi()->SetSessionExecutionMode(_ptr, mode);
    status.Check();
}

void Pyort::SessionOptions::EnableProfiling(const std::string& profileFilePrefix)
{
    Pyort::Status status = GetApi()->EnableProfiling(
        _ptr, StringToOrtString(profileFilePrefix).c_str());
    status.Check();
}

void Pyort::SessionOptions::DisableProfiling()
{
    Pyort::Status status = GetApi()->DisableProfiling(_ptr);
    status.Check();
}

void Pyort::SessionOptions::EnableMemPattern()
{
    Pyort::Status status = GetApi()->EnableMemPattern(_ptr);
    status.Check();
}

void Pyort::SessionOptions::DisableMemPattern()
{
    Pyort::Status status = GetApi()->DisableMemPattern(_ptr);
    status.Check();
}

void Pyort::SessionOptions::EnableCpuMemArena()
{
    Pyort::Status status = GetApi()->EnableCpuMemArena(_ptr);
    status.Check();
}

void Pyort::SessionOptions::DisableCpuMemArena()
{
    Pyort::Status status = GetApi()->DisableCpuMemArena(_ptr);
    status.Check();
}

void Pyort::SessionOptions::SetSessionLogId(const std::string& logId)
{
    Pyort::Status status = GetApi()->SetSessionLogId(_ptr, logId.c_str());
    status.Check();
}

void Pyort::SessionOptions::SetSessionLogVerbosityLevel(int level)
{
    Pyort::Status status = GetApi()->SetSessionLogVerbosityLevel(_ptr, level);
    status.Check();
}

void Pyort::SessionOptions::SetSessionLogSeverityLevel(int level)
{
    Pyort::Status status = GetApi()->SetSessionLogSeverityLevel(_ptr, level);
    status.Check();
}

void Pyort::SessionOptions::SetSessionGraphOptimizationLevel(
    GraphOptimizationLevel level)
{
    Pyort::Status status = GetApi()->SetSessionGraphOptimizationLevel(_ptr, level);
    status.Check();
}

void Pyort::SessionOptions::SetIntraOpNumThreads(int intraOpNumThreads)
{
    Pyort::Status status = GetApi()->SetIntraOpNumThreads(
        _ptr, intraOpNumThreads);
    status.Check();
}

void Pyort::SessionOptions::SetInterOpNumThreads(int interOpNumThreads)
{
    Pyort::Status status = GetApi()->SetInterOpNumThreads(
        _ptr, interOpNumThreads);
    status.Check();
}

Pyort::LibraryHandle Pyort::SessionOptions::RegisterCustomOpsLibrary(const std::string& libraryPath)
{
    void* handle = nullptr;
    Pyort::Status status = GetApi()->RegisterCustomOpsLibrary(
        _ptr, libraryPath.c_str(), &handle);
    status.Check();
    return LibraryHandle{ handle };
}

void Pyort::SessionOptions::AppendExecutionProvider_V2(
    const std::vector<EpDevice>& epDevices,
    const std::unordered_map<std::string, std::string>& epOptions)
{
    std::vector<const OrtEpDevice*> epDevicePtrs;
    epDevicePtrs.reserve(epDevices.size());
    for (const auto& device : epDevices)
    {
        epDevicePtrs.push_back(device);
    }
    std::vector<const char*> epOptionKeys;
    epOptionKeys.reserve(epOptions.size());
    std::vector<const char*> epOptionValues;
    epOptionValues.reserve(epOptions.size());
    for (const auto& [k, v] : epOptions)
    {
        epOptionKeys.push_back(k.c_str());
        epOptionValues.push_back(v.c_str());
    }
    Pyort::Status status = GetApi()->SessionOptionsAppendExecutionProvider_V2(
        _ptr,
        *Pyort::Env::GetSingleton(),
        epDevicePtrs.data(),
        epDevicePtrs.size(),
        epOptionKeys.data(),
        epOptionValues.data(),
        epOptionKeys.size());
    status.Check();
}

void Pyort::SessionOptions::SetEpSelectionPolicy(OrtExecutionProviderDevicePolicy policy)
{
    Pyort::Status status = GetApi()->SessionOptionsSetEpSelectionPolicy(_ptr, policy);
    status.Check();
}

void Pyort::SessionOptions::SetEpSelectionPolicyDelegate(pybind11::function delegate)
{
    ::EpSelectionDelegate delegateWrapper = [](
        const OrtEpDevice** ep_devices,
        size_t num_devices,
        const OrtKeyValuePairs* model_metadata,
        const OrtKeyValuePairs* runtime_metadata,
        const OrtEpDevice** selected,
        size_t max_selected,
        size_t* num_selected,
        void* state
    ) -> OrtStatus* {
        try
        {
            PyObject* raw = static_cast<PyObject*>(state);
            pybind11::function delegate = pybind11::reinterpret_borrow<pybind11::function>(raw);
            std::vector<Pyort::EpDevice> devices;
            devices.reserve(num_devices);
            for (size_t i = 0; i < num_devices; ++i)
            {
                devices.emplace_back(ep_devices[i]);
            }
            auto modelMetadata = Pyort::KeyValuePairsToMap(model_metadata);
            auto runtimeMetadata = Pyort::KeyValuePairsToMap(runtime_metadata);
            pybind11::object result = pybind11::none();
            {
                pybind11::gil_scoped_acquire acquire;
                result = delegate(devices, modelMetadata, runtimeMetadata, max_selected);
            }
            if (result.is_none())
            {
                *num_selected = 0;
                return nullptr;
            }
            std::vector<Pyort::EpDevice> selectedDevices = result.cast<std::vector<Pyort::EpDevice>>();
            if (selectedDevices.size() > max_selected)
            {
                throw std::runtime_error("The number of selected devices exceeds max_selected");
            }
            for (size_t i = 0; i < selectedDevices.size(); ++i)
            {
                selected[i] = selectedDevices[i];
            }
            *num_selected = selectedDevices.size();
            return nullptr;
        }
        catch (const std::exception& ex)
        {
            return GetApi()->CreateStatus(ORT_FAIL, ex.what());
        }
        catch (...)
        {
            return GetApi()->CreateStatus(ORT_FAIL, "Unknown error in EpSelectionDelegate");
        }
    };
    Pyort::Status status = GetApi()->SessionOptionsSetEpSelectionPolicyDelegate(
        _ptr,
        delegateWrapper,
        /** If the python function gets deleted somehow, this will cause an invalid access. */
        delegate.ptr());
    status.Check();
}

Pyort::ModelCompilationOptions Pyort::SessionOptions::CreateModelCompilationOptions() const
{
    OrtModelCompilationOptions* options = nullptr;
    Pyort::Status status = GetApi()->GetCompileApi()->CreateModelCompilationOptionsFromSessionOptions(
        *Env::GetSingleton(), _ptr, &options);
    status.Check();
    return ModelCompilationOptions{ options };
}

/** TypeInfo */

void Pyort::TypeInfo::ReleaseOrtType(OrtTypeInfo* ptr)
{
    GetApi()->ReleaseTypeInfo(ptr);
}

/** TensorTypeAndShapeInfo */

void Pyort::TensorTypeAndShapeInfo::ReleaseOrtType(OrtTensorTypeAndShapeInfo* ptr)
{
    GetApi()->ReleaseTensorTypeAndShapeInfo(ptr);
}

/** TensorInfo */

Pyort::TensorInfo::TensorInfo(const TypeInfo& typeInfo)
{
    const OrtTensorTypeAndShapeInfo* tensorInfo = nullptr;
    /** DO NOT free the tensorInfo. It's bind to the typeInfo */
    Pyort::Status status = GetApi()->CastTypeInfoToTensorInfo(typeInfo, &tensorInfo);
    status.Check();
    size_t dimCount = 0;
    status = GetApi()->GetDimensionsCount(tensorInfo, &dimCount);
    status.Check();
    shape.resize(dimCount);
    /** The value will be -1 if the dimension is not fixed. */
    status = GetApi()->GetDimensions(tensorInfo, shape.data(), dimCount);
    status.Check();
    std::vector<const char*> dimensionsRaw(dimCount, nullptr);
    status = GetApi()->GetSymbolicDimensions(tensorInfo, dimensionsRaw.data(), dimCount);
    status.Check();
    dimensions.reserve(dimCount);
    for (size_t j = 0; j < dimCount; ++j)
    {
        dimensions.emplace_back(dimensionsRaw[j] ? dimensionsRaw[j] : "");
    }
    ONNXTensorElementDataType type;
    status = GetApi()->GetTensorElementType(tensorInfo, &type);
    status.Check();
    dtype = Pyort::Value::OrtTypeToNpType(type);
}

/** RunOptions */

void Pyort::RunOptions::ReleaseOrtType(OrtRunOptions* ptr)
{
    GetApi()->ReleaseRunOptions(ptr);
}

Pyort::RunOptions::RunOptions()
    : OrtTypeWrapper<OrtRunOptions, RunOptions>(nullptr)
{
    Pyort::Status status = GetApi()->CreateRunOptions(&_ptr);
    status.Check();
}

void Pyort::RunOptions::SetRunLogVerbosityLevel(int level)
{
    Pyort::Status status = GetApi()->RunOptionsSetRunLogVerbosityLevel(_ptr, level);
    status.Check();
}

int Pyort::RunOptions::GetRunLogVerbosityLevel() const
{
    int level = 0;
    Pyort::Status status = GetApi()->RunOptionsGetRunLogVerbosityLevel(_ptr, &level);
    status.Check();
    return level;
}

void Pyort::RunOptions::SetRunLogSeverityLevel(int level)
{
    Pyort::Status status = GetApi()->RunOptionsSetRunLogSeverityLevel(_ptr, level);
    status.Check();
}

int Pyort::RunOptions::GetRunLogSeverityLevel() const
{
    int level = 0;
    Pyort::Status status = GetApi()->RunOptionsGetRunLogSeverityLevel(_ptr, &level);
    status.Check();
    return level;
}

void Pyort::RunOptions::SetRunTag(const std::string& tag)
{
    Pyort::Status status = GetApi()->RunOptionsSetRunTag(_ptr, tag.c_str());
    status.Check();
}

std::string Pyort::RunOptions::GetRunTag() const
{
    const char* tag = nullptr;
    Pyort::Status status = GetApi()->RunOptionsGetRunTag(_ptr, &tag);
    status.Check();
    return tag ? tag : "";
}

void Pyort::RunOptions::SetTerminate()
{
    Pyort::Status status = GetApi()->RunOptionsSetTerminate(_ptr);
    status.Check();
}

void Pyort::RunOptions::UnsetTerminate()
{
    Pyort::Status status = GetApi()->RunOptionsUnsetTerminate(_ptr);
    status.Check();
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

Pyort::Session::Session(const pybind11::bytes& modelBytes, const SessionOptions& options)
    : OrtTypeWrapper<OrtSession, Session>(nullptr)
{
    auto bytesView = static_cast<std::string_view>(modelBytes);
    OrtSession* session = nullptr;
    Pyort::Status status = GetApi()->CreateSessionFromArray(
        *Pyort::Env::GetSingleton(),
        reinterpret_cast<const void*>(bytesView.data()),
        bytesView.size(),
        options,
        &session);
    status.Check();
    _ptr = session;
}

void Pyort::Session::ReleaseOrtType(OrtSession* ptr)
{
    GetApi()->ReleaseSession(ptr);
}

std::unordered_map<std::string, Pyort::TensorInfo> Pyort::Session::GetInputInfo() const
{
    size_t inputCount = 0;
    Pyort::Status status = GetApi()->SessionGetInputCount(_ptr, &inputCount);
    status.Check();
    std::unordered_map<std::string, Pyort::TensorInfo> inputInfo;
    auto allocator = GetAllocator();
    for (size_t i = 0; i < inputCount; i++)
    {
        char* nameRaw = nullptr;
        status = GetApi()->SessionGetInputName(_ptr, i, allocator, &nameRaw);
        status.Check();
        std::string name{ nameRaw };
        allocator->Free(allocator, nameRaw);

        OrtTypeInfo* typeInfoRaw = nullptr;
        status = GetApi()->SessionGetInputTypeInfo(_ptr, i, &typeInfoRaw);
        status.Check();
        TypeInfo typeInfo{ typeInfoRaw };
        inputInfo.emplace(name, TensorInfo{ typeInfo });
    }
    return inputInfo;
}

std::unordered_map<std::string, Pyort::TensorInfo> Pyort::Session::GetOutputInfo() const
{
    size_t outputCount = 0;
    Pyort::Status status = GetApi()->SessionGetOutputCount(_ptr, &outputCount);
    status.Check();
    std::unordered_map<std::string, Pyort::TensorInfo> outputInfo;
    auto allocator = GetAllocator();
    for (size_t i = 0; i < outputCount; i++)
    {
        char* nameRaw = nullptr;
        status = GetApi()->SessionGetOutputName(_ptr, i, allocator, &nameRaw);
        status.Check();
        std::string name{ nameRaw };
        allocator->Free(allocator, nameRaw);

        OrtTypeInfo* typeInfoRaw = nullptr;
        status = GetApi()->SessionGetOutputTypeInfo(_ptr, i, &typeInfoRaw);
        status.Check();
        TypeInfo typeInfo{ typeInfoRaw };
        outputInfo.emplace(name, TensorInfo{ typeInfo });
    }
    return outputInfo;
}

std::unordered_map<std::string, pybind11::array> Pyort::Session::Run(
    const std::unordered_map<std::string, pybind11::array>& inputs,
    const std::optional<std::reference_wrapper<Pyort::RunOptions>>& runOptions) const
{
    /** Create input values */
    std::vector<const char*> inputNamesView;
    inputNamesView.reserve(inputs.size());
    std::vector<Value> inputValues;
    inputValues.reserve(inputs.size());
    std::vector<OrtValue*> inputValuesView;
    inputValuesView.reserve(inputs.size());
    for (const auto& pair : inputs)
    {
        inputNamesView.emplace_back(pair.first.c_str());
        Value value{ pair.second };
        inputValuesView.emplace_back(value);
        /** move won't affect the raw pointer in the view array. */
        inputValues.emplace_back(std::move(value));
    }
    /** Create output values (part 1) */
    auto outputInfo = GetOutputInfo();
    std::vector<const char*> outputNamesView;
    outputNamesView.reserve(outputInfo.size());
    /** Let ort allocate the output values as we may not known their shapes */
    std::vector<OrtValue*> outputValues(outputInfo.size(), nullptr);
    std::vector<Value> outputValuesWrapper;
    outputValuesWrapper.reserve(outputInfo.size());
    for (const auto& pair: outputInfo)
    {
        outputNamesView.emplace_back(pair.first.c_str());
    }
    /** Run the session */
    Pyort::Status status = GetApi()->Run(
        _ptr, runOptions.has_value() ? runOptions.value().get() : nullptr,
        inputNamesView.data(), inputValuesView.data(), inputs.size(),
        outputNamesView.data(), outputInfo.size(), outputValues.data());
    status.Check();
    /** Create output values (part 2) */
    for (auto value : outputValues)
    {
        /** safe guard the raw values first. */
        outputValuesWrapper.emplace_back(value);
    }
    std::unordered_map<std::string, pybind11::array> outputs;
    size_t i = 0;
    for (const auto& pair : outputInfo)
    {
        outputs[pair.first] = outputValuesWrapper[i++];
    }
    return outputs;
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
    : OrtTypeWrapper<OrtValue, Value>(nullptr)
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

Pyort::Value::operator pybind11::array()
{
    if (_npArray.has_value())
    {
        return *_npArray;
    }
    /** Create a new np array from _ptr */
    auto shape = GetShape();
    auto ortType = GetType();
    auto npType = OrtTypeToNpType(ortType);
    _npArray = pybind11::array(npType, shape);
    void* data = GetData();
    auto size = GetSize();
    memcpy(_npArray->mutable_data(), data, size);
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
    switch (npType.num()) {
        case pybind11::detail::npy_api::constants::NPY_BOOL_:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
        case pybind11::detail::npy_api::constants::NPY_BYTE_:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
        case pybind11::detail::npy_api::constants::NPY_UBYTE_:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
        case pybind11::detail::npy_api::constants::NPY_SHORT_:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
        case pybind11::detail::npy_api::constants::NPY_USHORT_:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
        case pybind11::detail::npy_api::constants::NPY_INT_:
        case pybind11::detail::npy_api::constants::NPY_LONG_:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
        case pybind11::detail::npy_api::constants::NPY_UINT_:
        case pybind11::detail::npy_api::constants::NPY_ULONG_:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
        case pybind11::detail::npy_api::constants::NPY_LONGLONG_:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
        case pybind11::detail::npy_api::constants::NPY_ULONGLONG_:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
        case pybind11::detail::npy_api::constants::NPY_FLOAT_:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        case pybind11::detail::npy_api::constants::NPY_DOUBLE_:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
        case 23: /** float16. Not defined in pybind11's included numpy header. */
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    }
    throw std::runtime_error("Unsupported NumPy data type: " + std::to_string(npType.num()));
}

pybind11::dtype Pyort::Value::OrtTypeToNpType(ONNXTensorElementDataType ortType)
{
    switch (ortType) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return pybind11::dtype::of<float>();
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            return pybind11::dtype::of<uint8_t>();
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            return pybind11::dtype::of<int8_t>();
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            return pybind11::dtype::of<uint16_t>();
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            return pybind11::dtype::of<int16_t>();
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            return pybind11::dtype::of<int32_t>();
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            return pybind11::dtype::of<int64_t>();
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            return pybind11::dtype::of<bool>();
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            /** 23: numpy float 16 */
            return pybind11::dtype(23);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            return pybind11::dtype::of<double>();
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            return pybind11::dtype::of<uint32_t>();
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            return pybind11::dtype::of<uint64_t>();
    }
    throw std::runtime_error("Unsupported ONNX tensor element data type: " + std::to_string(ortType));
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
