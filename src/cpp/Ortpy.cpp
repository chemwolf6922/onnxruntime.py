#include "Ortpy.h"
#include <cstring>
#include <string>
#include <map>
#include <nanobind/stl/function.h>

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

/** nanobind::dlpack::dtype modifications to be a map key */
namespace nanobind::dlpack {
    inline bool operator<(const dtype& a, const dtype& b) {
        if (a.code != b.code) return a.code < b.code;
        if (a.bits != b.bits) return a.bits < b.bits;
        return a.lanes < b.lanes;
    }
}

/** Global */

const OrtApi* Ortpy::GetApi()
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

OrtAllocator* Ortpy::GetAllocator()
{
    static OrtAllocator* allocator = nullptr;
    if (allocator == nullptr)
    {
        Ortpy::Status status = GetApi()->GetAllocatorWithDefaultOptions(&allocator);
        status.Check();
    }
    return allocator;
}

std::unordered_map<std::string, std::string> Ortpy::KeyValuePairsToMap(const OrtKeyValuePairs* pairs)
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

std::vector<std::string> Ortpy::GetAvailableProviders()
{
    char** providers = nullptr;
    int count = 0;
    Ortpy::Status status = GetApi()->GetAvailableProviders(&providers, &count);
    status.Check();
    std::vector<std::string> result;
    result.reserve(count);
    for (int i = 0; i < count; ++i)
    {
        result.emplace_back(providers[i]);
    }
    GetApi()->ReleaseAvailableProviders(providers, count);
    return result;
}

OrtCompiledModelCompatibility Ortpy::GetModelCompatibilityForEpDevices(
    const std::vector<EpDevice>& epDevices, const std::string& compatibilityInfo)
{
    std::vector<const OrtEpDevice*> ptrs;
    ptrs.reserve(epDevices.size());
    for (const auto& d : epDevices)
    {
        ptrs.push_back(d);
    }
    OrtCompiledModelCompatibility result;
    Ortpy::Status status = GetApi()->GetModelCompatibilityForEpDevices(
        ptrs.data(), ptrs.size(), compatibilityInfo.c_str(), &result);
    status.Check();
    return result;
}

/** Helper: split unordered_map into key/value arrays for ORT APIs */
static void MapToKeyValueArrays(
    const std::unordered_map<std::string, std::string>& map,
    std::vector<const char*>& keys,
    std::vector<const char*>& values)
{
    keys.reserve(map.size());
    values.reserve(map.size());
    for (const auto& [k, v] : map)
    {
        keys.push_back(k.c_str());
        values.push_back(v.c_str());
    }
}

/** Helper: RAII wrapper for ORT provider options using a release function */
template <typename T>
using OrtProviderOptionsPtr = std::unique_ptr<T, std::function<void(T*)>>;

template <typename T, typename ReleaseFn>
OrtProviderOptionsPtr<T> MakeProviderOptions(T* raw, ReleaseFn releaseFn)
{
    return OrtProviderOptionsPtr<T>(raw, [releaseFn](T* ptr) { releaseFn(ptr); });
}

/** HardwareDevice */

Ortpy::HardwareDevice::HardwareDevice(const OrtHardwareDevice* device)
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

Ortpy::EpDevice::EpDevice(const OrtEpDevice* epDevice)
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

Ortpy::EpDevice::operator const OrtEpDevice*() const
{
    return _ptr;
}

std::optional<Ortpy::MemoryInfo> Ortpy::EpDevice::GetMemoryInfo(OrtDeviceMemoryType memoryType) const
{
    const OrtMemoryInfo* mi = GetApi()->EpDevice_MemoryInfo(_ptr, memoryType);
    if (mi == nullptr)
    {
        return std::nullopt;
    }
    const char* name = nullptr;
    GetApi()->MemoryInfoGetName(mi, &name);
    OrtAllocatorType allocType;
    GetApi()->MemoryInfoGetType(mi, &allocType);
    int id = 0;
    GetApi()->MemoryInfoGetId(mi, &id);
    OrtMemType memType;
    GetApi()->MemoryInfoGetMemType(mi, &memType);
    return MemoryInfo{ name ? name : "Cpu", allocType, id, memType };
}

/** Status */

OrtErrorCode Ortpy::Status::GetErrorCode() const
{
    if (_ptr == nullptr)
    {
        return ORT_OK;
    }
    return GetApi()->GetErrorCode(_ptr);
}

std::string Ortpy::Status::GetErrorMessage() const
{
    if (_ptr == nullptr)
    {
        return "";
    }
    return GetApi()->GetErrorMessage(_ptr);
}

void Ortpy::Status::Check() const
{
    OrtErrorCode code = GetErrorCode();
    if (code != ORT_OK)
    {
        throw std::runtime_error(GetErrorMessage());
    }
}

void Ortpy::Status::ReleaseOrtType(OrtStatus* ptr)
{
    GetApi()->ReleaseStatus(ptr);
}

/** Env */

std::shared_ptr<Ortpy::Env> Ortpy::Env::_instance = nullptr;

std::shared_ptr<Ortpy::Env> Ortpy::Env::GetSingleton()
{
    if (!_instance) 
    {
        _instance = std::shared_ptr<Ortpy::Env>(new Ortpy::Env());
    }
    return _instance;
}

void Ortpy::Env::ReleaseSingleton()
{
    _instance.reset();
}

Ortpy::Env::Env()
    : OrtTypeWrapper<OrtEnv, Env>(nullptr)
{
    Ortpy::Status status = GetApi()->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "Ortpy", &_ptr);
    status.Check();
    /** Ignore the return value. */
    status = GetApi()->DisableTelemetryEvents(_ptr);
}

void Ortpy::Env::ReleaseOrtType(OrtEnv* ptr)
{
    GetApi()->ReleaseEnv(ptr);
}

void Ortpy::Env::RegisterExecutionProviderLibrary(const std::string& name, const std::string& path)
{
    Ortpy::Status status = GetApi()->RegisterExecutionProviderLibrary(_ptr, name.c_str(), StringToOrtString(path).c_str());
    status.Check();
}

void Ortpy::Env::UnregisterExecutionProviderLibrary(const std::string& name)
{
    Ortpy::Status status = GetApi()->UnregisterExecutionProviderLibrary(_ptr, name.c_str());
    status.Check();
}

std::vector<Ortpy::EpDevice> Ortpy::Env::GetEpDevices() const
{
    /** DO NOT free this. This is owned by onnxruntime. */
    const OrtEpDevice* const* devicesRaw = nullptr;
    size_t deviceCount = 0;
    Ortpy::Status status = GetApi()->GetEpDevices(_ptr, &devicesRaw, &deviceCount);
    status.Check();
    std::vector<Ortpy::EpDevice> devices;
    devices.reserve(deviceCount);
    for (size_t i = 0; i < deviceCount; ++i)
    {
        devices.emplace_back(devicesRaw[i]);
    }
    return devices;
}

void Ortpy::Env::UpdateLogLevel(OrtLoggingLevel level)
{
    Ortpy::Status status = GetApi()->UpdateEnvWithCustomLogLevel(_ptr, level);
    status.Check();
}

#if ORT_API_VERSION >= 24
std::vector<Ortpy::HardwareDevice> Ortpy::Env::GetHardwareDevices() const
{
    size_t numDevices = 0;
    Ortpy::Status status = GetApi()->GetNumHardwareDevices(_ptr, &numDevices);
    status.Check();
    std::vector<const OrtHardwareDevice*> devicesRaw(numDevices, nullptr);
    status = GetApi()->GetHardwareDevices(_ptr, devicesRaw.data(), numDevices);
    status.Check();
    std::vector<HardwareDevice> devices;
    devices.reserve(numDevices);
    for (size_t i = 0; i < numDevices; ++i)
    {
        devices.emplace_back(devicesRaw[i]);
    }
    return devices;
}

Ortpy::Env::DeviceEpIncompatibilityInfo Ortpy::Env::GetHardwareDeviceEpIncompatibilityDetails(
    const std::string& epName, const HardwareDevice& device) const
{
    /** We need the raw OrtHardwareDevice*. Re-enumerate to find matching device. */
    size_t numDevices = 0;
    Ortpy::Status status = GetApi()->GetNumHardwareDevices(_ptr, &numDevices);
    status.Check();
    std::vector<const OrtHardwareDevice*> devicesRaw(numDevices, nullptr);
    status = GetApi()->GetHardwareDevices(_ptr, devicesRaw.data(), numDevices);
    status.Check();
    const OrtHardwareDevice* matchedDevice = nullptr;
    for (size_t i = 0; i < numDevices; ++i)
    {
        if (GetApi()->HardwareDevice_VendorId(devicesRaw[i]) == device.vendorId &&
            GetApi()->HardwareDevice_DeviceId(devicesRaw[i]) == device.deviceId)
        {
            matchedDevice = devicesRaw[i];
            break;
        }
    }
    if (matchedDevice == nullptr)
    {
        throw std::runtime_error("Hardware device not found");
    }
    OrtDeviceEpIncompatibilityDetails* details = nullptr;
    status = GetApi()->GetHardwareDeviceEpIncompatibilityDetails(_ptr, epName.c_str(), matchedDevice, &details);
    status.Check();
    DeviceEpIncompatibilityInfo info{};
    status = GetApi()->DeviceEpIncompatibilityDetails_GetReasonsBitmask(details, &info.reasonsBitmask);
    status.Check();
    const char* notes = nullptr;
    status = GetApi()->DeviceEpIncompatibilityDetails_GetNotes(details, &notes);
    status.Check();
    info.notes = notes ? notes : "";
    status = GetApi()->DeviceEpIncompatibilityDetails_GetErrorCode(details, &info.errorCode);
    status.Check();
    GetApi()->ReleaseDeviceEpIncompatibilityDetails(details);
    return info;
}

std::optional<std::string> Ortpy::Env::GetCompatibilityInfoFromModel(
    const std::string& modelPath, const std::string& epType) const
{
    auto allocator = GetAllocator();
    char* compatInfo = nullptr;
    Ortpy::Status status = GetApi()->GetCompatibilityInfoFromModel(
        StringToOrtString(modelPath).c_str(), epType.c_str(), allocator, &compatInfo);
    status.Check();
    if (compatInfo == nullptr)
    {
        return std::nullopt;
    }
    std::string result{ compatInfo };
    allocator->Free(allocator, compatInfo);
    return result;
}

std::optional<std::string> Ortpy::Env::GetCompatibilityInfoFromModelBytes(
    const nanobind::bytes& modelData, const std::string& epType) const
{
    auto allocator = GetAllocator();
    char* compatInfo = nullptr;
    Ortpy::Status status = GetApi()->GetCompatibilityInfoFromModelBytes(
        modelData.data(), modelData.size(), epType.c_str(), allocator, &compatInfo);
    status.Check();
    if (compatInfo == nullptr)
    {
        return std::nullopt;
    }
    std::string result{ compatInfo };
    allocator->Free(allocator, compatInfo);
    return result;
}
#endif /** ORT_API_VERSION >= 24 */

/** ModelCompilationOptions */

void Ortpy::ModelCompilationOptions::ReleaseOrtType(OrtModelCompilationOptions* ptr)
{
    GetApi()->GetCompileApi()->ReleaseModelCompilationOptions(ptr);
}

void Ortpy::ModelCompilationOptions::SetInputModelPath(const std::string& path)
{
    Ortpy::Status status = GetApi()->GetCompileApi()->ModelCompilationOptions_SetInputModelPath(
        _ptr, StringToOrtString(path).c_str());
    status.Check();
}

void Ortpy::ModelCompilationOptions::SetInputModelFromBuffer(const nanobind::bytes& modelBytes)
{
    Ortpy::Status status = GetApi()->GetCompileApi()->ModelCompilationOptions_SetInputModelFromBuffer(
        _ptr,
        modelBytes.data(),
        modelBytes.size());
    status.Check();
}

void Ortpy::ModelCompilationOptions::SetOutputModelExternalInitializersFile(
    const std::string& path, size_t externalInitializerSizeThreshold)
{
    Ortpy::Status status = GetApi()->GetCompileApi()->ModelCompilationOptions_SetOutputModelExternalInitializersFile(
        _ptr,
        StringToOrtString(path).c_str(),
        externalInitializerSizeThreshold);
    status.Check();
}

void Ortpy::ModelCompilationOptions::SetEpContextEmbedMode(bool embedContext)
{
    Ortpy::Status status = GetApi()->GetCompileApi()->ModelCompilationOptions_SetEpContextEmbedMode(_ptr, embedContext);
    status.Check();
}

void Ortpy::ModelCompilationOptions::CompileModelToFile(const std::string& path)
{
    Ortpy::Status status = GetApi()->GetCompileApi()->ModelCompilationOptions_SetOutputModelPath(
        _ptr, StringToOrtString(path).c_str());
    status.Check();
    status = GetApi()->GetCompileApi()->CompileModel(*Env::GetSingleton(), _ptr);
    status.Check();
}

nanobind::bytes Ortpy::ModelCompilationOptions::CompileModelToBuffer()
{
    void* buffer = nullptr;
    size_t bufferSize = 0;
    Ortpy::Status status = GetApi()->GetCompileApi()->ModelCompilationOptions_SetOutputModelBuffer(
        _ptr, GetAllocator(), &buffer, &bufferSize); 
    status.Check();
    status = GetApi()->GetCompileApi()->CompileModel(*Env::GetSingleton(), _ptr);
    status.Check();
    nanobind::bytes result{ static_cast<const char*>(buffer), bufferSize };
    GetAllocator()->Free(GetAllocator(), buffer);
    return result;
}

void Ortpy::ModelCompilationOptions::SetFlags(uint32_t flags)
{
    Ortpy::Status status = GetApi()->GetCompileApi()->ModelCompilationOptions_SetFlags(_ptr, flags);
    status.Check();
}

void Ortpy::ModelCompilationOptions::SetEpContextBinaryInformation(
    const std::string& outputDirectory, const std::string& modelName)
{
    Ortpy::Status status = GetApi()->GetCompileApi()->ModelCompilationOptions_SetEpContextBinaryInformation(
        _ptr, StringToOrtString(outputDirectory).c_str(), StringToOrtString(modelName).c_str());
    status.Check();
}

void Ortpy::ModelCompilationOptions::SetGraphOptimizationLevel(GraphOptimizationLevel level)
{
    Ortpy::Status status = GetApi()->GetCompileApi()->ModelCompilationOptions_SetGraphOptimizationLevel(_ptr, level);
    status.Check();
}

/** LibraryHandle */

void Ortpy::LibraryHandle::ReleaseOrtType(void* ptr)
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

Ortpy::SessionOptions::SessionOptions()
    : OrtTypeWrapper<OrtSessionOptions, SessionOptions>(nullptr)
{
    Ortpy::Status status = GetApi()->CreateSessionOptions(&_ptr);
    status.Check();
}

void Ortpy::SessionOptions::ReleaseOrtType(OrtSessionOptions* ptr)
{
    GetApi()->ReleaseSessionOptions(ptr);
}

void Ortpy::SessionOptions::SetOptimizedModelFilePath(const std::string& path)
{
    Ortpy::Status status = GetApi()->SetOptimizedModelFilePath(
        _ptr, StringToOrtString(path).c_str());
    status.Check();
}

void Ortpy::SessionOptions::SetSessionExecutionMode(ExecutionMode mode)
{
    Ortpy::Status status = GetApi()->SetSessionExecutionMode(_ptr, mode);
    status.Check();
}

void Ortpy::SessionOptions::EnableProfiling(const std::string& profileFilePrefix)
{
    Ortpy::Status status = GetApi()->EnableProfiling(
        _ptr, StringToOrtString(profileFilePrefix).c_str());
    status.Check();
}

void Ortpy::SessionOptions::DisableProfiling()
{
    Ortpy::Status status = GetApi()->DisableProfiling(_ptr);
    status.Check();
}

void Ortpy::SessionOptions::EnableMemPattern()
{
    Ortpy::Status status = GetApi()->EnableMemPattern(_ptr);
    status.Check();
}

void Ortpy::SessionOptions::DisableMemPattern()
{
    Ortpy::Status status = GetApi()->DisableMemPattern(_ptr);
    status.Check();
}

void Ortpy::SessionOptions::EnableCpuMemArena()
{
    Ortpy::Status status = GetApi()->EnableCpuMemArena(_ptr);
    status.Check();
}

void Ortpy::SessionOptions::DisableCpuMemArena()
{
    Ortpy::Status status = GetApi()->DisableCpuMemArena(_ptr);
    status.Check();
}

void Ortpy::SessionOptions::SetSessionLogId(const std::string& logId)
{
    Ortpy::Status status = GetApi()->SetSessionLogId(_ptr, logId.c_str());
    status.Check();
}

void Ortpy::SessionOptions::SetSessionLogVerbosityLevel(int level)
{
    Ortpy::Status status = GetApi()->SetSessionLogVerbosityLevel(_ptr, level);
    status.Check();
}

void Ortpy::SessionOptions::SetSessionLogSeverityLevel(int level)
{
    Ortpy::Status status = GetApi()->SetSessionLogSeverityLevel(_ptr, level);
    status.Check();
}

void Ortpy::SessionOptions::SetSessionGraphOptimizationLevel(
    GraphOptimizationLevel level)
{
    Ortpy::Status status = GetApi()->SetSessionGraphOptimizationLevel(_ptr, level);
    status.Check();
}

void Ortpy::SessionOptions::SetIntraOpNumThreads(int intraOpNumThreads)
{
    Ortpy::Status status = GetApi()->SetIntraOpNumThreads(
        _ptr, intraOpNumThreads);
    status.Check();
}

void Ortpy::SessionOptions::SetInterOpNumThreads(int interOpNumThreads)
{
    Ortpy::Status status = GetApi()->SetInterOpNumThreads(
        _ptr, interOpNumThreads);
    status.Check();
}

Ortpy::LibraryHandle Ortpy::SessionOptions::RegisterCustomOpsLibrary(const std::string& libraryPath)
{
    void* handle = nullptr;
    Ortpy::Status status = GetApi()->RegisterCustomOpsLibrary(
        _ptr, libraryPath.c_str(), &handle);
    status.Check();
    return LibraryHandle{ handle };
}

int Ortpy::SessionOptions::TpTraverse(PyObject* self, visitproc visit, void* arg) noexcept
{
    try
    {        
        // On Python 3.9+, we must traverse the implicit dependency
        // of an object on its associated type object.
        #if PY_VERSION_HEX >= 0x03090000
            Py_VISIT(Py_TYPE(self));
        #endif /** PY_VERSION_HEX >= 0x03090000 */

        if (!nanobind::inst_ready(self))
        {
            return 0;
        }
        SessionOptions* options = nanobind::inst_ptr<SessionOptions>(self);
        nanobind::handle handle = nanobind::find(options->_delegate);
        Py_VISIT(handle.ptr());
        return 0;
    }
    catch(...)
    {
        return -1;
    }
}

int Ortpy::SessionOptions::TpClear(PyObject* self) noexcept
{
    try
    {
        SessionOptions* options = nanobind::inst_ptr<SessionOptions>(self);
        /** Break circular reference */
        options->_delegate = nullptr;
        return 0;
    }
    catch(...)
    {
        return -1;
    }
}

void Ortpy::SessionOptions::AppendExecutionProvider_V2(
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
    Ortpy::Status status = GetApi()->SessionOptionsAppendExecutionProvider_V2(
        _ptr,
        *Ortpy::Env::GetSingleton(),
        epDevicePtrs.data(),
        epDevicePtrs.size(),
        epOptionKeys.data(),
        epOptionValues.data(),
        epOptionKeys.size());
    status.Check();
}

void Ortpy::SessionOptions::SetEpSelectionPolicy(OrtExecutionProviderDevicePolicy policy)
{
    Ortpy::Status status = GetApi()->SessionOptionsSetEpSelectionPolicy(_ptr, policy);
    status.Check();
}

void Ortpy::SessionOptions::SetEpSelectionPolicyDelegate(const EpSelectionPolicyDelegate& delegate)
{
    if (delegate == nullptr)
    {
        throw std::invalid_argument("delegate cannot be null");
    }
    _delegate = delegate;
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
            Ortpy::SessionOptions* options = static_cast<Ortpy::SessionOptions*>(state);
            auto& delegate = options->_delegate;
            std::vector<Ortpy::EpDevice> devices;
            devices.reserve(num_devices);
            for (size_t i = 0; i < num_devices; ++i)
            {
                devices.emplace_back(ep_devices[i]);
            }
            auto modelMetadata = Ortpy::KeyValuePairsToMap(model_metadata);
            auto runtimeMetadata = Ortpy::KeyValuePairsToMap(runtime_metadata);
            std::vector<Ortpy::EpDevice> selectedDevices{};
            {
                nanobind::gil_scoped_acquire acquire;
                selectedDevices = delegate(devices, modelMetadata, runtimeMetadata, max_selected);
            }
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
    Ortpy::Status status = GetApi()->SessionOptionsSetEpSelectionPolicyDelegate(
        _ptr,
        delegateWrapper,
        /** If the session_options gets deleted somehow, this will cause an invalid access. */
        this);
    status.Check();
}

Ortpy::ModelCompilationOptions Ortpy::SessionOptions::CreateModelCompilationOptions() const
{
    OrtModelCompilationOptions* options = nullptr;
    Ortpy::Status status = GetApi()->GetCompileApi()->CreateModelCompilationOptionsFromSessionOptions(
        *Env::GetSingleton(), _ptr, &options);
    status.Check();
    return ModelCompilationOptions{ options };
}

void Ortpy::SessionOptions::RegisterCustomOpsLibrary_V2(const std::string& libraryName)
{
    Ortpy::Status status = GetApi()->RegisterCustomOpsLibrary_V2(
        _ptr, StringToOrtString(libraryName).c_str());
    status.Check();
}

void Ortpy::SessionOptions::RegisterCustomOpsUsingFunction(const std::string& registrationFuncName)
{
    Ortpy::Status status = GetApi()->RegisterCustomOpsUsingFunction(
        _ptr, registrationFuncName.c_str());
    status.Check();
}

void Ortpy::SessionOptions::EnableOrtCustomOps()
{
    Ortpy::Status status = GetApi()->EnableOrtCustomOps(_ptr);
    status.Check();
}

void Ortpy::SessionOptions::AddFreeDimensionOverride(const std::string& dimDenotation, int64_t dimValue)
{
    Ortpy::Status status = GetApi()->AddFreeDimensionOverride(
        _ptr, dimDenotation.c_str(), dimValue);
    status.Check();
}

void Ortpy::SessionOptions::AddFreeDimensionOverrideByName(const std::string& dimName, int64_t dimValue)
{
    Ortpy::Status status = GetApi()->AddFreeDimensionOverrideByName(
        _ptr, dimName.c_str(), dimValue);
    status.Check();
}

void Ortpy::SessionOptions::DisablePerSessionThreads()
{
    Ortpy::Status status = GetApi()->DisablePerSessionThreads(_ptr);
    status.Check();
}

void Ortpy::SessionOptions::AddSessionConfigEntry(const std::string& configKey, const std::string& configValue)
{
    Ortpy::Status status = GetApi()->AddSessionConfigEntry(
        _ptr, configKey.c_str(), configValue.c_str());
    status.Check();
}

bool Ortpy::SessionOptions::HasSessionConfigEntry(const std::string& configKey) const
{
    int out = 0;
    Ortpy::Status status = GetApi()->HasSessionConfigEntry(_ptr, configKey.c_str(), &out);
    status.Check();
    return out != 0;
}

std::string Ortpy::SessionOptions::GetSessionConfigEntry(const std::string& configKey) const
{
    /** First call to get size */
    size_t size = 0;
    Ortpy::Status status = GetApi()->GetSessionConfigEntry(_ptr, configKey.c_str(), nullptr, &size);
    status.Check();
    std::string value(size, '\0');
    status = GetApi()->GetSessionConfigEntry(_ptr, configKey.c_str(), value.data(), &size);
    status.Check();
    /** Remove trailing null */
    if (!value.empty() && value.back() == '\0')
    {
        value.pop_back();
    }
    return value;
}

std::unordered_map<std::string, std::string> Ortpy::SessionOptions::GetSessionOptionsConfigEntries() const
{
    OrtKeyValuePairs* pairs = nullptr;
    Ortpy::Status status = GetApi()->GetSessionOptionsConfigEntries(_ptr, &pairs);
    status.Check();
    auto result = KeyValuePairsToMap(pairs);
    GetApi()->ReleaseKeyValuePairs(pairs);
    return result;
}

void Ortpy::SessionOptions::SetDeterministicCompute(bool value)
{
    Ortpy::Status status = GetApi()->SetDeterministicCompute(_ptr, value);
    status.Check();
}

void Ortpy::SessionOptions::SetLoadCancellationFlag(bool cancel)
{
    Ortpy::Status status = GetApi()->SessionOptionsSetLoadCancellationFlag(_ptr, cancel);
    status.Check();
}

Ortpy::SessionOptions Ortpy::SessionOptions::Clone() const
{
    OrtSessionOptions* cloned = nullptr;
    Ortpy::Status status = GetApi()->CloneSessionOptions(_ptr, &cloned);
    status.Check();
    return SessionOptions{ cloned };
}

void Ortpy::SessionOptions::AppendExecutionProvider(
    const std::string& providerName,
    const std::unordered_map<std::string, std::string>& providerOptions)
{
    std::vector<const char*> keys;
    std::vector<const char*> values;
    MapToKeyValueArrays(providerOptions, keys, values);
    Ortpy::Status status = GetApi()->SessionOptionsAppendExecutionProvider(
        _ptr, providerName.c_str(),
        keys.data(), values.data(), keys.size());
    status.Check();
}

/** TypeInfo */

void Ortpy::TypeInfo::ReleaseOrtType(OrtTypeInfo* ptr)
{
    GetApi()->ReleaseTypeInfo(ptr);
}

ONNXType Ortpy::TypeInfo::GetOnnxType() const
{
    ONNXType type;
    Ortpy::Status status = GetApi()->GetOnnxTypeFromTypeInfo(_ptr, &type);
    status.Check();
    return type;
}

std::string Ortpy::TypeInfo::GetDenotation() const
{
    const char* denotation = nullptr;
    size_t len = 0;
    Ortpy::Status status = GetApi()->GetDenotationFromTypeInfo(_ptr, &denotation, &len);
    status.Check();
    return denotation ? std::string(denotation, len) : "";
}

/** TensorTypeAndShapeInfo */

void Ortpy::TensorTypeAndShapeInfo::ReleaseOrtType(OrtTensorTypeAndShapeInfo* ptr)
{
    GetApi()->ReleaseTensorTypeAndShapeInfo(ptr);
}

size_t Ortpy::TensorTypeAndShapeInfo::GetElementCount() const
{
    size_t count = 0;
    Ortpy::Status status = GetApi()->GetTensorShapeElementCount(_ptr, &count);
    status.Check();
    return count;
}

/** TensorInfo */

Ortpy::TensorInfo::TensorInfo(const TypeInfo& typeInfo)
{
    const OrtTensorTypeAndShapeInfo* tensorInfo = nullptr;
    /** DO NOT free the tensorInfo. It's bind to the typeInfo */
    Ortpy::Status status = GetApi()->CastTypeInfoToTensorInfo(typeInfo, &tensorInfo);
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
    dtype = Ortpy::Value::OrtTypeToNpType(type);
}

/** ModelMetadata */

void Ortpy::ModelMetadata::ReleaseOrtType(OrtModelMetadata* ptr)
{
    GetApi()->ReleaseModelMetadata(ptr);
}

std::string Ortpy::ModelMetadata::GetProducerName() const
{
    auto allocator = GetAllocator();
    char* value = nullptr;
    Ortpy::Status status = GetApi()->ModelMetadataGetProducerName(_ptr, allocator, &value);
    status.Check();
    std::string result{ value ? value : "" };
    allocator->Free(allocator, value);
    return result;
}

std::string Ortpy::ModelMetadata::GetGraphName() const
{
    auto allocator = GetAllocator();
    char* value = nullptr;
    Ortpy::Status status = GetApi()->ModelMetadataGetGraphName(_ptr, allocator, &value);
    status.Check();
    std::string result{ value ? value : "" };
    allocator->Free(allocator, value);
    return result;
}

std::string Ortpy::ModelMetadata::GetDomain() const
{
    auto allocator = GetAllocator();
    char* value = nullptr;
    Ortpy::Status status = GetApi()->ModelMetadataGetDomain(_ptr, allocator, &value);
    status.Check();
    std::string result{ value ? value : "" };
    allocator->Free(allocator, value);
    return result;
}

std::string Ortpy::ModelMetadata::GetDescription() const
{
    auto allocator = GetAllocator();
    char* value = nullptr;
    Ortpy::Status status = GetApi()->ModelMetadataGetDescription(_ptr, allocator, &value);
    status.Check();
    std::string result{ value ? value : "" };
    allocator->Free(allocator, value);
    return result;
}

std::string Ortpy::ModelMetadata::GetGraphDescription() const
{
    auto allocator = GetAllocator();
    char* value = nullptr;
    Ortpy::Status status = GetApi()->ModelMetadataGetGraphDescription(_ptr, allocator, &value);
    status.Check();
    std::string result{ value ? value : "" };
    allocator->Free(allocator, value);
    return result;
}

int64_t Ortpy::ModelMetadata::GetVersion() const
{
    int64_t value = 0;
    Ortpy::Status status = GetApi()->ModelMetadataGetVersion(_ptr, &value);
    status.Check();
    return value;
}

std::unordered_map<std::string, std::string> Ortpy::ModelMetadata::GetCustomMetadataMap() const
{
    auto allocator = GetAllocator();
    char** keys = nullptr;
    int64_t numKeys = 0;
    Ortpy::Status status = GetApi()->ModelMetadataGetCustomMetadataMapKeys(_ptr, allocator, &keys, &numKeys);
    status.Check();
    std::unordered_map<std::string, std::string> result;
    for (int64_t i = 0; i < numKeys; ++i)
    {
        std::string key{ keys[i] };
        allocator->Free(allocator, keys[i]);
        char* value = nullptr;
        status = GetApi()->ModelMetadataLookupCustomMetadataMap(_ptr, allocator, key.c_str(), &value);
        status.Check();
        result[key] = value ? value : "";
        allocator->Free(allocator, value);
    }
    allocator->Free(allocator, keys);
    return result;
}

std::optional<std::string> Ortpy::ModelMetadata::LookupCustomMetadata(const std::string& key) const
{
    auto allocator = GetAllocator();
    char* value = nullptr;
    Ortpy::Status status = GetApi()->ModelMetadataLookupCustomMetadataMap(_ptr, allocator, key.c_str(), &value);
    status.Check();
    if (value == nullptr)
    {
        return std::nullopt;
    }
    std::string result{ value };
    allocator->Free(allocator, value);
    return result;
}

/** RunOptions */

void Ortpy::RunOptions::ReleaseOrtType(OrtRunOptions* ptr)
{
    GetApi()->ReleaseRunOptions(ptr);
}

Ortpy::RunOptions::RunOptions()
    : OrtTypeWrapper<OrtRunOptions, RunOptions>(nullptr)
{
    Ortpy::Status status = GetApi()->CreateRunOptions(&_ptr);
    status.Check();
}

void Ortpy::RunOptions::SetRunLogVerbosityLevel(int level)
{
    Ortpy::Status status = GetApi()->RunOptionsSetRunLogVerbosityLevel(_ptr, level);
    status.Check();
}

int Ortpy::RunOptions::GetRunLogVerbosityLevel() const
{
    int level = 0;
    Ortpy::Status status = GetApi()->RunOptionsGetRunLogVerbosityLevel(_ptr, &level);
    status.Check();
    return level;
}

void Ortpy::RunOptions::SetRunLogSeverityLevel(int level)
{
    Ortpy::Status status = GetApi()->RunOptionsSetRunLogSeverityLevel(_ptr, level);
    status.Check();
}

int Ortpy::RunOptions::GetRunLogSeverityLevel() const
{
    int level = 0;
    Ortpy::Status status = GetApi()->RunOptionsGetRunLogSeverityLevel(_ptr, &level);
    status.Check();
    return level;
}

void Ortpy::RunOptions::SetRunTag(const std::string& tag)
{
    Ortpy::Status status = GetApi()->RunOptionsSetRunTag(_ptr, tag.c_str());
    status.Check();
}

std::string Ortpy::RunOptions::GetRunTag() const
{
    const char* tag = nullptr;
    Ortpy::Status status = GetApi()->RunOptionsGetRunTag(_ptr, &tag);
    status.Check();
    return tag ? tag : "";
}

void Ortpy::RunOptions::SetTerminate()
{
    Ortpy::Status status = GetApi()->RunOptionsSetTerminate(_ptr);
    status.Check();
}

void Ortpy::RunOptions::UnsetTerminate()
{
    Ortpy::Status status = GetApi()->RunOptionsUnsetTerminate(_ptr);
    status.Check();
}

void Ortpy::RunOptions::AddRunConfigEntry(const std::string& configKey, const std::string& configValue)
{
    Ortpy::Status status = GetApi()->AddRunConfigEntry(
        _ptr, configKey.c_str(), configValue.c_str());
    status.Check();
}

std::optional<std::string> Ortpy::RunOptions::GetRunConfigEntry(const std::string& configKey) const
{
    const char* value = GetApi()->GetRunConfigEntry(_ptr, configKey.c_str());
    if (value == nullptr)
    {
        return std::nullopt;
    }
    return std::string{ value };
}

void Ortpy::RunOptions::AddActiveLoraAdapter(const LoraAdapter& adapter)
{
    Ortpy::Status status = GetApi()->RunOptionsAddActiveLoraAdapter(_ptr, adapter);
    status.Check();
}

/** Session */

Ortpy::Session::Session(const std::string& modelPath, const SessionOptions& options)
    : OrtTypeWrapper<OrtSession, Session>(nullptr)
{
    OrtSession* session = nullptr;
    Ortpy::Status status = GetApi()->CreateSession(
        *Ortpy::Env::GetSingleton(),
        StringToOrtString(modelPath).c_str(),
        options,
        &session);
    status.Check();
    _ptr = session;
}

Ortpy::Session::Session(const nanobind::bytes& modelBytes, const SessionOptions& options)
    : OrtTypeWrapper<OrtSession, Session>(nullptr)
{
    OrtSession* session = nullptr;
    Ortpy::Status status = GetApi()->CreateSessionFromArray(
        *Ortpy::Env::GetSingleton(),
        modelBytes.data(),
        modelBytes.size(),
        options,
        &session);
    status.Check();
    _ptr = session;
}

void Ortpy::Session::ReleaseOrtType(OrtSession* ptr)
{
    GetApi()->ReleaseSession(ptr);
}

std::unordered_map<std::string, Ortpy::TensorInfo> Ortpy::Session::GetInputInfo() const
{
    size_t inputCount = 0;
    Ortpy::Status status = GetApi()->SessionGetInputCount(_ptr, &inputCount);
    status.Check();
    std::unordered_map<std::string, Ortpy::TensorInfo> inputInfo;
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

std::unordered_map<std::string, Ortpy::TensorInfo> Ortpy::Session::GetOutputInfo() const
{
    size_t outputCount = 0;
    Ortpy::Status status = GetApi()->SessionGetOutputCount(_ptr, &outputCount);
    status.Check();
    std::unordered_map<std::string, Ortpy::TensorInfo> outputInfo;
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

Ortpy::ModelMetadata Ortpy::Session::GetModelMetadata() const
{
    OrtModelMetadata* metadata = nullptr;
    Ortpy::Status status = GetApi()->SessionGetModelMetadata(_ptr, &metadata);
    status.Check();
    return ModelMetadata{ metadata };
}

std::string Ortpy::Session::EndProfiling() const
{
    auto allocator = GetAllocator();
    char* value = nullptr;
    Ortpy::Status status = GetApi()->SessionEndProfiling(_ptr, allocator, &value);
    status.Check();
    std::string result{ value ? value : "" };
    allocator->Free(allocator, value);
    return result;
}

uint64_t Ortpy::Session::GetProfilingStartTimeNs() const
{
    uint64_t value = 0;
    Ortpy::Status status = GetApi()->SessionGetProfilingStartTimeNs(_ptr, &value);
    status.Check();
    return value;
}

std::unordered_map<std::string, Ortpy::Value> Ortpy::Session::Run(
    const std::unordered_map<std::string, Ortpy::NpArray>& inputs,
    const std::optional<std::vector<std::string>>& outputNamesOpt,
    const std::optional<std::reference_wrapper<Ortpy::RunOptions>>& runOptionsOpt) const
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
    std::vector<std::string> outputNames;
    std::vector<const char*> outputNamesView;
    if (outputNamesOpt.has_value())
    {
        outputNames = outputNamesOpt.value();
    }
    else
    {
        auto outputInfo = GetOutputInfo();
        outputNames.reserve(outputInfo.size());
        for (const auto& pair : outputInfo)
        {
            outputNames.push_back(pair.first);
        }
    }
    outputNamesView.reserve(outputNames.size());
    for (const auto& name : outputNames)
    {
        outputNamesView.push_back(name.c_str());
    }
    /** Let ort allocate the output values as we may not known their shapes */
    std::vector<OrtValue*> outputValues(outputNamesView.size(), nullptr);
    std::vector<Value> outputValuesWrapper;
    outputValuesWrapper.reserve(outputNamesView.size());
    /** Run the session */
    OrtRunOptions* runOptions = runOptionsOpt.has_value()
        ? static_cast<OrtRunOptions*>(runOptionsOpt.value().get())
        : nullptr;
    Ortpy::Status status = GetApi()->Run(
        _ptr, runOptions,
        inputNamesView.data(), inputValuesView.data(), inputs.size(),
        outputNamesView.data(), outputNamesView.size(), outputValues.data());
    status.Check();
    /** Create output values (part 2) */
    for (auto value : outputValues)
    {
        /** safe guard the raw values first. */
        outputValuesWrapper.emplace_back(value);
    }
    std::unordered_map<std::string, Value> outputs;
    size_t i = 0;
    for (const auto& name : outputNames)
    {
        outputs.emplace(name, std::move(outputValuesWrapper[i++]));
    }
    return outputs;
}

/** Value */

Ortpy::Value::State::~State()
{
    if (ortValue != nullptr)
    {
        GetApi()->ReleaseValue(ortValue);
    }
}

Ortpy::Value::Value(OrtValue* ptr)
{
    if (ptr == nullptr)
    {
        /** Create an empty value */
        return;
    }
    _state->ortValue = ptr;
    auto npType = OrtTypeToNpType(GetType());
    auto ortShape = GetShape();
    std::vector<size_t> npShape(ortShape.begin(), ortShape.end());
    auto sharedStateHeldByNpArray = new std::shared_ptr<State>(_state);
    nanobind::capsule owner(sharedStateHeldByNpArray, [](void* p) noexcept {
        delete static_cast<std::shared_ptr<std::vector<uint8_t>>*>(p);
    });
    _state->npArray = NpArray(
        GetData(),
        npShape.size(),
        npShape.data(),
        owner,
        nullptr,
        npType);
}

Ortpy::Value::Value(const NpArray& npArray)
{
    /** Store the numpy array to keep the data alive. */
    _state->npArray = npArray;
    auto ortType = NpTypeToOrtType(npArray.dtype());
    std::vector<int64_t> ortShape;
    auto ndim = npArray.ndim();
    ortShape.reserve(ndim);
    for (int i = 0; i < ndim; i++)
    {
        ortShape.push_back(npArray.shape(i));
    }
    Ortpy::MemoryInfo memInfo{};
    Ortpy::Status status = GetApi()->CreateTensorWithDataAsOrtValue(
        memInfo,
        npArray.data(),
        npArray.nbytes(),
        ortShape.data(),
        ortShape.size(),
        ortType,
        &_state->ortValue);
    status.Check();
}

Ortpy::Value::Value(const std::vector<int64_t>& ortShape, ONNXTensorElementDataType ortType)
{
    Ortpy::Status status = GetApi()->CreateTensorAsOrtValue(
        GetAllocator(),
        ortShape.data(),
        ortShape.size(),
        ortType,
        &_state->ortValue);
    status.Check();
    auto npType = OrtTypeToNpType(ortType);
    std::vector<size_t> npShape(ortShape.begin(), ortShape.end());
    auto sharedStateHeldByNpArray = new std::shared_ptr<State>(_state);
    nanobind::capsule owner(sharedStateHeldByNpArray, [](void* p) noexcept {
        delete static_cast<std::shared_ptr<std::vector<uint8_t>>*>(p);
    });
    _state->npArray = NpArray(
        GetData(),
        npShape.size(),
        npShape.data(),
        owner,
        nullptr,
        npType);
}

Ortpy::Value::operator OrtValue*() const
{
    return _state->ortValue;
}

Ortpy::NpArray Ortpy::Value::ToNumpy() const
{
    if (!_state->npArray.has_value())
    {
        throw std::runtime_error("Value does not hold a numpy array");
    }
    return *(_state->npArray);
}

ONNXTensorElementDataType Ortpy::Value::GetType() const
{
    if (_state->ortValue == nullptr)
    {
        throw std::runtime_error("Value is empty");
    }
    OrtTensorTypeAndShapeInfo *info = nullptr;
    Ortpy::Status status = GetApi()->GetTensorTypeAndShape(_state->ortValue, &info);
    status.Check();
    ONNXTensorElementDataType type;
    status = GetApi()->GetTensorElementType(info, &type);
    status.Check();
    return type;
}

std::vector<int64_t> Ortpy::Value::GetShape() const
{
    if (_state->ortValue == nullptr)
    {
        throw std::runtime_error("Value is empty");
    }
    OrtTensorTypeAndShapeInfo *info = nullptr;
    Ortpy::Status status = GetApi()->GetTensorTypeAndShape(_state->ortValue, &info);
    status.Check();
    size_t dimCount = 0;
    status = GetApi()->GetDimensionsCount(info, &dimCount);
    status.Check();
    std::vector<int64_t> shape(dimCount);
    status = GetApi()->GetDimensions(info, shape.data(), dimCount);
    status.Check();
    return shape;
}

size_t Ortpy::Value::GetSize() const
{
    if (_state->ortValue == nullptr)
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

void* Ortpy::Value::GetData() const
{
    if (_state->ortValue == nullptr)
    {
        throw std::runtime_error("Value is empty");
    }
    void* data = nullptr;
    Ortpy::Status status = GetApi()->GetTensorMutableData(_state->ortValue, &data);
    if (data == nullptr)
    {
        throw std::runtime_error("Failed to get data from OrtValue.");
    }
    return data;
}

ONNXTensorElementDataType Ortpy::Value::NpTypeToOrtType(const nanobind::dlpack::dtype& npType)
{
    static std::map<nanobind::dlpack::dtype, ONNXTensorElementDataType> typeMap{};
    if (typeMap.empty())
    {
        typeMap[nanobind::dtype<bool>()] = ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
        typeMap[nanobind::dtype<int8_t>()] = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
        typeMap[nanobind::dtype<uint8_t>()] = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
        typeMap[nanobind::dtype<int16_t>()] = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
        typeMap[nanobind::dtype<uint16_t>()] = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
        typeMap[nanobind::dtype<int32_t>()] = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
        typeMap[nanobind::dtype<uint32_t>()] = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
        typeMap[nanobind::dtype<int64_t>()] = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
        typeMap[nanobind::dtype<uint64_t>()] = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
        typeMap[nanobind::dtype<float>()] = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        typeMap[nanobind::dtype<double>()] = ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
        typeMap[{ static_cast<uint8_t>(nanobind::dlpack::dtype_code::Float), 16, 1 }] = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    }
    auto it = typeMap.find(npType);
    if (it == typeMap.end())
    {
        throw std::runtime_error("Unsupported NumPy data type: " + std::to_string(npType.code) + ", " + std::to_string(npType.bits) + ", " + std::to_string(npType.lanes));
    }
    return it->second;
}

nanobind::dlpack::dtype Ortpy::Value::OrtTypeToNpType(ONNXTensorElementDataType ortType)
{
    switch (ortType) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return nanobind::dtype<float>();
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            return nanobind::dtype<uint8_t>();
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            return nanobind::dtype<int8_t>();
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            return nanobind::dtype<uint16_t>();
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            return nanobind::dtype<int16_t>();
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            return nanobind::dtype<int32_t>();
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            return nanobind::dtype<int64_t>();
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            return nanobind::dtype<bool>();
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            return { static_cast<uint8_t>(nanobind::dlpack::dtype_code::Float), 16, 1 };
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            return nanobind::dtype<double>();
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            return nanobind::dtype<uint32_t>();
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            return nanobind::dtype<uint64_t>();
    }
    throw std::runtime_error("Unsupported ONNX tensor element data type: " + std::to_string(ortType));
}

std::string Ortpy::Value::NpTypeToName(const nanobind::dlpack::dtype& npType)
{
    static std::map<nanobind::dlpack::dtype, std::string> typeMap{};
    if (typeMap.empty())
    {
        typeMap[nanobind::dtype<bool>()] = "bool";
        typeMap[nanobind::dtype<int8_t>()] = "int8";
        typeMap[nanobind::dtype<uint8_t>()] = "uint8";
        typeMap[nanobind::dtype<int16_t>()] = "int16";
        typeMap[nanobind::dtype<uint16_t>()] = "uint16";
        typeMap[nanobind::dtype<int32_t>()] = "int32";
        typeMap[nanobind::dtype<uint32_t>()] = "uint32";
        typeMap[nanobind::dtype<int64_t>()] = "int64";
        typeMap[nanobind::dtype<uint64_t>()] = "uint64";
        typeMap[nanobind::dtype<float>()] = "float32";
        typeMap[nanobind::dtype<double>()] = "float64";
        typeMap[{ static_cast<uint8_t>(nanobind::dlpack::dtype_code::Float), 16, 1 }] = "float16";
    }
    auto it = typeMap.find(npType);
    if (it == typeMap.end())
    {
        throw std::runtime_error("Unsupported NumPy data type: " + std::to_string(npType.code) + ", " + std::to_string(npType.bits) + ", " + std::to_string(npType.lanes));
    }
    return it->second;
}

size_t Ortpy::Value::GetSizeOfOrtType(ONNXTensorElementDataType ortType)
{
    switch (ortType) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return sizeof(float);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            return sizeof(uint8_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            return sizeof(int8_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            return sizeof(uint16_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            return sizeof(int16_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            return sizeof(int32_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            return sizeof(int64_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            return sizeof(bool);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            return 2;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            return sizeof(double);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            return sizeof(uint32_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            return sizeof(uint64_t);
    }
    throw std::runtime_error("Unsupported ONNX tensor element data type: " + std::to_string(ortType));
}

/** MemoryInfo */

Ortpy::MemoryInfo::MemoryInfo()
    : OrtTypeWrapper<OrtMemoryInfo, MemoryInfo>(nullptr)
{
    Ortpy::Status status = GetApi()->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &_ptr);
    status.Check();
}

Ortpy::MemoryInfo::MemoryInfo(const std::string& name, OrtAllocatorType allocatorType,
                               int deviceId, OrtMemType memType)
    : OrtTypeWrapper<OrtMemoryInfo, MemoryInfo>(nullptr)
{
    Ortpy::Status status = GetApi()->CreateMemoryInfo(
        name.c_str(), allocatorType, deviceId, memType, &_ptr);
    status.Check();
}

std::string Ortpy::MemoryInfo::GetName() const
{
    const char* name = nullptr;
    Ortpy::Status status = GetApi()->MemoryInfoGetName(_ptr, &name);
    status.Check();
    return name ? name : "";
}

int Ortpy::MemoryInfo::GetDeviceId() const
{
    int id = 0;
    Ortpy::Status status = GetApi()->MemoryInfoGetId(_ptr, &id);
    status.Check();
    return id;
}

OrtMemType Ortpy::MemoryInfo::GetMemType() const
{
    OrtMemType memType;
    Ortpy::Status status = GetApi()->MemoryInfoGetMemType(_ptr, &memType);
    status.Check();
    return memType;
}

OrtAllocatorType Ortpy::MemoryInfo::GetAllocatorType() const
{
    OrtAllocatorType type;
    Ortpy::Status status = GetApi()->MemoryInfoGetType(_ptr, &type);
    status.Check();
    return type;
}

OrtMemoryInfoDeviceType Ortpy::MemoryInfo::GetDeviceType() const
{
    OrtMemoryInfoDeviceType type;
    GetApi()->MemoryInfoGetDeviceType(_ptr, &type);
    return type;
}

bool Ortpy::MemoryInfo::operator==(const MemoryInfo& other) const
{
    int result = 0;
    Ortpy::Status status = GetApi()->CompareMemoryInfo(_ptr, other._ptr, &result);
    status.Check();
    return result == 0;
}

void Ortpy::MemoryInfo::ReleaseOrtType(OrtMemoryInfo* ptr)
{
    GetApi()->ReleaseMemoryInfo(ptr);
}

/** IoBinding */

void Ortpy::IoBinding::ReleaseOrtType(OrtIoBinding* ptr)
{
    GetApi()->ReleaseIoBinding(ptr);
}

Ortpy::IoBinding::IoBinding(const Session& session)
    : OrtTypeWrapper<OrtIoBinding, IoBinding>(nullptr)
{
    Ortpy::Status status = GetApi()->CreateIoBinding(session, &_ptr);
    status.Check();
}

void Ortpy::IoBinding::BindInput(const std::string& name, const Value& value)
{
    _boundInputValues.push_back(value);
    Ortpy::Status status = GetApi()->BindInput(_ptr, name.c_str(), value);
    status.Check();
}

void Ortpy::IoBinding::BindOutput(const std::string& name, const Value& value)
{
    _boundOutputValues.push_back(value);
    Ortpy::Status status = GetApi()->BindOutput(_ptr, name.c_str(), value);
    status.Check();
}

void Ortpy::IoBinding::BindOutputToDevice(const std::string& name, const MemoryInfo& memInfo)
{
    Ortpy::Status status = GetApi()->BindOutputToDevice(_ptr, name.c_str(), memInfo);
    status.Check();
}

std::unordered_map<std::string, Ortpy::Value> Ortpy::IoBinding::GetOutputs()
{
    auto allocator = GetAllocator();

    /** Get output names */
    char* namesBuffer = nullptr;
    size_t* lengths = nullptr;
    size_t count = 0;
    Ortpy::Status status = GetApi()->GetBoundOutputNames(_ptr, allocator, &namesBuffer, &lengths, &count);
    status.Check();

    std::vector<std::string> names;
    names.reserve(count);
    size_t offset = 0;
    for (size_t i = 0; i < count; ++i)
    {
        names.emplace_back(namesBuffer + offset, lengths[i]);
        offset += lengths[i];
    }
    allocator->Free(allocator, namesBuffer);
    allocator->Free(allocator, lengths);

    /** Get output values */
    OrtValue** values = nullptr;
    size_t valueCount = 0;
    status = GetApi()->GetBoundOutputValues(_ptr, allocator, &values, &valueCount);
    status.Check();

    std::unordered_map<std::string, Value> result;
    for (size_t i = 0; i < valueCount; ++i)
    {
        result.emplace(names[i], Value{ values[i] });
    }
    allocator->Free(allocator, values);

    return result;
}

void Ortpy::IoBinding::ClearInputs()
{
    _boundInputValues.clear();
    GetApi()->ClearBoundInputs(_ptr);
}

void Ortpy::IoBinding::ClearOutputs()
{
    _boundOutputValues.clear();
    GetApi()->ClearBoundOutputs(_ptr);
}

void Ortpy::IoBinding::SynchronizeInputs()
{
    Ortpy::Status status = GetApi()->SynchronizeBoundInputs(_ptr);
    status.Check();
}

void Ortpy::IoBinding::SynchronizeOutputs()
{
    Ortpy::Status status = GetApi()->SynchronizeBoundOutputs(_ptr);
    status.Check();
}

/** Session::IoBinding methods */

Ortpy::IoBinding Ortpy::Session::CreateIoBinding() const
{
    return IoBinding{ *this };
}

void Ortpy::Session::RunWithBinding(
    IoBinding& binding,
    const std::optional<std::reference_wrapper<RunOptions>>& runOptionsOpt) const
{
    OrtRunOptions* runOptions = runOptionsOpt.has_value()
        ? static_cast<OrtRunOptions*>(runOptionsOpt.value().get())
        : nullptr;
    Ortpy::Status status = GetApi()->RunWithBinding(_ptr, runOptions, binding);
    status.Check();
}

std::unordered_map<std::string, Ortpy::MemoryInfo> Ortpy::Session::GetMemoryInfoForInputs() const
{
    size_t inputCount = 0;
    Ortpy::Status status = GetApi()->SessionGetInputCount(_ptr, &inputCount);
    status.Check();
    std::vector<const OrtMemoryInfo*> memInfos(inputCount, nullptr);
    status = GetApi()->SessionGetMemoryInfoForInputs(_ptr, memInfos.data(), inputCount);
    status.Check();
    auto allocator = GetAllocator();
    std::unordered_map<std::string, MemoryInfo> result;
    for (size_t i = 0; i < inputCount; ++i)
    {
        char* nameRaw = nullptr;
        status = GetApi()->SessionGetInputName(_ptr, i, allocator, &nameRaw);
        status.Check();
        std::string name{ nameRaw };
        allocator->Free(allocator, nameRaw);
        /** Create a copy of the borrowed MemoryInfo */
        const char* miName = nullptr;
        GetApi()->MemoryInfoGetName(memInfos[i], &miName);
        OrtAllocatorType miAllocType;
        GetApi()->MemoryInfoGetType(memInfos[i], &miAllocType);
        int miId = 0;
        GetApi()->MemoryInfoGetId(memInfos[i], &miId);
        OrtMemType miMemType;
        GetApi()->MemoryInfoGetMemType(memInfos[i], &miMemType);
        result.emplace(name, MemoryInfo{ miName ? miName : "Cpu", miAllocType, miId, miMemType });
    }
    return result;
}

std::unordered_map<std::string, Ortpy::MemoryInfo> Ortpy::Session::GetMemoryInfoForOutputs() const
{
    size_t outputCount = 0;
    Ortpy::Status status = GetApi()->SessionGetOutputCount(_ptr, &outputCount);
    status.Check();
    std::vector<const OrtMemoryInfo*> memInfos(outputCount, nullptr);
    status = GetApi()->SessionGetMemoryInfoForOutputs(_ptr, memInfos.data(), outputCount);
    status.Check();
    auto allocator = GetAllocator();
    std::unordered_map<std::string, MemoryInfo> result;
    for (size_t i = 0; i < outputCount; ++i)
    {
        char* nameRaw = nullptr;
        status = GetApi()->SessionGetOutputName(_ptr, i, allocator, &nameRaw);
        status.Check();
        std::string name{ nameRaw };
        allocator->Free(allocator, nameRaw);
        const char* miName = nullptr;
        GetApi()->MemoryInfoGetName(memInfos[i], &miName);
        OrtAllocatorType miAllocType;
        GetApi()->MemoryInfoGetType(memInfos[i], &miAllocType);
        int miId = 0;
        GetApi()->MemoryInfoGetId(memInfos[i], &miId);
        OrtMemType miMemType;
        GetApi()->MemoryInfoGetMemType(memInfos[i], &miMemType);
        result.emplace(name, MemoryInfo{ miName ? miName : "Cpu", miAllocType, miId, miMemType });
    }
    return result;
}

std::unordered_map<std::string, Ortpy::EpDevice> Ortpy::Session::GetEpDeviceForInputs() const
{
    size_t inputCount = 0;
    Ortpy::Status status = GetApi()->SessionGetInputCount(_ptr, &inputCount);
    status.Check();
    std::vector<const OrtEpDevice*> epDevices(inputCount, nullptr);
    status = GetApi()->SessionGetEpDeviceForInputs(_ptr, epDevices.data(), inputCount);
    status.Check();
    auto allocator = GetAllocator();
    std::unordered_map<std::string, EpDevice> result;
    for (size_t i = 0; i < inputCount; ++i)
    {
        if (epDevices[i] == nullptr) continue;
        char* nameRaw = nullptr;
        status = GetApi()->SessionGetInputName(_ptr, i, allocator, &nameRaw);
        status.Check();
        std::string name{ nameRaw };
        allocator->Free(allocator, nameRaw);
        result.emplace(name, EpDevice{ epDevices[i] });
    }
    return result;
}

#if ORT_API_VERSION >= 24
std::unordered_map<std::string, Ortpy::EpDevice> Ortpy::Session::GetEpDeviceForOutputs() const
{
    size_t outputCount = 0;
    Ortpy::Status status = GetApi()->SessionGetOutputCount(_ptr, &outputCount);
    status.Check();
    std::vector<const OrtEpDevice*> epDevices(outputCount, nullptr);
    status = GetApi()->SessionGetEpDeviceForOutputs(_ptr, epDevices.data(), outputCount);
    status.Check();
    auto allocator = GetAllocator();
    std::unordered_map<std::string, EpDevice> result;
    for (size_t i = 0; i < outputCount; ++i)
    {
        if (epDevices[i] == nullptr) continue;
        char* nameRaw = nullptr;
        status = GetApi()->SessionGetOutputName(_ptr, i, allocator, &nameRaw);
        status.Check();
        std::string name{ nameRaw };
        allocator->Free(allocator, nameRaw);
        result.emplace(name, EpDevice{ epDevices[i] });
    }
    return result;
}

std::vector<Ortpy::EpAssignedSubgraph> Ortpy::Session::GetEpGraphAssignmentInfo() const
{
    const OrtEpAssignedSubgraph* const* subgraphsRaw = nullptr;
    size_t numSubgraphs = 0;
    Ortpy::Status status = GetApi()->Session_GetEpGraphAssignmentInfo(_ptr, &subgraphsRaw, &numSubgraphs);
    status.Check();
    std::vector<EpAssignedSubgraph> result;
    result.reserve(numSubgraphs);
    for (size_t i = 0; i < numSubgraphs; ++i)
    {
        EpAssignedSubgraph subgraph;
        const char* epName = nullptr;
        status = GetApi()->EpAssignedSubgraph_GetEpName(subgraphsRaw[i], &epName);
        status.Check();
        subgraph.epName = epName ? epName : "";
        const OrtEpAssignedNode* const* nodesRaw = nullptr;
        size_t numNodes = 0;
        status = GetApi()->EpAssignedSubgraph_GetNodes(subgraphsRaw[i], &nodesRaw, &numNodes);
        status.Check();
        subgraph.nodes.reserve(numNodes);
        for (size_t j = 0; j < numNodes; ++j)
        {
            EpAssignedNode node;
            const char* val = nullptr;
            status = GetApi()->EpAssignedNode_GetName(nodesRaw[j], &val);
            status.Check();
            node.name = val ? val : "";
            status = GetApi()->EpAssignedNode_GetDomain(nodesRaw[j], &val);
            status.Check();
            node.domain = val ? val : "";
            status = GetApi()->EpAssignedNode_GetOperatorType(nodesRaw[j], &val);
            status.Check();
            node.operatorType = val ? val : "";
            subgraph.nodes.push_back(std::move(node));
        }
        result.push_back(std::move(subgraph));
    }
    return result;
}
#endif /** ORT_API_VERSION >= 24 */

/** LoraAdapter */

void Ortpy::LoraAdapter::ReleaseOrtType(OrtLoraAdapter* ptr)
{
    GetApi()->ReleaseLoraAdapter(ptr);
}

Ortpy::LoraAdapter::LoraAdapter(const std::string& adapterFilePath)
    : OrtTypeWrapper<OrtLoraAdapter, LoraAdapter>(nullptr)
{
    Ortpy::Status status = GetApi()->CreateLoraAdapter(
        StringToOrtString(adapterFilePath).c_str(), nullptr, &_ptr);
    status.Check();
}

Ortpy::LoraAdapter::LoraAdapter(const nanobind::bytes& adapterBytes)
    : OrtTypeWrapper<OrtLoraAdapter, LoraAdapter>(nullptr)
{
    Ortpy::Status status = GetApi()->CreateLoraAdapterFromArray(
        adapterBytes.data(), adapterBytes.size(), nullptr, &_ptr);
    status.Check();
}
