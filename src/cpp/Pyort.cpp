#include "Pyort.h"
#include <cstring>
#include <string>
#include <map>
#include <nanobind/stl/function.h>

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

/** nanobind::dlpack::dtype modifications to be a map key */
namespace nanobind::dlpack {
    inline bool operator<(const dtype& a, const dtype& b) {
        if (a.code != b.code) return a.code < b.code;
        if (a.bits != b.bits) return a.bits < b.bits;
        return a.lanes < b.lanes;
    }
}

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

void Pyort::ModelCompilationOptions::SetInputModelFromBuffer(const nanobind::bytes& modelBytes)
{
    Pyort::Status status = GetApi()->GetCompileApi()->ModelCompilationOptions_SetInputModelFromBuffer(
        _ptr,
        modelBytes.data(),
        modelBytes.size());
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

nanobind::bytes Pyort::ModelCompilationOptions::CompileModelToBuffer()
{
    void* buffer = nullptr;
    size_t bufferSize = 0;
    Pyort::Status status = GetApi()->GetCompileApi()->ModelCompilationOptions_SetOutputModelBuffer(
        _ptr, GetAllocator(), &buffer, &bufferSize); 
    status.Check();
    status = GetApi()->GetCompileApi()->CompileModel(*Env::GetSingleton(), _ptr);
    status.Check();
    nanobind::bytes result{ static_cast<const char*>(buffer), bufferSize };
    GetAllocator()->Free(GetAllocator(), buffer);
    return result;
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

int Pyort::SessionOptions::TpTraverse(PyObject* self, visitproc visit, void* arg) noexcept
{
    try
    {        
        // On Python 3.9+, we must traverse the implicit dependency
        // of an object on its associated type object.
        #if PY_VERSION_HEX >= 0x03090000
            Py_VISIT(Py_TYPE(self));
        #endif

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

int Pyort::SessionOptions::TpClear(PyObject* self) noexcept
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

void Pyort::SessionOptions::SetEpSelectionPolicyDelegate(const EpSelectionPolicyDelegate& delegate)
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
            Pyort::SessionOptions* options = static_cast<Pyort::SessionOptions*>(state);
            auto& delegate = options->_delegate;
            std::vector<Pyort::EpDevice> devices;
            devices.reserve(num_devices);
            for (size_t i = 0; i < num_devices; ++i)
            {
                devices.emplace_back(ep_devices[i]);
            }
            auto modelMetadata = Pyort::KeyValuePairsToMap(model_metadata);
            auto runtimeMetadata = Pyort::KeyValuePairsToMap(runtime_metadata);
            std::vector<Pyort::EpDevice> selectedDevices{};
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
    Pyort::Status status = GetApi()->SessionOptionsSetEpSelectionPolicyDelegate(
        _ptr,
        delegateWrapper,
        /** If the session_options gets deleted somehow, this will cause an invalid access. */
        this);
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

std::unordered_map<std::string, Pyort::NpArray> Pyort::Session::Run(
    const std::unordered_map<std::string, Pyort::NpArray>& inputs) const
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
        _ptr, nullptr,
        inputNamesView.data(), inputValuesView.data(), inputs.size(),
        outputNamesView.data(), outputInfo.size(), outputValues.data());
    status.Check();
    /** Create output values (part 2) */
    for (auto value : outputValues)
    {
        /** safe guard the raw values first. */
        outputValuesWrapper.emplace_back(value);
    }
    std::unordered_map<std::string, NpArray> outputs;
    size_t i = 0;
    for (const auto& pair : outputInfo)
    {
        outputs[pair.first] = outputValuesWrapper[i++];
    }
    return outputs;
}

/** Value */

Pyort::Value::State::~State()
{
    if (ortValue != nullptr)
    {
        GetApi()->ReleaseValue(ortValue);
    }
}

Pyort::Value::Value(OrtValue* ptr)
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

Pyort::Value::Value(const NpArray& npArray)
{
    /** npArray (indirectly) holds the data, the Value holds a reference. */
    auto ortType = NpTypeToOrtType(npArray.dtype());
    std::vector<int64_t> ortShape;
    auto ndim = npArray.ndim();
    ortShape.reserve(ndim);
    for (int i = 0; i < ndim; i++)
    {
        ortShape.push_back(npArray.shape(i));
    }
    Pyort::MemoryInfo memInfo{};
    Pyort::Status status = GetApi()->CreateTensorWithDataAsOrtValue(
        memInfo,
        npArray.data(),
        npArray.nbytes(),
        ortShape.data(),
        ortShape.size(),
        ortType,
        &_state->ortValue);
    status.Check();
}

Pyort::Value::Value(const std::vector<int64_t>& ortShape, ONNXTensorElementDataType ortType)
{
    Pyort::Status status = GetApi()->CreateTensorAsOrtValue(
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

Pyort::Value::operator Pyort::NpArray() const
{
    if (!_state->npArray.has_value())
    {
        throw std::runtime_error("Value does not hold a numpy array");
    }
    return *(_state->npArray);
}

Pyort::Value::operator OrtValue*() const
{
    return _state->ortValue;
}

ONNXTensorElementDataType Pyort::Value::GetType() const
{
    if (_state->ortValue == nullptr)
    {
        throw std::runtime_error("Value is empty");
    }
    OrtTensorTypeAndShapeInfo *info = nullptr;
    Pyort::Status status = GetApi()->GetTensorTypeAndShape(_state->ortValue, &info);
    status.Check();
    ONNXTensorElementDataType type;
    status = GetApi()->GetTensorElementType(info, &type);
    status.Check();
    return type;
}

std::vector<int64_t> Pyort::Value::GetShape() const
{
    if (_state->ortValue == nullptr)
    {
        throw std::runtime_error("Value is empty");
    }
    OrtTensorTypeAndShapeInfo *info = nullptr;
    Pyort::Status status = GetApi()->GetTensorTypeAndShape(_state->ortValue, &info);
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

void* Pyort::Value::GetData() const
{
    if (_state->ortValue == nullptr)
    {
        throw std::runtime_error("Value is empty");
    }
    void* data = nullptr;
    Pyort::Status status = GetApi()->GetTensorMutableData(_state->ortValue, &data);
    if (data == nullptr)
    {
        throw std::runtime_error("Failed to get data from OrtValue.");
    }
    return data;
}

ONNXTensorElementDataType Pyort::Value::NpTypeToOrtType(const nanobind::dlpack::dtype& npType)
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

nanobind::dlpack::dtype Pyort::Value::OrtTypeToNpType(ONNXTensorElementDataType ortType)
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

std::string Pyort::Value::NpTypeToName(const nanobind::dlpack::dtype& npType)
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

size_t Pyort::Value::GetSizeOfOrtType(ONNXTensorElementDataType ortType)
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
