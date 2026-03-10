#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <memory>
#include <unordered_map>
#include <vector>
#include <optional>
#include <functional>

/** Use the C API for maximum compatibility */
#include <onnxruntime_c_api.h>

namespace Ortpy
{
    using NpArray = nanobind::ndarray<nanobind::numpy, nanobind::device::cpu, nanobind::c_contig>;

    const OrtApi* GetApi();
    OrtAllocator* GetAllocator();
    std::unordered_map<std::string, std::string> KeyValuePairsToMap(const OrtKeyValuePairs* pairs);
    std::vector<std::string> GetAvailableProviders();

    template <typename T, typename Derived>
    class OrtTypeWrapper
    {
    public:
        OrtTypeWrapper(T* ptr)
            : _ptr(ptr)
        {
        }
        ~OrtTypeWrapper()
        {
            if (_ptr)
            {
                Derived::ReleaseOrtType(_ptr);
                _ptr = nullptr;
            }
        }
        OrtTypeWrapper(const OrtTypeWrapper&) = delete;
        OrtTypeWrapper& operator=(const OrtTypeWrapper&) = delete;
        OrtTypeWrapper(OrtTypeWrapper&& other) noexcept
            : _ptr(other._ptr)
        {
            other._ptr = nullptr;
        }
        OrtTypeWrapper& operator=(OrtTypeWrapper&& other) noexcept
        {
            if (this != &other)
            {
                if (_ptr)
                {
                    Derived::ReleaseOrtType(_ptr);
                }
                _ptr = other._ptr;
                other._ptr = nullptr;
            }
            return *this;
        }
        operator T*() const
        {
            return _ptr;
        }
    protected:
        T* _ptr{ nullptr };
    };

    class Status : public OrtTypeWrapper<OrtStatus, Status>
    {
    public:
        static void ReleaseOrtType(OrtStatus* ptr);
        using OrtTypeWrapper::OrtTypeWrapper;
        OrtErrorCode GetErrorCode() const;
        std::string GetErrorMessage() const;
        void Check() const;
    };

    struct HardwareDevice
    {
        OrtHardwareDeviceType type;
        uint32_t vendorId;
        std::string vendor;
        uint32_t deviceId;
        std::unordered_map<std::string, std::string> metadata;
        HardwareDevice(const OrtHardwareDevice* device);
        HardwareDevice() = default;
    };

    class MemoryInfo;

    struct EpDevice
    {
        std::string epName;
        std::string epVendor;
        std::unordered_map<std::string, std::string> epMetadata;
        std::unordered_map<std::string, std::string> epOptions;
        HardwareDevice device;
        EpDevice(const OrtEpDevice* epDevice);
        EpDevice() = default;
        operator const OrtEpDevice*() const;
        /** Declared after MemoryInfo is complete. Implemented in Ortpy.cpp. */
        std::optional<MemoryInfo> GetMemoryInfo(OrtDeviceMemoryType memoryType) const;
    private:
        const OrtEpDevice* _ptr{ nullptr };
    };

    OrtCompiledModelCompatibility GetModelCompatibilityForEpDevices(
        const std::vector<EpDevice>& epDevices, const std::string& compatibilityInfo);

    class ThreadingOptions : public OrtTypeWrapper<OrtThreadingOptions, ThreadingOptions>
    {
    public:
        static void ReleaseOrtType(OrtThreadingOptions* ptr);
        ThreadingOptions();
        void SetIntraOpNumThreads(int numThreads);
        void SetInterOpNumThreads(int numThreads);
        void SetSpinControl(bool allowSpinning);
        void SetDenormalAsZero();
        void SetIntraOpThreadAffinity(const std::string& affinity);
    };

    class Env : public OrtTypeWrapper<OrtEnv, Env>
    {
    public:
        using LoggingFunction = std::function<
            void(OrtLoggingLevel severity,
                 const std::string& category,
                 const std::string& logid,
                 const std::string& codeLocation,
                 const std::string& message)>;

        static std::shared_ptr<Env> GetSingleton();
        static void ReleaseSingleton();
        static void ReleaseOrtType(OrtEnv* ptr);

        /** Explicit env creation — must be called before any other API that creates the env. */
        static void CreateEnv(
            OrtLoggingLevel logLevel = ORT_LOGGING_LEVEL_WARNING,
            const std::string& logId = "Ortpy",
            const LoggingFunction& loggingFunc = nullptr,
            const ThreadingOptions* threadingOpts = nullptr
#if ORT_API_VERSION >= 24
            , const std::unordered_map<std::string, std::string>& configEntries = {}
#endif /** ORT_API_VERSION >= 24 */
        );

        void RegisterExecutionProviderLibrary(const std::string& name, const std::string& path);
        void UnregisterExecutionProviderLibrary(const std::string& name);
        std::vector<EpDevice> GetEpDevices() const;
        void UpdateLogLevel(OrtLoggingLevel level);
#if ORT_API_VERSION >= 24
        std::vector<HardwareDevice> GetHardwareDevices() const;

        struct DeviceEpIncompatibilityInfo
        {
            uint32_t reasonsBitmask;
            std::string notes;
            int32_t errorCode;
        };
        DeviceEpIncompatibilityInfo GetHardwareDeviceEpIncompatibilityDetails(
            const std::string& epName, const HardwareDevice& device) const;
        std::optional<std::string> GetCompatibilityInfoFromModel(
            const std::string& modelPath, const std::string& epType) const;
        std::optional<std::string> GetCompatibilityInfoFromModelBytes(
            const nanobind::bytes& modelData, const std::string& epType) const;
#endif /** ORT_API_VERSION >= 24 */
    private:
        static std::shared_ptr<Env> _instance;
        LoggingFunction _loggingFunction;
        Env();
        Env(OrtLoggingLevel logLevel, const std::string& logId,
            const LoggingFunction& loggingFunc,
            const ThreadingOptions* threadingOpts
#if ORT_API_VERSION >= 24
            , const std::unordered_map<std::string, std::string>& configEntries
#endif /** ORT_API_VERSION >= 24 */
        );
    };

    class ModelCompilationOptions : public OrtTypeWrapper<OrtModelCompilationOptions, ModelCompilationOptions>
    {
    public:
        static void ReleaseOrtType(OrtModelCompilationOptions* ptr);
        static int TpTraverse(PyObject* self, visitproc visit, void* arg) noexcept;
        static int TpClear(PyObject* self) noexcept;
        using OrtTypeWrapper::OrtTypeWrapper;
        void SetInputModelPath(const std::string& path);
        void SetInputModelFromBuffer(const nanobind::bytes& modelBytes);
        void SetOutputModelExternalInitializersFile(
            const std::string& path, size_t externalInitializerSizeThreshold);
        void SetEpContextEmbedMode(bool embedContext);
        void CompileModelToFile(const std::string& path);
        nanobind::bytes CompileModelToBuffer();
        void SetFlags(uint32_t flags);
        void SetEpContextBinaryInformation(
            const std::string& outputDirectory, const std::string& modelName);
        void SetGraphOptimizationLevel(GraphOptimizationLevel level);
        using WriteFunction = std::function<void(const nanobind::bytes& buffer)>;
        void SetOutputModelWriteFunc(const WriteFunction& writeFunc);
    private:
        WriteFunction _writeFunc { nullptr };
    };

    class LibraryHandle : public OrtTypeWrapper<void, LibraryHandle>
    {
    public:
        static void ReleaseOrtType(void* ptr);
        using OrtTypeWrapper::OrtTypeWrapper;
    };

    class Value;

    class SessionOptions : public OrtTypeWrapper<OrtSessionOptions, SessionOptions>
    {
    public:
        static void ReleaseOrtType(OrtSessionOptions* ptr);
        static int TpTraverse(PyObject* self, visitproc visit, void* arg) noexcept;
        static int TpClear(PyObject* self) noexcept;

        SessionOptions();
        using OrtTypeWrapper::OrtTypeWrapper;
        void SetOptimizedModelFilePath(const std::string& path);
        void SetSessionExecutionMode(ExecutionMode mode);
        void EnableProfiling(const std::string& profileFilePrefix);
        void DisableProfiling();
        void EnableMemPattern();
        void DisableMemPattern();
        void EnableCpuMemArena();
        void DisableCpuMemArena();
        void SetSessionLogId(const std::string& logId);
        void SetSessionLogVerbosityLevel(int level);
        void SetSessionLogSeverityLevel(int level);
        void SetSessionGraphOptimizationLevel(GraphOptimizationLevel level);
        void SetIntraOpNumThreads(int intraOpNumThreads);
        void SetInterOpNumThreads(int interOpNumThreads);
        LibraryHandle RegisterCustomOpsLibrary(const std::string& libraryPath);
        void RegisterCustomOpsLibrary_V2(const std::string& libraryName);
        void RegisterCustomOpsUsingFunction(const std::string& registrationFuncName);
        void EnableOrtCustomOps();
        void AddFreeDimensionOverride(const std::string& dimDenotation, int64_t dimValue);
        void AddFreeDimensionOverrideByName(const std::string& dimName, int64_t dimValue);
        void DisablePerSessionThreads();
        void AddSessionConfigEntry(const std::string& configKey, const std::string& configValue);
        bool HasSessionConfigEntry(const std::string& configKey) const;
        std::string GetSessionConfigEntry(const std::string& configKey) const;
        std::unordered_map<std::string, std::string> GetSessionOptionsConfigEntries() const;
        void SetDeterministicCompute(bool value);
        void SetLoadCancellationFlag(bool cancel);
        void AddInitializer(const std::string& name, const Value& value);
        void AddExternalInitializers(
            const std::unordered_map<std::string, Value>& initializers);
        void AddExternalInitializersFromFilesInMemory(
            const std::unordered_map<std::string, nanobind::bytes>& files);
        SessionOptions Clone() const;
        void AppendExecutionProvider(
            const std::string& providerName,
            const std::unordered_map<std::string, std::string>& providerOptions);
        void AppendExecutionProvider_V2(
            const std::vector<EpDevice>& epDevices,
            const std::unordered_map<std::string, std::string>& epOptions);
        void SetEpSelectionPolicy(OrtExecutionProviderDevicePolicy policy);
        using EpSelectionPolicyDelegate = std::function<
            std::vector<EpDevice>(
                const std::vector<EpDevice>& epDevices,
                const std::unordered_map<std::string, std::string>& modelMetadata,
                const std::unordered_map<std::string, std::string>& runtimeMetadata,
                size_t max_selected)>;
        void SetEpSelectionPolicyDelegate(const EpSelectionPolicyDelegate& delegate);
        using LoggingFunction = std::function<
            void(OrtLoggingLevel severity,
                 const std::string& category,
                 const std::string& logid,
                 const std::string& codeLocation,
                 const std::string& message)>;
        void SetUserLoggingFunction(const LoggingFunction& loggingFunction);
        ModelCompilationOptions CreateModelCompilationOptions() const;
    private:
        EpSelectionPolicyDelegate _delegate { nullptr };
        LoggingFunction _loggingFunction { nullptr };
    };

    class TypeInfo : public OrtTypeWrapper<OrtTypeInfo, TypeInfo>
    {
    public:
        static void ReleaseOrtType(OrtTypeInfo* ptr);
        using OrtTypeWrapper::OrtTypeWrapper;

        /** Type discriminator */
        ONNXType GetOnnxType() const;
        std::string GetDenotation() const;

        /** Tensor accessors (throw if not TENSOR/SPARSETENSOR) */
        std::vector<int64_t> GetShape() const;
        std::vector<std::string> GetSymbolicDimensions() const;
        std::string GetElementType() const;

        /** Map accessors (throw if not MAP) */
        std::string GetMapKeyType() const;
        TypeInfo GetMapValueType() const;

        /** Sequence accessor (throw if not SEQUENCE) */
        TypeInfo GetSequenceElementType() const;

        /** Optional accessor (throw if not OPTIONAL) */
        TypeInfo GetOptionalContainedType() const;
    };

    class ModelMetadata : public OrtTypeWrapper<OrtModelMetadata, ModelMetadata>
    {
    public:
        static void ReleaseOrtType(OrtModelMetadata* ptr);
        using OrtTypeWrapper::OrtTypeWrapper;
        std::string GetProducerName() const;
        std::string GetGraphName() const;
        std::string GetDomain() const;
        std::string GetDescription() const;
        std::string GetGraphDescription() const;
        int64_t GetVersion() const;
        std::unordered_map<std::string, std::string> GetCustomMetadataMap() const;
        std::optional<std::string> LookupCustomMetadata(const std::string& key) const;
    };

    class LoraAdapter;

    class RunOptions : public OrtTypeWrapper<OrtRunOptions, RunOptions>
    {
    public:
        static void ReleaseOrtType(OrtRunOptions* ptr);
        RunOptions();
        void SetRunLogVerbosityLevel(int level);
        int GetRunLogVerbosityLevel() const;
        void SetRunLogSeverityLevel(int level);
        int GetRunLogSeverityLevel() const;
        void SetRunTag(const std::string& tag);
        std::string GetRunTag() const;
        void SetTerminate();
        void UnsetTerminate();
        void AddRunConfigEntry(const std::string& configKey, const std::string& configValue);
        std::optional<std::string> GetRunConfigEntry(const std::string& configKey) const;
        void AddActiveLoraAdapter(const LoraAdapter& adapter);
    };

    class Value;
    class IoBinding;
    class LoraAdapter;

#if ORT_API_VERSION >= 24
    struct EpAssignedNode
    {
        std::string name;
        std::string domain;
        std::string operatorType;
    };

    struct EpAssignedSubgraph
    {
        std::string epName;
        std::vector<EpAssignedNode> nodes;
    };
#endif /** ORT_API_VERSION >= 24 */

    class Session : public OrtTypeWrapper<OrtSession, Session>
    {
    public:
        static void ReleaseOrtType(OrtSession* ptr);
        Session(const std::string& modelPath, const SessionOptions& options);
        Session(const nanobind::bytes& modelBytes, const SessionOptions& options);

        std::unordered_map<std::string, TypeInfo> GetInputInfo() const;
        std::unordered_map<std::string, TypeInfo> GetOutputInfo() const;
        std::unordered_map<std::string, TypeInfo> GetOverridableInitializerInfo() const;
        ModelMetadata GetModelMetadata() const;
        std::string EndProfiling() const;
        uint64_t GetProfilingStartTimeNs() const;
        std::unordered_map<std::string, MemoryInfo> GetMemoryInfoForInputs() const;
        std::unordered_map<std::string, MemoryInfo> GetMemoryInfoForOutputs() const;
        std::unordered_map<std::string, EpDevice> GetEpDeviceForInputs() const;
#if ORT_API_VERSION >= 24
        std::unordered_map<std::string, EpDevice> GetEpDeviceForOutputs() const;
        std::vector<EpAssignedSubgraph> GetEpGraphAssignmentInfo() const;
#endif /** ORT_API_VERSION >= 24 */
        IoBinding CreateIoBinding() const;
        void RunWithBinding(IoBinding& binding, const RunOptions* runOptions = nullptr) const;
        std::unordered_map<std::string, Value> Run(
            const std::unordered_map<std::string, NpArray>& inputs,
            const std::optional<std::vector<std::string>>& outputNames = std::nullopt,
            const RunOptions* runOptions = nullptr) const;
        std::unordered_map<std::string, Value> RunWithOrtValues(
            const std::unordered_map<std::string, Value>& inputs,
            const std::optional<std::vector<std::string>>& outputNames = std::nullopt,
            const RunOptions* runOptions = nullptr) const;
    };

    class Value
    {
    public:
        static ONNXTensorElementDataType NpTypeToOrtType(const nanobind::dlpack::dtype& npType);
        static nanobind::dlpack::dtype OrtTypeToNpType(ONNXTensorElementDataType type);
        static std::string NpTypeToName(const nanobind::dlpack::dtype& npType);
        static size_t GetSizeOfOrtType(ONNXTensorElementDataType type);

        Value(OrtValue* ptr);
        Value(const NpArray& NpArray);
        Value(const std::vector<int64_t>& shape, ONNXTensorElementDataType type);
        static Value FromStrings(const std::vector<std::string>& strings,
            const std::optional<std::vector<int64_t>>& shape = std::nullopt);

        /** Introspection */
        bool IsTensor() const;
        ONNXType GetValueType() const;
        bool HasValue() const;
        std::optional<MemoryInfo> GetTensorMemoryInfo() const;
        size_t GetTensorSizeInBytes() const;

        /** Tensor data access (numeric tensors only) */
        NpArray ToNumpy() const;
        operator OrtValue*() const;
        ONNXTensorElementDataType GetType() const;
        std::vector<int64_t> GetShape() const;
        size_t GetSize() const;
        void* GetData() const;

        /** String tensor access */
        std::vector<std::string> GetStrings() const;

        /** Map/Sequence access */
        Value GetElement(int index) const;
        size_t GetCount() const;
    private:
        struct State
        {
            /** 
             * Stores the data when the value is created by ort or binding code.
             * A view / reference to the data if not.
             */
            OrtValue* ortValue{ nullptr };
            /** 
             * Stores the data when the value is created by python.
             * A view / reference to the data if not.
             */
            std::optional<NpArray> npArray { std::nullopt };
            State() = default;
            State(const State&) = delete;
            State& operator=(const State&) = delete;
            State(State&&) noexcept = delete;
            State& operator=(State&&) noexcept = delete;
            ~State();
        };
        std::shared_ptr<State> _state{ std::make_shared<State>() };
    };

    class MemoryInfo : public OrtTypeWrapper<OrtMemoryInfo, MemoryInfo>
    {
    public:
        static void ReleaseOrtType(OrtMemoryInfo* ptr);
        MemoryInfo();
        MemoryInfo(const std::string& name, OrtAllocatorType allocatorType,
                   int deviceId, OrtMemType memType);
        std::string GetName() const;
        int GetDeviceId() const;
        OrtMemType GetMemType() const;
        OrtAllocatorType GetAllocatorType() const;
        OrtMemoryInfoDeviceType GetDeviceType() const;
        bool operator==(const MemoryInfo& other) const;
    };

    class IoBinding : public OrtTypeWrapper<OrtIoBinding, IoBinding>
    {
    public:
        static void ReleaseOrtType(OrtIoBinding* ptr);
        IoBinding(const Session& session);
        void BindInput(const std::string& name, const Value& value);
        void BindOutput(const std::string& name, const Value& value);
        void BindOutputToDevice(const std::string& name, const MemoryInfo& memInfo);
        std::unordered_map<std::string, Value> GetOutputs();
        void ClearInputs();
        void ClearOutputs();
        void SynchronizeInputs();
        void SynchronizeOutputs();
    private:
        std::vector<Value> _boundInputValues;
        std::vector<Value> _boundOutputValues;
    };

    class LoraAdapter : public OrtTypeWrapper<OrtLoraAdapter, LoraAdapter>
    {
    public:
        static void ReleaseOrtType(OrtLoraAdapter* ptr);
        LoraAdapter(const std::string& adapterFilePath);
        LoraAdapter(const nanobind::bytes& adapterBytes);
    };

}
