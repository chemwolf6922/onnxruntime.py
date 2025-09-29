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
    private:
        const OrtEpDevice* _ptr{ nullptr };
    };

    class Env : public OrtTypeWrapper<OrtEnv, Env>
    {
    public:
        static std::shared_ptr<Env> GetSingleton();
        static void ReleaseOrtType(OrtEnv* ptr);
        void RegisterExecutionProviderLibrary(const std::string& name, const std::string& path);
        void UnregisterExecutionProviderLibrary(const std::string& name);
        std::vector<EpDevice> GetEpDevices() const;
    private:
        static std::shared_ptr<Env> _instance;
        Env();
    };

    class ModelCompilationOptions : public OrtTypeWrapper<OrtModelCompilationOptions, ModelCompilationOptions>
    {
    public:
        static void ReleaseOrtType(OrtModelCompilationOptions* ptr);
        using OrtTypeWrapper::OrtTypeWrapper;
        void SetInputModelPath(const std::string& path);
        void SetInputModelFromBuffer(const nanobind::bytes& modelBytes);
        void SetOutputModelExternalInitializersFile(
            const std::string& path, size_t externalInitializerSizeThreshold);
        void SetEpContextEmbedMode(bool embedContext);
        void CompileModelToFile(const std::string& path);
        nanobind::bytes CompileModelToBuffer();
    };

    class LibraryHandle : public OrtTypeWrapper<void, LibraryHandle>
    {
    public:
        static void ReleaseOrtType(void* ptr);
        using OrtTypeWrapper::OrtTypeWrapper;
    };

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
        ModelCompilationOptions CreateModelCompilationOptions() const;
    private:
        EpSelectionPolicyDelegate _delegate { nullptr };
    };

    class TypeInfo : public OrtTypeWrapper<OrtTypeInfo, TypeInfo>
    {
    public:
        static void ReleaseOrtType(OrtTypeInfo* ptr);
        using OrtTypeWrapper::OrtTypeWrapper;
    };

    class TensorTypeAndShapeInfo : public OrtTypeWrapper<OrtTensorTypeAndShapeInfo, TensorTypeAndShapeInfo>
    {
    public:
        static void ReleaseOrtType(OrtTensorTypeAndShapeInfo* ptr);
        using OrtTypeWrapper::OrtTypeWrapper;
    };

    struct TensorInfo
    {
        std::vector<int64_t> shape;
        std::vector<std::string> dimensions;
        nanobind::dlpack::dtype dtype;

        TensorInfo() = default;
        TensorInfo(const TypeInfo& typeInfo);
    };

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
    };

    class Session : public OrtTypeWrapper<OrtSession, Session>
    {
    public:
        static void ReleaseOrtType(OrtSession* ptr);
        Session(const std::string& modelPath, const SessionOptions& options);
        Session(const nanobind::bytes& modelBytes, const SessionOptions& options);

        std::unordered_map<std::string, TensorInfo> GetInputInfo() const;
        std::unordered_map<std::string, TensorInfo> GetOutputInfo() const;
        std::unordered_map<std::string, NpArray> Run(
            const std::unordered_map<std::string, NpArray>& inputs,
            const std::optional<std::reference_wrapper<RunOptions>>& runOptions) const;
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

        operator NpArray() const;
        operator OrtValue*() const;
        ONNXTensorElementDataType GetType() const;
        std::vector<int64_t> GetShape() const;
        size_t GetSize() const;
        void* GetData() const;
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
    };
}
