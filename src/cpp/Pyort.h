#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <memory>
#include <unordered_map>
#include <vector>
#include <optional>

/** Use the C API for maximum compatibility */
#include <onnxruntime_c_api.h>

namespace Pyort
{
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
        Pyort::HardwareDevice device;
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

    class SessionOptions : public OrtTypeWrapper<OrtSessionOptions, SessionOptions>
    {
    public:
        static void ReleaseOrtType(OrtSessionOptions* ptr);
        SessionOptions();
        using OrtTypeWrapper::OrtTypeWrapper;
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
        pybind11::dtype dtype;
    };

    class Session : public OrtTypeWrapper<OrtSession, Session>
    {
    public:
        static void ReleaseOrtType(OrtSession* ptr);
        Session(const std::string& modelPath, const SessionOptions& options);

        std::unordered_map<std::string, TensorInfo> GetInputInfo() const;
        std::unordered_map<std::string, TensorInfo> GetOutputInfo() const;
        std::unordered_map<std::string, pybind11::array> Run(
            const std::unordered_map<std::string, pybind11::array>& inputs) const;
    };

    class Value : public OrtTypeWrapper<OrtValue, Value>
    {
    public:
        static ONNXTensorElementDataType NpTypeToOrtType(const pybind11::dtype& npType);
        static pybind11::dtype OrtTypeToNpType(ONNXTensorElementDataType type);
        static void ReleaseOrtType(OrtValue* ptr);
        using OrtTypeWrapper::OrtTypeWrapper;
        Value(const pybind11::array& npArray);
        Value(const std::vector<int64_t>& shape, ONNXTensorElementDataType type);

        operator pybind11::array();
        ONNXTensorElementDataType GetType() const;
        std::vector<int64_t> GetShape() const;
        size_t GetSize() const;
        void* GetData() const;
    private:
        std::optional<pybind11::array> _npArray{ std::nullopt };
    };

    class MemoryInfo : public OrtTypeWrapper<OrtMemoryInfo, MemoryInfo>
    {
    public:
        static void ReleaseOrtType(OrtMemoryInfo* ptr);
        MemoryInfo();
    };
}
