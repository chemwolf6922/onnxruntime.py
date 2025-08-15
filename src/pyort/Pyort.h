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

    class Env : public OrtTypeWrapper<OrtEnv, Env>
    {
    public:
        static std::shared_ptr<Env> GetSingleton();
        static void ReleaseOrtType(OrtEnv* ptr);
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

    class Session : public OrtTypeWrapper<OrtSession, Session>
    {
    public:
        static void ReleaseOrtType(OrtSession* ptr);
        Session(const std::string& modelPath, const SessionOptions& options);

        std::unordered_map<std::string, pybind11::array> Run(
            const std::unordered_map<std::string, pybind11::array>& inputs);
    };

    class Value : public OrtTypeWrapper<OrtValue, Value>
    {
    public:
        static void ReleaseOrtType(OrtValue* ptr);
        Value(const pybind11::array& npArray);
        Value(const std::vector<int64_t>& shape, ONNXTensorElementDataType type);

        operator pybind11::array() const;
        ONNXTensorElementDataType GetType() const;
        std::vector<int64_t> GetShape() const;
        size_t GetSize() const;
        void* GetData() const;
    private:
        static ONNXTensorElementDataType NpTypeToOrtType(const pybind11::dtype& npType);
        static pybind11::dtype OrtTypeToNpType(ONNXTensorElementDataType type);

        /** Data is always stored in _npArray. _ptr is just a shell. */
        std::optional<pybind11::array> _npArray;
    };

    class MemoryInfo : public OrtTypeWrapper<OrtMemoryInfo, MemoryInfo>
    {
    public:
        static void ReleaseOrtType(OrtMemoryInfo* ptr);
        MemoryInfo();
    };
}