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

    class Status
    {
    public:
        Status(OrtStatus* status);
        ~Status();

        OrtErrorCode GetErrorCode() const;
        std::string GetErrorMessage() const;
        void Check() const;
        operator OrtStatus*() const;
    private:
        OrtStatus* _status{ nullptr };
    };

    class Env
    {
    public:
        static std::shared_ptr<Env> GetSingleton();
        ~Env();
        operator OrtEnv*() const;
    private:
        static std::shared_ptr<Env> _instance;
        OrtEnv* _env{ nullptr };
        Env();
    };

    class SessionOptions
    {
    public:
        SessionOptions();
        ~SessionOptions();
        operator OrtSessionOptions*() const;
    private:
        OrtSessionOptions* _options{ nullptr };
    };

    class Session
    {
    public:
        Session(const std::string& modelPath, const SessionOptions& options);
        ~Session();
        operator OrtSession*() const;
        std::unordered_map<std::string, pybind11::array> Run(
            const std::unordered_map<std::string, pybind11::array>& inputs);
    private:
        OrtSession* _session{ nullptr };
    };

    class Value
    {
    public:
        explicit Value(const pybind11::array& source);
        ~Value();
        explicit operator pybind11::array() const;
    private:
        static Value Create(std::vector<int64_t> shape, ONNXTensorElementDataType type);
        static Value CreateRef(
            std::vector<int64_t> shape, ONNXTensorElementDataType type,
            void* data, size_t dataSize);

        std::optional<pybind11::array> _source { std::nullopt };
        OrtValue* _value{ nullptr };

        Value(OrtValue* value);
    };

    class MemoryInfo
    {
    public:
        /** No parameter allowed. DO NOT mess with this. */
        MemoryInfo();
        ~MemoryInfo();
        operator OrtMemoryInfo*() const;
    private:
        OrtMemoryInfo* _memoryInfo{ nullptr };
    };

    
}