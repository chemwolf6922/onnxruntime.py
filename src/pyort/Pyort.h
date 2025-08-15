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
        Status(const Status&) = delete;
        Status& operator=(const Status&) = delete;
        Status(Status&& other) noexcept;
        Status& operator=(Status&& other) noexcept;

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
        Env(const Env&) = delete;
        Env& operator=(const Env&) = delete;
        Env(Env&&) = delete;
        Env& operator=(Env&&) = delete;

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
        SessionOptions(const SessionOptions&) = delete;
        SessionOptions& operator=(const SessionOptions&) = delete;
        SessionOptions(SessionOptions&& other) noexcept;
        SessionOptions& operator=(SessionOptions&& other) noexcept;

        operator OrtSessionOptions*() const;
    private:
        OrtSessionOptions* _options{ nullptr };
    };

    class Session
    {
    public:
        Session(const std::string& modelPath, const SessionOptions& options);
        ~Session();
        Session(const Session&) = delete;
        Session& operator=(const Session&) = delete;
        Session(Session&& other) noexcept;
        Session& operator=(Session&& other) noexcept;

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
        Value(const Value&) = delete;
        Value& operator=(const Value&) = delete;
        Value(Value&& other) noexcept;
        Value& operator=(Value&& other) noexcept;

        operator OrtValue*() const;
        explicit operator pybind11::array() const;
        ONNXTensorElementDataType GetType() const;
        std::vector<int64_t> GetShape() const;
        size_t GetSize() const;
        void* GetData() const;
        OrtValue* Detach();
    private:
        static Value Create(std::vector<int64_t> shape, ONNXTensorElementDataType type);
        static Value CreateRef(
            std::vector<int64_t> shape, ONNXTensorElementDataType type,
            void* data, size_t dataSize);

        std::optional<pybind11::array> _source { std::nullopt };
        OrtValue* _value{ nullptr };

        explicit Value(OrtValue* value);
    };

    class MemoryInfo
    {
    public:
        /** No parameter allowed. DO NOT mess with this. */
        MemoryInfo();
        ~MemoryInfo();
        MemoryInfo(const MemoryInfo&) = delete;
        MemoryInfo& operator=(const MemoryInfo&) = delete;
        MemoryInfo(MemoryInfo&& other) noexcept;
        MemoryInfo& operator=(MemoryInfo&& other) noexcept;

        operator OrtMemoryInfo*() const;
    private:
        OrtMemoryInfo* _memoryInfo{ nullptr };
    };
}