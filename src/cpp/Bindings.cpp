#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <memory>
#include <optional>

/** Use the C API for maximum compatibility */
#include <onnxruntime_c_api.h>

#include "Pyort.h"

#ifndef PYORT_VERSION
#define PYORT_VERSION "0.0"
#endif

PYBIND11_MODULE(_pyort, m) {
    m.doc() = "onnxruntime binding build upon C API.";
    m.attr("__version__") = PYORT_VERSION;
    m.attr("ORT_API_VERSION") = ORT_API_VERSION;

    pybind11::enum_<ExecutionMode>(m, "ExecutionMode")
        .value("SEQUENTIAL", ORT_SEQUENTIAL)
        .value("PARALLEL", ORT_PARALLEL);

    pybind11::enum_<GraphOptimizationLevel>(m, "GraphOptimizationLevel")
        .value("DISABLE_ALL", ORT_DISABLE_ALL)
        .value("ENABLE_BASIC", ORT_ENABLE_BASIC)
        .value("ENABLE_EXTENDED", ORT_ENABLE_EXTENDED)
        .value("ENABLE_ALL", ORT_ENABLE_ALL);

    pybind11::enum_<OrtHardwareDeviceType>(m, "HardwareDeviceType")
        .value("CPU", OrtHardwareDeviceType_CPU)
        .value("GPU", OrtHardwareDeviceType_GPU)
        .value("NPU", OrtHardwareDeviceType_NPU);

    pybind11::enum_<OrtExecutionProviderDevicePolicy>(m, "ExecutionProviderDevicePolicy")
        .value("DEFAULT", OrtExecutionProviderDevicePolicy_DEFAULT)
        .value("PREFER_CPU", OrtExecutionProviderDevicePolicy_PREFER_CPU)
        .value("PREFER_NPU", OrtExecutionProviderDevicePolicy_PREFER_NPU)
        .value("PREFER_GPU", OrtExecutionProviderDevicePolicy_PREFER_GPU)
        .value("MAX_PERFORMANCE", OrtExecutionProviderDevicePolicy_MAX_PERFORMANCE)
        .value("MAX_EFFICIENCY", OrtExecutionProviderDevicePolicy_MAX_EFFICIENCY)
        .value("MIN_OVERALL_POWER", OrtExecutionProviderDevicePolicy_MIN_OVERALL_POWER);

    pybind11::class_<Pyort::HardwareDevice>(m, "HardwareDevice")
        .def_readonly("type", &Pyort::HardwareDevice::type)
        .def_readonly("vendor_id", &Pyort::HardwareDevice::vendorId)
        .def_readonly("vendor", &Pyort::HardwareDevice::vendor)
        .def_readonly("device_id", &Pyort::HardwareDevice::deviceId)
        .def_readonly("metadata", &Pyort::HardwareDevice::metadata);

    pybind11::class_<Pyort::EpDevice>(m, "EpDevice")
        .def_readonly("ep_name", &Pyort::EpDevice::epName)
        .def_readonly("ep_vendor", &Pyort::EpDevice::epVendor)
        .def_readonly("ep_metadata", &Pyort::EpDevice::epMetadata)
        .def_readonly("ep_options", &Pyort::EpDevice::epOptions)
        .def_readonly("device", &Pyort::EpDevice::device);

    m.def("register_execution_provider_library", [](const std::string& name, const std::string& path) -> void {
        Pyort::Env::GetSingleton()->RegisterExecutionProviderLibrary(name, path);
    });

    m.def("unregister_execution_provider_library", [](const std::string& name) -> void {
        Pyort::Env::GetSingleton()->UnregisterExecutionProviderLibrary(name);
    });

    m.def("get_ep_devices", []() -> std::vector<Pyort::EpDevice> {
        return Pyort::Env::GetSingleton()->GetEpDevices();
    });

    pybind11::class_<Pyort::ModelCompilationOptions>(m, "ModelCompilationOptions")
        .def("set_input_model_path",
            &Pyort::ModelCompilationOptions::SetInputModelPath,
            pybind11::arg("path"))
        .def("set_input_model_from_buffer",
            &Pyort::ModelCompilationOptions::SetInputModelFromBuffer,
            pybind11::arg("model_bytes"))
        .def("set_output_model_external_initializers_file",
            &Pyort::ModelCompilationOptions::SetOutputModelExternalInitializersFile,
            pybind11::arg("path"),
            pybind11::arg("external_initializer_size_threshold"))
        .def("set_ep_context_embed_mode",
            &Pyort::ModelCompilationOptions::SetEpContextEmbedMode,
            pybind11::arg("embed_context"))
        .def("compile_model_to_file",
            &Pyort::ModelCompilationOptions::CompileModelToFile,
            pybind11::arg("path"))
        .def("compile_model_to_buffer", &Pyort::ModelCompilationOptions::CompileModelToBuffer);

    pybind11::class_<Pyort::SessionOptions>(m, "SessionOptions")
        .def(pybind11::init<>())
        .def("set_optimized_model_file_path",
            &Pyort::SessionOptions::SetOptimizedModelFilePath,
            pybind11::arg("path"))
        .def("set_session_execution_mode",
            &Pyort::SessionOptions::SetSessionExecutionMode,
            pybind11::arg("mode"))
        .def("enable_profiling",
            &Pyort::SessionOptions::EnableProfiling,
            pybind11::arg("profile_file_prefix"))
        .def("disable_profiling", &Pyort::SessionOptions::DisableProfiling)
        .def("enable_mem_pattern", &Pyort::SessionOptions::EnableMemPattern)
        .def("disable_mem_pattern", &Pyort::SessionOptions::DisableMemPattern)
        .def("enable_cpu_mem_arena", &Pyort::SessionOptions::EnableCpuMemArena)
        .def("disable_cpu_mem_arena", &Pyort::SessionOptions::DisableCpuMemArena)
        .def("set_session_log_id",
            &Pyort::SessionOptions::SetSessionLogId,
            pybind11::arg("log_id"))
        .def("set_session_log_verbosity_level",
            &Pyort::SessionOptions::SetSessionLogVerbosityLevel,
            pybind11::arg("level"))
        .def("set_session_log_severity_level",
            &Pyort::SessionOptions::SetSessionLogSeverityLevel,
            pybind11::arg("level"))
        .def("set_session_graph_optimization_level",
            &Pyort::SessionOptions::SetSessionGraphOptimizationLevel,
            pybind11::arg("level"))
        .def("set_intra_op_num_threads",
            &Pyort::SessionOptions::SetIntraOpNumThreads,
            pybind11::arg("intra_op_num_threads"))
        .def("set_inter_op_num_threads",
            &Pyort::SessionOptions::SetInterOpNumThreads,
            pybind11::arg("inter_op_num_threads"))
        .def("register_custom_ops_library",
            &Pyort::SessionOptions::RegisterCustomOpsLibrary,
            pybind11::arg("library_path"))
        .def("append_execution_provider_v2",
            &Pyort::SessionOptions::AppendExecutionProvider_V2,
            pybind11::arg("ep_devices"),
            pybind11::arg("options"))
        .def("set_ep_selection_policy",
            &Pyort::SessionOptions::SetEpSelectionPolicy,
            pybind11::arg("policy"))
        .def("set_ep_selection_policy_delegate",
            &Pyort::SessionOptions::SetEpSelectionPolicyDelegate,
            pybind11::arg("delegate"),
            R"pbdoc(
set_ep_selection_policy_delegate(delegate: Callable[[List[EpDevice], Dict[str, str], Dict[str, str], int]])
)pbdoc")
        .def("create_model_compilation_options", &Pyort::SessionOptions::CreateModelCompilationOptions);

    pybind11::class_<Pyort::TensorInfo>(m, "TensorInfo")
        .def_readonly("shape", &Pyort::TensorInfo::shape)
        .def_readonly("dimensions", &Pyort::TensorInfo::dimensions)
        .def_readonly("dtype", &Pyort::TensorInfo::dtype);

    pybind11::class_<Pyort::RunOptions>(m, "RunOptions")
        .def(pybind11::init<>())
        .def_property("run_log_verbosity_level",
            &Pyort::RunOptions::GetRunLogVerbosityLevel,
            &Pyort::RunOptions::SetRunLogVerbosityLevel)
        .def_property("run_log_severity_level",
            &Pyort::RunOptions::GetRunLogSeverityLevel,
            &Pyort::RunOptions::SetRunLogSeverityLevel)
        .def_property("run_tag",
            &Pyort::RunOptions::GetRunTag,
            &Pyort::RunOptions::SetRunTag)
        .def("set_terminate", &Pyort::RunOptions::SetTerminate)
        .def("unset_terminate", &Pyort::RunOptions::UnsetTerminate);

    pybind11::class_<Pyort::Session, std::shared_ptr<Pyort::Session>>(m, "Session")
        .def(pybind11::init<const std::string&, const Pyort::SessionOptions&>(),
             pybind11::arg("model_path"),
             pybind11::arg("options"))
        .def(pybind11::init<const pybind11::bytes&, const Pyort::SessionOptions&>(),
             pybind11::arg("model_bytes"),
             pybind11::arg("options"))
        .def("get_input_info", &Pyort::Session::GetInputInfo)
        .def("get_output_info", &Pyort::Session::GetOutputInfo)
        .def("run",
             &Pyort::Session::Run,
             pybind11::arg("inputs"),
             pybind11::arg("run_options") = std::nullopt);
}
