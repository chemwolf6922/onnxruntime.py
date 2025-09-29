#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/function.h>
#include <memory>
#include <optional>

/** Use the C API for maximum compatibility */
#include <onnxruntime_c_api.h>

#include "Ortpy.h"

#ifndef ORTPY_VERSION
#define ORTPY_VERSION "0.0"
#endif

static PyType_Slot sessionOptionsSlots[] = {
    { Py_tp_traverse, (void*) &Ortpy::SessionOptions::TpTraverse },
    { Py_tp_clear, (void*) &Ortpy::SessionOptions::TpClear },
    { 0, nullptr }
};

NB_MODULE(_ortpy, m) {
    m.doc() = "onnxruntime binding build upon C API.";
    m.attr("__version__") = ORTPY_VERSION;
    m.attr("ORT_API_VERSION") = ORT_API_VERSION;

    nanobind::enum_<ExecutionMode>(m, "ExecutionMode")
        .value("SEQUENTIAL", ORT_SEQUENTIAL)
        .value("PARALLEL", ORT_PARALLEL);

    nanobind::enum_<GraphOptimizationLevel>(m, "GraphOptimizationLevel")
        .value("DISABLE_ALL", ORT_DISABLE_ALL)
        .value("ENABLE_BASIC", ORT_ENABLE_BASIC)
        .value("ENABLE_EXTENDED", ORT_ENABLE_EXTENDED)
        .value("ENABLE_ALL", ORT_ENABLE_ALL);

    nanobind::enum_<OrtHardwareDeviceType>(m, "HardwareDeviceType")
        .value("CPU", OrtHardwareDeviceType_CPU)
        .value("GPU", OrtHardwareDeviceType_GPU)
        .value("NPU", OrtHardwareDeviceType_NPU);

    nanobind::enum_<OrtExecutionProviderDevicePolicy>(m, "ExecutionProviderDevicePolicy")
        .value("DEFAULT", OrtExecutionProviderDevicePolicy_DEFAULT)
        .value("PREFER_CPU", OrtExecutionProviderDevicePolicy_PREFER_CPU)
        .value("PREFER_NPU", OrtExecutionProviderDevicePolicy_PREFER_NPU)
        .value("PREFER_GPU", OrtExecutionProviderDevicePolicy_PREFER_GPU)
        .value("MAX_PERFORMANCE", OrtExecutionProviderDevicePolicy_MAX_PERFORMANCE)
        .value("MAX_EFFICIENCY", OrtExecutionProviderDevicePolicy_MAX_EFFICIENCY)
        .value("MIN_OVERALL_POWER", OrtExecutionProviderDevicePolicy_MIN_OVERALL_POWER);

    nanobind::class_<Ortpy::HardwareDevice>(m, "HardwareDevice")
        .def_ro("type", &Ortpy::HardwareDevice::type)
        .def_ro("vendor_id", &Ortpy::HardwareDevice::vendorId)
        .def_ro("vendor", &Ortpy::HardwareDevice::vendor)
        .def_ro("device_id", &Ortpy::HardwareDevice::deviceId)
        .def_ro("metadata", &Ortpy::HardwareDevice::metadata);

    nanobind::class_<Ortpy::EpDevice>(m, "EpDevice")
        .def_ro("ep_name", &Ortpy::EpDevice::epName)
        .def_ro("ep_vendor", &Ortpy::EpDevice::epVendor)
        .def_ro("ep_metadata", &Ortpy::EpDevice::epMetadata)
        .def_ro("ep_options", &Ortpy::EpDevice::epOptions)
        .def_ro("device", &Ortpy::EpDevice::device);

    m.def("register_execution_provider_library", [](const std::string& name, const std::string& path) -> void {
        Ortpy::Env::GetSingleton()->RegisterExecutionProviderLibrary(name, path);
    });

    m.def("unregister_execution_provider_library", [](const std::string& name) -> void {
        Ortpy::Env::GetSingleton()->UnregisterExecutionProviderLibrary(name);
    });

    m.def("get_ep_devices", []() -> std::vector<Ortpy::EpDevice> {
        return Ortpy::Env::GetSingleton()->GetEpDevices();
    });

    nanobind::class_<Ortpy::ModelCompilationOptions>(m, "ModelCompilationOptions")
        .def("set_input_model_path",
            &Ortpy::ModelCompilationOptions::SetInputModelPath,
            nanobind::arg("path"))
        .def("set_input_model_from_buffer",
            &Ortpy::ModelCompilationOptions::SetInputModelFromBuffer,
            nanobind::arg("model_bytes"))
        .def("set_output_model_external_initializers_file",
            &Ortpy::ModelCompilationOptions::SetOutputModelExternalInitializersFile,
            nanobind::arg("path"),
            nanobind::arg("external_initializer_size_threshold"))
        .def("set_ep_context_embed_mode",
            &Ortpy::ModelCompilationOptions::SetEpContextEmbedMode,
            nanobind::arg("embed_context"))
        .def("compile_model_to_file",
            &Ortpy::ModelCompilationOptions::CompileModelToFile,
            nanobind::arg("path"))
        .def("compile_model_to_buffer", &Ortpy::ModelCompilationOptions::CompileModelToBuffer);

    nanobind::class_<Ortpy::SessionOptions>(m, "SessionOptions", nanobind::type_slots(sessionOptionsSlots))
        .def(nanobind::init<>())
        .def("set_optimized_model_file_path",
            &Ortpy::SessionOptions::SetOptimizedModelFilePath,
            nanobind::arg("path"))
        .def("set_session_execution_mode",
            &Ortpy::SessionOptions::SetSessionExecutionMode,
            nanobind::arg("mode"))
        .def("enable_profiling",
            &Ortpy::SessionOptions::EnableProfiling,
            nanobind::arg("profile_file_prefix"))
        .def("disable_profiling", &Ortpy::SessionOptions::DisableProfiling)
        .def("enable_mem_pattern", &Ortpy::SessionOptions::EnableMemPattern)
        .def("disable_mem_pattern", &Ortpy::SessionOptions::DisableMemPattern)
        .def("enable_cpu_mem_arena", &Ortpy::SessionOptions::EnableCpuMemArena)
        .def("disable_cpu_mem_arena", &Ortpy::SessionOptions::DisableCpuMemArena)
        .def("set_session_log_id",
            &Ortpy::SessionOptions::SetSessionLogId,
            nanobind::arg("log_id"))
        .def("set_session_log_verbosity_level",
            &Ortpy::SessionOptions::SetSessionLogVerbosityLevel,
            nanobind::arg("level"))
        .def("set_session_log_severity_level",
            &Ortpy::SessionOptions::SetSessionLogSeverityLevel,
            nanobind::arg("level"))
        .def("set_session_graph_optimization_level",
            &Ortpy::SessionOptions::SetSessionGraphOptimizationLevel,
            nanobind::arg("level"))
        .def("set_intra_op_num_threads",
            &Ortpy::SessionOptions::SetIntraOpNumThreads,
            nanobind::arg("intra_op_num_threads"))
        .def("set_inter_op_num_threads",
            &Ortpy::SessionOptions::SetInterOpNumThreads,
            nanobind::arg("inter_op_num_threads"))
        .def("register_custom_ops_library",
            &Ortpy::SessionOptions::RegisterCustomOpsLibrary,
            nanobind::arg("library_path"))
        .def("append_execution_provider_v2",
            &Ortpy::SessionOptions::AppendExecutionProvider_V2,
            nanobind::arg("ep_devices"),
            nanobind::arg("options"))
        .def("set_ep_selection_policy",
            &Ortpy::SessionOptions::SetEpSelectionPolicy,
            nanobind::arg("policy"))
        .def("set_ep_selection_policy_delegate",
            &Ortpy::SessionOptions::SetEpSelectionPolicyDelegate,
            nanobind::arg("delegate"))
        .def("create_model_compilation_options", &Ortpy::SessionOptions::CreateModelCompilationOptions);

    nanobind::class_<Ortpy::TensorInfo>(m, "TensorInfo")
        .def_ro("shape", &Ortpy::TensorInfo::shape)
        .def_ro("dimensions", &Ortpy::TensorInfo::dimensions)
        .def_prop_ro("dtype",
            [](const Ortpy::TensorInfo &self) -> std::string {
                return Ortpy::Value::NpTypeToName(self.dtype);
            });

    nanobind::class_<Ortpy::RunOptions>(m, "RunOptions")
        .def(nanobind::init<>())
        .def_prop_rw("run_log_verbosity_level",
            &Ortpy::RunOptions::GetRunLogVerbosityLevel,
            &Ortpy::RunOptions::SetRunLogVerbosityLevel)
        .def_prop_rw("run_log_severity_level",
            &Ortpy::RunOptions::GetRunLogSeverityLevel,
            &Ortpy::RunOptions::SetRunLogSeverityLevel)
        .def_prop_rw("run_tag",
            &Ortpy::RunOptions::GetRunTag,
            &Ortpy::RunOptions::SetRunTag)
        .def("set_terminate", &Ortpy::RunOptions::SetTerminate)
        .def("unset_terminate", &Ortpy::RunOptions::UnsetTerminate);

    nanobind::class_<Ortpy::Session>(m, "Session")
        .def(nanobind::init<const std::string&, const Ortpy::SessionOptions&>(),
            nanobind::arg("model_path"),
            nanobind::arg("options"))
        .def(nanobind::init<const nanobind::bytes&, const Ortpy::SessionOptions&>(),
            nanobind::arg("model_bytes"),
            nanobind::arg("options"))
        .def("get_input_info", &Ortpy::Session::GetInputInfo)
        .def("get_output_info", &Ortpy::Session::GetOutputInfo)
        .def("run",
            &Ortpy::Session::Run,
            nanobind::arg("inputs"),
            nanobind::arg("run_options") = std::nullopt);
}
