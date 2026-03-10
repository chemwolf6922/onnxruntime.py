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
#endif /** ORTPY_VERSION */

static PyType_Slot sessionOptionsSlots[] = {
    { Py_tp_traverse, (void*) &Ortpy::SessionOptions::TpTraverse },
    { Py_tp_clear, (void*) &Ortpy::SessionOptions::TpClear },
    { 0, nullptr }
};

static PyType_Slot modelCompilationOptionsSlots[] = {
    { Py_tp_traverse, (void*) &Ortpy::ModelCompilationOptions::TpTraverse },
    { Py_tp_clear, (void*) &Ortpy::ModelCompilationOptions::TpClear },
    { 0, nullptr }
};

NB_MODULE(_ortpy, m) {
    m.doc() = "onnxruntime binding build upon C API.";
    m.attr("__version__") = ORTPY_VERSION;
    m.attr("ORT_API_VERSION") = ORT_API_VERSION;

    /** Clean up at exit */
    m.def("_release_env", []() { Ortpy::Env::ReleaseSingleton(); });
    auto atexit = nanobind::module_::import_("atexit");
    atexit.attr("register")(m.attr("_release_env"));

    nanobind::enum_<OrtLoggingLevel>(m, "LogLevel")
        .value("VERBOSE", ORT_LOGGING_LEVEL_VERBOSE)
        .value("INFO", ORT_LOGGING_LEVEL_INFO)
        .value("WARNING", ORT_LOGGING_LEVEL_WARNING)
        .value("ERROR", ORT_LOGGING_LEVEL_ERROR)
        .value("FATAL", ORT_LOGGING_LEVEL_FATAL);

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

    nanobind::enum_<OrtAllocatorType>(m, "AllocatorType")
        .value("INVALID", OrtInvalidAllocator)
        .value("DEVICE", OrtDeviceAllocator)
        .value("ARENA", OrtArenaAllocator);

    nanobind::enum_<OrtMemType>(m, "MemType")
        .value("CPU_INPUT", OrtMemTypeCPUInput)
        .value("CPU_OUTPUT", OrtMemTypeCPUOutput)
        .value("DEFAULT", OrtMemTypeDefault);

    nanobind::enum_<OrtDeviceMemoryType>(m, "DeviceMemoryType")
        .value("DEFAULT", OrtDeviceMemoryType_DEFAULT)
        .value("HOST_ACCESSIBLE", OrtDeviceMemoryType_HOST_ACCESSIBLE);

    nanobind::enum_<OrtCompiledModelCompatibility>(m, "CompiledModelCompatibility")
        .value("EP_NOT_APPLICABLE", OrtCompiledModelCompatibility_EP_NOT_APPLICABLE)
        .value("EP_SUPPORTED_OPTIMAL", OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL)
        .value("EP_SUPPORTED_PREFER_RECOMPILATION", OrtCompiledModelCompatibility_EP_SUPPORTED_PREFER_RECOMPILATION)
        .value("EP_UNSUPPORTED", OrtCompiledModelCompatibility_EP_UNSUPPORTED);

    nanobind::enum_<ONNXType>(m, "ONNXType")
        .value("UNKNOWN", ONNX_TYPE_UNKNOWN)
        .value("TENSOR", ONNX_TYPE_TENSOR)
        .value("SEQUENCE", ONNX_TYPE_SEQUENCE)
        .value("MAP", ONNX_TYPE_MAP)
        .value("OPAQUE", ONNX_TYPE_OPAQUE)
        .value("SPARSETENSOR", ONNX_TYPE_SPARSETENSOR)
        .value("OPTIONAL", ONNX_TYPE_OPTIONAL);

    nanobind::enum_<OrtMemoryInfoDeviceType>(m, "MemoryInfoDeviceType")
        .value("CPU", OrtMemoryInfoDeviceType_CPU)
        .value("GPU", OrtMemoryInfoDeviceType_GPU)
        .value("FPGA", OrtMemoryInfoDeviceType_FPGA)
        .value("NPU", OrtMemoryInfoDeviceType_NPU);

    nanobind::class_<Ortpy::ThreadingOptions>(m, "ThreadingOptions")
        .def(nanobind::init<>())
        .def("set_intra_op_num_threads",
            &Ortpy::ThreadingOptions::SetIntraOpNumThreads,
            nanobind::arg("num_threads"))
        .def("set_inter_op_num_threads",
            &Ortpy::ThreadingOptions::SetInterOpNumThreads,
            nanobind::arg("num_threads"))
        .def("set_spin_control",
            &Ortpy::ThreadingOptions::SetSpinControl,
            nanobind::arg("allow_spinning"))
        .def("set_denormal_as_zero", &Ortpy::ThreadingOptions::SetDenormalAsZero)
        .def("set_intra_op_thread_affinity",
            &Ortpy::ThreadingOptions::SetIntraOpThreadAffinity,
            nanobind::arg("affinity"));

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
        .def_ro("device", &Ortpy::EpDevice::device)
        .def("get_memory_info",
            &Ortpy::EpDevice::GetMemoryInfo,
            nanobind::arg("memory_type"));

#if ORT_API_VERSION >= 24
    nanobind::class_<Ortpy::EpAssignedNode>(m, "EpAssignedNode")
        .def_ro("name", &Ortpy::EpAssignedNode::name)
        .def_ro("domain", &Ortpy::EpAssignedNode::domain)
        .def_ro("operator_type", &Ortpy::EpAssignedNode::operatorType);

    nanobind::class_<Ortpy::EpAssignedSubgraph>(m, "EpAssignedSubgraph")
        .def_ro("ep_name", &Ortpy::EpAssignedSubgraph::epName)
        .def_ro("nodes", &Ortpy::EpAssignedSubgraph::nodes);
#endif /** ORT_API_VERSION >= 24 */

    m.def("create_env", &Ortpy::Env::CreateEnv,
        nanobind::arg("log_level") = ORT_LOGGING_LEVEL_WARNING,
        nanobind::arg("log_id") = "Ortpy",
        nanobind::arg("logging_function") = nullptr,
        nanobind::arg("threading_options") = nullptr
#if ORT_API_VERSION >= 24
        , nanobind::arg("config_entries") = std::unordered_map<std::string, std::string>{}
#endif /** ORT_API_VERSION >= 24 */
    );

    m.def("register_execution_provider_library", [](const std::string& name, const std::string& path) {
        Ortpy::Env::GetSingleton()->RegisterExecutionProviderLibrary(name, path);
    }, nanobind::arg("name"), nanobind::arg("path"));

    m.def("unregister_execution_provider_library", [](const std::string& name) {
        Ortpy::Env::GetSingleton()->UnregisterExecutionProviderLibrary(name);
    }, nanobind::arg("name"));

    m.def("get_ep_devices", []() {
        return Ortpy::Env::GetSingleton()->GetEpDevices();
    });

    m.def("get_available_providers", &Ortpy::GetAvailableProviders);

    m.def("get_model_compatibility_for_ep_devices",
        &Ortpy::GetModelCompatibilityForEpDevices,
        nanobind::arg("ep_devices"),
        nanobind::arg("compatibility_info"));

    m.def("set_log_level", [](OrtLoggingLevel level) {
        Ortpy::Env::GetSingleton()->UpdateLogLevel(level);
    }, nanobind::arg("level"));

#if ORT_API_VERSION >= 24
    m.def("get_hardware_devices", []() {
        return Ortpy::Env::GetSingleton()->GetHardwareDevices();
    });

    nanobind::class_<Ortpy::Env::DeviceEpIncompatibilityInfo>(m, "DeviceEpIncompatibilityInfo")
        .def_ro("reasons_bitmask", &Ortpy::Env::DeviceEpIncompatibilityInfo::reasonsBitmask)
        .def_ro("notes", &Ortpy::Env::DeviceEpIncompatibilityInfo::notes)
        .def_ro("error_code", &Ortpy::Env::DeviceEpIncompatibilityInfo::errorCode);

    m.def("get_hardware_device_ep_incompatibility_details",
        [](const std::string& epName, const Ortpy::HardwareDevice& device) {
            return Ortpy::Env::GetSingleton()->GetHardwareDeviceEpIncompatibilityDetails(epName, device);
        }, nanobind::arg("ep_name"), nanobind::arg("device"));

    m.def("get_compatibility_info_from_model",
        [](const std::string& modelPath, const std::string& epType) {
            return Ortpy::Env::GetSingleton()->GetCompatibilityInfoFromModel(modelPath, epType);
        }, nanobind::arg("model_path"), nanobind::arg("ep_type"));

    m.def("get_compatibility_info_from_model_bytes",
        [](const nanobind::bytes& modelData, const std::string& epType) {
            return Ortpy::Env::GetSingleton()->GetCompatibilityInfoFromModelBytes(modelData, epType);
        }, nanobind::arg("model_data"), nanobind::arg("ep_type"));
#endif /** ORT_API_VERSION >= 24 */

    nanobind::class_<Ortpy::Value>(m, "Value")
        .def(nanobind::init<const Ortpy::NpArray&>(),
            nanobind::arg("numpy_array"))
        .def_static("from_strings", &Ortpy::Value::FromStrings,
            nanobind::arg("strings"),
            nanobind::arg("shape") = std::nullopt)
        .def("numpy", &Ortpy::Value::ToNumpy)
        .def_prop_ro("shape", &Ortpy::Value::GetShape)
        .def_prop_ro("dtype",
            [](const Ortpy::Value& self) -> std::string {
                return Ortpy::Value::NpTypeToName(
                    Ortpy::Value::OrtTypeToNpType(self.GetType()));
            })
        .def_prop_ro("is_tensor", &Ortpy::Value::IsTensor)
        .def_prop_ro("value_type", &Ortpy::Value::GetValueType)
        .def_prop_ro("has_value", &Ortpy::Value::HasValue)
        .def("get_tensor_memory_info", &Ortpy::Value::GetTensorMemoryInfo)
        .def("get_tensor_size_in_bytes", &Ortpy::Value::GetTensorSizeInBytes)
        .def("get_strings", &Ortpy::Value::GetStrings)
        .def("__getitem__", &Ortpy::Value::GetElement,
            nanobind::arg("index"))
        .def("__len__", &Ortpy::Value::GetCount);

    nanobind::class_<Ortpy::MemoryInfo>(m, "MemoryInfo")
        .def(nanobind::init<>())
        .def(nanobind::init<const std::string&, OrtAllocatorType, int, OrtMemType>(),
            nanobind::arg("name"),
            nanobind::arg("allocator_type"),
            nanobind::arg("device_id"),
            nanobind::arg("mem_type"))
        .def_prop_ro("name", &Ortpy::MemoryInfo::GetName)
        .def_prop_ro("device_id", &Ortpy::MemoryInfo::GetDeviceId)
        .def_prop_ro("mem_type", &Ortpy::MemoryInfo::GetMemType)
        .def_prop_ro("allocator_type", &Ortpy::MemoryInfo::GetAllocatorType)
        .def_prop_ro("device_type", &Ortpy::MemoryInfo::GetDeviceType)
        .def("__eq__", &Ortpy::MemoryInfo::operator==);

    nanobind::class_<Ortpy::IoBinding>(m, "IoBinding")
        .def("bind_input",
            &Ortpy::IoBinding::BindInput,
            nanobind::arg("name"),
            nanobind::arg("value"))
        .def("bind_output",
            &Ortpy::IoBinding::BindOutput,
            nanobind::arg("name"),
            nanobind::arg("value"))
        .def("bind_output_to_device",
            &Ortpy::IoBinding::BindOutputToDevice,
            nanobind::arg("name"),
            nanobind::arg("memory_info"))
        .def("get_outputs", &Ortpy::IoBinding::GetOutputs)
        .def("clear_inputs", &Ortpy::IoBinding::ClearInputs)
        .def("clear_outputs", &Ortpy::IoBinding::ClearOutputs)
        .def("synchronize_inputs", &Ortpy::IoBinding::SynchronizeInputs)
        .def("synchronize_outputs", &Ortpy::IoBinding::SynchronizeOutputs);

    nanobind::class_<Ortpy::ModelCompilationOptions>(m, "ModelCompilationOptions",
            nanobind::type_slots(modelCompilationOptionsSlots))
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
        .def("compile_model_to_buffer", &Ortpy::ModelCompilationOptions::CompileModelToBuffer)
        .def("set_flags",
            &Ortpy::ModelCompilationOptions::SetFlags,
            nanobind::arg("flags"))
        .def("set_ep_context_binary_information",
            &Ortpy::ModelCompilationOptions::SetEpContextBinaryInformation,
            nanobind::arg("output_directory"),
            nanobind::arg("model_name"))
        .def("set_graph_optimization_level",
            &Ortpy::ModelCompilationOptions::SetGraphOptimizationLevel,
            nanobind::arg("level"))
        .def("set_output_model_write_func",
            &Ortpy::ModelCompilationOptions::SetOutputModelWriteFunc,
            nanobind::arg("write_func"));

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
        .def("register_custom_ops_library_v2",
            &Ortpy::SessionOptions::RegisterCustomOpsLibrary_V2,
            nanobind::arg("library_name"))
        .def("register_custom_ops_using_function",
            &Ortpy::SessionOptions::RegisterCustomOpsUsingFunction,
            nanobind::arg("registration_func_name"))
        .def("enable_ort_custom_ops", &Ortpy::SessionOptions::EnableOrtCustomOps)
        .def("add_free_dimension_override",
            &Ortpy::SessionOptions::AddFreeDimensionOverride,
            nanobind::arg("dim_denotation"),
            nanobind::arg("dim_value"))
        .def("add_free_dimension_override_by_name",
            &Ortpy::SessionOptions::AddFreeDimensionOverrideByName,
            nanobind::arg("dim_name"),
            nanobind::arg("dim_value"))
        .def("disable_per_session_threads", &Ortpy::SessionOptions::DisablePerSessionThreads)
        .def("add_session_config_entry",
            &Ortpy::SessionOptions::AddSessionConfigEntry,
            nanobind::arg("config_key"),
            nanobind::arg("config_value"))
        .def("has_session_config_entry",
            &Ortpy::SessionOptions::HasSessionConfigEntry,
            nanobind::arg("config_key"))
        .def("get_session_config_entry",
            &Ortpy::SessionOptions::GetSessionConfigEntry,
            nanobind::arg("config_key"))
        .def("get_session_config_entries", &Ortpy::SessionOptions::GetSessionOptionsConfigEntries)
        .def("set_deterministic_compute",
            &Ortpy::SessionOptions::SetDeterministicCompute,
            nanobind::arg("value"))
        .def("set_load_cancellation_flag",
            &Ortpy::SessionOptions::SetLoadCancellationFlag,
            nanobind::arg("cancel"))
        .def("add_initializer",
            &Ortpy::SessionOptions::AddInitializer,
            nanobind::arg("name"),
            nanobind::arg("value"),
            nanobind::keep_alive<1, 3>())
        .def("add_external_initializers",
            &Ortpy::SessionOptions::AddExternalInitializers,
            nanobind::arg("initializers"))
        .def("add_external_initializers_from_files_in_memory",
            &Ortpy::SessionOptions::AddExternalInitializersFromFilesInMemory,
            nanobind::arg("files"))
        .def("clone", &Ortpy::SessionOptions::Clone)
        .def("append_execution_provider",
            &Ortpy::SessionOptions::AppendExecutionProvider,
            nanobind::arg("provider_name"),
            nanobind::arg("provider_options"))
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
        .def("set_user_logging_function",
            &Ortpy::SessionOptions::SetUserLoggingFunction,
            nanobind::arg("logging_function"))
        .def("create_model_compilation_options", &Ortpy::SessionOptions::CreateModelCompilationOptions);

    nanobind::class_<Ortpy::ModelMetadata>(m, "ModelMetadata")
        .def_prop_ro("producer_name", &Ortpy::ModelMetadata::GetProducerName)
        .def_prop_ro("graph_name", &Ortpy::ModelMetadata::GetGraphName)
        .def_prop_ro("domain", &Ortpy::ModelMetadata::GetDomain)
        .def_prop_ro("description", &Ortpy::ModelMetadata::GetDescription)
        .def_prop_ro("graph_description", &Ortpy::ModelMetadata::GetGraphDescription)
        .def_prop_ro("version", &Ortpy::ModelMetadata::GetVersion)
        .def_prop_ro("custom_metadata_map", &Ortpy::ModelMetadata::GetCustomMetadataMap)
        .def("lookup_custom_metadata",
            &Ortpy::ModelMetadata::LookupCustomMetadata,
            nanobind::arg("key"));

    nanobind::class_<Ortpy::TypeInfo>(m, "TypeInfo")
        .def_prop_ro("onnx_type", &Ortpy::TypeInfo::GetOnnxType)
        .def_prop_ro("denotation", &Ortpy::TypeInfo::GetDenotation)
        /** Tensor accessors */
        .def_prop_ro("shape", &Ortpy::TypeInfo::GetShape)
        .def_prop_ro("dimensions", &Ortpy::TypeInfo::GetSymbolicDimensions)
        .def_prop_ro("dtype", &Ortpy::TypeInfo::GetElementType)
        /** Map accessors */
        .def_prop_ro("map_key_type", &Ortpy::TypeInfo::GetMapKeyType)
        .def_prop_ro("map_value_type", &Ortpy::TypeInfo::GetMapValueType)
        /** Sequence accessor */
        .def_prop_ro("sequence_element_type", &Ortpy::TypeInfo::GetSequenceElementType)
        /** Optional accessor */
        .def_prop_ro("optional_contained_type", &Ortpy::TypeInfo::GetOptionalContainedType);

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
        .def("unset_terminate", &Ortpy::RunOptions::UnsetTerminate)
        .def("add_run_config_entry",
            &Ortpy::RunOptions::AddRunConfigEntry,
            nanobind::arg("config_key"),
            nanobind::arg("config_value"))
        .def("get_run_config_entry",
            &Ortpy::RunOptions::GetRunConfigEntry,
            nanobind::arg("config_key"))
        .def("add_active_lora_adapter",
            &Ortpy::RunOptions::AddActiveLoraAdapter,
            nanobind::arg("adapter"));

    nanobind::class_<Ortpy::Session>(m, "Session")
        .def(nanobind::init<const std::string&, const Ortpy::SessionOptions&>(),
            nanobind::arg("model_path"),
            nanobind::arg("options"))
        .def(nanobind::init<const nanobind::bytes&, const Ortpy::SessionOptions&>(),
            nanobind::arg("model_bytes"),
            nanobind::arg("options"))
        .def("get_input_info", &Ortpy::Session::GetInputInfo)
        .def("get_output_info", &Ortpy::Session::GetOutputInfo)
        .def("get_overridable_initializer_info", &Ortpy::Session::GetOverridableInitializerInfo)
        .def("get_model_metadata", &Ortpy::Session::GetModelMetadata)
        .def("end_profiling", &Ortpy::Session::EndProfiling)
        .def("get_profiling_start_time_ns", &Ortpy::Session::GetProfilingStartTimeNs)
        .def("get_memory_info_for_inputs", &Ortpy::Session::GetMemoryInfoForInputs)
        .def("get_memory_info_for_outputs", &Ortpy::Session::GetMemoryInfoForOutputs)
        .def("get_ep_device_for_inputs", &Ortpy::Session::GetEpDeviceForInputs)
#if ORT_API_VERSION >= 24
        .def("get_ep_device_for_outputs", &Ortpy::Session::GetEpDeviceForOutputs)
        .def("get_ep_graph_assignment_info", &Ortpy::Session::GetEpGraphAssignmentInfo)
#endif /** ORT_API_VERSION >= 24 */
        .def("create_io_binding", &Ortpy::Session::CreateIoBinding,
            nanobind::keep_alive<0, 1>())
        .def("run_with_binding",
            &Ortpy::Session::RunWithBinding,
            nanobind::arg("io_binding"),
            nanobind::arg("run_options") = nullptr)
        .def("run",
            &Ortpy::Session::Run,
            nanobind::arg("inputs"),
            nanobind::arg("output_names") = std::nullopt,
            nanobind::arg("run_options") = nullptr)
        .def("run_with_ort_values",
            &Ortpy::Session::RunWithOrtValues,
            nanobind::arg("inputs"),
            nanobind::arg("output_names") = std::nullopt,
            nanobind::arg("run_options") = nullptr);

    nanobind::class_<Ortpy::LoraAdapter>(m, "LoraAdapter")
        .def(nanobind::init<const std::string&>(),
            nanobind::arg("adapter_file_path"))
        .def(nanobind::init<const nanobind::bytes&>(),
            nanobind::arg("adapter_bytes"));
}
