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

#include "Pyort.h"

#ifndef PYORT_VERSION
#define PYORT_VERSION "0.0"
#endif

static PyType_Slot sessionOptionsSlots[] = {
    { Py_tp_traverse, (void*) &Pyort::SessionOptions::TpTraverse },
    { Py_tp_clear, (void*) &Pyort::SessionOptions::TpClear },
    { 0, nullptr }
};

NB_MODULE(_pyort, m) {
    m.doc() = "onnxruntime binding build upon C API.";
    m.attr("__version__") = PYORT_VERSION;
    m.attr("ORT_API_VERSION") = ORT_API_VERSION;

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

    nanobind::class_<Pyort::HardwareDevice>(m, "HardwareDevice")
        .def_ro("type", &Pyort::HardwareDevice::type)
        .def_ro("vendor_id", &Pyort::HardwareDevice::vendorId)
        .def_ro("vendor", &Pyort::HardwareDevice::vendor)
        .def_ro("device_id", &Pyort::HardwareDevice::deviceId)
        .def_ro("metadata", &Pyort::HardwareDevice::metadata);

    nanobind::class_<Pyort::EpDevice>(m, "EpDevice")
        .def_ro("ep_name", &Pyort::EpDevice::epName)
        .def_ro("ep_vendor", &Pyort::EpDevice::epVendor)
        .def_ro("ep_metadata", &Pyort::EpDevice::epMetadata)
        .def_ro("ep_options", &Pyort::EpDevice::epOptions)
        .def_ro("device", &Pyort::EpDevice::device);

    m.def("register_execution_provider_library", [](const std::string& name, const std::string& path) -> void {
        Pyort::Env::GetSingleton()->RegisterExecutionProviderLibrary(name, path);
    });

    m.def("unregister_execution_provider_library", [](const std::string& name) -> void {
        Pyort::Env::GetSingleton()->UnregisterExecutionProviderLibrary(name);
    });

    m.def("get_ep_devices", []() -> std::vector<Pyort::EpDevice> {
        return Pyort::Env::GetSingleton()->GetEpDevices();
    });

    nanobind::class_<Pyort::ModelCompilationOptions>(m, "ModelCompilationOptions")
        .def("set_input_model_path",
            &Pyort::ModelCompilationOptions::SetInputModelPath,
            nanobind::arg("path"))
        .def("set_input_model_from_buffer",
            &Pyort::ModelCompilationOptions::SetInputModelFromBuffer,
            nanobind::arg("model_bytes"))
        .def("set_output_model_external_initializers_file",
            &Pyort::ModelCompilationOptions::SetOutputModelExternalInitializersFile,
            nanobind::arg("path"),
            nanobind::arg("external_initializer_size_threshold"))
        .def("set_ep_context_embed_mode",
            &Pyort::ModelCompilationOptions::SetEpContextEmbedMode,
            nanobind::arg("embed_context"))
        .def("compile_model_to_file",
            &Pyort::ModelCompilationOptions::CompileModelToFile,
            nanobind::arg("path"))
        .def("compile_model_to_buffer", &Pyort::ModelCompilationOptions::CompileModelToBuffer);

    nanobind::class_<Pyort::SessionOptions>(m, "SessionOptions", nanobind::type_slots(sessionOptionsSlots))
        .def(nanobind::init<>())
        .def("append_execution_provider_v2",
            &Pyort::SessionOptions::AppendExecutionProvider_V2,
            nanobind::arg("ep_devices"),
            nanobind::arg("options"))
        .def("set_ep_selection_policy",
            &Pyort::SessionOptions::SetEpSelectionPolicy,
            nanobind::arg("policy"))
        .def("set_ep_selection_policy_delegate",
            &Pyort::SessionOptions::SetEpSelectionPolicyDelegate,
            nanobind::arg("delegate"),
            R"pbdoc(
set_ep_selection_policy_delegate(delegate: Callable[[List[EpDevice], Dict[str, str], Dict[str, str], int]])
            )pbdoc")
        .def("create_model_compilation_options", &Pyort::SessionOptions::CreateModelCompilationOptions);

    nanobind::class_<Pyort::TensorInfo>(m, "TensorInfo")
        .def_ro("shape", &Pyort::TensorInfo::shape)
        .def_ro("dimensions", &Pyort::TensorInfo::dimensions)
        .def_prop_ro("dtype",
            [](const Pyort::TensorInfo &self) -> nanobind::object {
                return Pyort::Value::NpTypeToPythonObject(self.dtype);
            });

    nanobind::class_<Pyort::Session>(m, "Session")
        .def(nanobind::init<const std::string&, const Pyort::SessionOptions&>(),
             nanobind::arg("model_path"),
             nanobind::arg("options"))
        .def("get_input_info", &Pyort::Session::GetInputInfo)
        .def("get_output_info", &Pyort::Session::GetOutputInfo)
        .def("run",
             &Pyort::Session::Run,
             nanobind::arg("inputs"));
}
