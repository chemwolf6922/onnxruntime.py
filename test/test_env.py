"""Tests for module-level functions and Env-related APIs."""
import pytest

import ortpy as ort


class TestModuleAttributes:
    def test_version_exists(self):
        # __version__ is on the _ortpy extension module
        import ortpy._ortpy as _ortpy
        assert hasattr(_ortpy, "__version__")
        assert isinstance(_ortpy.__version__, str)

    def test_ort_api_version(self):
        assert hasattr(ort, "ORT_API_VERSION")
        assert isinstance(ort.ORT_API_VERSION, int)
        assert ort.ORT_API_VERSION >= 23


class TestGetAvailableProviders:
    def test_returns_list(self):
        providers = ort.get_available_providers()
        assert isinstance(providers, list)

    def test_cpu_provider_available(self):
        providers = ort.get_available_providers()
        assert "CPUExecutionProvider" in providers


class TestSetLogLevel:
    def test_set_all_levels(self):
        for level in [
            ort.LogLevel.VERBOSE,
            ort.LogLevel.INFO,
            ort.LogLevel.WARNING,
            ort.LogLevel.ERROR,
            ort.LogLevel.FATAL,
        ]:
            ort.set_log_level(level)

    def test_restore_default(self):
        ort.set_log_level(ort.LogLevel.WARNING)  # restore sensible default


class TestGetEpDevices:
    def test_returns_list(self):
        devices = ort.get_ep_devices()
        assert isinstance(devices, list)

    def test_at_least_one_device(self):
        """CPU EP should always provide at least one device."""
        devices = ort.get_ep_devices()
        assert len(devices) >= 1

    def test_device_fields(self):
        devices = ort.get_ep_devices()
        for d in devices:
            assert isinstance(d.ep_name, str)
            assert len(d.ep_name) > 0
            assert isinstance(d.ep_vendor, str)
            assert isinstance(d.ep_metadata, dict)
            assert isinstance(d.ep_options, dict)
            assert isinstance(d.device, ort.HardwareDevice)

    def test_ep_device_memory_info(self):
        devices = ort.get_ep_devices()
        # At least one device should support DEFAULT memory
        for d in devices:
            mem = d.get_memory_info(ort.DeviceMemoryType.DEFAULT)
            if mem is not None:
                assert isinstance(mem, ort.MemoryInfo)
                return
        # If none had memory info, that's still valid


class TestHardwareDevices:
    def test_get_hardware_devices(self):
        if not hasattr(ort, "get_hardware_devices"):
            pytest.skip("get_hardware_devices not available (ORT API < 24)")
        devices = ort.get_hardware_devices()
        assert isinstance(devices, list)
        assert len(devices) >= 1  # At least CPU

    def test_hardware_device_fields(self):
        if not hasattr(ort, "get_hardware_devices"):
            pytest.skip("get_hardware_devices not available (ORT API < 24)")
        devices = ort.get_hardware_devices()
        for d in devices:
            assert isinstance(d.type, ort.HardwareDeviceType)
            assert isinstance(d.vendor, str)
            assert isinstance(d.device_id, int)
            assert isinstance(d.vendor_id, int)
            assert isinstance(d.metadata, dict)


class TestEnums:
    """Verify all exposed enums have the expected members."""

    def test_log_level(self):
        assert ort.LogLevel.VERBOSE is not None
        assert ort.LogLevel.INFO is not None
        assert ort.LogLevel.WARNING is not None
        assert ort.LogLevel.ERROR is not None
        assert ort.LogLevel.FATAL is not None

    def test_execution_mode(self):
        assert ort.ExecutionMode.SEQUENTIAL is not None
        assert ort.ExecutionMode.PARALLEL is not None

    def test_graph_optimization_level(self):
        assert ort.GraphOptimizationLevel.DISABLE_ALL is not None
        assert ort.GraphOptimizationLevel.ENABLE_BASIC is not None
        assert ort.GraphOptimizationLevel.ENABLE_EXTENDED is not None
        assert ort.GraphOptimizationLevel.ENABLE_ALL is not None

    def test_hardware_device_type(self):
        assert ort.HardwareDeviceType.CPU is not None
        assert ort.HardwareDeviceType.GPU is not None
        assert ort.HardwareDeviceType.NPU is not None

    def test_ep_device_policy(self):
        assert ort.ExecutionProviderDevicePolicy.DEFAULT is not None
        assert ort.ExecutionProviderDevicePolicy.PREFER_CPU is not None

    def test_onnx_type(self):
        assert ort.ONNXType.TENSOR is not None
        assert ort.ONNXType.MAP is not None
        assert ort.ONNXType.SEQUENCE is not None
        assert ort.ONNXType.OPTIONAL is not None
        assert ort.ONNXType.UNKNOWN is not None

    def test_compiled_model_compatibility(self):
        assert ort.CompiledModelCompatibility.EP_NOT_APPLICABLE is not None
        assert ort.CompiledModelCompatibility.EP_SUPPORTED_OPTIMAL is not None
        assert ort.CompiledModelCompatibility.EP_UNSUPPORTED is not None

    def test_device_memory_type(self):
        assert ort.DeviceMemoryType.DEFAULT is not None
        assert ort.DeviceMemoryType.HOST_ACCESSIBLE is not None
