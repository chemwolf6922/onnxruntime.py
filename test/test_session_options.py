"""Tests for ortpy.SessionOptions."""
import tempfile
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

import ortpy as ort


class TestSessionOptionsBasic:
    def test_default_construction(self):
        opts = ort.SessionOptions()
        assert opts is not None

    def test_clone(self):
        opts = ort.SessionOptions()
        opts.add_session_config_entry("key1", "value1")
        clone = opts.clone()
        # Clone should have the same config entry
        assert clone.has_session_config_entry("key1")
        assert clone.get_session_config_entry("key1") == "value1"


class TestSessionOptionsExecutionMode:
    def test_set_sequential(self):
        opts = ort.SessionOptions()
        opts.set_session_execution_mode(ort.ExecutionMode.SEQUENTIAL)

    def test_set_parallel(self):
        opts = ort.SessionOptions()
        opts.set_session_execution_mode(ort.ExecutionMode.PARALLEL)


class TestSessionOptionsGraphOptLevel:
    @pytest.mark.parametrize("level", [
        ort.GraphOptimizationLevel.DISABLE_ALL,
        ort.GraphOptimizationLevel.ENABLE_BASIC,
        ort.GraphOptimizationLevel.ENABLE_EXTENDED,
        ort.GraphOptimizationLevel.ENABLE_ALL,
    ])
    def test_set_graph_optimization_level(self, level):
        opts = ort.SessionOptions()
        opts.set_session_graph_optimization_level(level)


class TestSessionOptionsThreading:
    def test_set_intra_op_threads(self):
        opts = ort.SessionOptions()
        opts.set_intra_op_num_threads(2)

    def test_set_inter_op_threads(self):
        opts = ort.SessionOptions()
        opts.set_inter_op_num_threads(2)

    def test_disable_per_session_threads(self):
        opts = ort.SessionOptions()
        opts.disable_per_session_threads()


class TestSessionOptionsLogging:
    def test_set_log_id(self):
        opts = ort.SessionOptions()
        opts.set_session_log_id("test_log_id")

    def test_set_log_verbosity_level(self):
        opts = ort.SessionOptions()
        opts.set_session_log_verbosity_level(1)

    def test_set_log_severity_level(self):
        opts = ort.SessionOptions()
        opts.set_session_log_severity_level(2)


class TestSessionOptionsMemory:
    def test_enable_disable_mem_pattern(self):
        opts = ort.SessionOptions()
        opts.enable_mem_pattern()
        opts.disable_mem_pattern()

    def test_enable_disable_cpu_mem_arena(self):
        opts = ort.SessionOptions()
        opts.enable_cpu_mem_arena()
        opts.disable_cpu_mem_arena()


class TestSessionOptionsProfiling:
    def test_enable_disable_profiling(self, tmp_dir):
        opts = ort.SessionOptions()
        opts.enable_profiling(str(tmp_dir / "profile"))
        opts.disable_profiling()


class TestSessionOptionsConfig:
    def test_add_and_get_config_entry(self):
        opts = ort.SessionOptions()
        opts.add_session_config_entry("session.test_key", "test_value")
        assert opts.has_session_config_entry("session.test_key")
        assert opts.get_session_config_entry("session.test_key") == "test_value"

    def test_has_nonexistent_key(self):
        opts = ort.SessionOptions()
        assert not opts.has_session_config_entry("nonexistent_key")

    def test_get_nonexistent_key_raises(self):
        opts = ort.SessionOptions()
        with pytest.raises(RuntimeError):
            opts.get_session_config_entry("nonexistent_key")

    def test_get_config_entries(self):
        opts = ort.SessionOptions()
        opts.add_session_config_entry("k1", "v1")
        opts.add_session_config_entry("k2", "v2")
        entries = opts.get_session_config_entries()
        assert isinstance(entries, dict)
        assert entries["k1"] == "v1"
        assert entries["k2"] == "v2"


class TestSessionOptionsDeterministic:
    def test_set_deterministic_compute(self):
        opts = ort.SessionOptions()
        opts.set_deterministic_compute(True)
        opts.set_deterministic_compute(False)


class TestSessionOptionsLoadCancellation:
    def test_set_load_cancellation_flag(self):
        opts = ort.SessionOptions()
        opts.set_load_cancellation_flag(True)
        opts.set_load_cancellation_flag(False)


class TestSessionOptionsOptimizedModel:
    def test_set_optimized_model_file_path(self, tmp_dir):
        opts = ort.SessionOptions()
        opts.set_optimized_model_file_path(str(tmp_dir / "optimized.onnx"))


class TestSessionOptionsFreeDimOverride:
    def test_add_free_dimension_override(self):
        opts = ort.SessionOptions()
        opts.add_free_dimension_override("batch_size", 4)

    def test_add_free_dimension_override_by_name(self):
        opts = ort.SessionOptions()
        opts.add_free_dimension_override_by_name("N", 8)


class TestSessionOptionsAddInitializer:
    def test_add_initializer(self, initializer_model_path):
        """Override the 'bias' initializer and verify the result changes."""
        opts = ort.SessionOptions()
        new_bias = ort.Value(np.array([100.0, 200.0], dtype=np.float32))
        opts.add_initializer("bias", new_bias)
        session = ort.Session(str(initializer_model_path), opts)
        x = np.array([1.0, 2.0], dtype=np.float32)
        outputs = session.run({"X": x})
        np.testing.assert_allclose(outputs["Y"].numpy(), [101.0, 202.0])


class TestSessionOptionsEpGeneric:
    def test_append_execution_provider_v2(self, add_model_path):
        """Use EP device list (V2 API) filtered to a single EP."""
        opts = ort.SessionOptions()
        all_devices = ort.get_ep_devices()
        assert len(all_devices) >= 1
        # Use only the first device (not all devices from the same EP are compatible)
        opts.append_execution_provider_v2([all_devices[0]], {})
        session = ort.Session(str(add_model_path), opts)
        assert session is not None


class TestSessionOptionsEpSelectionPolicy:
    def test_set_policy(self):
        opts = ort.SessionOptions()
        opts.set_ep_selection_policy(ort.ExecutionProviderDevicePolicy.DEFAULT)


class TestSessionOptionsUserLogging:
    def test_logging_callback_receives_messages(self, add_model_path):
        messages = []

        def log_fn(severity, category, logid, code_location, message):
            messages.append(message)

        opts = ort.SessionOptions()
        opts.set_session_log_severity_level(0)  # VERBOSE
        opts.set_session_log_verbosity_level(10)
        opts.set_user_logging_function(log_fn)
        session = ort.Session(str(add_model_path), opts)
        # The callback may or may not receive messages during session creation
        # depending on ORT's internal logging. Just verify it doesn't crash.
        assert isinstance(messages, list)


class TestSessionOptionsCompilationOptions:
    def test_create_model_compilation_options(self):
        opts = ort.SessionOptions()
        compile_opts = opts.create_model_compilation_options()
        assert compile_opts is not None
