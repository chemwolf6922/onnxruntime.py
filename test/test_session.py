"""Tests for ortpy.Session — creation, inference, profiling, and queries."""
import tempfile
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

import ortpy as ort


# ---------------------------------------------------------------------------
# Session creation
# ---------------------------------------------------------------------------

class TestSessionCreation:
    def test_create_from_path(self, add_model_path):
        session = ort.Session(str(add_model_path), ort.SessionOptions())
        assert session is not None

    def test_create_from_bytes(self, add_model_bytes):
        session = ort.Session(add_model_bytes, ort.SessionOptions())
        assert session is not None

    def test_invalid_path_raises(self):
        with pytest.raises(RuntimeError):
            ort.Session("nonexistent_model.onnx", ort.SessionOptions())

    def test_invalid_bytes_raises(self):
        with pytest.raises(RuntimeError):
            ort.Session(b"not_a_valid_model", ort.SessionOptions())


# ---------------------------------------------------------------------------
# Input/Output info
# ---------------------------------------------------------------------------

class TestSessionInfo:
    def test_input_info_add_model(self, add_session):
        info = add_session.get_input_info()
        assert len(info) == 2
        assert "A" in info
        assert "B" in info
        assert info["A"].onnx_type == ort.ONNXType.TENSOR
        assert info["A"].shape == [2]

    def test_output_info_add_model(self, add_session):
        info = add_session.get_output_info()
        assert len(info) == 1
        assert "C" in info
        assert info["C"].shape == [2]

    def test_input_count_matmul(self, matmul_model_path):
        session = ort.Session(str(matmul_model_path), ort.SessionOptions())
        info = session.get_input_info()
        assert len(info) == 2
        assert "X" in info
        assert "W" in info
        assert info["X"].shape == [1, 4]
        assert info["W"].shape == [4, 2]

    def test_overridable_initializer_info(self, initializer_model_path):
        session = ort.Session(str(initializer_model_path), ort.SessionOptions())
        info = session.get_overridable_initializer_info()
        assert "bias" in info
        assert info["bias"].shape == [2]

    def test_memory_info_for_inputs(self, add_session):
        mem_info = add_session.get_memory_info_for_inputs()
        assert "A" in mem_info
        assert "B" in mem_info
        assert mem_info["A"].name == "Cpu"

    def test_memory_info_for_outputs(self, add_session):
        mem_info = add_session.get_memory_info_for_outputs()
        assert "C" in mem_info

    def test_ep_device_for_inputs(self, add_session):
        ep_devices = add_session.get_ep_device_for_inputs()
        assert "A" in ep_devices
        assert ep_devices["A"].ep_name is not None

    def test_ep_device_for_outputs(self, add_session):
        if not hasattr(add_session, "get_ep_device_for_outputs"):
            pytest.skip("get_ep_device_for_outputs not available (ORT API < 24)")
        ep_devices = add_session.get_ep_device_for_outputs()
        assert "C" in ep_devices

    def test_ep_graph_assignment_info(self, add_model_path):
        if not hasattr(ort.Session, "get_ep_graph_assignment_info"):
            pytest.skip("get_ep_graph_assignment_info not available (ORT API < 24)")
        opts = ort.SessionOptions()
        opts.add_session_config_entry(
            "session.record_ep_graph_assignment_info", "1"
        )
        session = ort.Session(str(add_model_path), opts)
        info = session.get_ep_graph_assignment_info()
        assert isinstance(info, list)
        assert len(info) > 0
        assert info[0].ep_name is not None
        assert len(info[0].nodes) > 0


# ---------------------------------------------------------------------------
# Inference — run()
# ---------------------------------------------------------------------------

class TestSessionRun:
    def test_add_model(self, add_session):
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0], dtype=np.float32)
        outputs = add_session.run({"A": a, "B": b})
        np.testing.assert_allclose(outputs["C"].numpy(), [4.0, 6.0])

    def test_run_specific_outputs(self, two_output_model_path):
        session = ort.Session(str(two_output_model_path), ort.SessionOptions())
        a = np.array([5.0, 3.0], dtype=np.float32)
        b = np.array([1.0, 2.0], dtype=np.float32)
        # Request only "Sum"
        outputs = session.run({"A": a, "B": b}, output_names=["Sum"])
        assert "Sum" in outputs
        assert "Diff" not in outputs
        np.testing.assert_allclose(outputs["Sum"].numpy(), [6.0, 5.0])

    def test_run_all_outputs(self, two_output_model_path):
        session = ort.Session(str(two_output_model_path), ort.SessionOptions())
        a = np.array([5.0, 3.0], dtype=np.float32)
        b = np.array([1.0, 2.0], dtype=np.float32)
        outputs = session.run({"A": a, "B": b})
        assert "Sum" in outputs
        assert "Diff" in outputs
        np.testing.assert_allclose(outputs["Sum"].numpy(), [6.0, 5.0])
        np.testing.assert_allclose(outputs["Diff"].numpy(), [4.0, 1.0])

    def test_matmul_model(self, matmul_model_path):
        session = ort.Session(str(matmul_model_path), ort.SessionOptions())
        x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        w = np.ones((4, 2), dtype=np.float32)
        outputs = session.run({"X": x, "W": w})
        np.testing.assert_allclose(outputs["Y"].numpy(), [[10.0, 10.0]])

    def test_multi_type_inputs(self, multi_type_model_path):
        session = ort.Session(str(multi_type_model_path), ort.SessionOptions())
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3, 4], dtype=np.int64)
        outputs = session.run({"A": a, "B": b})
        np.testing.assert_allclose(outputs["C"].numpy(), [4.0, 6.0])

    def test_dynamic_shape(self, dynamic_model_path):
        opts = ort.SessionOptions()
        opts.add_free_dimension_override_by_name("N", 2)
        session = ort.Session(str(dynamic_model_path), opts)
        inp = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        outputs = session.run({"input": inp})
        np.testing.assert_array_equal(outputs["output"].numpy(), inp)

    def test_identity_float(self, identity_session):
        inp = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        outputs = identity_session.run({"input": inp})
        np.testing.assert_array_equal(outputs["output"].numpy(), inp)


# ---------------------------------------------------------------------------
# Inference — run_with_ort_values()
# ---------------------------------------------------------------------------

class TestSessionRunWithOrtValues:
    def test_add_model(self, add_session):
        a = ort.Value(np.array([1.0, 2.0], dtype=np.float32))
        b = ort.Value(np.array([3.0, 4.0], dtype=np.float32))
        outputs = add_session.run_with_ort_values({"A": a, "B": b})
        np.testing.assert_allclose(outputs["C"].numpy(), [4.0, 6.0])

    def test_specific_outputs(self, two_output_model_path):
        session = ort.Session(str(two_output_model_path), ort.SessionOptions())
        a = ort.Value(np.array([5.0, 3.0], dtype=np.float32))
        b = ort.Value(np.array([1.0, 2.0], dtype=np.float32))
        outputs = session.run_with_ort_values(
            {"A": a, "B": b}, output_names=["Diff"]
        )
        assert "Diff" in outputs
        assert "Sum" not in outputs
        np.testing.assert_allclose(outputs["Diff"].numpy(), [4.0, 1.0])


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------

class TestSessionProfiling:
    def test_profiling_round_trip(self, add_model_path, tmp_dir):
        opts = ort.SessionOptions()
        opts.enable_profiling(str(tmp_dir / "prof"))
        session = ort.Session(str(add_model_path), opts)
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0], dtype=np.float32)
        session.run({"A": a, "B": b})
        profile_file = session.end_profiling()
        assert Path(profile_file).exists()

    def test_profiling_start_time_ns(self, add_model_path, tmp_dir):
        opts = ort.SessionOptions()
        opts.enable_profiling(str(tmp_dir / "prof"))
        session = ort.Session(str(add_model_path), opts)
        t = session.get_profiling_start_time_ns()
        assert isinstance(t, int)
        assert t > 0


# ---------------------------------------------------------------------------
# Session from bytes
# ---------------------------------------------------------------------------

class TestSessionFromBytes:
    def test_inference_from_bytes(self, add_model_bytes):
        session = ort.Session(add_model_bytes, ort.SessionOptions())
        a = np.array([10.0, 20.0], dtype=np.float32)
        b = np.array([30.0, 40.0], dtype=np.float32)
        outputs = session.run({"A": a, "B": b})
        np.testing.assert_allclose(outputs["C"].numpy(), [40.0, 60.0])


# ---------------------------------------------------------------------------
# PrepackedWeightsContainer
# ---------------------------------------------------------------------------

class TestPrepackedWeightsContainer:
    def test_create(self):
        container = ort.PrepackedWeightsContainer()
        assert container is not None

    def test_session_with_prepacked_weights_from_path(self, add_model_path):
        container = ort.PrepackedWeightsContainer()
        session = ort.Session(str(add_model_path), ort.SessionOptions(),
                              prepacked_weights=container)
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0], dtype=np.float32)
        outputs = session.run({"A": a, "B": b})
        np.testing.assert_allclose(outputs["C"].numpy(), [4.0, 6.0])

    def test_session_with_prepacked_weights_from_bytes(self, add_model_bytes):
        container = ort.PrepackedWeightsContainer()
        session = ort.Session(add_model_bytes, ort.SessionOptions(),
                              prepacked_weights=container)
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0], dtype=np.float32)
        outputs = session.run({"A": a, "B": b})
        np.testing.assert_allclose(outputs["C"].numpy(), [4.0, 6.0])

    def test_shared_container_across_sessions(self, add_model_path):
        container = ort.PrepackedWeightsContainer()
        s1 = ort.Session(str(add_model_path), ort.SessionOptions(),
                         prepacked_weights=container)
        s2 = ort.Session(str(add_model_path), ort.SessionOptions(),
                         prepacked_weights=container)
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0], dtype=np.float32)
        np.testing.assert_allclose(
            s1.run({"A": a, "B": b})["C"].numpy(), [4.0, 6.0])
        np.testing.assert_allclose(
            s2.run({"A": a, "B": b})["C"].numpy(), [4.0, 6.0])
