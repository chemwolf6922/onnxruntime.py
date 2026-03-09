"""Tests for ortpy.TypeInfo — tensor, map, sequence, and optional type introspection."""
import tempfile
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

import ortpy as ort


# ---------------------------------------------------------------------------
# Helpers to create models with specific type signatures
# ---------------------------------------------------------------------------

def _save_model(model: onnx.ModelProto, path: Path) -> Path:
    onnx.save(model, str(path))
    return path


def _make_map_output_model() -> onnx.ModelProto:
    """ZipMap produces seq<map<int64, float>> output from a 2D float input."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
    # Build the sequence<map<int64, float>> type for Y manually
    Y = onnx.helper.make_value_info("Y", onnx.TypeProto())
    seq_type = Y.type.sequence_type
    map_type = seq_type.elem_type.map_type
    map_type.key_type = TensorProto.INT64
    map_type.value_type.tensor_type.elem_type = TensorProto.FLOAT

    graph = helper.make_graph(
        [helper.make_node("ZipMap", ["X"], ["Y"],
                          classlabels_int64s=[0, 1, 2],
                          domain="ai.onnx.ml")],
        "zipmap_graph",
        [X],
        [Y],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 13),
            helper.make_opsetid("ai.onnx.ml", 1),
        ],
    )
    model.ir_version = 7
    return model


# ---------------------------------------------------------------------------
# Tensor TypeInfo
# ---------------------------------------------------------------------------

class TestTypeInfoTensor:
    def test_onnx_type(self, add_model_path):
        session = ort.Session(str(add_model_path), ort.SessionOptions())
        info = session.get_input_info()
        ti = info["A"]
        assert ti.onnx_type == ort.ONNXType.TENSOR

    def test_shape(self, add_model_path):
        session = ort.Session(str(add_model_path), ort.SessionOptions())
        ti = session.get_input_info()["A"]
        assert ti.shape == [2]

    def test_element_type_float(self, add_model_path):
        session = ort.Session(str(add_model_path), ort.SessionOptions())
        ti = session.get_input_info()["A"]
        assert ti.dtype == "float32"

    def test_dtype_string(self, add_model_path):
        session = ort.Session(str(add_model_path), ort.SessionOptions())
        ti = session.get_input_info()["A"]
        assert ti.dtype == "float32"

    def test_multidim_shape(self, matmul_model_path):
        session = ort.Session(str(matmul_model_path), ort.SessionOptions())
        ti = session.get_input_info()["X"]
        assert ti.shape == [1, 4]

    def test_int64_element_type(self, multi_type_model_path):
        session = ort.Session(str(multi_type_model_path), ort.SessionOptions())
        ti = session.get_input_info()["B"]
        assert ti.dtype == "int64"

    def test_symbolic_dimensions(self, dynamic_model_path):
        session = ort.Session(str(dynamic_model_path), ort.SessionOptions())
        ti = session.get_input_info()["input"]
        dims = ti.dimensions
        assert len(dims) == 2
        # First dim is symbolic "N", second is literal "3"
        assert dims[0] == "N"

    def test_denotation(self, add_model_path):
        session = ort.Session(str(add_model_path), ort.SessionOptions())
        ti = session.get_input_info()["A"]
        # Most models don't set denotation, so it should be empty
        assert ti.denotation == ""


# ---------------------------------------------------------------------------
# Map/Sequence TypeInfo (via ZipMap model)
# ---------------------------------------------------------------------------

class TestTypeInfoMapSequence:
    @pytest.fixture
    def zipmap_session(self, tmp_dir):
        model = _make_map_output_model()
        path = _save_model(model, tmp_dir / "zipmap.onnx")
        return ort.Session(str(path), ort.SessionOptions())

    def test_zipmap_output_is_sequence(self, zipmap_session):
        out_info = zipmap_session.get_output_info()
        ti = out_info["Y"]
        assert ti.onnx_type == ort.ONNXType.SEQUENCE

    def test_zipmap_sequence_element_is_map(self, zipmap_session):
        ti = zipmap_session.get_output_info()["Y"]
        elem_ti = ti.sequence_element_type
        assert elem_ti.onnx_type == ort.ONNXType.MAP

    def test_zipmap_map_key_type(self, zipmap_session):
        ti = zipmap_session.get_output_info()["Y"]
        map_ti = ti.sequence_element_type
        assert map_ti.map_key_type == "int64"

    def test_zipmap_map_value_type(self, zipmap_session):
        ti = zipmap_session.get_output_info()["Y"]
        map_ti = ti.sequence_element_type
        val_ti = map_ti.map_value_type
        assert val_ti.onnx_type == ort.ONNXType.TENSOR
        assert val_ti.dtype == "float32"

    def test_zipmap_inference_and_access(self, zipmap_session):
        """Run ZipMap and access the sequence/map output values."""
        x = np.array([[0.1, 0.5, 0.4]], dtype=np.float32)
        outputs = zipmap_session.run({"X": x})
        seq_val = outputs["Y"]
        # Should be a sequence of length 1 (one row)
        assert len(seq_val) == 1
        map_val = seq_val[0]
        # Map has 3 entries (keys 0, 1, 2)
        assert len(map_val) == 2  # map has key-value pair: __getitem__(0)=keys, __getitem__(1)=values
