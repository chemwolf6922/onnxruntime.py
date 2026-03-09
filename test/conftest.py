"""Shared fixtures for ortpy tests.

All test models are created in-memory using the `onnx` helper API so that
no files need to be downloaded and no specific hardware is required.
"""
import tempfile
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

import ortpy as ort


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def _make_add_model() -> onnx.ModelProto:
    """output = A + B  (float32, shape [2])."""
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [2])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [2])
    C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [2])
    graph = helper.make_graph(
        [helper.make_node("Add", ["A", "B"], ["C"])],
        "add_graph", [A, B], [C],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _make_matmul_model() -> onnx.ModelProto:
    """output = X @ W  (float32, X=[1,4], W=[4,2], Y=[1,2])."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    W = helper.make_tensor_value_info("W", TensorProto.FLOAT, [4, 2])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["X", "W"], ["Y"])],
        "matmul_graph", [X, W], [Y],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _make_identity_model(dtype=TensorProto.FLOAT, shape=None) -> onnx.ModelProto:
    """output = Identity(input)  — pass-through, configurable type and shape."""
    if shape is None:
        shape = [3]
    inp = helper.make_tensor_value_info("input", dtype, shape)
    out = helper.make_tensor_value_info("output", dtype, shape)
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"])],
        "identity_graph", [inp], [out],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _make_model_with_metadata() -> onnx.ModelProto:
    """Identity model with rich metadata fields."""
    model = _make_identity_model()
    model.producer_name = "ortpy-test"
    model.domain = "test.domain"
    model.doc_string = "Test model description"
    model.model_version = 42
    model.graph.doc_string = "Test graph description"
    entry = model.metadata_props.add()
    entry.key = "key1"
    entry.value = "value1"
    entry = model.metadata_props.add()
    entry.key = "key2"
    entry.value = "value2"
    onnx.checker.check_model(model)
    return model


def _make_model_with_initializer() -> onnx.ModelProto:
    """Y = X + bias, where bias is an overridable initializer."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])
    bias_init = numpy_helper.from_array(
        np.array([10.0, 20.0], dtype=np.float32), name="bias"
    )
    # Declare bias as an input so it becomes an overridable initializer
    bias_input = helper.make_tensor_value_info("bias", TensorProto.FLOAT, [2])
    graph = helper.make_graph(
        [helper.make_node("Add", ["X", "bias"], ["Y"])],
        "init_graph",
        [X, bias_input],  # both are inputs
        [Y],
        initializer=[bias_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _make_dynamic_shape_model() -> onnx.ModelProto:
    """Identity with dynamic batch dim: input shape [N, 3]."""
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3])
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"])],
        "dynamic_graph", [inp], [out],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _make_multi_type_model() -> onnx.ModelProto:
    """Two inputs of different types: float32 and int64."""
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [2])
    B = helper.make_tensor_value_info("B", TensorProto.INT64, [2])
    # Cast B to float, then add
    B_cast = helper.make_tensor_value_info("B_cast", TensorProto.FLOAT, [2])
    C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [2])
    graph = helper.make_graph(
        [
            helper.make_node("Cast", ["B"], ["B_cast"], to=TensorProto.FLOAT),
            helper.make_node("Add", ["A", "B_cast"], ["C"]),
        ],
        "multi_type_graph", [A, B], [C],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _make_two_output_model() -> onnx.ModelProto:
    """Two outputs: Sum = A + B, Diff = A - B."""
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [2])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [2])
    S = helper.make_tensor_value_info("Sum", TensorProto.FLOAT, [2])
    D = helper.make_tensor_value_info("Diff", TensorProto.FLOAT, [2])
    graph = helper.make_graph(
        [
            helper.make_node("Add", ["A", "B"], ["Sum"]),
            helper.make_node("Sub", ["A", "B"], ["Diff"]),
        ],
        "two_output_graph", [A, B], [S, D],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir():
    """Provides a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def add_model_path(tmp_dir) -> Path:
    path = tmp_dir / "add.onnx"
    onnx.save(_make_add_model(), str(path))
    return path


@pytest.fixture
def add_model_bytes() -> bytes:
    return _make_add_model().SerializeToString()


@pytest.fixture
def identity_model_path(tmp_dir) -> Path:
    path = tmp_dir / "identity.onnx"
    onnx.save(_make_identity_model(), str(path))
    return path


@pytest.fixture
def metadata_model_path(tmp_dir) -> Path:
    path = tmp_dir / "metadata.onnx"
    onnx.save(_make_model_with_metadata(), str(path))
    return path


@pytest.fixture
def initializer_model_path(tmp_dir) -> Path:
    path = tmp_dir / "initializer.onnx"
    onnx.save(_make_model_with_initializer(), str(path))
    return path


@pytest.fixture
def dynamic_model_path(tmp_dir) -> Path:
    path = tmp_dir / "dynamic.onnx"
    onnx.save(_make_dynamic_shape_model(), str(path))
    return path


@pytest.fixture
def matmul_model_path(tmp_dir) -> Path:
    path = tmp_dir / "matmul.onnx"
    onnx.save(_make_matmul_model(), str(path))
    return path


@pytest.fixture
def multi_type_model_path(tmp_dir) -> Path:
    path = tmp_dir / "multi_type.onnx"
    onnx.save(_make_multi_type_model(), str(path))
    return path


@pytest.fixture
def two_output_model_path(tmp_dir) -> Path:
    path = tmp_dir / "two_output.onnx"
    onnx.save(_make_two_output_model(), str(path))
    return path


@pytest.fixture
def add_session(add_model_path) -> ort.Session:
    return ort.Session(str(add_model_path), ort.SessionOptions())


@pytest.fixture
def identity_session(identity_model_path) -> ort.Session:
    return ort.Session(str(identity_model_path), ort.SessionOptions())
