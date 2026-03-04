"""Basic smoke test: create a simple ONNX model that adds two numbers, then load and run it with ortpy."""
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

import ortpy as ort


def create_add_model(path: Path) -> None:
    """Create a minimal ONNX model: output = A + B (element-wise, float32, shape [2])."""
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [2])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [2])
    C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [2])

    add_node = helper.make_node("Add", inputs=["A", "B"], outputs=["C"])

    graph = helper.make_graph([add_node], "add_graph", [A, B], [C])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7

    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def test_add_model() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / "add.onnx"
        create_add_model(model_path)

        session_options = ort.SessionOptions()
        session = ort.Session(str(model_path), session_options)

        # Verify input/output info
        input_info = session.get_input_info()
        assert "A" in input_info
        assert "B" in input_info
        assert input_info["A"].shape == [2]
        assert input_info["A"].dtype == "float32"

        output_info = session.get_output_info()
        assert "C" in output_info

        # Run inference
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0], dtype=np.float32)
        outputs = session.run({"A": a, "B": b})

        assert "C" in outputs
        c = outputs["C"].numpy()
        expected = np.array([4.0, 6.0], dtype=np.float32)
        np.testing.assert_allclose(c, expected, rtol=1e-5)
