"""Tests for ortpy.ModelCompilationOptions (CompileApi)."""
import tempfile
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

import ortpy as ort


class TestModelCompilationOptionsCreation:
    def test_create(self):
        opts = ort.SessionOptions()
        compile_opts = opts.create_model_compilation_options()
        assert compile_opts is not None


class TestModelCompilationOptionsSetters:
    def test_set_input_model_path(self, add_model_path):
        opts = ort.SessionOptions()
        co = opts.create_model_compilation_options()
        co.set_input_model_path(str(add_model_path))

    def test_set_input_model_from_buffer(self, add_model_bytes):
        opts = ort.SessionOptions()
        co = opts.create_model_compilation_options()
        co.set_input_model_from_buffer(add_model_bytes)

    def test_set_ep_context_embed_mode(self):
        opts = ort.SessionOptions()
        co = opts.create_model_compilation_options()
        co.set_ep_context_embed_mode(True)
        co.set_ep_context_embed_mode(False)

    def test_set_flags(self):
        opts = ort.SessionOptions()
        co = opts.create_model_compilation_options()
        co.set_flags(0)

    def test_set_graph_optimization_level(self):
        opts = ort.SessionOptions()
        co = opts.create_model_compilation_options()
        co.set_graph_optimization_level(ort.GraphOptimizationLevel.ENABLE_ALL)

    @pytest.mark.skip(reason="ORT validates directory with wchar path on Windows — may fail in temp dirs")
    def test_set_ep_context_binary_information(self, tmp_dir):
        opts = ort.SessionOptions()
        co = opts.create_model_compilation_options()
        out_dir = tmp_dir / "binary_output"
        out_dir.mkdir()
        co.set_ep_context_binary_information(str(out_dir), "test_model")

    def test_set_output_model_external_initializers_file(self, tmp_dir):
        opts = ort.SessionOptions()
        co = opts.create_model_compilation_options()
        co.set_output_model_external_initializers_file(
            str(tmp_dir / "ext_init.bin"), 1024
        )


class TestModelCompilationCompile:
    def test_compile_to_file(self, add_model_path, tmp_dir):
        opts = ort.SessionOptions()
        co = opts.create_model_compilation_options()
        co.set_input_model_path(str(add_model_path))
        output_path = str(tmp_dir / "compiled.onnx")
        co.compile_model_to_file(output_path)
        assert Path(output_path).exists()
        assert Path(output_path).stat().st_size > 0

    def test_compile_to_buffer(self, add_model_path):
        opts = ort.SessionOptions()
        co = opts.create_model_compilation_options()
        co.set_input_model_path(str(add_model_path))
        buf = co.compile_model_to_buffer()
        assert isinstance(buf, bytes)
        assert len(buf) > 0

    def test_compiled_model_is_loadable(self, add_model_path, tmp_dir):
        """Compile a model, then load and run the compiled version."""
        opts = ort.SessionOptions()
        co = opts.create_model_compilation_options()
        co.set_input_model_path(str(add_model_path))
        output_path = str(tmp_dir / "compiled.onnx")
        co.compile_model_to_file(output_path)

        session = ort.Session(output_path, ort.SessionOptions())
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0], dtype=np.float32)
        outputs = session.run({"A": a, "B": b})
        np.testing.assert_allclose(outputs["C"].numpy(), [4.0, 6.0])

    def test_compile_from_buffer(self, add_model_bytes, tmp_dir):
        opts = ort.SessionOptions()
        co = opts.create_model_compilation_options()
        co.set_input_model_from_buffer(add_model_bytes)
        output_path = str(tmp_dir / "compiled_from_buf.onnx")
        co.compile_model_to_file(output_path)
        assert Path(output_path).exists()


class TestModelCompilationWriteFunc:
    def test_write_func_is_settable(self, add_model_path):
        """Verify the write func can be set without errors."""
        chunks = []

        def write_fn(data: bytes):
            chunks.append(data)

        opts = ort.SessionOptions()
        co = opts.create_model_compilation_options()
        co.set_input_model_path(str(add_model_path))
        co.set_output_model_write_func(write_fn)
        # Setting the write func itself should succeed
        # Whether it's invoked depends on the EP and compilation path
