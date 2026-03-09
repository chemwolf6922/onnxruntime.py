"""Tests for ortpy.IoBinding."""
import numpy as np
import pytest

import ortpy as ort


class TestIoBindingCreation:
    def test_create_io_binding(self, add_session):
        binding = add_session.create_io_binding()
        assert binding is not None


class TestIoBindingInference:
    def test_bind_and_run(self, add_session):
        binding = add_session.create_io_binding()

        a = ort.Value(np.array([1.0, 2.0], dtype=np.float32))
        b = ort.Value(np.array([3.0, 4.0], dtype=np.float32))

        binding.bind_input("A", a)
        binding.bind_input("B", b)

        mem = ort.MemoryInfo()
        binding.bind_output_to_device("C", mem)

        add_session.run_with_binding(binding)

        outputs = binding.get_outputs()
        assert "C" in outputs
        np.testing.assert_allclose(outputs["C"].numpy(), [4.0, 6.0])

    def test_bind_output_value(self, add_session):
        binding = add_session.create_io_binding()

        a = ort.Value(np.array([1.0, 2.0], dtype=np.float32))
        b = ort.Value(np.array([3.0, 4.0], dtype=np.float32))
        c = ort.Value(np.zeros(2, dtype=np.float32))

        binding.bind_input("A", a)
        binding.bind_input("B", b)
        binding.bind_output("C", c)

        add_session.run_with_binding(binding)

        outputs = binding.get_outputs()
        assert "C" in outputs
        np.testing.assert_allclose(outputs["C"].numpy(), [4.0, 6.0])

    def test_bind_with_run_options(self, add_session):
        binding = add_session.create_io_binding()
        a = ort.Value(np.array([1.0, 2.0], dtype=np.float32))
        b = ort.Value(np.array([3.0, 4.0], dtype=np.float32))
        binding.bind_input("A", a)
        binding.bind_input("B", b)
        binding.bind_output_to_device("C", ort.MemoryInfo())

        run_opts = ort.RunOptions()
        run_opts.run_tag = "binding_test"
        add_session.run_with_binding(binding, run_options=run_opts)

        outputs = binding.get_outputs()
        np.testing.assert_allclose(outputs["C"].numpy(), [4.0, 6.0])


class TestIoBindingClear:
    def test_clear_inputs(self, add_session):
        binding = add_session.create_io_binding()
        a = ort.Value(np.array([1.0, 2.0], dtype=np.float32))
        binding.bind_input("A", a)
        binding.clear_inputs()

    def test_clear_outputs(self, add_session):
        binding = add_session.create_io_binding()
        binding.bind_output_to_device("C", ort.MemoryInfo())
        binding.clear_outputs()


class TestIoBindingSynchronize:
    def test_synchronize_inputs(self, add_session):
        binding = add_session.create_io_binding()
        a = ort.Value(np.array([1.0, 2.0], dtype=np.float32))
        binding.bind_input("A", a)
        binding.synchronize_inputs()

    def test_synchronize_outputs(self, add_session):
        binding = add_session.create_io_binding()
        binding.bind_output_to_device("C", ort.MemoryInfo())
        binding.synchronize_outputs()


class TestIoBindingReuse:
    def test_rebind_and_rerun(self, add_session):
        """Bind, run, clear, rebind with new data, run again."""
        binding = add_session.create_io_binding()
        mem = ort.MemoryInfo()

        # First run
        a1 = ort.Value(np.array([1.0, 2.0], dtype=np.float32))
        b1 = ort.Value(np.array([3.0, 4.0], dtype=np.float32))
        binding.bind_input("A", a1)
        binding.bind_input("B", b1)
        binding.bind_output_to_device("C", mem)
        add_session.run_with_binding(binding)
        out1 = binding.get_outputs()["C"].numpy().copy()

        # Clear and rebind
        binding.clear_inputs()
        binding.clear_outputs()

        a2 = ort.Value(np.array([10.0, 20.0], dtype=np.float32))
        b2 = ort.Value(np.array([30.0, 40.0], dtype=np.float32))
        binding.bind_input("A", a2)
        binding.bind_input("B", b2)
        binding.bind_output_to_device("C", mem)
        add_session.run_with_binding(binding)
        out2 = binding.get_outputs()["C"].numpy()

        np.testing.assert_allclose(out1, [4.0, 6.0])
        np.testing.assert_allclose(out2, [40.0, 60.0])
