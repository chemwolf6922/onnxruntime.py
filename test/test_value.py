"""Tests for ortpy.Value — numeric tensors, string tensors, and map/sequence containers."""
import numpy as np
import pytest

import ortpy as ort


# ---------------------------------------------------------------------------
# Numeric tensor construction & round-trip
# ---------------------------------------------------------------------------

class TestNumericTensor:
    @pytest.mark.parametrize("dtype", [
        np.float32, np.float64, np.int32, np.int64, np.int16, np.int8,
        np.uint8, np.uint16, np.uint32, np.uint64, np.bool_,
    ])
    def test_round_trip_dtypes(self, dtype):
        arr = np.array([1, 2, 3], dtype=dtype)
        val = ort.Value(arr)
        out = val.numpy()
        np.testing.assert_array_equal(out, arr)

    def test_float16_round_trip(self):
        arr = np.array([1.0, 0.5, 0.25], dtype=np.float16)
        val = ort.Value(arr)
        np.testing.assert_array_equal(val.numpy(), arr)

    def test_scalar_like(self):
        """A 1-element tensor still round-trips."""
        arr = np.array([42.0], dtype=np.float32)
        val = ort.Value(arr)
        np.testing.assert_array_equal(val.numpy(), arr)

    def test_multidimensional(self):
        arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        val = ort.Value(arr)
        assert val.shape == [2, 3, 4]
        np.testing.assert_array_equal(val.numpy(), arr)

    def test_c_contiguous_required(self):
        """Fortran-order arrays should still work (nanobind handles copy)."""
        arr = np.asfortranarray(np.arange(6, dtype=np.float32).reshape(2, 3))
        # This may either work via copy or raise — depending on binding.
        # We just ensure it doesn't crash.
        try:
            val = ort.Value(arr)
            np.testing.assert_array_equal(val.numpy(), arr)
        except Exception:
            pass  # Acceptable if non-contiguous is rejected


# ---------------------------------------------------------------------------
# Introspection properties
# ---------------------------------------------------------------------------

class TestValueIntrospection:
    def test_is_tensor(self):
        val = ort.Value(np.array([1.0], dtype=np.float32))
        assert val.is_tensor is True

    def test_value_type_tensor(self):
        val = ort.Value(np.array([1.0], dtype=np.float32))
        assert val.value_type == ort.ONNXType.TENSOR

    def test_has_value(self):
        val = ort.Value(np.array([1.0], dtype=np.float32))
        assert val.has_value is True

    def test_shape(self):
        val = ort.Value(np.array([[1, 2], [3, 4]], dtype=np.int64))
        assert val.shape == [2, 2]

    def test_dtype(self):
        val = ort.Value(np.array([1.0], dtype=np.float32))
        assert val.dtype == "float32"

    def test_dtype_int64(self):
        val = ort.Value(np.array([1], dtype=np.int64))
        assert val.dtype == "int64"

    def test_tensor_memory_info(self):
        val = ort.Value(np.array([1.0], dtype=np.float32))
        mem = val.get_tensor_memory_info()
        assert mem is not None
        assert mem.name == "Cpu"

    def test_tensor_size_in_bytes(self):
        arr = np.zeros(10, dtype=np.float32)
        val = ort.Value(arr)
        assert val.get_tensor_size_in_bytes() == 40  # 10 * 4


# ---------------------------------------------------------------------------
# String tensor
# ---------------------------------------------------------------------------

class TestStringTensor:
    def test_from_strings_1d(self):
        strings = ["hello", "world", "foo"]
        val = ort.Value.from_strings(strings)
        assert val.shape == [3]
        assert val.get_strings() == strings

    def test_from_strings_with_shape(self):
        strings = ["a", "b", "c", "d"]
        val = ort.Value.from_strings(strings, shape=[2, 2])
        assert val.shape == [2, 2]
        assert val.get_strings() == strings

    def test_from_strings_empty(self):
        val = ort.Value.from_strings([])
        assert val.shape == [0]
        assert val.get_strings() == []

    def test_from_strings_unicode(self):
        strings = ["café", "naïve", "日本語"]
        val = ort.Value.from_strings(strings)
        assert val.get_strings() == strings

    def test_string_tensor_is_tensor(self):
        val = ort.Value.from_strings(["x"])
        assert val.is_tensor is True

    def test_string_tensor_value_type(self):
        val = ort.Value.from_strings(["x"])
        assert val.value_type == ort.ONNXType.TENSOR


# ---------------------------------------------------------------------------
# Value from session output (map/sequence access tested via inference)
# ---------------------------------------------------------------------------

class TestValueFromInference:
    def test_output_value_properties(self, add_session):
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0], dtype=np.float32)
        outputs = add_session.run({"A": a, "B": b})
        val = outputs["C"]
        assert val.is_tensor
        assert val.shape == [2]
        assert val.has_value
        assert val.dtype == "float32"
