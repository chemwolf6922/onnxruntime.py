"""Tests for SyncStream, SharedAllocator, CopyTensors, and Value.create_empty.

These tests exercise the CPU paths of the device APIs. GPU-specific behavior
(e.g., actual async copies) requires GPU hardware and is not tested here.
"""
import numpy as np
import pytest

import ortpy as ort


class TestSharedAllocator:
    def test_get_cpu_allocator(self):
        """CPU shared allocator should be available by default."""
        mem = ort.MemoryInfo()
        allocator = ort.SharedAllocator.get(mem)
        assert allocator is not None

    def test_get_returns_shared_allocator_type(self):
        mem = ort.MemoryInfo()
        allocator = ort.SharedAllocator.get(mem)
        assert isinstance(allocator, ort.SharedAllocator)


class TestValueCreateEmpty:
    def test_create_empty_float32(self):
        mem = ort.MemoryInfo()
        allocator = ort.SharedAllocator.get(mem)
        value = ort.Value.create_empty([2, 3], "float32", allocator)
        assert value.is_tensor
        assert value.shape == [2, 3]

    def test_create_empty_int64(self):
        mem = ort.MemoryInfo()
        allocator = ort.SharedAllocator.get(mem)
        value = ort.Value.create_empty([4], "int64", allocator)
        assert value.is_tensor
        assert value.shape == [4]

    def test_create_empty_has_numpy_for_cpu(self):
        """CPU-allocated empty tensor should be accessible via numpy."""
        mem = ort.MemoryInfo()
        allocator = ort.SharedAllocator.get(mem)
        value = ort.Value.create_empty([2, 2], "float32", allocator)
        arr = value.numpy()
        assert arr.shape == (2, 2)
        assert arr.dtype == np.float32

    def test_create_empty_scalar(self):
        mem = ort.MemoryInfo()
        allocator = ort.SharedAllocator.get(mem)
        value = ort.Value.create_empty([], "float32", allocator)
        assert value.is_tensor
        assert value.shape == []


class TestCopyTensors:
    def test_copy_cpu_to_cpu_not_supported(self):
        """CPU-to-CPU copy is not supported by CopyTensors (requires cross-device EP)."""
        src_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        src = ort.Value(src_arr)

        mem = ort.MemoryInfo()
        allocator = ort.SharedAllocator.get(mem)
        dst = ort.Value.create_empty([2, 2], "float32", allocator)

        # CopyTensors requires a data transfer EP (e.g., CUDA). CPU EP doesn't provide one.
        with pytest.raises(RuntimeError, match="Data transfer"):
            ort.copy_tensors([src], [dst])

    def test_copy_with_none_stream_not_supported(self):
        """Passing None as stream still fails on CPU (no data transfer EP)."""
        src = ort.Value(np.array([1.0], dtype=np.float32))
        mem = ort.MemoryInfo()
        allocator = ort.SharedAllocator.get(mem)
        dst = ort.Value.create_empty([1], "float32", allocator)
        with pytest.raises(RuntimeError, match="Data transfer"):
            ort.copy_tensors([src], [dst], None)


class TestRunOptionsSetSyncStream:
    def test_set_sync_stream_exists(self):
        """RunOptions should have set_sync_stream method."""
        ro = ort.RunOptions()
        assert hasattr(ro, "set_sync_stream")

