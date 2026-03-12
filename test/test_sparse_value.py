"""Tests for ortpy.Value — sparse tensor construction and read-back."""
import numpy as np
import pytest

import ortpy as ort


# ---------------------------------------------------------------------------
# COO sparse tensor
# ---------------------------------------------------------------------------

class TestSparseCoo:
    def test_create_and_readback(self):
        """Round-trip a COO sparse tensor."""
        dense_shape = (4, 4)
        values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        # Flat COO indices: (0,1), (2,3), (3,0)
        indices = np.array([0, 1, 2, 3, 3, 0], dtype=np.int64)

        val = ort.Value.from_sparse_coo(dense_shape, values, indices)

        assert val.is_sparse_tensor
        assert not val.is_tensor
        assert val.sparse_format == ort.SparseFormat.COO
        assert list(val.sparse_dense_shape) == [4, 4]

        np.testing.assert_array_equal(np.asarray(val.get_sparse_values()), values)
        # ORT returns COO indices as (nnz, 2) shape
        expected_indices = np.array([[0, 1], [2, 3], [3, 0]], dtype=np.int64)
        np.testing.assert_array_equal(np.asarray(val.get_sparse_indices()), expected_indices)

    def test_2d_indices(self):
        """COO with 2-D paired indices (nnz, 2)."""
        dense_shape = (3, 3)
        values = np.array([10.0, 20.0], dtype=np.float64)
        indices = np.array([0, 0, 1, 2], dtype=np.int64)  # (0,0), (1,2)

        val = ort.Value.from_sparse_coo(dense_shape, values, indices)
        np.testing.assert_array_equal(np.asarray(val.get_sparse_values()), values)

    def test_integer_values(self):
        """COO with integer element type."""
        dense_shape = (2, 3)
        values = np.array([5, 10], dtype=np.int32)
        indices = np.array([0, 1, 1, 2], dtype=np.int64)

        val = ort.Value.from_sparse_coo(dense_shape, values, indices)
        assert val.is_sparse_tensor
        np.testing.assert_array_equal(np.asarray(val.get_sparse_values()), values)

    def test_coo_rejects_inner_outer_indices(self):
        """COO tensors should reject CSR-style index accessors."""
        dense_shape = (3, 3)
        values = np.array([1.0], dtype=np.float32)
        indices = np.array([0, 0], dtype=np.int64)
        val = ort.Value.from_sparse_coo(dense_shape, values, indices)
        with pytest.raises(RuntimeError):
            val.get_sparse_inner_indices()
        with pytest.raises(RuntimeError):
            val.get_sparse_outer_indices()


# ---------------------------------------------------------------------------
# CSR sparse tensor
# ---------------------------------------------------------------------------

class TestSparseCsr:
    def test_create_and_readback(self):
        """Round-trip a CSR sparse tensor."""
        dense_shape = (3, 4)
        values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        inner_indices = np.array([1, 3, 0], dtype=np.int64)   # column indices
        outer_indices = np.array([0, 1, 2, 3], dtype=np.int64)  # row pointers

        val = ort.Value.from_sparse_csr(dense_shape, values, inner_indices, outer_indices)

        assert val.is_sparse_tensor
        assert val.sparse_format == ort.SparseFormat.CSRC
        assert list(val.sparse_dense_shape) == [3, 4]

        np.testing.assert_array_equal(np.asarray(val.get_sparse_values()), values)
        np.testing.assert_array_equal(np.asarray(val.get_sparse_inner_indices()), inner_indices)
        np.testing.assert_array_equal(np.asarray(val.get_sparse_outer_indices()), outer_indices)

    def test_csr_rejects_generic_indices(self):
        """CSR tensors should reject COO-style index accessor."""
        dense_shape = (3, 4)
        values = np.array([1.0], dtype=np.float32)
        inner = np.array([0], dtype=np.int64)
        outer = np.array([0, 0, 0, 1], dtype=np.int64)
        val = ort.Value.from_sparse_csr(dense_shape, values, inner, outer)
        with pytest.raises(RuntimeError):
            val.get_sparse_indices()


# ---------------------------------------------------------------------------
# Block-sparse tensor
# ---------------------------------------------------------------------------

class TestSparseBlock:
    def test_create_and_readback(self):
        """Round-trip a block-sparse tensor."""
        dense_shape = (4, 4)
        # 2 blocks of 2x2
        values = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
        # Block-sparse indices: (num_blocks, 2) with [row_block_idx, col_block_idx]
        indices = np.array([[0, 0], [1, 1]], dtype=np.int32)

        val = ort.Value.from_sparse_block(dense_shape, values, indices)

        assert val.is_sparse_tensor
        assert val.sparse_format == ort.SparseFormat.BLOCK_SPARSE
        assert list(val.sparse_dense_shape) == [4, 4]

        np.testing.assert_array_equal(np.asarray(val.get_sparse_values()), values)
        np.testing.assert_array_equal(np.asarray(val.get_sparse_indices()), indices)


# ---------------------------------------------------------------------------
# Query on non-sparse values
# ---------------------------------------------------------------------------

class TestSparseOnDense:
    def test_dense_is_not_sparse(self):
        """A dense tensor should report is_sparse_tensor=False."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        val = ort.Value(arr)
        assert not val.is_sparse_tensor
        assert val.sparse_format is None
        assert val.sparse_dense_shape is None

    def test_get_sparse_values_raises_on_dense(self):
        """Calling sparse methods on a dense tensor should raise."""
        arr = np.array([1.0], dtype=np.float32)
        val = ort.Value(arr)
        with pytest.raises(RuntimeError):
            val.get_sparse_values()
        with pytest.raises(RuntimeError):
            val.get_sparse_indices()
        with pytest.raises(RuntimeError):
            val.get_sparse_inner_indices()
        with pytest.raises(RuntimeError):
            val.get_sparse_outer_indices()
