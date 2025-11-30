# -*- coding:utf-8 -*-
###
# File: test_fvdbtensor.py
# Created Date: Sunday, November 30th 2025, 5:28:11 pm
# Author: iYuqinL
# -----
# Last Modified:
# Modified By:
# -----
# Copyright © 2025 iYuqinL Holding Limited
#
# All shall be well and all shall be well and all manner of things shall be well.
# Nope...we're doomed!
# -----
# HISTORY:
# Date      	By	Comments
# ----------	---	----------------------------------------------------------
###
import unittest
import torch
from typing import Any, Tuple
from parameterized import parameterized
import math

# Assuming fvdb, GridBatch, JaggedTensor, and fVDBTensor (and helper functions)
# are correctly imported from the user's context/provided code.
import fvdb
from fvdb import GridBatch, JaggedTensor
import fvdbnn
from fvdbnn import fVDBTensor
from fvdbnn.utils.utils import get_gpu_compute_capacity


# --- Test Environment Setup ---
# Define the list of devices: CPU is always included, 'cuda' is included if available.
TEST_DEVICES = ['cpu']

if torch.cuda.is_available():
    # Note: get_gpu_compute_capacity is assumed to be defined by the user's import path.
    major_cc, minor_cc = get_gpu_compute_capacity(torch.device('cuda'))
    if major_cc >= 7:
        TEST_DEVICES.append('cuda')

# Prepare parameters for parameterized tests
DEVICE_PARAMS = [(device,) for device in TEST_DEVICES]


# --- Helper for creating test instances ---

def create_real_fvdb_tensor(
    batch_size: int,
    channels: int,
    size: Tuple[int, int, int],
    device: str
) -> fvdbnn.fVDBTensor:
    """Creates a valid fVDBTensor using the fVDBTensor_from_dense helper."""
    dense_data = torch.rand(batch_size, channels, *size, device=device)
    # This relies on the fVDBTensor_from_dense function provided by the user
    return fvdbnn.fVDBTensor_from_dense(dense_data)


# --- Unit Test Class ---

class TestFVDBTensor(unittest.TestCase):

    def setUp(self):
        """Common setup before each test."""
        self.batch_size = 4
        self.channels = 2
        self.dense_size = (8, 8, 8)  # W, H, D

    # -----------------------------
    # 1. Initialization and Checks
    # -----------------------------

    @parameterized.expand(DEVICE_PARAMS)
    def test_initialization_success_and_properties(self, device):
        """Test successful initialization and basic property access."""
        tensor = create_real_fvdb_tensor(
            self.batch_size, self.channels, self.dense_size, device)

        self.assertIsInstance(tensor, fVDBTensor)
        self.assertIsInstance(tensor.grid, GridBatch)
        self.assertIsInstance(tensor.data, JaggedTensor)

        # Check basic properties
        self.assertEqual(len(tensor), self.batch_size)
        self.assertEqual(tensor.grid_count, self.batch_size)
        self.assertEqual(tensor.data.jdata.size(1), self.channels)
        self.assertEqual(tensor.device.type, torch.device(device).type)

    def test_initialization_value_error_manual_mismatch(self):
        """Test initialization failure due to mismatched voxel count."""
        # Create valid components but tamper with one to trigger the check
        tensor = create_real_fvdb_tensor(
            self.batch_size, self.channels, self.dense_size, 'cpu')

        # Attempt to create a new fVDBTensor with mismatched total_voxels count
        # (This is tricky with real fvdb, but we test the wrapper's logic)
        mismatched_data = tensor.data.clone()
        mismatched_data.jdata = torch.randn(tensor.grid.total_voxels, self.channels)

        # --- Define the MismatchedGridBatch class here ---
        # It MUST inherit from the actual fvdb.GridBatch class.
        # It will override the total_voxels property to inject the mismatch.
        class MismatchedGridBatch(GridBatch):
            def __init__(self, original_grid, mismatched_voxel_count):
                # We must call the parent's __init__ or initialize its internals.
                # Since fvdb.GridBatch internal initialization is complex,
                # we will initialize essential attributes by copying and overriding the mismatch.
                
                # This is a safe way to create a shallow copy and bypass complex internal C/C++ init,
                # but relies on the GridBatch not having side-effects in its __init__.
                self.__dict__ = original_grid.__dict__.copy() 
                self._mismatched_voxel_count = mismatched_voxel_count

            @property
            def total_voxels(self) -> int:
                """Override the property to return the intended mismatched count."""
                return self._mismatched_voxel_count

        mismatched_grid = MismatchedGridBatch(tensor.grid, tensor.grid.total_voxels + 1)

        with self.assertRaisesRegex(ValueError, "same total voxel count"):
            fVDBTensor(mismatched_grid, tensor.data)

    # -----------------------------
    # 2. Basic Tensor Operations
    # -----------------------------

    def test_getitem_and_slicing(self):
        """Test indexing and slicing."""
        tensor = create_real_fvdb_tensor(
            self.batch_size, self.channels, self.dense_size, 'cpu')

        # Test basic indexing (returns a single grid)
        sliced_tensor = tensor[1]
        self.assertIsInstance(sliced_tensor, fVDBTensor)
        self.assertEqual(len(sliced_tensor), 1)

        # Test slice (returns a batch of grids)
        sliced_batch = tensor[1:3]
        self.assertIsInstance(sliced_batch, fVDBTensor)
        self.assertEqual(len(sliced_batch), 2)

    def test_to_dense(self):
        """Test conversion to dense tensor and shape."""
        tensor = create_real_fvdb_tensor(
            self.batch_size, self.channels, self.dense_size, 'cpu')

        # Note: inject_to_dense_cmajor requires the grid to know its dense size,
        # which is usually inferred during from_dense or stored.
        dense_out = tensor.to_dense()

        self.assertIsInstance(dense_out, torch.Tensor)

        # Expected shape: (B, C, W, H, D)
        expected_shape = (self.batch_size, self.channels, *self.dense_size)
        self.assertEqual(dense_out.shape, expected_shape)

    def test_is_same(self):
        """Test the is_same comparison logic based on grid address."""
        tensor1 = create_real_fvdb_tensor(
            self.batch_size, self.channels, self.dense_size, 'cpu')

        # Create a second tensor that shares the same grid topology (using clone)
        tensor_shared_grid = fVDBTensor(tensor1.grid, tensor1.data.clone())

        # 1. Comparison with shared GridBatch reference
        self.assertTrue(
            tensor1.is_same(tensor_shared_grid), "Should be same if grid is shared")

        # 2. Comparison with GridBatch object itself
        self.assertTrue(
            tensor1.is_same(tensor1.grid), "Should match against its own GridBatch")

        # 3. Comparison with a different GridBatch (different address)
        tensor_diff_grid = create_real_fvdb_tensor(
            self.batch_size, self.channels, (4, 4, 4), 'cpu')
        self.assertFalse(
            tensor1.is_same(tensor_diff_grid), "Should be false if topology differs")

    # -----------------------------
    # 3. Arithmetic Operations
    # -----------------------------

    def test_binary_operations_grid_preservation(self):
        """Test binary ops maintain grid reference and return new data."""
        tensor1 = create_real_fvdb_tensor(
            self.batch_size, self.channels, self.dense_size, 'cpu')
        scalar = 5.0

        # Test Tensor-Scalar operation
        result_mul_scalar = tensor1 * scalar
        self.assertIsInstance(result_mul_scalar, fVDBTensor)

        # Check 1: Grid topology must be preserved (shared reference)
        self.assertIs(tensor1.grid, result_mul_scalar.grid)

        # Check 2: Data must be a new object (out-of-place operation)
        self.assertIsNot(tensor1.data, result_mul_scalar.data)

    def test_inplace_operations_data_modification(self):
        """Test inplace ops modify data in place and return self."""
        tensor = create_real_fvdb_tensor(
            self.batch_size, self.channels, self.dense_size, 'cpu')

        # Capture initial data object reference
        initial_data_ref = tensor.data

        # Inplace addition with a scalar
        result = tensor.__iadd__(1.0)  # Or tensor += 1.0

        self.assertIs(result, tensor)  # Must return self

        # Check 2: Data object must remain the same (inplace)
        self.assertIs(tensor.data, initial_data_ref)

    def test_unary_operations_inplace_vs_outplace(self):
        """Test out-of-place (sqrt) vs. inplace (sqrt_) behavior."""
        tensor = create_real_fvdb_tensor(
            self.batch_size, self.channels, self.dense_size, 'cpu')

        # Out-of-place: sqrt()
        out_result = tensor.sqrt()
        self.assertIsInstance(out_result, fVDBTensor)
        self.assertIsNot(out_result.data, tensor.data)
        self.assertIs(out_result.grid, tensor.grid)

        # Inplace: sqrt_()
        inplace_result = tensor.sqrt_()
        self.assertIs(inplace_result, tensor)
        self.assertIs(inplace_result.data, tensor.data)

    # -----------------------------
    # 4. Spatial Cache
    # -----------------------------

    def test_spatial_cache_logic(self):
        """Test cache registration and retrieval based on grid identity."""
        tensor = create_real_fvdb_tensor(1, 1, (10, 10, 10), 'cpu')

        # 1. Register cache
        tensor.register_spatial_cache("Voxel_ID", 42)

        # 2. Retrieve cache
        self.assertEqual(tensor.get_spatial_cache("Voxel_ID"), 42)

        # 3. Change topology (simulated by creating a new tensor with different voxsize)
        tensor_diff_scale = create_real_fvdb_tensor(1, 1, (20, 20, 20), 'cpu')

        # 4. Check that cache is not retrieved for different scale
        self.assertIsNone(tensor_diff_scale.get_spatial_cache("Voxel_ID"))

        # 5. Check retrieval of all caches for current scale
        tensor_diff_scale.register_spatial_cache("Voxel_ID_2", 99)
        all_cache = tensor_diff_scale.get_spatial_cache()
        self.assertIn("Voxel_ID_2", all_cache)
        self.assertNotIn("Voxel_ID", all_cache)

    # -----------------------------
    # 5. Helper function tests (jcat)
    # -----------------------------

    def test_jcat_channels_concatenation(self):
        """
        Test jcat for fVDBTensor which should concatenate along 
        the channel dimension (dim != 0).
        """
        # Create two tensors with the same grid/topology but different features
        tensor1 = create_real_fvdb_tensor(self.batch_size, 2, self.dense_size, 'cpu')
        tensor2 = fVDBTensor(tensor1.grid, tensor1.data.clone())  # Same grid, new data

        # Concatenate along channel dimension
        # (default for jcat on fVDBTensor is assumed to be dim=None/1)
        # Note: The provided jcat implementation assumes dim=None
        # implies channel cat if not GridBatch/JaggedTensor

        # Check 1: dim=None (assumed to be channel/feature concatenation)
        result_cat_default = fvdbnn.jcat([tensor1, tensor2])
        self.assertIsInstance(result_cat_default, fVDBTensor)
        self.assertEqual(result_cat_default.num_tensors, 8)

        # Check 2: dim=1 (explicit channel/feature concatenation)
        result_cat_dim1 = fvdbnn.jcat([tensor1, tensor2], dim=1)
        self.assertEqual(result_cat_dim1.data.jdata.size(1), 4)
        self.assertIs(result_cat_dim1.grid, tensor1.grid)

        # Check 3: Invalid dim=0 concatenation
        with self.assertRaisesRegex(ValueError, "does not support dim=0"):
            fvdbnn.jcat([tensor1, tensor2], dim=0)


# --- Execution ---
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
