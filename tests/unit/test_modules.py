# -*- coding:utf-8 -*-
###
# File: test_modules.py
# Created Date: Sunday, November 30th 2025, 6:46:19 pm
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
import torch.nn as nn
import torch.nn.functional as F
from parameterized import parameterized
from typing import Tuple, Any

import fvdb
# from fvdb import GridBatch, JaggedTensor
import fvdbnn
from fvdbnn import fVDBTensor
from fvdbnn.utils.utils import get_gpu_compute_capacity, check_same_grid


# --- Helper for creating test instances ---

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


def create_real_fvdb_tensor(
    batch_size: int,
    channels: int,
    size: Tuple[int, int, int],
    device: str
) -> fVDBTensor:
    """Creates a valid fVDBTensor manually for testing."""
    # 1. Create dense torch data
    # Create non-zero data to ensure sparsity generation is non-trivial
    dense_data = torch.rand(batch_size, channels, *size, device=device) * 5.0
    # Set some voxels to 0 to introduce sparsity
    dense_data[:, :, 0:2, 0:2, 0:2] = 0.0
    tensor = fvdbnn.fVDBTensor_from_dense(dense_data)
    return tensor


# --- Unit Test Class ---


class TestElementwiseAndNormModules(unittest.TestCase):

    ELEMENTWISE_MODULES = [
        (fvdbnn.ELUFVDB, (0.5,)), (fvdbnn.CELUFVDB, (0.5,)),
        (fvdbnn.GELUFVDB, ()), (fvdbnn.ReLUFVDB, ()),
        (fvdbnn.LeakyReLUFVDB, (0.1,)), (fvdbnn.SELUFVDB, ()),
        (fvdbnn.SiLUFVDB, ()), (fvdbnn.TanhFVDB, ()),
        (fvdbnn.SigmoidFVDB, ()), (fvdbnn.DropoutFVDB, (0.5,))
    ]

    def setUp(self):
        """Common setup before each test."""
        self.batch_size = 2
        self.channels_in = 16
        self.channels_out = 32
        self.dense_size = (16, 16, 16)  # Larger size to test pooling/convs better

    @parameterized.expand(
        [(m, args, d) for m, args in ELEMENTWISE_MODULES for d in TEST_DEVICES])
    def test_elementwise_modules(self, module_cls, module_args, device):
        """Test all ElementwiseMixin-based modules 
        for correct operation and topology preservation."""

        instance = module_cls(*module_args).to(device)
        tensor_in = create_real_fvdb_tensor(
            self.batch_size, self.channels_in, self.dense_size, device)

        if module_cls is fvdbnn.DropoutFVDB:
            instance.train()

        tensor_out: fVDBTensor = instance(tensor_in)

        self.assertIsInstance(tensor_out, fVDBTensor, "Output must be fVDBTensor")
        self.assertIs(tensor_out.grid, tensor_in.grid,
                      "Grid reference must be preserved")
        self.assertEqual(
            tensor_out.data.jdata.shape, tensor_in.data.jdata.shape,
            "Feature shape must be unchanged")

        if module_cls is not fvdbnn.DropoutFVDB:
            base_module_cls = module_cls.__bases__[1]
            base_instance = base_module_cls(*module_args).to(device)
            expected_data = base_instance(tensor_in.data.jdata)
            self.assertTrue(
                torch.allclose(tensor_out.data.jdata, expected_data, atol=1e-5),
                f"{module_cls.__name__} calculation is incorrect."
            )

    # Assuming the norm tests are correct from the prior conversation and
    # we use the mock definitions if the original implementations aren't fully available.
    @parameterized.expand(DEVICE_PARAMS)
    def test_layer_norm_32_fvdb(self, device):
        """Test LayerNorm32FVDB's forward pass logic and dtype handling."""
        # 1. Setup
        norm_layer = fvdbnn.LayerNorm32FVDB(self.channels_in).to(device)
        tensor_in = create_real_fvdb_tensor(
            self.batch_size, self.channels_in, self.dense_size, device)

        # Ensure we test with a non-default dtype
        tensor_in_f16 = tensor_in.to(torch.float16)

        # 2. Forward Pass
        tensor_out: fVDBTensor = norm_layer(tensor_in_f16)

        # 3. Checks

        # Check A: Output Type and Topology Preservation
        self.assertIsInstance(tensor_out, fVDBTensor)
        self.assertIs(tensor_out.grid, tensor_in.grid,
                      "Grid reference must be preserved")
        self.assertEqual(
            tensor_out.data.jdata.shape, tensor_in_f16.data.jdata.shape)

        # Check B: Data Type Reversion
        self.assertEqual(tensor_out.data.jdata.dtype, torch.float16,
                         "Output dtype must match input dtype (float16)")

        # Check C: Functional Correctness (comparing against raw PyTorch LayerNorm)
        raw_output_f32 = F.layer_norm(
            tensor_in_f16.data.jdata.float(), norm_layer.normalized_shape,
            norm_layer.weight.float(), norm_layer.bias.float(), norm_layer.eps
        ).half()  # Convert expected output back to float16 for comparison

        self.assertTrue(
            torch.allclose(tensor_out.data.jdata, raw_output_f32, atol=1e-3, rtol=1e-3),
            "LayerNorm32FVDB calculation is incorrect or dtype conversion failed."
        )

    @parameterized.expand(DEVICE_PARAMS)
    def test_group_norm_32_fvdb(self, device):
        """Test GroupNorm32FVDB's forward pass and per-batch normalization logic."""

        num_groups = 4  # Must divide channels
        norm_layer = fvdbnn.GroupNorm32FVDB(num_groups, self.channels_in).to(device)
        tensor_in = create_real_fvdb_tensor(
            self.batch_size, self.channels_in, self.dense_size, device)

        # 1. Pre-check: Input channels assertion
        # If we change input channels, it should raise an error
        tensor_in_bad_channels = create_real_fvdb_tensor(
            self.batch_size, self.channels_in + 1, self.dense_size, device)
        with self.assertRaisesRegex(AssertionError, "same number of channels"):
            norm_layer(tensor_in_bad_channels)

        # 2. Forward Pass
        tensor_out: fVDBTensor = norm_layer(tensor_in)

        # 3. Checks
        self.assertIsInstance(tensor_out, fVDBTensor)
        self.assertIs(tensor_out.grid, tensor_in.grid)

        # Check B: Per-batch normalization (The logic of GroupNorm32FVDB)

        # For a single batch,
        # the group norm applied to the dense data should match the output.
        b = 0
        feat_raw = tensor_in.data.jdata[
            tensor_in.data.joffsets[b]: tensor_in.data.joffsets[b+1]].float()

        # GroupNorm implementation requires C, L format (Channels, Length)
        feat_c_l = feat_raw.transpose(0, 1).contiguous()
        feat_c_l = feat_c_l.reshape(1, self.channels_in, -1)

        # Apply the original PyTorch GroupNorm function
        expected_normed = F.group_norm(
            feat_c_l, num_groups, norm_layer.weight.float(),
            norm_layer.bias.float(), norm_layer.eps
        )

        # Reshape back to L, C format
        expected_data = expected_normed.reshape(
            self.channels_in, -1).transpose(0, 1).to(tensor_in.data.jdata.dtype)

        actual_data = tensor_out.data.jdata[
            tensor_out.data.joffsets[b]: tensor_out.data.joffsets[b+1]]

        self.assertTrue(
            torch.allclose(actual_data, expected_data),
            "GroupNorm32FVDB failed per-batch calculation check."
        )

    @parameterized.expand(DEVICE_PARAMS)
    def test_batch_norm_32_fvdb(self, device):
        """
        Test BatchNorm32FVDB, 
        verifying it processes the entire flattened JaggedTensor.
        """
        # 1. Setup
        norm_layer = fvdbnn.BatchNorm32FVDB(self.channels_in).to(device)
        tensor_in: fVDBTensor = create_real_fvdb_tensor(
            self.batch_size, self.channels_in, self.dense_size, device)

        # Set to training mode to ensure running stats are updated
        norm_layer.train()

        # 2. Forward Pass
        tensor_out: fVDBTensor = norm_layer(tensor_in)

        # 3. Checks
        self.assertIsInstance(tensor_out, fVDBTensor)
        self.assertIs(tensor_out.grid, tensor_in.grid)
        self.assertEqual(tensor_out.data.jdata.dtype, tensor_in.data.jdata.dtype)

        # Check B: Functional Correctness (comparing against raw PyTorch BatchNorm1d)

        # The BatchNorm32FVDB.super_forward is essentially
        # F.batch_norm applied to the flattened data.
        # It treats the entire JaggedTensor.jdata (shape: N_total_voxels x C)
        # as a single batch (N_total_voxels).

        # Prepare data in (N, C, L) format for PyTorch's
        # F.batch_norm (treating N=1, L=N_total_voxels)
        # Note: torch.nn.BatchNorm1d typically expects (N, C, L).
        # If we treat total voxels as N, we need (N_voxels, C).
        # Since the implementation applies BN to jdata (N_voxels, C) and
        # BN1d expects (N, L), it relies on the internal implementation of
        # super_forward to handle the 2D tensor.

        # Check if running stats were updated (in training mode)
        # Note: Running mean/var should NOT be all zeros after the first training step.
        self.assertFalse(torch.all(norm_layer.running_mean == 0.0),
                         "Running mean should have been updated.")

        # Compare output against expected raw PyTorch output (using batch statistics)
        expected_output = F.batch_norm(
            tensor_in.data.jdata.transpose(0, 1).unsqueeze(0).float(),
            norm_layer.running_mean.float(),
            norm_layer.running_var.float(),
            norm_layer.weight.float(),
            norm_layer.bias.float(),
            True,  # Use batch statistics for calculation during training
            norm_layer.momentum,
            norm_layer.eps,
        ).squeeze(0).transpose(0, 1).to(tensor_in.data.jdata.dtype)  # Convert back to (N, C)

        self.assertTrue(
            torch.allclose(tensor_out.data.jdata, expected_output, atol=1e-4, rtol=1e-4),
            "BatchNorm32FVDB calculation failed against raw F.batch_norm."
        )

    # Skipping MultiHeadRMSNorm since its implementation was complex and
    # testing it requires another fVDBTensor


class TestFVDBModules(unittest.TestCase):

    def setUp(self):
        """Common setup before each test."""
        self.batch_size = 2
        self.channels_in = 16
        self.channels_out = 32
        self.dense_size = (16, 16, 16)  # Larger size to test pooling/convs better

    # -----------------------------------------------------
    # 1. Linear Layers (ElementwiseMixin + Standard Linear)
    # -----------------------------------------------------

    @parameterized.expand(DEVICE_PARAMS)
    def test_linear_fvdb(self, device):
        """Test LinearFVDB for correct transformation and topology preservation."""

        # 1. Setup
        linear_layer = fvdbnn.LinearFVDB(
            self.channels_in, self.channels_out, bias=True).to(device)
        tensor_in = create_real_fvdb_tensor(
            self.batch_size, self.channels_in, self.dense_size, device)

        # Test with a different dtype (e.g., half precision) to check casting logic
        tensor_in_f16 = tensor_in.to(torch.float16)

        # 2. Forward Pass
        tensor_out: fVDBTensor = linear_layer(tensor_in_f16)

        # 3. Checks

        # Check A: Output Type and Topology Preservation
        self.assertIsInstance(tensor_out, fVDBTensor, "Output must be fVDBTensor")
        self.assertIs(tensor_out.grid, tensor_in_f16.grid,
                      "Grid reference must be preserved")

        # Check B: Output Feature Shape and Dtype
        expected_shape = (tensor_in_f16.data.jdata.size(0), self.channels_out)
        self.assertEqual(tensor_out.data.jdata.shape, expected_shape,
                         "Output feature shape mismatch")
        self.assertEqual(tensor_out.data.dtype, torch.float16,
                         "Output dtype must match input dtype (float16)")

        # Check C: Functional Correctness (comparing against raw PyTorch function)
        weight_f16 = linear_layer.weight.to(dtype=torch.float16)
        bias_f16 = linear_layer.bias.to(dtype=torch.float16)
        # pylint: disable=not-callable
        expected_data = F.linear(tensor_in_f16.data.jdata, weight_f16, bias_f16)

        self.assertTrue(
            torch.allclose(tensor_out.data.jdata, expected_data, atol=1e-3, rtol=1e-3),
            "LinearFVDB calculation is incorrect."
        )

    @parameterized.expand(DEVICE_PARAMS)
    def test_cast_2_intype_linear(self, device):
        """Test Cast2IntypeLinear for correct dtype casting on raw tensors."""

        # 1. Setup
        linear_layer = fvdbnn.Cast2IntypeLinear(
            self.channels_in, self.channels_out, bias=True).to(device)

        # Create input tensor in float16
        # (different from default nn.Parameter dtype, usually float32)
        input_tensor = torch.rand(10, self.channels_in, device=device).half()

        # 2. Forward Pass
        output_tensor: torch.Tensor = linear_layer(input_tensor)

        # 3. Checks

        # Check A: Output Dtype
        self.assertEqual(output_tensor.dtype, torch.float16,
                         "Output dtype must match input dtype (float16)")

        # Check B: Weight/Bias Dtype in Forward
        self.assertEqual(linear_layer.weight.dtype, torch.float32,
                         "Original weight dtype must remain float32")

        # Check C: Functional Correctness
        weight_f16 = linear_layer.weight.to(dtype=torch.float16)
        bias_f16 = linear_layer.bias.to(dtype=torch.float16)
        # pylint: disable=not-callable
        expected_output = F.linear(input_tensor, weight_f16, bias_f16)

        self.assertTrue(
            torch.allclose(output_tensor, expected_output, atol=1e-3, rtol=1e-3),
            "Cast2IntypeLinear calculation is incorrect after casting."
        )

    # -----------------------------------------------------
    # 2. Pooling and Upsampling Layers
    # -----------------------------------------------------

    POOLING_MODULES = [fvdbnn.MaxPoolFVDB, fvdbnn.AvgPoolFVDB]

    @parameterized.expand([(m, d) for m in POOLING_MODULES for d in TEST_DEVICES])
    def test_pooling_fvdb(self, module_cls, device):
        """Test MaxPoolFVDB and AvgPoolFVDB for structural changes (pooling)."""

        # 1. Setup: Stride 2, Kernel 2 should halve the grid resolution
        # Note: fvdb.nn.MaxPool/AvgPool only take kernel_size, stride is implicitly 2
        # for downsampling or handled internally by fvdb. We will rely on default behavior.
        # We assume the default pooling downsamples the grid resolution by 2.
        pooling_layer = module_cls(kernel_size=2).to(device)
        tensor_in = create_real_fvdb_tensor(
            self.batch_size, self.channels_in, self.dense_size, device)

        # 2. Forward Pass
        tensor_out: fVDBTensor = pooling_layer(tensor_in)

        # 3. Checks

        # Check A: Output Type and Grid Change
        self.assertIsInstance(tensor_out, fVDBTensor, "Output must be fVDBTensor")
        self.assertIsNot(
            tensor_out.grid, tensor_in.grid,
            "Pooling must create a new grid object (downsampled)")

        # Check B: Feature Count Reduction (Approximate check, as sparsity varies)
        # Downsampling by 2^3 = 8 factor should reduce voxel count significantly
        input_voxels = tensor_in.grid.total_voxels
        output_voxels = tensor_out.grid.total_voxels

        # The downsampled voxel count should be less than the input count
        self.assertLess(output_voxels, input_voxels,
                        "Voxel count must decrease after pooling.")

        # Check C: Feature Shape
        self.assertEqual(tensor_out.data.jdata.size(1), self.channels_in,
                         "Output channel count must be preserved.")

    @parameterized.expand(DEVICE_PARAMS)
    def test_upsampling_fvdb(self, device):
        """Test UpsamplingNearestFVDB for structural changes (upsampling)."""

        # 1. Setup: Upsampling typically increases resolution by 2
        upsampling_layer = fvdbnn.UpsamplingNearestFVDB(scale_factor=2).to(device)
        tensor_in = create_real_fvdb_tensor(
            self.batch_size, self.channels_in, self.dense_size, device)

        # 2. Forward Pass
        tensor_out = upsampling_layer(tensor_in)

        # 3. Checks

        # Check A: Output Type and Grid Change
        self.assertIsInstance(tensor_out, fVDBTensor, "Output must be fVDBTensor")
        self.assertIsNot(
            tensor_out.grid, tensor_in.grid,
            "Upsampling must create a new grid object (upsampled)")

        # Check B: Feature Count Increase (Approximate check)
        input_voxels = tensor_in.grid.total_voxels
        output_voxels = tensor_out.grid.total_voxels

        # The upsampled voxel count should be greater than the input count
        self.assertGreater(output_voxels, input_voxels,
                           "Voxel count must increase after upsampling.")

        # Check C: Feature Shape
        self.assertEqual(tensor_out.data.jdata.size(1), self.channels_in,
                         "Output channel count must be preserved.")

    # -----------------------------------------------------
    # 3. Sparse Convolutions Layers
    # -----------------------------------------------------

    CONV_MODULES = [fvdbnn.SparseConv3dFVDB, fvdbnn.SparseConvTranspose3dFVDB]
    # CONV_MODULES = [fvdbnn.SparseConv3dFVDB]

    @parameterized.expand([(m, d) for m in CONV_MODULES for d in TEST_DEVICES])
    def test_sparse_conv_fvdb_forward(self, module_cls, device):
        """Test SparseConv3dFVDB and SparseConvTranspose3dFVDB for basic execution."""

        # 1. Setup
        kernel_size = 3
        # Stride 2 for Conv, Stride 1 for ConvTranspose
        # only stride = 1 is supported for now
        stride = 1 if module_cls is fvdbnn.SparseConv3dFVDB else 1

        conv_layer = module_cls(
            in_channels=self.channels_in,
            out_channels=self.channels_out,
            kernel_size=kernel_size,
            stride=stride
        ).to(device)
        tensor_in = create_real_fvdb_tensor(
            self.batch_size, self.channels_in, self.dense_size, device)

        # 2. Forward Pass
        tensor_out: fVDBTensor = conv_layer(tensor_in)

        # 3. Checks

        # Check A: Output Type and Grid Change
        self.assertIsInstance(tensor_out, fVDBTensor, "Output must be fVDBTensor")
        self.assertIsNot(tensor_out.grid, tensor_in.grid,
                         "Convolution must create a new grid object")

        # Check B: Output Channel Count
        self.assertEqual(
            tensor_out.data.jdata.size(1),
            self.channels_out, "Output channel count mismatch.")

        # Check C: Plan Caching - Must be registered after the first call
        conv_key = (f"conv3d_ks{conv_layer.kernel_size}_stride{conv_layer.stride}"
                    if module_cls is fvdbnn.SparseConv3dFVDB else
                    f"conv3d_transpose3d_ks{conv_layer.kernel_size}_stride{conv_layer.stride}")
        # print(tensor_in.get_spatial_cache())
        plan_cache: fvdb.ConvolutionPlan = tensor_in.get_spatial_cache(conv_key)
        self.assertIsInstance(plan_cache, fvdb.ConvolutionPlan,
                              "Convolution Plan must be cached.")

        # Check D: Second call uses the cached plan
        tensor_out_cached: fVDBTensor = conv_layer(tensor_in)
        self.assertTrue(
            check_same_grid(tensor_out_cached.grid, plan_cache.target_grid_batch),
            "Cached grid should be used in second forward pass.")
        self.assertTrue(
            torch.allclose(tensor_out.data.jdata, tensor_out_cached.data.jdata),
            "Second execution with cached plan must yield identical results."
        )


# --- Execution ---
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
