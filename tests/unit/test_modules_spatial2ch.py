# -*- coding:utf-8 -*-
###
# File: test_spatial2ch.py
# Created Date: Sunday, November 30th 2025, 3:47:46 pm
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
from parameterized import parameterized

import fvdb
import fvdbnn
from fvdbnn import fVDBTensor_from_dense
from fvdbnn.modules.spatial2ch import (
    DownSamplingSpatial2ChannelFVDB,
    UpSamplingChannel2SpatialFVDB
)
from fvdbnn.utils.utils import get_gpu_compute_capacity


# --- Test Environment Setup ---
# Define the list of devices: CPU is always included, 'cuda' is included if available.
TEST_DEVICES = ['cpu']

if torch.cuda.is_available():
    major_cc, minor_cc = get_gpu_compute_capacity(torch.device('cuda'))
    if major_cc >= 7:
        TEST_DEVICES.append('cuda')

# Prepare parameters for parameterized tests
DEVICE_PARAMS = [(device,) for device in TEST_DEVICES]


# --- Helper for creating test instances ---

def create_test_fvdb_tensor(
    batch_size: int,
    channels: int,
    size: tuple,
    device: str
) -> fvdbnn.fVDBTensor:
    """Creates a valid fVDBTensor using the fVDBTensor_from_dense helper."""
    dense_data = torch.rand(batch_size, channels, *size, device=device)
    return fvdbnn.fVDBTensor_from_dense(dense_data)


# --- Unit Test Class ---

class TestSpatial2Ch(unittest.TestCase):

    def setUp(self):
        """Common setup before each test."""
        self.batch_size = 2
        self.channels = 4
        self.dense_size = (16, 16, 16)  # W, H, D
        self.scale_factor = 2

    # -----------------------------
    # DownSamplingSpatial2ChannelFVDB Tests
    # -----------------------------

    def test_downsampling_initialization(self):
        """Test valid and invalid initialization parameters for downsampling."""
        # Valid: scale_factor=2
        _ = DownSamplingSpatial2ChannelFVDB(scale_factor=2)

        # Valid: with in_channels, out_channels
        _ = DownSamplingSpatial2ChannelFVDB(scale_factor=2, in_channels=4, out_channels=8)

        # Valid: with middle_channels
        _ = DownSamplingSpatial2ChannelFVDB(
            scale_factor=2, in_channels=4, middle_channels=6, out_channels=8
        )

        # Invalid: scale_factor <= 1
        with self.assertRaises(AssertionError):
            DownSamplingSpatial2ChannelFVDB(scale_factor=1)

        # Invalid: non-integer scale_factor
        with self.assertRaises(AssertionError):
            DownSamplingSpatial2ChannelFVDB(scale_factor=1.5)

        # Invalid: unknown middle_proj_type
        with self.assertRaises(ValueError):
            DownSamplingSpatial2ChannelFVDB(
                scale_factor=2, in_channels=4, middle_channels=16,
                middle_proj_type="invalid")

        # Invalid: unknown out_proj_type
        with self.assertRaises(ValueError):
            DownSamplingSpatial2ChannelFVDB(
                scale_factor=2, in_channels=4, out_proj_type="invalid"
            )

    @parameterized.expand(DEVICE_PARAMS)
    def test_downsampling_shape_consistency(self, device):
        """Test output shapes for downsampling."""
        scale_factor = 2
        downsampling = DownSamplingSpatial2ChannelFVDB(scale_factor=scale_factor)

        x = create_test_fvdb_tensor(self.batch_size, self.channels, self.dense_size, device)

        # Test without projections
        x_down = downsampling(x)

        # Verify the output is a fVDBTensor
        self.assertIsInstance(x_down, fvdbnn.fVDBTensor)

        # The output grid should be coarsened by scale_factor
        expected_size = tuple(s // scale_factor for s in self.dense_size)
        dense_out = x_down.to_dense()
        self.assertEqual(
            dense_out.shape,
            (self.batch_size, self.channels * (scale_factor ** 3),
             *expected_size))

    @parameterized.expand(DEVICE_PARAMS)
    def test_downsampling_with_projections(self, device):
        """Test downsampling with middle and out projections."""
        scale_factor = 2
        in_channels = 4
        middle_channels = 2
        out_channels = 8

        # Test with linear projections
        downsampling_linear = DownSamplingSpatial2ChannelFVDB(
            scale_factor=scale_factor,
            in_channels=in_channels,
            middle_channels=middle_channels,
            out_channels=out_channels,
            middle_proj_type="linear",
            out_proj_type="linear"
        )

        x = create_test_fvdb_tensor(self.batch_size, in_channels, self.dense_size, device)
        x_down_linear = downsampling_linear(x)

        dense_out_linear = x_down_linear.to_dense()
        expected_size = tuple(s // scale_factor for s in self.dense_size)
        self.assertEqual(dense_out_linear.shape, (self.batch_size, out_channels, *expected_size))

    # -----------------------------
    # UpSamplingChannel2SpatialFVDB Tests
    # -----------------------------

    def test_upsampling_initialization(self):
        """Test valid and invalid initialization parameters for upsampling."""
        # Valid: scale_factor=2
        _ = UpSamplingChannel2SpatialFVDB(scale_factor=2)

        # Valid: with in_channels, out_channels
        _ = UpSamplingChannel2SpatialFVDB(scale_factor=2, in_channels=8, out_channels=4)

        # Valid: with middle_channels
        _ = UpSamplingChannel2SpatialFVDB(
            scale_factor=2, in_channels=8, middle_channels=16, out_channels=4
        )

        # Invalid: scale_factor <= 1
        with self.assertRaises(AssertionError):
            UpSamplingChannel2SpatialFVDB(scale_factor=1)

        # Invalid: non-integer scale_factor
        with self.assertRaises(AssertionError):
            UpSamplingChannel2SpatialFVDB(scale_factor=1.5)

        # Invalid: unknown middle_proj_type
        with self.assertRaises(ValueError):
            UpSamplingChannel2SpatialFVDB(
                scale_factor=2, in_channels=8, middle_channels=16,
                middle_proj_type="invalid"
            )

        # Invalid: unknown out_proj_type
        with self.assertRaises(ValueError):
            UpSamplingChannel2SpatialFVDB(
                scale_factor=2, in_channels=8, out_proj_type="invalid"
            )

    @parameterized.expand(DEVICE_PARAMS)
    def test_upsampling_shape_consistency(self, device):
        """Test output shapes for upsampling."""
        scale_factor = 2
        channels = 8  # Should be divisible by scale_factor**3

        upsampling = UpSamplingChannel2SpatialFVDB(scale_factor=scale_factor)

        # Create downsampled input
        down_size = tuple(s // scale_factor for s in self.dense_size)
        x_down = create_test_fvdb_tensor(self.batch_size, channels, down_size, device)

        # Test without projections
        x_up = upsampling(x_down)

        # Verify the output is a fVDBTensor
        self.assertIsInstance(x_up, fvdbnn.fVDBTensor)

        # The output grid should be refined by scale_factor
        dense_out = x_up.to_dense()
        self.assertEqual(dense_out.shape, (self.batch_size, channels //
                         (scale_factor**3), *self.dense_size))

    @parameterized.expand(DEVICE_PARAMS)
    def test_upsampling_with_projections(self, device):
        """Test upsampling with middle and out projections."""
        scale_factor = 2
        in_channels = 8  # Should be divisible by scale_factor**3
        middle_channels = 16
        out_channels = 4

        # Test with linear projections
        upsampling_linear = UpSamplingChannel2SpatialFVDB(
            scale_factor=scale_factor,
            in_channels=in_channels,
            middle_channels=middle_channels,
            out_channels=out_channels,
            middle_proj_type="linear",
            out_proj_type="linear"
        )

        # Create downsampled input
        down_size = tuple(s // scale_factor for s in self.dense_size)
        x_down = create_test_fvdb_tensor(self.batch_size, in_channels, down_size, device)

        x_up_linear = upsampling_linear(x_down)

        dense_out_linear = x_up_linear.to_dense()
        self.assertEqual(dense_out_linear.shape,
                         (self.batch_size, out_channels, *self.dense_size))

    @parameterized.expand(DEVICE_PARAMS)
    def test_downsampling_upsampling_consistency(self, device):
        """Test that downsampling followed by upsampling preserves the original shape."""
        scale_factor = 2
        channels = 8  # Should be divisible by scale_factor**3

        downsampling = DownSamplingSpatial2ChannelFVDB(scale_factor=scale_factor)
        upsampling = UpSamplingChannel2SpatialFVDB(scale_factor=scale_factor)

        # Create input tensor
        x = create_test_fvdb_tensor(self.batch_size, channels, self.dense_size, device)

        # Downsample then upsample
        x_down = downsampling(x)
        x_up = upsampling(x_down)

        # Verify the output shape matches the original
        dense_original = x.to_dense()
        dense_up = x_up.to_dense()

        self.assertEqual(dense_up.shape, dense_original.shape)

    # -----------------------------
    # Gradient Tests
    # -----------------------------

    @parameterized.expand(DEVICE_PARAMS)
    def test_downsampling_gradients(self, device):
        """Ensure gradients propagate through the downsampling module."""
        scale_factor = 2
        in_channels = 4
        out_channels = 8

        downsampling = DownSamplingSpatial2ChannelFVDB(
            scale_factor=scale_factor, in_channels=in_channels, out_channels=out_channels
        )

        x = create_test_fvdb_tensor(self.batch_size, in_channels, self.dense_size, device)
        x.data.jdata.requires_grad = True

        out = downsampling(x)
        loss = out.data.jdata.sum()
        loss.backward()

        self.assertIsNotNone(x.data.jdata.grad)
        self.assertFalse(torch.isnan(x.data.jdata.grad).any())

    @parameterized.expand(DEVICE_PARAMS)
    def test_upsampling_gradients(self, device):
        """Ensure gradients propagate through the upsampling module."""
        scale_factor = 2
        in_channels = 8  # Should be divisible by scale_factor**3
        out_channels = 4

        upsampling = UpSamplingChannel2SpatialFVDB(
            scale_factor=scale_factor, in_channels=in_channels, out_channels=out_channels
        )

        down_size = tuple(s // scale_factor for s in self.dense_size)
        x_down = create_test_fvdb_tensor(self.batch_size, in_channels, down_size, device)
        x_down.data.jdata.requires_grad = True

        out = upsampling(x_down)
        loss = out.data.jdata.sum()
        loss.backward()

        self.assertIsNotNone(x_down.data.jdata.grad)
        self.assertFalse(torch.isnan(x_down.data.jdata.grad).any())

    @parameterized.expand(DEVICE_PARAMS)
    def test_downupsampling_data_consistency(self, device):
        """Test that downsampling followed by upsampling preserves the original shape."""
        scale_factor = 2

        downsampling = DownSamplingSpatial2ChannelFVDB(scale_factor=scale_factor)
        upsampling = UpSamplingChannel2SpatialFVDB(scale_factor=scale_factor)

        dense_data = torch.rand(
            self.batch_size, 8, *self.dense_size, device=device)
        dense_vdbx = fvdbnn.fVDBTensor_from_dense(dense_data)

        dense_data_flat = dense_data.reshape(self.batch_size, 8, -1)
        dense_data_flat = torch.stack(dense_vdbx.data.unbind(), dim=0)  # (B, N, C)
        self.assertTrue(
            torch.isclose(dense_data_flat.reshape(-1, 8), dense_vdbx.jdata).all())

        masked_data_list, mask_list = [], []
        for i in range(self.batch_size):
            mask_i = torch.randn(dense_data_flat.shape[1], device=device) >= 0.0
            # print(f"mask_i.sum()={mask_i.sum()}")
            masked_data_i = dense_data_flat[i, mask_i]
            # print(f"masked_data_i.shape={masked_data_i.shape}")
            masked_data_list.append(masked_data_i)  # [N, C]
            mask_flat = mask_i.reshape(-1)
            mask_list.append(mask_flat)  # [N]
        masked_data = torch.cat(masked_data_list, dim=0)  # (NN, C)
        mask_flat = torch.cat(mask_list, dim=0)  # [NN]
        mask_jagged = dense_vdbx.grid.jagged_like(mask_flat)

        vdbx_grid = dense_vdbx.grid
        vdbx_data = dense_vdbx.data
        masked_vdbx_data, masked_vdbx_grid = (
            vdbx_grid.refine(1, vdbx_data, mask=mask_jagged))
        masked_vdbx = fvdbnn.fVDBTensor(masked_vdbx_grid, masked_vdbx_data)
        # print(f"masked_vdbx.rshape={masked_vdbx.rshape}")

        self.assertTrue(torch.isclose(masked_vdbx.jdata, masked_data).all())

        downsampled_vdbx = downsampling.forward(dense_vdbx)
        upsampled_vdbx = upsampling.forward(downsampled_vdbx, fine_data=masked_vdbx)
        self.assertEqual(upsampled_vdbx.rshape, masked_vdbx.rshape)
        self.assertTrue(torch.isclose(upsampled_vdbx.jdata, masked_vdbx.jdata).all())


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
