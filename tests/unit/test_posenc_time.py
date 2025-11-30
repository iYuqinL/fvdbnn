# -*- coding:utf-8 -*-
###
# File: test_posenc_time.py
# Created Date: Sunday, November 30th 2025, 5:17:47 pm
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
import torch
import torch.nn as nn
import unittest
from parameterized import parameterized

from fvdbnn.utils.utils import get_gpu_compute_capacity
from fvdbnn.posenc import TimestepEmbedder

# --- Unit Test Class with Parameterization ---

# Define the list of devices: CPU is always included, 'cuda' is included if available.
TEST_DEVICES = ['cpu']

# Helper function placeholder for environment check
if torch.cuda.is_available():
    # Attempt to get compute capability if the function is available
    try:
        # Assuming get_gpu_compute_capacity returns (major, minor)
        major_cc, minor_cc = get_gpu_compute_capacity(torch.device('cuda'))
        if major_cc >= 7:
            TEST_DEVICES.append('cuda')
    except (ImportError, NameError):
        # Fallback if get_gpu_compute_capacity is not easily accessible
        TEST_DEVICES.append('cuda')


# Prepare parameters for parameterized tests
DEVICE_PARAMS = [(device,) for device in TEST_DEVICES]


class TestTimestepEmbedder(unittest.TestCase):

    def setUp(self):
        # Common setup before each test
        self.batch_size = 4
        self.hidden_size = 64
        self.freq_size = 32

    @parameterized.expand(DEVICE_PARAMS)
    def test_shape_and_device(self, device):
        """Test output shape and device correctness."""

        model = TimestepEmbedder(
            hidden_size=self.hidden_size,
            frequency_embedding_size=self.freq_size
        ).to(device)

        # Simulate Flow Matching input t ~ [0, 1]
        t = torch.rand(self.batch_size, device=device)

        output = model(t)

        # Check 1: Output shape
        expected_shape = (self.batch_size, self.hidden_size)
        self.assertEqual(output.shape, expected_shape,
                         f"Failed on {device}: shape mismatch")

        # Check 2: Output device
        self.assertEqual(output.device.type, torch.device(device).type,
                         f"Failed on {device}: output device incorrect")

    def test_odd_frequency_dimension(self):
        """Test robustness when frequency_embedding_size is odd."""
        odd_freq_size = 33
        model = TimestepEmbedder(self.hidden_size, odd_freq_size)
        t = torch.rand(self.batch_size)

        # Verify no error and correct shape
        output = model(t)
        self.assertEqual(output.shape, (self.batch_size, self.hidden_size))

    @parameterized.expand(DEVICE_PARAMS)
    def test_gradient_flow(self, device):
        """Test backward pass to ensure MLP parameters can be updated."""

        model = TimestepEmbedder(self.hidden_size, self.freq_size).to(device)
        t = torch.rand(self.batch_size, device=device, requires_grad=False)

        output = model(t)
        loss = output.mean()
        loss.backward()

        # Check gradient of the first MLP layer's weight
        first_layer_grad = model.mlp[0].weight.grad
        self.assertIsNotNone(first_layer_grad, "Gradient should not be None")

        # Ensure gradient is not all zeros (implies a learning signal)
        self.assertNotEqual(
            first_layer_grad.sum().item(), 0.0, "Gradient should not be all zero")

    def test_extreme_values(self):
        """Test boundary cases like t=0 and t=1."""
        model = TimestepEmbedder(self.hidden_size, self.freq_size)
        t = torch.tensor([0.0, 1.0, 0.5])  # Common boundaries
        output = model(t)

        # Check for NaN or Inf
        self.assertFalse(torch.isnan(output).any(), "Output contains NaN")
        self.assertFalse(torch.isinf(output).any(), "Output contains Inf")


# --- Execution ---
if __name__ == '__main__':
    # Use this pattern to run tests correctly with parameterized and unittest.main()
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
