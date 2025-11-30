# -*- coding:utf-8 -*-
###
# File: test_posenc_ape.py
# Created Date: Sunday, November 30th 2025, 4:43:56 pm
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
from fvdbnn.posenc import AbsolutePositionEmbedder

# --- Unit Test Class with Parameterization ---
# Define the list of devices: CPU is always included, 'cuda' is included if available.
TEST_DEVICES = ['cpu']

if torch.cuda.is_available():
    major_cc, minor_cc = get_gpu_compute_capacity(torch.device('cuda'))
    if major_cc >= 7:
        TEST_DEVICES.append('cuda')

class TestAbsolutePositionEmbedder(unittest.TestCase):

    # Define the shape and configuration test cases.
    TEST_CONFIGS = [
        # (channels, in_channels, input_shape)
        (128, 3, (5, 2, 10, 3)), # Standard Multi-Batch (Padding)
        (8, 3, (10, 3)),        # Truncation Test (Required dim=12. Target=8)
        (32, 4, (2, 5, 4)),     # Exactly matching dimension
    ]

    # Use parameterized.expand to combine all configurations with all available devices.
    @parameterized.expand(
        [(c, i, s, d) for c, i, s in TEST_CONFIGS for d in TEST_DEVICES])
    def test_shape_batching_and_padding(
        self, channels, in_channels, input_shape, device):
        """
        Tests correctness of output shape across different input dimensions, 
        padding/truncation modes, and devices (CPU/CUDA).
        """
        
        target_device = torch.device(device)
        embedder = AbsolutePositionEmbedder(
            channels=channels, in_channels=in_channels).to(target_device)
        
        # Move input data to the target device
        x = torch.randn(input_shape, device=target_device)
        output = embedder(x)
        
        # Expected shape is original dimensions minus the last (D), 
        # plus the output channels
        expected_shape = input_shape[:-1] + (channels,)
        
        self.assertEqual(output.shape, expected_shape, 
                         f"Shape test failed for device {device}. "
                         f"Expected {expected_shape}, got {output.shape}")
        
        # Verify output device
        self.assertEqual(output.device.type, target_device.type, 
                         f"Output device mismatch. "
                         f"Expected {target_device.type}, got {output.device.type}")

    # For simple tests that do not require parameterization, keep them separate.
    def test_zero_position_encoding(self):
        """Verifies the output values when the input coordinates are zero."""
        channels, in_channels = 10, 2
        embedder = AbsolutePositionEmbedder(channels=channels, in_channels=in_channels)
        x = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        output = embedder(x) # Shape (1, 10)

        # In _sin_cos_embedding, the order is 
        # [sin(w0), sin(w1), ..., cos(w0), cos(w1), ...]
        # For input 0.0, sin terms are 0, cos terms are 1. freq_dim=2.
        # Single coord embed: [0, 0, 1, 1]
        # Since input [0.0, 0.0] is flattened to [0.0, 0.0], the final 8-dim output is:
        # [0, 0, 1, 1] (from 1st 0.0) + [0, 0, 1, 1] (from 2nd 0.0)
        expected_non_padded = torch.tensor(
            [0., 0., 1., 1., 0., 0., 1., 1.], dtype=torch.float32)

        torch.testing.assert_close(
            output[0, :8], expected_non_padded, atol=1e-6, rtol=1e-6)
        self.assertTrue(
            torch.all(output[0, 8:] == 0.0), "Padded dimensions should be zero.")

    def test_in_channels_assertion(self):
        """Tests the assertion for mismatched spatial dimensions (D vs in_channels)."""
        embedder = AbsolutePositionEmbedder(channels=32, in_channels=3)
        # Input has 2 spatial dimensions, but embedder expects 3
        x = torch.randn(10, 2)
        with self.assertRaisesRegex(
            AssertionError, 
            "Input last dimension \\(2\\) must match number of input channels \\(3\\)"):
            embedder(x)
            
    # We can add a simple determinism test as well
    def test_determinism(self):
        """Ensures that identical inputs result in identical outputs."""
        embedder = AbsolutePositionEmbedder(channels=32, in_channels=1) 
        x = torch.tensor([[5.0], [5.0]], dtype=torch.float32)
        
        out_a = embedder(x)
        out_b = embedder(x)
        
        self.assertTrue(torch.equal(out_a, out_b), 
                        "Embedder output must be deterministic.")


# --- Execution ---
if __name__ == '__main__':
    # Use this pattern to run tests correctly with parameterized and unittest.main()
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
