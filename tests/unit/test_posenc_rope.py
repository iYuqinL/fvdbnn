# -*- coding:utf-8 -*-
###
# File: test_rope.py
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
import torch.nn.functional as F
from fvdbnn.posenc import RotaryPositionEncoding


class TestRotaryPositionEncoding(unittest.TestCase):

    def setUp(self):
        # Set seed for reproducibility
        torch.manual_seed(42)

    def test_initialization_assertions(self):
        """
        Test valid and invalid initialization parameters.
        """
        # Valid: 64 is divisible by 2 and 2*1
        _ = RotaryPositionEncoding(feat_edim=64, spatial_dim=1)

        # Valid: 192 is divisible by 2*3=6
        _ = RotaryPositionEncoding(feat_edim=192, spatial_dim=3)

        # Invalid: Odd feature dimension
        with self.assertRaises(AssertionError):
            RotaryPositionEncoding(feat_edim=63, spatial_dim=1)

    def test_shape_consistency_1d(self):
        """
        Test output shapes for standard sequence (1D) usage.
        """
        batch_size, seq_len, dim = 2, 10, 32
        rope = RotaryPositionEncoding(feat_edim=dim, spatial_dim=1)

        q = torch.randn(batch_size, seq_len, dim)
        k = torch.randn(batch_size, seq_len, dim)

        # Test without passing indices (auto-generation)
        q_out, k_out = rope(q, k)

        self.assertEqual(q_out.shape, q.shape)
        self.assertEqual(k_out.shape, k.shape)

    def test_shape_consistency_3d(self):
        """
        Test output shapes for multi-dimensional (3D) usage.
        """
        batch_size, num_points, dim = 2, 100, 96  # 96 is divisible by 2*3=6
        spatial_dim = 3
        rope = RotaryPositionEncoding(feat_edim=dim, spatial_dim=spatial_dim)

        q = torch.randn(batch_size, num_points, dim)
        k = torch.randn(batch_size, num_points, dim)
        indices = torch.rand(batch_size, num_points, spatial_dim)

        q_out, k_out = rope(q, k, indices=indices)

        self.assertEqual(q_out.shape, q.shape)
        self.assertEqual(k_out.shape, k.shape)

    def test_zero_position_identity(self):
        """
        Test that at position 0, no rotation occurs (output == input).
        cos(0) = 1, sin(0) = 0.
        """
        dim = 32
        rope = RotaryPositionEncoding(feat_edim=dim, spatial_dim=1)

        # Create a tensor where the first token is at index 0
        x = torch.randn(1, 5, dim)

        # Manually create indices starting at 0
        indices = torch.arange(5).float().unsqueeze(0).unsqueeze(-1)  # (1, 5, 1)

        x_out = rope(x, indices=indices)

        # Check only the first token (index 0)
        # Verify x_out[0, 0] is approximately equal to x[0, 0]
        self.assertTrue(torch.allclose(x_out[0, 0], x[0, 0], atol=1e-5))

    def test_relative_position_invariance_1d(self):
        """
        Test the core property of RoPE:
        Dot product depends only on relative distance.
        (Rot(q, t) . Rot(k, t+delta)) should be constant for any t.
        """
        dim = 64
        rope = RotaryPositionEncoding(feat_edim=dim, spatial_dim=1)

        q_base = torch.randn(1, 1, dim)  # A single query vector
        k_base = torch.randn(1, 1, dim)  # A single key vector

        delta = 5  # Relative distance
        t1 = 0    # Absolute position 1
        t2 = 10   # Absolute position 2

        # Case 1: q at t1, k at t1 + delta
        idx_q1 = torch.tensor([[[t1]]]).float()
        idx_k1 = torch.tensor([[[t1 + delta]]]).float()

        q_rot1 = rope(q_base, indices=idx_q1)
        k_rot1 = rope(k_base, indices=idx_k1)
        score1 = torch.sum(q_rot1 * k_rot1)

        # Case 2: q at t2, k at t2 + delta
        idx_q2 = torch.tensor([[[t2]]]).float()
        idx_k2 = torch.tensor([[[t2 + delta]]]).float()

        q_rot2 = rope(q_base, indices=idx_q2)
        k_rot2 = rope(k_base, indices=idx_k2)
        score2 = torch.sum(q_rot2 * k_rot2)

        # The dot products should be effectively identical
        self.assertTrue(torch.allclose(score1, score2, atol=1e-5),
                        f"Scores mismatch: {score1.item()} vs {score2.item()}")

    def test_multidimensional_independence(self):
        """
        Test that in the 3D implementation, changing coordinate X does not affect 
        the embedding part corresponding to coordinate Y or Z.
        This validates the 'block splitting' logic.
        """
        dim = 60  # 60 / (2*3) = 10 frequencies per block
        spatial_dim = 3
        rope = RotaryPositionEncoding(feat_edim=dim, spatial_dim=spatial_dim)

        # Block size = dim / spatial_dim = 20
        block_size = dim // spatial_dim

        x = torch.randn(1, 1, dim)

        # Two indices: Only change the first dimension (X)
        # Index A: [0, 0, 0]
        idx_a = torch.tensor([[[0, 0, 0]]]).float()
        # Index B: [10, 0, 0] (Change X only)
        idx_b = torch.tensor([[[10, 0, 0]]]).float()

        out_a = rope(x, indices=idx_a)
        out_b = rope(x, indices=idx_b)

        # Split outputs into 3 blocks: X-block, Y-block, Z-block
        # X-block: features 0:20 -> Should change (because X coord changed)
        # Y-block: features 20:40 -> Should NOT change (because Y coord is same 0->0)
        # Z-block: features 40:60 -> Should NOT change (because Z coord is same 0->0)

        diff_x_block = (out_a[..., 0:block_size] -
                        out_b[..., 0:block_size]).abs().sum()
        diff_y_block = (out_a[..., block_size:2*block_size] -
                        out_b[..., block_size:2*block_size]).abs().sum()
        diff_z_block = (out_a[..., 2*block_size:] -
                        out_b[..., 2*block_size:]).abs().sum()

        self.assertGreater(diff_x_block.item(), 1e-4,
                           "X block should have rotated")
        self.assertTrue(torch.allclose(diff_y_block, torch.tensor(0.0), atol=1e-5),
                        "Y block should stay constant")
        self.assertTrue(torch.allclose(diff_z_block, torch.tensor(0.0), atol=1e-5),
                        "Z block should stay constant")

    def test_gradients(self):
        """
        Ensure gradients propagate through the module.
        """
        dim = 32
        rope = RotaryPositionEncoding(feat_edim=dim, spatial_dim=1)
        x = torch.randn(1, 10, dim, requires_grad=True)

        out = rope(x)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())


if __name__ == '__main__':
    # You need to define the class RotaryPositionEncoding before running this,
    # or import it. Here we assume the class structure is available.
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
