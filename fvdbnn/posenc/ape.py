# -*- coding:utf-8 -*-
###
# File: ape.py
# Created Date: Sunday, November 16th 2025, 12:15:02 am
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

__all__ = ['AbsolutePositionEmbedder']


class AbsolutePositionEmbedder(nn.Module):
    """
    Embeds spatial positions into vector representations using a sinusoidal approach.

    This implementation is a generalization of the standard sinusoidal positional 
    encoding for D-dim coordinates, supporting batched inputs of shape (..., N, D).
    """

    def __init__(self, channels: int, in_channels: int = 3):
        """
        Initializes the AbsolutePositionEmbedder.

        Args:
            channels (int): The output dimension (embedding size).
            in_channels (int): The dimension of the input coordinates (D).
        """
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels

        # Calculate the number of frequency terms per spatial coordinate value.
        # Total output dimension (channels) must be >= in_channels * 2 * freq_dim
        # Note: The implementation means 2 * freq_dim is the embedding size
        #       for a single coordinate value (x, or y, or z, etc.)
        self.freq_dim = channels // in_channels // 2
        if self.freq_dim == 0:
            # If the output dimension is not large enough,
            # we set a minimal embedding size.
            # Truncation/Padding logic in forward will handle dimension mismatch.
            # We enforce a minimal dimension of 1 for freq_dim to avoid division by zero
            # and to allow at least one sin/cos pair per coordinate value.
            self.freq_dim = 1
            print(f"[Warning] channels ({channels}) is too small "
                  f"for in_channels ({in_channels}). "
                  f"Minimum recommended channels for encoding is {in_channels * 2}. "
                  f"Using minimal freq_dim=1, resulting embedding size "
                  f"before padding/truncation: {in_channels * 2}.")

        # Create the frequency scaling factors (omegas)
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        # freqs = 1 / (10000^(2i / D_model))
        self.freqs = 1.0 / (10000 ** self.freqs)

    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def _sin_cos_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal position embeddings.

        Args:
            x (torch.Tensor): A 1-D Tensor of N indices (total of coordinate values).

        Returns:
            torch.Tensor: An (N, 2 * freq_dim) Tensor of positional embeddings.
        """
        self.freqs = self.freqs.to(x.device)
        # The core operation: x * freqs (N, 1) @ (1, freq_dim) -> (N, freq_dim)
        # torch.outer is equivalent to x[:, None] * self.freqs[None, :]
        out = torch.outer(x, self.freqs)
        out = torch.cat([torch.sin(out), torch.cos(out)], dim=-1)
        return out

    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generates absolute positional embeddings, supporting batched inputs.
        The last dimension D is assumed to be the spatial dimension (in_channels).

        Args:
            x (torch.Tensor): (..., N, D) tensor of spatial positions, D is in_channels.

        Returns:
            torch.Tensor: (..., N, channels) tensor of positional embeddings.
        """
        *batch_dims, N, D = x.shape
        # 1. Assertion and Device Move
        assert D == self.in_channels, (
            f"Input last dimension ({D}) must match number of "
            f"input channels ({self.in_channels})")
        original_shape_without_D = batch_dims + [N]

        # 2. Reshape Input: (..., N, D) -> (Total Elements, D)
        x_flat = x.reshape(-1, D).to(torch.float32)
        Total_Elements, _ = x_flat.shape

        # 3. Apply Embedding Logic (Identical to your original forward)
        # Flattens all coordinate values into a single stream
        embed_flat_coords = self._sin_cos_embedding(x_flat.reshape(-1))

        # Reshapes back to (Total Elements, D * (2 * freq_dim))
        embed_flat_positions = embed_flat_coords.reshape(Total_Elements, -1)

        # 4. Padding/Truncation
        current_dim = embed_flat_positions.shape[1]
        target_dim = self.channels
        if current_dim < target_dim:
            # Pad with zeros if the calculated dimension is less than the target channels
            pad_emb = torch.zeros(
                Total_Elements, target_dim - current_dim, device=x.device)
            embed_final = torch.cat([embed_flat_positions, pad_emb], dim=-1)
        elif current_dim > target_dim:
            # Truncate if the calculated dimension exceeds the target channels
            embed_final = embed_flat_positions[:, :target_dim]
        else:
            embed_final = embed_flat_positions

        # 5. Reshape Output: (Total Elements, channels) -> (..., N, channels)
        # The output shape is the original shape (..., N) + the new channel dimension
        output_shape = original_shape_without_D + [target_dim]

        return embed_final.reshape(output_shape)
