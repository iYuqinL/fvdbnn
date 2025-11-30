# -*- coding:utf-8 -*-
###
# File: rope.py
# Created Date: Sunday, November 16th 2025, 12:16:31 am
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
from typing import Optional, Tuple
import torch
import torch.nn as nn


__all__ = ['RotaryPositionEncoding']


class RotaryPositionEncoding(nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation, generalized for multi-dimensional 
    spatial coordinates (e.g., 3D positions in a point cloud).

    The feature dimension 'feat_edim' (D) is split into 'spatial_dim' (C) blocks. 
    Each block (of size D/C) is independently rotated using phases derived from the 
    corresponding spatial coordinate.

    Args
    ----
    feat_edim (int): token feature dim.

    spatial_dim (int): spatial dimensions.
    """

    def __init__(self, feat_edim: int, spatial_dim: int = 3):
        super(RotaryPositionEncoding, self).__init__()

        assert feat_edim % 2 == 0, "Hidden size must be divisible by 2"
        self.feat_edim = feat_edim
        self.spatial_dim = spatial_dim
        self.freq_dim = feat_edim // 2 // spatial_dim
        # Calculate the base frequencies for all spatial dims
        freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        freqs = 1.0 / (10000 ** freqs)
        self.register_buffer("freqs", freqs)

    def _get_phases(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Calculates the complex rotation phases.

        Args
        ----
        indices (tensor): Flattened tensor of spatial positions, shape (X, ). 
        X = prod(batch_dims) * N * C.

        Returns
        -------
        phases (tensor): Complex rotation phases. Shape (X, F) where each element is 
        a complex number exp(i * phase), F is self.freq_dim.
        """
        freqs = self.freqs.type(torch.float32)    # pylint: disable=E1101
        # indices: (X, ) self.freqs: (F) --> (X, F)
        phases = torch.outer(indices, freqs)
        # phases: (X, F) complex phases (polar(1, phase) = cos(phase) + i*sin(phase))
        phases = torch.polar(torch.ones_like(phases), phases)
        return phases

    def _rotary_embedding(self, x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        """
        Applies the Rotary Position Embedding to the input tokens.

        Args
        ----
        x (tensor): tokens, (..., D).

        phases (tensor): Complex rotation phases, shape (..., D//2).

        Returns
        -------
        x_embed (tensor): Rotated tokens, shape (..., D).
        """
        # x_complex: (..., D//2) 
        # Converts (..., D//2, 2) real/imag parts to (..., D//2) complex numbers
        x_complex = torch.view_as_complex(
            x.float().reshape(*x.shape[:-1], x.shape[-1]//2, 2))
        # Hadamard product (complex multiplication) for rotation
        x_rotated = x_complex * phases
        # x_embed: (..., D//2, 2); Converts back to (real, imag) real tensor
        x_embed = torch.view_as_real(x_rotated)
        # x_embed: (..., D); Reshapes to original feature dimension D
        x_embed = x_embed.reshape(*x_rotated.shape[:-1], x.shape[-1]).to(x.dtype)
        return x_embed

    def forward(
        self,
        tokens: torch.Tensor,
        k_tokens: torch.Tensor = None,
        indices: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies Rotary Position Embedding to query (tokens)
        and optionally key (k_tokens).

        Args
        ----
        tokens (Tensor): tensor of tokens,
        (..., D) or  (..., N, D) when indices is None.

        k_tokens (Tensor): tensor of keys,
        (..., D) or  (..., N, D) when indices is None.

        indices (Tensor): (..., C) or (..., N, C),
        tensor of spatial positions, last dim if for spatial dimension.

        Returns
        -------
        tokens_embed (tensor): (..., D) or (..., N, D)

        Example
        -------
        >>> # Standard 1D RoPE (like for NLP/sequences)
        >>> rope_1d = RotaryPositionEncoding(feat_edim=64, spatial_dim=1)
        >>> # tokens shape: (batch_size, seq_len, feat_dim)
        >>> q = torch.randn(1, 10, 64) 
        >>> q_embed = rope_1d(q)
        >>> # For 3D RoPE (e.g., for point clouds)
        >>> rope_3d = RotaryPositionEncoding(feat_edim=192, spatial_dim=3)
        >>> # feat_edim=192 is split into 3 blocks of 64 features (32 complex pairs)
        >>> # Indices shape: (batch_size, num_points, 3)
        >>> indices_3d = torch.randint(0, 100, (1, 100, 3)).float() 
        >>> q_3d = torch.randn(1, 100, 192)
        >>> k_3d = torch.randn(1, 100, 192)
        >>> q_embed_3d, k_embed_3d = rope_3d(q_3d, k_3d, indices_3d)
        """
        intype = tokens.dtype
        tokens = tokens.type(torch.float32)

        if indices is not None:
            assert indices.shape[-1] == self.spatial_dim

        if indices is None:
            # If no indices provided, assume 1D sequence and generate range (0, ..., N-1)
            # N is the sequence length (tokens.shape[-2])
            indices = torch.arange(
                tokens.shape[-2], dtype=torch.float32, device=tokens.device)
            if len(tokens.shape) > 2:
                indices = indices.unsqueeze(0).expand(tokens.shape[:-2] + (-1,))
            # Add spatial_dim=1 for consistency,
            # if tokens is (..., N, D) -> indices is (..., N, 1)
            indices = indices.unsqueeze(-1)
        indices = indices.type(torch.float32)
        
        # X = product of leading dimensions (including N) * C
        # Flattens (..., N, C) to (X, C) -> then to (X, ) 
        # where X is total number of coordinate values
        # phases: (X, F), X is total num of single coordinate values across all dims.
        phases = self._get_phases(indices.reshape(-1))
        # (..., N, C*F), F = feat_edim//2//spatial_dim
        # C*F = spatial_dim * (feat_edim//2//spatial_dim) = feat_edim//2 = D//2
        phases = phases.reshape(*indices.shape[:-1], indices.shape[-1]*self.freq_dim)
        if phases.shape[-1] < self.feat_edim // 2:
            padshape = list(phases.shape[:-1]) + [self.feat_edim//2-phases.shape[-1]]
            pads = torch.polar(
                torch.ones(padshape, dtype=torch.float32, device=phases.device),
                torch.zeros(padshape, dtype=torch.float32, device=phases.device))

            phases = torch.cat([phases, pads], dim=-1)

        tokens_embed = self._rotary_embedding(tokens, phases)
        tokens_embed = tokens_embed.type(intype)
        if k_tokens is not None:
            k_tokens_embed = self._rotary_embedding(k_tokens, phases)
            k_tokens_embed = k_tokens_embed.type(intype)
            return tokens_embed, k_tokens_embed

        return tokens_embed
