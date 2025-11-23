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
        freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        freqs = 1.0 / (10000 ** freqs)
        self.register_buffer("freqs", freqs)

    def _get_phases(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        indices (tensor): tensor of spatial positions, has flatten to (X, ).

        Returns
        -------
        phases (tensor): (X, F)
        """
        freqs = self.freqs.type(torch.float32)    # pylint: disable=E1101
        # indices: (X, ) self.freqs: (F) --> (X, F)
        phases = torch.outer(indices, freqs)
        phases = torch.polar(torch.ones_like(phases), phases)
        return phases

    def _rotary_embedding(self, x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        x (tensor): tokens, (..., D).

        phases (tensor): (..., D//2)

        Returns
        -------
        x_embed (tensor): (..., D)
        """
        # complex: (..., D//2)
        x_complex = torch.view_as_complex(
            x.float().reshape(*x.shape[:-1], x.shape[-1]//2, 2))
        x_rotated = x_complex * phases
        x_embed = torch.view_as_real(x_rotated)
        x_embed = x_embed.reshape(*x_rotated.shape[:-1], x.shape[-1]).to(x.dtype)
        return x_embed

    def forward(self, tokens: torch.Tensor,
                k_tokens: torch.Tensor = None,
                indices: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args
        ----
        tokens (Tensor): tensor of tokens,
        (..., D) or  (..., N, D) when indices is None.

        indices (Tensor): (..., C) or (..., N, C),
        tensor of spatial positions, last dim if for spatial dimension.

        Returns
        -------
        tokens_embed (tensor): (..., D) or (..., N, D)
        """
        intype = tokens.dtype
        tokens = tokens.type(torch.float32)

        if indices is not None:
            assert indices.shape[-1] == self.spatial_dim

        if indices is None:
            indices = torch.arange(
                tokens.shape[-2], dtype=torch.float32, device=tokens.device)
            if len(tokens.shape) > 2:
                indices = indices.unsqueeze(0).expand(tokens.shape[:-2] + (-1,))
        indices = indices.type(torch.float32)

        # (X, F), X = prod(...) * N * C
        phases = self._get_phases(indices.reshape(-1))
        # (..., N, C*F), F = feat_edim//2//spatial_dim
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
