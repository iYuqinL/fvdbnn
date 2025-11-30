# -*- coding:utf-8 -*-
###
# File: time.py
# Created Date: Sunday, November 16th 2025, 12:18:16 am
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
import numpy as np
import torch
import torch.nn as nn

__all__ = ['TimestepEmbedder']


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.

    This module first creates a sinusoidal embedding (positional encoding) 
    for the timesteps and then projects it using an MLP.

    Args:
        hidden_size (int): The dimension of the final output embedding.
        frequency_embedding_size (int): The dimension of the sinusoidal 
            embedding before passing through the MLP. Defaults to 256.

    Example:
        >>> device = 'cuda' if torch.cuda.is_available() else 'cpu'
        >>> # Initialize embedder
        >>> embedder = TimestepEmbedder(hidden_size=512, frequency_embedding_size=256).to(device)
        >>> # Create dummy timesteps (e.g., batch size 4, values usually in [0, 1] for Flow Matching)
        >>> t = torch.rand(4, device=device)
        >>> # Forward pass
        >>> t_emb = embedder(t)
        >>> print(t_emb.shape)
        torch.Size([4, 512])
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super(TimestepEmbedder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor,
        dim: int,
        max_period: int = 10000
    ) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.

        Args:
            t (torch.Tensor): A 1-D Tensor of N indices (timesteps), 
                one per batch element. Shape: (N,). 
                usually in range [0, 1] for Flow Matching.
            dim (int): The dimension of the output embedding.
            max_period (int): Controls the minimum frequency of the embeddings. 
                Defaults to 10000.

        Returns:
            torch.Tensor: An (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) *
            torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) /
            half
        )
        # Create args: (N, 1) * (1, dim/2) -> (N, dim/2)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2 != 0:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

        return embedding

    def forward(self, t):
        """
        Args:
            t (torch.Tensor): Input timesteps tensor of shape (N,).

        Returns:
            torch.Tensor: Output embeddings of shape (N, hidden_size).
        """
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
