# -*- coding:utf-8 -*-
###
# File: spatial2ch.py
# Created Date: Monday, December 22nd 2025, 10:56:59 pm
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
from typing import Optional, Callable
import torch
import torch.nn as nn

import fvdb

from . import fVDBTensor, LinearFVDB


class DownSamplingSpatial2ChannelFVDB(nn.Module):
    def __init__(
        self,
        scale_factor: int,
        in_channels: int = None,
        middle_channels: int = None,
        out_channels: int = None,
    ):
        super(DownSamplingSpatial2ChannelFVDB, self).__init__()
        self.scale_factor = scale_factor
        assert isinstance(scale_factor, int) and scale_factor > 1

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.middle_channels = middle_channels

        if self.in_channels is not None:
            if self.out_channels is None:
                self.out_channels = self.in_channels
            cur_inchs = self.in_channels
            if self.middle_channels is not None:
                self.middle_proj = LinearFVDB(cur_inchs, middle_channels, bias=False)
                cur_inchs = middle_channels

            self.out_proj = LinearFVDB(cur_inchs, out_channels, bias=False)

    def forward(self, x: fVDBTensor):
        """
        """
        if self.in_channels is not None and self.middle_channels is not None:
            x: fVDBTensor = self.middle_proj(x)

        in_grid = x.grid
        in_data = x.data.jdata  # (N_up, C)
        device, dtype = in_data.device, in_data.dtype
        C = in_data.shape[1]
        S = self.scale_factor ** 3

        in_index = in_grid.ijk_to_index(in_grid.ijk.int(), cumulative=True).jdata
        assert torch.all(in_index[1:] > in_index[:-1]), (
            f"input grid must be sorted in canonical order got {in_index[:10]} ...")

        down_grid = in_grid.coarsened_grid(self.scale_factor)

        up_ijk = in_grid.ijk.float()
        down_ijk = (up_ijk / self.scale_factor).floor()
        # range [0, scale_factor-1]
        local_ijk = up_ijk.long() - (down_ijk.long() * self.scale_factor)
        # down_index: (num_upvoxs,)
        down_index = down_grid.ijk_to_index(down_ijk.int(), cumulative=True).jdata
        assert not (down_index == -1).any(), f"Downsampled index contains -1"

        local_ijk = local_ijk.jdata.long()  # (num_upvoxs, 3)
        assert (local_ijk >= 0).all() and (local_ijk < self.scale_factor).all(), (
            f"local_ijk must be in range [0, {self.scale_factor - 1}], "
            f"but got {local_ijk.min()} and {local_ijk.max()}.")
        # local_index: (num_upvoxs,) range [0, scale_factor**3 - 1]
        local_index = (local_ijk[..., 0] * (self.scale_factor ** 2) +
                       local_ijk[..., 1] * (self.scale_factor ** 1) +
                       local_ijk[..., 2] * (self.scale_factor ** 0)).long()

        assert len(x.data.rshape) == 2, (
            f"x.data must have 2 dimensions, got {len(x.data.rshape)}")

        # flat target index = down_index * S + local_index
        flat_idx = (down_index * S + local_index).long()   # shape (N_up,)
        assert flat_idx.unique().numel() == flat_idx.numel(), (
            f"duplicate mapping detected, got {flat_idx.unique().numel()} "
            f"unique indices out of {flat_idx.numel()} total indices.")
        flat_target = torch.zeros(
            (down_grid.total_voxels * S, C), dtype=dtype, device=device)
        # scatter_add_: index must have same shape as src when dim=0,
        # so expand flat_idx to (N_up, C)
        index_for_scatter = flat_idx.unsqueeze(1).expand(-1, C)    # (N_up, C)
        # x.data shape assumed (N_up, C)
        flat_target.scatter_add_(0, index_for_scatter, in_data)
        down_data_3d = flat_target.view(down_grid.total_voxels, S, C)  # [V_down, S, C]
        down_data = down_data_3d.view(down_grid.total_voxels, S * C)  # [V_down, S*C]

        down_x = fVDBTensor(
            down_grid, down_grid.jagged_like(down_data),
            spatial_cache=x.spatial_cache)

        if self.in_channels is not None:
            down_x: fVDBTensor = self.out_proj(down_x)

        return down_x


class UpSamplingChannel2SpatialFVDB(nn.Module):
    def __init__(
        self,
        scale_factor: int,
        in_channels: int = None,
        middle_channels: int = None,
        out_channels: int = None,
    ):
        super(UpSamplingChannel2SpatialFVDB, self).__init__()
        self.scale_factor = scale_factor
        assert isinstance(scale_factor, int) and scale_factor > 1

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.middle_channels = middle_channels

        if self.in_channels is not None:
            if self.out_channels is None:
                self.out_channels = self.in_channels
            cur_inchs = self.in_channels
            if self.middle_channels is not None:
                self.middle_proj = LinearFVDB(cur_inchs, middle_channels, bias=False)
                cur_inchs = middle_channels

            self.out_proj = LinearFVDB(cur_inchs, out_channels, bias=False)

    def forward(self, x: fVDBTensor):
        """
        """
        if self.in_channels is not None and self.middle_channels is not None:
            x: fVDBTensor = self.middle_proj(x)
        in_index = x.grid.ijk_to_index(x.grid.ijk.int(), cumulative=True).jdata
        assert torch.all(in_index[1:] > in_index[:-1]), (
            f"input grid must be monotonically increasing, got {in_index[:10]} ...")

        in_grid = x.grid
        in_data = x.data.jdata  # (N_down, total_C)
        # device, dtype = in_data.device, in_data.dtype
        S = self.scale_factor ** 3

        up_grid = in_grid.refined_grid(subdiv_factor=self.scale_factor)
        up_index = up_grid.ijk_to_index(up_grid.ijk, cumulative=True).jdata
        assert torch.all(up_index[1:] - up_index[:-1] > 0), (
            f"up_index must be monotonically increasing, but got {up_index[:10]}...")

        # Calculate indices for upsampling
        up_ijk = up_grid.ijk.float()
        down_ijk = (up_ijk / self.scale_factor).floor()
        # range [0, scale_factor-1]
        local_ijk = up_ijk.long() - (down_ijk.long() * self.scale_factor)

        down_index = in_grid.ijk_to_index(down_ijk.int(), cumulative=True).jdata
        assert not (down_index == -1).any(), f"Downsampled index contains -1"

        local_ijk = local_ijk.jdata.long()  # (num_upvoxs, 3)
        assert (local_ijk >= 0).all() and (local_ijk < self.scale_factor).all(), (
            f"local_ijk must be in range [0, {self.scale_factor - 1}], "
            f"but got {local_ijk.min()} and {local_ijk.max()}.")
        # local_index: (num_upvoxs,) range [0, scale_factor**3 - 1]
        local_index = (local_ijk[..., 0] * (self.scale_factor ** 2) +
                       local_ijk[..., 1] * (self.scale_factor ** 1) +
                       local_ijk[..., 2] * (self.scale_factor ** 0)).long()

        assert len(x.data.rshape) == 2, (
            f"x.data must have 2 dimensions, got {len(x.data.rshape)}")
        assert in_data.shape[1] % S == 0, (
            f"channels {in_data.shape[1]} must be divisible by scale_factor^3 ({S})")
        C = in_data.shape[1] // S

        flat_in_data = in_data.view(-1, C)  # (N_down * S, C)
        flat_idx = (down_index * S + local_index).long()
        assert flat_idx.unique().numel() == flat_idx.numel(), (
            f"duplicate mapping detected, got {flat_idx.unique().numel()} "
            f"unique indices out of {flat_idx.numel()} total indices.")
        # (N_up, C); N_up == N_down * S
        up_data = torch.index_select(flat_in_data, 0, flat_idx)  # (N_up, C)

        up_x = fVDBTensor(
            up_grid, up_grid.jagged_like(up_data),
            spatial_cache=x.spatial_cache)

        if self.in_channels is not None:
            up_x: fVDBTensor = self.out_proj(up_x)

        return up_x
