# -*- coding:utf-8 -*-
###
# File: pooling.py
# Created Date: Thursday, January 1st 1970, 8:00:00 am
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
from typing import Union
import torch
import torch.nn as nn
import fvdb

from .vdbtensor import fVDBTensor


__all__ = ["MaxPoolFVDB","AvgPoolFVDB", "UpsamplingNearestFVDB",
           "convert_lowres_chdata_to_highres_data"]


class MaxPoolFVDB(fvdb.nn.MaxPool):
    def forward(
        self,
        input: fVDBTensor,
        coarse_data: Union[fVDBTensor, fvdb.GridBatch] = None,
    ) -> fVDBTensor:
        spatial_cache = input.spatial_cache
        if isinstance(coarse_data, fVDBTensor):
            coarse_grid = coarse_data.grid
        elif isinstance(coarse_data, fvdb.GridBatch):
            coarse_grid = coarse_data
        else:
            coarse_grid = None
        
        new_data, new_grid = super().forward(input.data, input.grid, coarse_grid)
        
        new_tensor = fVDBTensor(new_grid, new_data, spatial_cache)
        return new_tensor



class AvgPoolFVDB(fvdb.nn.AvgPool):
    def forward(
        self,
        input: fVDBTensor,
        coarse_data: Union[fVDBTensor, fvdb.GridBatch] = None,
    ) -> fVDBTensor:
        spatial_cache = input.spatial_cache
        if isinstance(coarse_data, fVDBTensor):
            coarse_grid = coarse_data.grid
        elif isinstance(coarse_data, fvdb.GridBatch):
            coarse_grid = coarse_data
        else:
            coarse_grid = None

        new_data, new_grid = super().forward(input.data, input.grid, coarse_grid)

        new_tensor = fVDBTensor(new_grid, new_data, spatial_cache)
        return new_tensor


class UpsamplingNearestFVDB(fvdb.nn.UpsamplingNearest):
    def forward(
        self,
        input: fVDBTensor,
        mask: fvdb.JaggedTensor = None,
        fine_data: Union[fVDBTensor, fvdb.GridBatch] = None,
        fine_mask: Union[fVDBTensor, fvdb.JaggedTensor] = None,
    ) -> fVDBTensor:
        spatial_cache = input.spatial_cache
        if isinstance(fine_data, fVDBTensor):
            fine_grid = fine_data.grid
        elif isinstance(fine_data, fvdb.GridBatch):
            fine_grid = fine_data
        else:
            assert fine_data is None
            fine_grid = None

        if fine_grid is None and fine_mask is not None:
            if isinstance(fine_mask, fVDBTensor):
                fine_mask = fine_mask.data
            assert isinstance(fine_mask, fvdb.JaggedTensor), (
                f"fine_mask must be a jagged tensor, got {type(fine_mask)}")

            up_grid = input.grid.refined_grid(
                subdiv_factor=self.scale_factor, mask=mask)
            fine_grid = up_grid.refined_grid(1, mask=fine_mask)

        new_data, new_grid = super().forward(
            input.data, input.grid, mask=mask, fine_grid=fine_grid)
        new_tensor = fVDBTensor(new_grid, new_data, spatial_cache)
        return new_tensor


def convert_lowres_chdata_to_highres_data(
    lowres_grid: fvdb.GridBatch,
    lowres_data: fvdb.JaggedTensor,
    scale_factor: int = 2,
) -> tuple[fvdb.GridBatch, fvdb.JaggedTensor]:
    """
    Convert downsampled channel data to upsampled channel data.
    
    Args:
        lowres_grid (fvdb.GridBatch): Low resolution grid batch.
        lowres_data (fvdb.JaggedTensor): Low resolution data.
            eshape[0] (i.e rshape[1]) must equal to (scale_factor ** 3).
        scale_factor (int, optional): Upsampling scale factor. Defaults to 2.

    Returns:
        fvdb.JaggedTensor: Upsampled channel data.
    """
    assert scale_factor > 1, (
        f"scale_factor must be greater than 1, got {scale_factor}")
    assert isinstance(scale_factor, int), (
        f"scale_factor must be an integer, got {type(scale_factor)}")

    assert lowres_grid.total_voxels == lowres_data.rshape[0], (
        f"lowres_grid must have the same total voxels as lowres_data, "
        f"but got {lowres_grid.total_voxels} and {lowres_data.rshape[0]}.")

    up_grid = lowres_grid.refined_grid(subdiv_factor=scale_factor)

    up_ijk = up_grid.ijk.float()
    down_ijk = (up_ijk / scale_factor).floor()
    local_ijk = up_ijk - (down_ijk * scale_factor)  # range [0, scale_factor-1]

    down_index = lowres_grid.ijk_to_index(down_ijk.int(), cumulative=True).jdata
    assert not (down_index == -1).any(), f"Downsampled index contains -1"

    num_upvoxs = up_grid.total_voxels
    # (num_upvoxs, scale_factor ** 3, *)
    up_data_from_lowres = lowres_data.jdata[down_index]
    if up_data_from_lowres.ndim == 2:
        assert up_data_from_lowres.shape[1] % (scale_factor ** 3) == 0, (
            f"up_data_from_lowres must have shape[1] divisible by "
            f"{scale_factor ** 3}, but got {up_data_from_lowres.shape[1]}.")
        up_data_from_lowres = (
            up_data_from_lowres.reshape(num_upvoxs, scale_factor ** 3, -1))

    assert (up_data_from_lowres.ndim >= 3 and
            (up_data_from_lowres.shape[1] == (scale_factor ** 3))), (
        f"up_data_from_lowres must be a jagged tensor with shape "
        f"(num_upvoxs, {scale_factor ** 3}, *), "
        f"but got shape {up_data_from_lowres.shape}.")

    local_ijk = local_ijk.jdata.long()  # (num_upvoxs, 3)
    assert (local_ijk >= 0).all() and (local_ijk < scale_factor).all(), (
        f"local_ijk must be in range [0, {scale_factor - 1}], "
        f"but got {local_ijk.min()} and {local_ijk.max()}.")

    local_index = (local_ijk[..., 0] * (scale_factor ** 2) +
                   local_ijk[..., 1] * (scale_factor ** 1) +
                   local_ijk[..., 2] * (scale_factor ** 0)).long()

    expanded_index = local_index.view(-1, 1, *(1,) * (up_data_from_lowres.ndim - 2))
    expanded_index = expanded_index.expand(-1, -1, *(up_data_from_lowres.shape[2:]))

    selected_data = torch.gather(up_data_from_lowres, dim=1, index=expanded_index)
    up_data = up_grid.jagged_like(selected_data.squeeze(dim=1))

    return up_grid, up_data
