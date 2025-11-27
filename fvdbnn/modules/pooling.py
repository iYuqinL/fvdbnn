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
import torch.nn as nn
import fvdb

from .vdbtensor import fVDBTensor


__all__ = ["MaxPoolFVDB","AvgPoolFVDB", "UpsamplingNearestFVDB"]


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
    ) -> fVDBTensor:
        spatial_cache = input.spatial_cache
        if isinstance(fine_data, fVDBTensor):
            fine_grid = fine_data.grid
        elif isinstance(fine_data, fvdb.GridBatch):
            fine_grid = fine_data
        else:
            fine_grid = None

        new_data, new_grid = super().forward(
            input.data, input.grid, mask=mask, fine_grid=fine_grid)

        new_tensor = fVDBTensor(new_grid, new_data, spatial_cache)
        return new_tensor
