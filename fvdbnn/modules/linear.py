# -*- coding:utf-8 -*-
###
# File: linear.py
# Created Date: Saturday, November 22nd 2025, 12:35:01 am
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
import torch.nn.functional as F

from .vdbtensor import fVDBTensor
from .utils import ElementwiseMixin


__all__ = ["LinearFVDB", "Cast2IntypeLinear"]


class LinearFVDB(ElementwiseMixin, nn.Linear):
    r"""
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.
    """
    def forward(self, input: fVDBTensor) -> fVDBTensor:
        assert isinstance(input, fVDBTensor), "Input should have type fVDBTensor"
        intype = input.data.jdata.dtype
        weight = (self.weight.to(dtype=intype)
                  if intype != self.weight.dtype else self.weight)
        bias = None
        if hasattr(self, "bias") and self.bias is not None:
            bias = (self.bias.to(dtype=intype)
                    if self.bias.dtype != intype else self.bias)
        # print(f"fvnn.Linear, weight.dtype={weight.dtype}, input.dtype={intype}")
        res = F.linear(input.data.jdata, weight, bias)  # pylint: disable=not-callable

        spatial_cache = input.spatial_cache
        return fVDBTensor(input.grid, input.data.jagged_like(res), spatial_cache)


class Cast2IntypeLinear(nn.Linear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        intype = x.dtype
        weight = (self.weight.to(intype)
                  if self.weight.dtype != intype else self.weight)
        bias = None
        if hasattr(self, 'bias') and self.bias is not None:
            bias = (self.bias.to(intype)
                    if self.bias.dtype != intype else self.bias)
        ret = F.linear(x, weight, bias)  # pylint: disable=not-callable
        return ret

