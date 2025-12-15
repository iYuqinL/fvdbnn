# -*- coding:utf-8 -*-
###
# File: conv3d.py
# Created Date: Sunday, November 23rd 2025, 3:27:17 pm
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
from torch import nn
import fvdb

from .vdbtensor import fVDBTensor


__all__ = ["SparseConv3dFVDB", "SparseConvTranspose3dFVDB"]


class SparseConv3dFVDB(fvdb.nn.SparseConv3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        bias: bool = True,
        disable_conv_output_padding: bool = True,
    ):
        super(SparseConv3dFVDB, self).__init__(
            in_channels, out_channels, kernel_size, stride, bias)
        self.disable_conv_output_padding = disable_conv_output_padding

        self.weight.data = self.weight.data.contiguous()
        if self.bias is not None:
            self.bias.data = self.bias.data.contiguous()

    def forward(self, input: fVDBTensor,
                plan: fvdb.ConvolutionPlan = None) -> fVDBTensor:
        """
        Args
        ----
        input (fVDBTensor): Input fvdb sparse tensor.

        plan (fvdb.ConvolutionPlan): Convolution plan. Defaults to None.

        Returns
        -------
        output (fVDBTensor): Output fvdb sparse tensor.
        """
        if plan is None:
            conv_key = f"conv3d_ks{self.kernel_size}_stride{self.stride}"
            plan = input.get_spatial_cache(conv_key)
            # print(f"\nconv3d_ks{self.kernel_size}_stride{self.stride} plan: {plan}")
            if plan is None:
                if self.disable_conv_output_padding:
                    target_grid = input.grid
                else:
                    target_grid = input.grid.conv_grid(self.kernel_size, self.stride)
                plan = fvdb.ConvolutionPlan.from_grid_batch(
                    self.kernel_size, self.stride, input.grid, target_grid=target_grid)
                input.register_spatial_cache(conv_key, plan)

        if not plan.valid_usage(
                self.in_channels, self.out_channels,
                self.kernel_size, self.stride, transposed=False):
            raise ValueError(
                f"Convolution plan {conv_key} is not valid for "
                f"input with shape {input.rshape} and "
                f"kernel size {self.kernel_size} and stride {self.stride}.")

        intype = input.data.dtype
        weight = self.weight
        if intype != weight.dtype:
            weight = weight.to(intype)
        bias = self.bias
        if bias is not None and intype != bias.dtype:
            bias = bias.to(intype)

        outgrid = plan.target_grid_batch
        outdata = plan.execute(input.data, weight)
        if self.bias is not None:
            outdata.jdata = outdata.jdata + bias

        output = fVDBTensor(outgrid, outdata, spatial_cache=input.spatial_cache)
        return output


class SparseConvTranspose3dFVDB(fvdb.nn.SparseConvTranspose3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        bias: bool = True,
    ):
        super(SparseConvTranspose3dFVDB, self).__init__(
            in_channels, out_channels, kernel_size, stride, bias)

    def forward(self, input: fVDBTensor,
                plan: fvdb.ConvolutionPlan = None) -> fVDBTensor:
        """
        Args
        ----
        input (fVDBTensor): Input fvdb sparse tensor.

        plan (fvdb.ConvolutionPlan): Convolution plan. Defaults to None.

        Returns
        -------
        output (fVDBTensor): Output fvdb sparse tensor.
        """
        if plan is None:
            conv_key = f"conv3d_transpose3d_ks{self.kernel_size}_stride{self.stride}"
            plan = input.get_spatial_cache(conv_key)
            # print(f"\nconv3d_transpose3d_ks{self.kernel_size}"
            #       f"_stride{self.stride} plan: {plan}")
            if plan is None:
                assert torch.all(self.stride == 1), "Only stride 1 is supported for now."
                target_grid = input.grid
                plan = fvdb.ConvolutionPlan.from_grid_batch_transposed(
                    self.kernel_size, self.stride, input.grid, target_grid)
                input.register_spatial_cache(conv_key, plan)

        if not plan.valid_usage(
                self.in_channels, self.out_channels,
                self.kernel_size, self.stride, transposed=True):
            raise ValueError(
                f"Convolution plan {conv_key} is not valid for "
                f"input with shape {input.rshape} and "
                f"kernel size {self.kernel_size} and stride {self.stride}.")

        intype = input.data.dtype
        
        # --- FIX START ---
        # The weight is stored as (C_out, C_in, K, K, K) in _SparseConv3dBase.
        # For transposed convolution, the fvdb backend (sparse_transpose_conv_3d) 
        # expects the weight to be (C_in, C_out, K, K, K). We permute the first two dims.
        weight = self.weight
        # Handle both KxKxK (5D) and 1x1x1 (2D) kernels
        if weight.dim() == 5:
            # (C_out, C_in, K, K, K) -> (C_in, C_out, K, K, K)
            weight = weight.permute(1, 0, 2, 3, 4)
        elif weight.dim() == 2:
            # (C_out, C_in) -> (C_in, C_out)
            weight = weight.transpose(0, 1)
        # --- FIX END ---

        if intype != weight.dtype:
            weight = weight.to(intype)
        bias = self.bias
        if bias is not None and intype != bias.dtype:
            bias = bias.to(intype)

        outgrid = plan.target_grid_batch
        outdata = plan.execute(input.data, weight)
        if self.bias is not None:
            outdata.jdata = outdata.jdata + bias

        output = fVDBTensor(outgrid, outdata, spatial_cache=input.spatial_cache)
        return output
