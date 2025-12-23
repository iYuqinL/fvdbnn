# -*- coding:utf-8 -*-
###
# File: conv3d.py
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

from typing import Optional, Callable, Literal
import torch
import torch.nn as nn

import fvdb
from torch.utils.checkpoint import checkpoint

from ..modules import fVDBTensor, SparseConv3dFVDB, LinearFVDB
from ..modules import MaxPoolFVDB, AvgPoolFVDB, DownSamplingSpatial2ChannelFVDB
from ..modules import LayerNorm32FVDB, GroupNorm32FVDB
from ..modules import SiLUFVDB


__all__ = ["BasicResBlock3DFVDB",
           "BottleneckResBlock3DFVDB",
           "SparseConvOutputHeadFVDB"]


class BasicResBlock3DFVDB(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        pooling: Literal["avg", "max", "s2c"] = "avg",
        act_layer=SiLUFVDB,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        midplanes: Optional[int] = None,
        disable_conv_output_padding: bool = True,
        checkpointing=False,
    ) -> None:
        super(BasicResBlock3DFVDB, self).__init__()
        if norm_layer is None:
            def norm_layer(chs): return GroupNorm32FVDB(1, chs)
        self.checkpointing = checkpointing
        self.disable_conv_output_padding = disable_conv_output_padding
        self.pooling_type = pooling

        self.norm1 = norm_layer(inplanes)
        self.act1 = act_layer()
        midplanes = planes if midplanes is None else midplanes
        if pooling is not None and pooling in ["s2c"]:
            midplanes = midplanes // 4
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        conv1_stride = stride if pooling is None else 1
        self.conv1 = SparseConv3dFVDB(
            inplanes, midplanes, 3, stride=conv1_stride, bias=False, 
            disable_conv_output_padding=disable_conv_output_padding)

        self.pooling = None
        if (stride != 1) and (pooling is not None):
            assert pooling in ["avg", "max", "s2c"], f"pooling={pooling}"
            if pooling == "s2c":
                self.pooling = DownSamplingSpatial2ChannelFVDB(stride)
                midplanes = midplanes * 8
            else:
                pooling_cls = MaxPoolFVDB if pooling == "max" else AvgPoolFVDB
                self.pooling = pooling_cls(kernel_size=stride)

        self.norm2 = norm_layer(midplanes)
        self.act2 = act_layer()
        self.conv2 = SparseConv3dFVDB(
            midplanes, planes, 3, stride=1, bias=False, 
            disable_conv_output_padding=disable_conv_output_padding)

        if stride != 1 or inplanes != planes*self.expansion:
            inplanes_mul = 8 if self.pooling_type == "s2c" else 1
            self.skip_connection = nn.Sequential(
                # SparseConv3dFVDB(
                #     inplanes, planes*self.expansion, 1, 1, bias=False),
                LinearFVDB(inplanes*inplanes_mul, planes*self.expansion, bias=False),
                norm_layer(planes*self.expansion)
            )
        else:
            self.skip_connection = nn.Identity()

        self.stride = stride

    def forward(self, x: fVDBTensor,
                downres_grid: fvdb.GridBatch = None) -> fVDBTensor:
        # if x.dtype != self.conv1.weight.dtype:
        #     x = x.type(self.conv1.weight.dtype)

        if self.checkpointing and self.training:
            grid, data, spatial_cache = checkpoint(
                self._forward, x.grid, x.data, x.spatial_cache,
                downres_grid, use_reentrant=False)
        else:
            grid, data, spatial_cache = self._forward(
                x.grid, x.data, x.spatial_cache, downres_grid=downres_grid)

        out = fVDBTensor(grid, data, spatial_cache=spatial_cache)
        return out

    def _forward(
            self,
            grid: fvdb.GridBatch,
            data: fvdb.JaggedTensor,
            spatial_cache: Optional[dict] = None,
            downres_grid: fvdb.GridBatch = None):
        x = fVDBTensor(grid, data, spatial_cache=spatial_cache)
        out = x.clone()
        
        if not self.disable_conv_output_padding:
            target_grid = x.grid.conv_grid(
                self.conv1.kernel_size, stride=self.conv1.stride)
        else:
            target_grid = x.grid
        plan1 = fvdb.ConvolutionPlan.from_grid_batch(
            self.conv1.kernel_size, self.conv1.stride, x.grid, target_grid)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.conv1(out, plan=plan1)

        if self.pooling is not None:
            out = self.pooling(out, coarse_data=downres_grid)

        if not self.disable_conv_output_padding:
            target_grid = out.grid.conv_grid(
                self.conv2.kernel_size, stride=self.conv2.stride)
        else:
            target_grid = out.grid

        plan2 = fvdb.ConvolutionPlan.from_grid_batch(
            self.conv2.kernel_size, self.conv2.stride, out.grid, target_grid)
        out = self.norm2(out)
        out = self.act2(out)
        out = self.conv2(out, plan=plan2)

        if self.pooling is not None:
            x = self.pooling(x, coarse_data=downres_grid)
        x: fVDBTensor = self.skip_connection(x)

        if not self.disable_conv_output_padding:
            new_x = out.grid.inject_from(x.grid, x.data, default_value=0.0)
            x = fVDBTensor(out.grid, new_x, spatial_cache=out.spatial_cache)

        out: fVDBTensor = out + x

        curgrid, curdata, curspatial_cache = out.grid, out.data, out.spatial_cache
        return curgrid, curdata, curspatial_cache


class BottleneckResBlock3DFVDB(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        pooling: Literal["avg", "max", "s2c"] = "avg",
        groups: int = 1,
        base_width: int = 64,
        act_layer=SiLUFVDB,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        disable_conv_output_padding: bool = True,
        checkpointing=False,
    ) -> None:
        super(BottleneckResBlock3DFVDB, self).__init__()
        if norm_layer is None:
            def norm_layer(chs): return GroupNorm32FVDB(1, chs)
        self.checkpointing = checkpointing
        self.disable_conv_output_padding = disable_conv_output_padding
        self.pooling_type = pooling

        width = int(planes * (base_width / 64.0)) * groups
        ori_width = width
        if pooling == "s2c":
            width = int(width / 2)

        self.norm1 = norm_layer(inplanes)
        self.act1 = act_layer()
        # self.conv1 = SparseConv3dFVDB(
        #   inplanes, width, 1, 1, bias=False, disable_conv_output_padding=True)
        self.conv1 = LinearFVDB(inplanes, width, bias=False)

        self.pooling = None
        if (stride != 1) and (pooling is not None):
            assert pooling in ["avg", "max", "s2c"]
            if pooling == "s2c":
                self.pooling = DownSamplingSpatial2ChannelFVDB(stride)
                width = width * 8
            else:
                pooling_cls = MaxPoolFVDB if pooling == "max" else AvgPoolFVDB
                self.pooling = pooling_cls(kernel_size=stride)

        self.norm2 = norm_layer(width)
        self.act2 = act_layer()
        conv2_stride = stride if pooling is None else 1
        self.conv2 = SparseConv3dFVDB(
            width, width, 3, stride=conv2_stride, bias=True,
            disable_conv_output_padding=disable_conv_output_padding)
        # self.conv2 = conv3x3(width, width, stride, groups, dilation)

        self.norm3 = norm_layer(width)
        self.act3 = act_layer()
        # self.conv3 = SparseConv3dFVDB(
        #     width, planes*self.expansion, 1, 1, bias=False,
        #     disable_conv_output_padding=True)
        self.conv3 = LinearFVDB(width, planes*self.expansion, bias=False)
        # self.conv3 = conv1x1(width, planes * self.expansion)

        if stride != 1 or inplanes != planes*self.expansion:
            inplanes_mul = 8 if self.pooling_type == "s2c" else 1
            self.skip_connection = nn.Sequential(
                # SparseConv3dFVDB(
                #     inplanes, planes*self.expansion, 1, 1, bias=False),
                LinearFVDB(inplanes*inplanes_mul, planes*self.expansion, bias=False),
                norm_layer(planes*self.expansion)
            )
        else:
            self.skip_connection = nn.Identity()
            # fvdbnn.MaxPool()
        self.stride = stride

    def forward(self, x: fVDBTensor,
                downres_grid: fvdb.GridBatch = None) -> fVDBTensor:
        # if x.dtype != self.conv1.weight.dtype:
        #     x = x.type(self.conv1.weight.dtype)

        if self.checkpointing and self.training:
            grid, data, spatial_cache = checkpoint(
                self._forward, x.grid, x.data, x.spatial_cache,
                downres_grid, use_reentrant=False)
        else:
            grid, data, spatial_cache = self._forward(
                x.grid, x.data, x.spatial_cache, downres_grid=downres_grid)

        out = fVDBTensor(grid, data, spatial_cache=spatial_cache)
        return out

    def _forward(
        self,
        grid: fvdb.GridBatch,
        data: fvdb.JaggedTensor,
        spatial_cache: Optional[dict] = None,
        downres_grid: fvdb.GridBatch = None
    ):
        x = fVDBTensor(grid, data, spatial_cache=spatial_cache)
        out = x.clone()

        # grid_padded = x.grid.conv_grid(
        #     self.conv1.kernel_size, stride=self.conv1.stride)
        # plan1 = fvdb.ConvolutionPlan.from_grid_batch(
        #     self.conv1.kernel_size, self.conv1.stride, x.grid, target_grid=x.grid)

        # resnet v2
        out = self.norm1(out)
        out = self.act1(out)
        out = self.conv1(out)

        if self.pooling is not None:
            out = self.pooling(out, coarse_data=downres_grid)

        if not self.disable_conv_output_padding:
            target_grid = out.grid.conv_grid(
                self.conv2.kernel_size, stride=self.conv2.stride)
        else:
            target_grid = out.grid
        plan2 = fvdb.ConvolutionPlan.from_grid_batch(
            self.conv2.kernel_size, self.conv2.stride, out.grid, target_grid=target_grid)

        out = self.norm2(out)
        out = self.act2(out)
        out = self.conv2(out, plan=plan2)

        # plan3 = fvdb.ConvolutionPlan.from_grid_batch(
        #     self.conv3.kernel_size, self.conv3.stride, out.grid, target_grid=out.grid)
        out = self.norm3(out)
        out = self.act3(out)
        out = self.conv3(out)

        if self.pooling is not None:
            x = self.pooling(x, coarse_data=downres_grid)

        x: fVDBTensor = self.skip_connection(x)

        if not self.disable_conv_output_padding:
            new_x = out.grid.inject_from(x.grid, x.data, default_value=0.0)
            x = fVDBTensor(out.grid, new_x, spatial_cache=out.spatial_cache)

        out: fVDBTensor = out + x

        curgrid, curdata, curspatial_cache = out.grid, out.data, out.spatial_cache

        return curgrid, curdata, curspatial_cache


class SparseConvOutputHeadFVDB(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups, checkpointing=False):
        super(SparseConvOutputHeadFVDB, self).__init__()
        self.checkpointing = checkpointing

        num_groups = max(1, min(num_groups, in_channels//16))
        self.norm1 = GroupNorm32FVDB(num_groups, in_channels)
        self.conv1 = SparseConv3dFVDB(in_channels, in_channels, 3, 1, bias=False)
        self.act1 = SiLUFVDB(inplace=True)

        # self.outconv = SparseConv3dFVDB(in_channels, out_channels, 1, bias=True)
        self.outconv = LinearFVDB(in_channels, out_channels, bias=True)

    def forward(self, x: fVDBTensor):
        # if x.dtype != self.conv1.weight.dtype:
        #     x = x.type(self.conv1.weight.dtype)

        grids, feats, spatial_cache = x.grid, x.data, x.spatial_cache
        if self.training and self.checkpointing:
            grids, feats, spatial_cache = checkpoint(
                self._forward, grids, feats, spatial_cache, use_reentrant=False)
        else:
            grids, feats, spatial_cache = self._forward(grids, feats, spatial_cache)

        out = fVDBTensor(grids, feats, spatial_cache=spatial_cache)
        return out

    def _forward(
        self,
        grids: fvdb.GridBatch,
        feats: fvdb.JaggedTensor,
        spatial_cache: Optional[dict] = None,
    ):
        x = fVDBTensor(grids, feats, spatial_cache=spatial_cache)
        out = self.norm1(x)
        # out = out.type(torch.float32)
        # grid_padded = x.grid.conv_grid(
        #     self.conv1.kernel_size, stride=self.conv1.stride)
        plan1 = fvdb.ConvolutionPlan.from_grid_batch(
            self.conv1.kernel_size, self.conv1.stride, x.grid, target_grid=x.grid)

        out = self.conv1(out, plan=plan1)
        out = self.act1(out)

        # plan2 = fvdb.ConvolutionPlan.from_grid_batch(
        #     self.outconv.kernel_size, 
        #     self.outconv.stride, out.grid, target_grid=out.grid)

        # out: fVDBTensor = self.outconv(out, plan=plan2)
        out: fVDBTensor = self.outconv(out)

        out = out.type(x.dtype)

        return out.grid, out.data, out.spatial_cache
