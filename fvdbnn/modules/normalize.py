# -*- coding:utf-8 -*-
###
# File: normalize.py
# Created Date: Saturday, November 22nd 2025, 12:36:34 am
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
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import fvnn_module
from .vdbtensor import fVDBTensor


@fvnn_module
class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        x (tensor): (..., heads, dim)
        """
        norm_x = F.normalize(x.float(), dim=-1)
        norm_x = (norm_x * self.gamma * self.scale).to(x.dtype)
        return norm_x


@fvnn_module
class LayerNorm32FVDB(nn.LayerNorm):
    """
    Applies Layer Normalization over a :class:`JaggedTensor` batch of features.
    See :class:`~torch.nn.LayerNorm` for detailed information on Layer Normalization.

    Args
    -----
    normalized_shape (int or list or torch.Size): 
    input shape from an expected input of size

    eps (float, optional): a value added to the denominator for numerical stability.
    Default: 1e-5.

    elementwise_affine (bool, optional): a boolean value that when set to ``True``,
    this module has learnable affine parameters. Default: ``True``

    device (torch.device, optional): device on which the module is allocated.
    Default: ``None``

    dtype (torch.dtype, optional): data type of the module parameters. Default: ``None``
    """
    def forward(self, input: fVDBTensor) -> fVDBTensor:
        """
        Apply Layer Normalization to the input :class:`JaggedTensor`.

        Args
        ----
        input (fVDBTensor): Input features to be normalized.

        Returns
        -------
        result (fVDBTensor): The result of the layer normalization.
        """
        input_data = input.data.jdata
        original_dtype = input_data.dtype
        input_data = input_data.to(torch.float32)
        intype = input_data.dtype
        weight, bias = self.weight, self.bias
        if weight is not None and weight.dtype != intype:
            weight = weight.to(intype)
        if bias is not None and bias.dtype != intype:
            bias = bias.to(intype)
        normed_x = F.layer_norm(
            input_data, self.normalized_shape, weight, bias, self.eps)
        normed_x = normed_x.to(original_dtype)
        spatial_cache = input.spatial_cache

        ret = fVDBTensor(input.grid, input.grid.jagged_like(normed_x),
                         spatial_cache=spatial_cache)
        return ret


@fvnn_module
class GroupNorm32FVDB(nn.GroupNorm):
    """
    Applies Group Normalization over a 
    :class:`JaggedTensor` batch of features associated with a :class:`GridBatch`.
    See :class:`~torch.nn.GroupNorm` for detailed information on Group Normalization.

    Args
    -----
    num_groups (int): number of groups to separate the channels into

    num_channels (int): number of channels in the input :class:`JaggedTensor`

    eps (float, optional): a value added to the denominator for numerical stability.
    Default: 1e-5.

    affine (bool, optional): a boolean value that when set to ``True``,
    this module has learnable affine parameters. Default: ``True``

    device (torch.device, optional): device on which the module is allocated.
    Default: ``None``

    dtype (torch.dtype, optional): data type of the module parameters. Default: ``None``
    """

    def super_forward(self, input: torch.Tensor,
                      weight: torch.Tensor = None,
                      bias: torch.Tensor = None) -> torch.Tensor:
        return F.group_norm(input, self.num_groups, weight, bias, self.eps)

    def forward(self, input: fVDBTensor) -> fVDBTensor:
        """
        Apply Group Normalization to the input :class:`JaggedTensor`
        using the provided :class:`GridBatch`.

        Args:
            data (JaggedTensor): Input features to be normalized.
            grid (GridBatch): The grid batch corresponding to ``data``.

        Returns:
            result (JaggedTensor): The result of the group normalization.
        """
        num_channels = input.data.jdata.size(1)
        assert num_channels == self.num_channels, (
            f"Input feature should have the same number of channels as GroupNorm")
        num_batches = input.grid.grid_count

        flat_data, flat_offsets = input.data.jdata, input.data.joffsets
        origin_dtype = flat_data.dtype
        flat_data = flat_data.to(torch.float32)
        intype = flat_data.dtype

        weight, bias = self.weight, self.bias
        if weight is not None and weight.dtype != intype:
            weight = weight.to(intype)
        if bias is not None and bias.dtype != intype:
            bias = bias.to(intype)

        result_data = torch.empty_like(flat_data)
        for b in range(num_batches):
            feat = flat_data[flat_offsets[b]: flat_offsets[b + 1]]
            if feat.size(0) != 0:
                feat = feat.transpose(0, 1).contiguous().reshape(1, num_channels, -1)
                feat = self.super_forward(feat, weight, bias)
                feat = feat.reshape(num_channels, -1).transpose(0, 1)

                result_data[flat_offsets[b]: flat_offsets[b + 1]] = feat

        result_data = result_data.to(origin_dtype)
        spatial_cache = input.spatial_cache
        ret = fVDBTensor(input.grid, input.grid.jagged_like(result_data),
                         spatial_cache=spatial_cache)
        return ret


@fvnn_module
class BatchNorm32FVDB(nn.BatchNorm1d):
    """
    Applies Batch Normalization over a :class:`JaggedTensor`
    batch of features associated with a :class:`GridBatch`.
    See :class:`~torch.nn.BatchNorm1d` for detailed information on Batch Normalization.

    .. seealso::
    :class:`fvdb.nn.SyncBatchNorm` for distributed batch normalization
    across multiple processes.

    Args
    ----
    num_features (int): number of features in the input :class:`JaggedTensor`

    eps (float, optional): a value added to the denominator for numerical stability.
    Default: 1e-5.

    momentum (float, optional): the value used for
    the running_mean and running_var computation. Default: 0.1

    affine (bool, optional): a boolean value that when set to ``True``,
    this module has learnable affine parameters. Default: ``True``

    track_running_stats (bool, optional): a boolean value that when set to ``True``,
    this module tracks the running mean and variance, and when set to ``False``,
    this module does not track such statistics and always uses batch statistics
    in both training and eval modes. Default: ``True``

    device (torch.device, optional): device on which the module is allocated.
    Default: ``None``

    dtype (torch.dtype, optional): data type of the module parameters. Default: ``None``
    """

    def super_forward(self, input: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization
        rather than the buffers. Mini-batch stats are used in training mode,
        and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode.
        Thus they only need to be passed when the update should occur
        (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        origin_dtype = input.dtype
        input = input.to(torch.float32)
        intype = input.dtype
        running_mean = self.running_mean
        if running_mean is not None and running_mean.dtype != intype:
            running_mean = running_mean.to(intype)
        running_var = self.running_var
        if running_var is not None and running_var.dtype != intype:
            running_var = running_var.to(intype)
        weight = self.weight
        if weight is not None and weight.dtype != intype:
            weight = weight.to(intype)
        bias = self.bias
        if bias is not None and bias.dtype != intype:
            bias = bias.to(intype)

        ret = F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            running_mean if not self.training or self.track_running_stats else None,
            running_var if not self.training or self.track_running_stats else None,
            weight,
            bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )
        ret = ret.to(origin_dtype)
        return ret

    def forward(self, input: fVDBTensor) -> fVDBTensor:
        num_channels = input.data.jdata.size(1)
        assert num_channels == self.num_features, (
            f"Input feature should have the same number of channels as BatchNorm")
        result_data = self.super_forward(input.data.jdata)
        spatial_cache = input.spatial_cache
        return fVDBTensor(input.grid, input.grid.jagged_like(result_data),
                          spatial_cache=spatial_cache)


@fvnn_module
class SyncBatchNorm32FVDB(nn.SyncBatchNorm):
    """
    Applies distributed Batch Normalization over a
    :class:`JaggedTensor` batch of features associated with a :class:`GridBatch`.
    See :class:`~torch.nn.SyncBatchNorm`
    for detailed information on distributed batch normalization.

    .. note::
        Only supports :class:`~torch.nn.DistributedDataParallel`
        (DDP) with single GPU per process. Use
        :meth:`fvdb.nn.SyncBatchNorm.convert_sync_batchnorm()` to convert
        :attr:`BatchNorm` layer to :class:`SyncBatchNorm` before wrapping
        Network with DDP.

    .. seealso::
        :class:`fvdb.nn.BatchNorm` for non-distributed batch normalization.

    Args
    ----
    num_features (int): number of features in the input :class:`JaggedTensor`

    eps (float, optional): a value added to the denominator for numerical stability.
    Default: 1e-5.

    momentum (float, optional): the value used for
    the running_mean and running_var computation. Default: 0.1

    affine (bool, optional): a boolean value that when set to ``True``,
    this module has learnable affine parameters. Default: ``True``

    track_running_stats (bool, optional): a boolean value that when set to ``True``,
    this module tracks the running mean and variance, and when set to ``False``,
    this module does not track such statistics and always uses batch statistics
    in both training and eval modes. Default: ``True``

    process_group (Any, optional): the process group to scope synchronization.
    Default: ``None``

    device (torch.device, optional): device on which the module is allocated.
    Default: ``None``

    dtype (torch.dtype, optional): data type of the module parameters. Default: ``None``.
    """
    
    def super_forward(self, input: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(input)
        self._check_non_zero_input_channels(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            assert self.num_batches_tracked is not None
            self.num_batches_tracked.add_(1)
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization
        rather than the buffers. Mini-batch stats are used in training mode,
        and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode.
        Thus they only need to be passed when the update should occur
        (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        # If buffers are not to be tracked, ensure that they won't be updated
        running_mean = (
            self.running_mean if not self.training or self.track_running_stats else None
        )
        running_var = (
            self.running_var if not self.training or self.track_running_stats else None
        )

        # Don't sync batchnorm stats in inference mode (model.eval()).
        need_sync = (
            bn_training
            and self.training
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        if need_sync:
            # currently only GPU/PrivateUse1 input is supported
            if input.device.type not in [
                "cuda",
                "xpu",
                torch._C._get_privateuse1_backend_name(),
            ]:
                raise ValueError(
                    "SyncBatchNorm expected input tensor to be on GPU or XPU or "
                    f"{torch._C._get_privateuse1_backend_name()}"
                )

            process_group = torch.distributed.group.WORLD
            if self.process_group:
                process_group = self.process_group
            world_size = torch.distributed.get_world_size(process_group)
            need_sync = world_size > 1

        origin_dtype = input.dtype
        input = input.to(torch.float32)
        intype = input.dtype
        if running_mean is not None and running_mean.dtype != intype:
            running_mean = running_mean.to(intype)
        if running_var is not None and running_var.dtype != intype:
            running_var = running_var.to(intype)
        weight, bias = self.weight, self.bias
        if weight is not None and weight.dtype != intype:
            weight = weight.to(intype)
        if bias is not None and bias.dtype != intype:
            bias = bias.to(intype)
        
        # fallback to framework BN when synchronization is not necessary
        if not need_sync:
            ret = F.batch_norm(
                input,
                running_mean,
                running_var,
                weight,
                bias,
                bn_training,
                exponential_average_factor,
                self.eps,
            )
        else:
            assert bn_training
            from torch.nn.modules._functions import SyncBatchNorm as sync_batch_norm
            ret = sync_batch_norm.apply(
                input,
                weight,
                bias,
                running_mean,
                running_var,
                self.eps,
                exponential_average_factor,
                process_group,  # type: ignore[possibly-undefined]
                world_size,  # type: ignore[possibly-undefined]
            )
        ret = ret.to(origin_dtype)
        return ret

    def forward(self, input: fVDBTensor) -> fVDBTensor:
        """
        Apply Synchronized Batch Normalization to the input
        :class:`JaggedTensor` using the provided :class:`GridBatch`.

        Args:
            data (JaggedTensor): Input features to be normalized.
            grid (GridBatch): The grid batch corresponding to ``data``.

        Returns:
            result (JaggedTensor): The result of the synchronized batch normalization.
        """
        num_channels = input.data.jdata.size(1)
        assert num_channels == self.num_features, (
            f"Input feature should have the same number of channels as BatchNorm")
        result_data = self.forward(input.data.jdata)
        spatial_cache = input.spatial_cache
        return fVDBTensor(input.grid, input.grid.jagged_like(result_data),
                          spatial_cache=spatial_cache)

    @classmethod
    def convert_sync_batchnorm(
        cls, module: nn.Module, process_group: Any = None) -> nn.Module:
        """
        Helper function to convert :attr:`fvdb.nn.BatchNorm`
        layer in the model to :attr:`fvdb.nn.SyncBatchNorm` layer.

        Args
        ----
        module (nn.Module): Module for which all :attr:`fvdb.nn.BatchNorm`
        layers will be converted to :attr:`fvdb.nn.SyncBatchNorm` layers.

        process_group (Any): process group to scope synchronization,
        default is the whole world.

        Returns:
        sync_batch_norm (torch.nn.Module):  The original module with the converted
        :attr:`fvdb.nn.SyncBatchNorm` layers.

        Example::

            >>> # Network with fvdb.nn.SyncBatchNorm layer
            >>> module = fvdb.nn.Sequential(
            >>>            fvdb.nn.Linear(20, 100),
            >>>            fvdb.nn.BatchNorm(100)
            >>>          )
            >>> # creating process group (optional)
            >>> # process_ids is a list of int identifying rank ids.
            >>> process_group = torch.distributed.new_group(process_ids)
            >>> sync_bn_module = fvdb.nn.SyncBatchNorm.convert_sync_batchnorm(module, process_group)

        """
        module_output = module
        if isinstance(module, BatchNorm32FVDB):
            module_output = cls(
                module.num_features,
                module.eps,
                module.momentum,
                module.affine,
                module.track_running_stats,
                process_group,
            )
            if module.affine:
                with torch.no_grad():
                    module_output.weight = module.weight
                    module_output.bias = module.bias
            module_output.running_mean = module.running_mean
            module_output.running_var = module.running_var
            module_output.num_batches_tracked = module.num_batches_tracked
            module_output.training = module.training
            if hasattr(module, "qconfig"):
                module_output.qconfig = module.qconfig
        for name, child in module.named_children():
            module_output.add_module(name, cls.convert_sync_batchnorm(child, process_group))
        del module
        return module_output
