# -*- coding:utf-8 -*-
###
# File: vdbtensor.py
# Created Date: Saturday, November 22nd 2025, 12:31:22 am
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
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, Union

import torch

import fvdb
from fvdb import GridBatch, JaggedTensor
from fvdb.types import Vec3dBatch, Vec3dBatchOrScalar, Vec3i

JaggedTensorOrTensor = Union[torch.Tensor, JaggedTensor]


@dataclass
class fVDBTensor:
    """
    A fVDBTensor is a thin wrapper around a GridBatch and
    its corresponding feature JaggedTensor, conceptually denoting a batch of
    sparse tensors along with its topology.
    It works as the input and output arguments of fvdb's neural network layers.
    One can simply construct a fVDBTensor from a GridBatch and a JaggedTensor,
    or from a dense tensor using from_dense().
    """

    grid: GridBatch
    data: JaggedTensor

    # store the scaling factor of the grid for spatial cache
    # grid spatial related cache
    spatial_cache: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.grid, GridBatch):
            raise TypeError("grid should be of type GridBatch")
        if not isinstance(self.data, JaggedTensor):
            raise TypeError("data must be a JaggedTensor or a torch.Tensor")
        if self.grid.grid_count != len(self.data):
            raise ValueError("grid and feature should have the same batch size")
        if self.grid.total_voxels != self.data.jdata.size(0):
            raise ValueError("grid and feature should have the same total voxel count")

    def __getitem__(self, idx):
        return fVDBTensor(self.grid[idx], self.data[idx],
                         spatial_cache=self.spatial_cache)

    def __len__(self):
        return self.grid.grid_count

    def to_dense(self) -> torch.Tensor:
        """
        Returns
        --------
        dense_data (torch.Tensor): Dense :class:`torch.Tensor`
        containing the sparse data with shape
        ``(batch_size, channels*, dense_size_x, dense_size_y, dense_size_z)``.
        """
        return self.grid.inject_to_dense_cmajor(self.data)

    def is_same(self, other: Union["fVDBTensor", GridBatch]):
        if isinstance(other, fVDBTensor):
            return (self.grid.address == other.grid.address and
                    self.grid_count == other.grid_count)
        elif isinstance(other, GridBatch):
            return (self.grid.address == other.address and
                    self.grid_count == other.grid_count)
        else:
            raise TypeError(
                f"Invalid argument 'other' to is_same, must " + 
                f"be a fVDBTensor or GridBatch but got {type(other)}")

    # -----------------------------
    # Arithmetic and math functions
    # -----------------------------
    def __add__(self, other):
        return self._binop(other, lambda a, b: a + b)

    def __sub__(self, other):
        return self._binop(other, lambda a, b: a - b)

    def __mul__(self, other):
        return self._binop(other, lambda a, b: a * b)

    def __pow__(self, other):
        return self._binop(other, lambda a, b: a**b)

    def __neg__(self):
        return fVDBTensor(self.grid, -self.data,
                         spatial_cache=self.spatial_cache)

    def __truediv__(self, other):
        return self._binop(other, lambda a, b: a / b)

    def __floordiv__(self, other):
        return self._binop(other, lambda a, b: a // b)

    def __mod__(self, other):
        return self._binop(other, lambda a, b: a % b)

    def __gt__(self, other):
        return self._binop(other, lambda a, b: a > b)

    def __lt__(self, other):
        return self._binop(other, lambda a, b: a < b)

    def __ge__(self, other):
        return self._binop(other, lambda a, b: a >= b)

    def __le__(self, other):
        return self._binop(other, lambda a, b: a <= b)

    def __eq__(self, other):
        return self._binop(other, lambda a, b: a == b)

    def __ne__(self, other):
        return self._binop(other, lambda a, b: a != b)

    def __iadd__(self, other):
        def inplace_add(a, b):
            a += b

        return self._binop_inplace(other, inplace_add)

    def __isub__(self, other):
        def inplace_sub(a, b):
            a -= b

        return self._binop_inplace(other, inplace_sub)

    def __imul__(self, other):
        def inplace_mul(a, b):
            a *= b

        return self._binop_inplace(other, inplace_mul)

    def __ipow__(self, other):
        def inplace_pow(a, b):
            a **= b

        return self._binop_inplace(other, inplace_pow)

    def __itruediv__(self, other):
        def inplace_truediv(a, b):
            a /= b

        return self._binop_inplace(other, inplace_truediv)

    def __ifloordiv__(self, other):
        def inplace_floordiv(a, b):
            a //= b

        return self._binop_inplace(other, inplace_floordiv)

    def __imod__(self, other):
        def inplace_mod(a, b):
            a %= b

        return self._binop_inplace(other, inplace_mod)

    def sqrt(self):
        return fVDBTensor(self.grid, self.data.sqrt(),
                         spatial_cache=self.spatial_cache)

    def abs(self):
        return fVDBTensor(self.grid, self.data.abs(),
                         spatial_cache=self.spatial_cache)

    def round(self):
        return fVDBTensor(self.grid, self.data.round(),
                         spatial_cache=self.spatial_cache)

    def floor(self):
        return fVDBTensor(self.grid, self.data.floor(),
                         spatial_cache=self.spatial_cache)

    def ceil(self):
        return fVDBTensor(self.grid, self.data.ceil(),
                         spatial_cache=self.spatial_cache)

    def sqrt_(self):
        self.data.sqrt_()
        return self

    def abs_(self):
        self.data.abs_()
        return self

    def round_(self):
        self.data.round_()
        return self

    def floor_(self):
        self.data.floor_()
        return self

    def ceil_(self):
        self.data.ceil_()
        return self

    def _binop(self, other, op):
        if isinstance(other, fVDBTensor):
            return fVDBTensor(self.grid, op(self.data, other.data),
                             spatial_cache=self.spatial_cache)
        else:
            return fVDBTensor(self.grid, op(self.data, other),
                             spatial_cache=self.spatial_cache)

    def _binop_inplace(self, other, op):
        if isinstance(other, fVDBTensor):
            op(self.data, other.data)
            return self
        else:
            op(self.data, other)
            return self

    # -----------------------
    # Interpolation functions
    # -----------------------

    def sample_bezier(self, points: JaggedTensorOrTensor) -> JaggedTensor:
        return self.grid.sample_bezier(points, self.data)

    def sample_bezier_with_grad(
            self, points: JaggedTensorOrTensor) -> Tuple[JaggedTensor, JaggedTensor]:
        return self.grid.sample_bezier_with_grad(points, self.data)

    def sample_trilinear(self, points: JaggedTensorOrTensor) -> JaggedTensor:
        return self.grid.sample_trilinear(points, self.data)

    def sample_trilinear_with_grad(
            self, points: JaggedTensorOrTensor) -> Tuple[JaggedTensor, JaggedTensor]:
        return self.grid.sample_trilinear_with_grad(points, self.data)

    def cpu(self):
        return fVDBTensor(
            self.grid.to("cpu"), self.data.cpu(),
            spatial_cache=self.spatial_cache)

    def cuda(self):
        return fVDBTensor(
            self.grid.to("cuda"), self.data.cuda(),
            spatial_cache=self.spatial_cache)

    def to(self, device_or_dtype: Any):
        return fVDBTensor(
            self.grid.to(device_or_dtype),
            self.data.to(device_or_dtype),
            spatial_cache=self.spatial_cache
        )

    def detach(self):
        return fVDBTensor(self.grid, self.data.detach(),
                         spatial_cache=self.spatial_cache)

    def type(self, arg0: torch.dtype):
        return fVDBTensor(self.grid, self.data.type(arg0),
                         spatial_cache=self.spatial_cache)

    def requires_grad_(self, required_grad):
        self.data.requires_grad_(required_grad)
        return self

    def clone(self):
        return fVDBTensor(self.grid, self.data.clone(),
                         spatial_cache=self.spatial_cache)

    @property
    def num_tensors(self):
        return self.data.num_tensors

    @property
    def is_cuda(self):
        return self.data.is_cuda

    @property
    def is_cpu(self):
        return self.data.is_cpu

    @property
    def device(self):
        return self.data.device

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def jidx(self):
        return self.data.jidx

    @property
    def jlidx(self):
        return self.data.jlidx

    @property
    def joffsets(self):
        return self.data.joffsets

    @property
    def jdata(self):
        return self.data.jdata

    @property
    def rshape(self):
        return self.data.rshape

    @property
    def lshape(self):
        return self.data.lshape

    @property
    def ldim(self):
        return self.data.ldim

    @property
    def eshape(self):
        return self.data.eshape

    @property
    def edim(self):
        return self.data.edim

    @property
    def requires_grad(self):
        return self.data.requires_grad

    @property
    def cum_voxels(self) -> torch.LongTensor:
        return self.grid.cum_voxels

    @property
    def grid_count(self) -> int:
        return self.grid.grid_count

    @property
    def ijk(self) -> JaggedTensor:
        return self.grid.ijk

    @property
    def num_voxels(self) -> torch.LongTensor:
        return self.grid.num_voxels

    @property
    def origins(self) -> torch.FloatTensor:
        return self.grid.origins

    @property
    def total_voxels(self) -> int:
        return self.grid.total_voxels

    @property
    def voxel_sizes(self) -> torch.FloatTensor:
        return self.grid.voxel_sizes

    @property
    def total_leaf_nodes(self) -> int:
        return self.grid.total_leaf_nodes

    @property
    def num_leaf_nodes(self) -> torch.LongTensor:
        return self.grid.num_leaf_nodes

    @property
    def grid_to_world_matrices(self) -> torch.FloatTensor:
        return self.grid.voxel_to_world_matrices

    @property
    def world_to_grid_matrices(self) -> torch.FloatTensor:
        return self.grid.world_to_voxel_matrices

    @property
    def bbox(self) -> torch.IntTensor:
        return self.grid.bboxes

    @property
    def dual_bbox(self) -> torch.IntTensor:
        return self.grid.dual_bboxes

    @property
    def total_bbox(self) -> torch.IntTensor:
        return self.grid.total_bbox

    def _grid_identity(self):
        num_grid = self.grid.grid_count
        total_voxel = self.grid.total_voxels
        voxsize = self.grid.voxel_sizes[:, 0].tolist()
        voxsize = [round(v, 6) for v in voxsize]
        ident = str(num_grid) + ":" + str(total_voxel) + str(voxsize)
        return ident

    def register_spatial_cache(self, key, value) -> None:
        """
        Register a spatial cache.
        The spatial cache can be any thing you want to cache.
        The registery and retrieval of the cache is based on current scale.
        """
        grid_identity = self._grid_identity()
        if grid_identity not in self.spatial_cache:
            self.spatial_cache[grid_identity] = {}
        self.spatial_cache[grid_identity][key] = value

    def get_spatial_cache(self, key=None):
        """
        Get a spatial cache.
        """
        grid_identity = self._grid_identity()
        cur_scale_cache = self.spatial_cache.get(grid_identity, {})
        if key is None:
            return cur_scale_cache
        return cur_scale_cache.get(key, None)


def fVDBTensor_from_dense(
    dense_data: torch.Tensor,
    ijk_min: Optional[Vec3i] = None,
    voxel_sizes: Optional[Vec3dBatchOrScalar] = None,
    origins: Optional[Vec3dBatch] = None,
) -> fVDBTensor:
    """
    Create a fVDBTensor from a dense tensor.

    Args
    -----
    dense_data (torch.Tensor): Dense :class:`torch.Tensor` containing the sparse data with
    shape ``(batch_size, channels*, dense_size_x, dense_size_y, dense_size_z)``.

    ijk_min (Optional[Vec3i]): The minimum index of the dense tensor. Defaults to None.

    voxel_sizes (Optional[Vec3dBatchOrScalar]): The voxel size of the grid. Defaults to None.

    origins (Optional[Vec3dBatch]): The origin of the grid. Defaults to None.

    Returns
    -------
    fVDBTensor: The fVDBTensor created from the dense tensor.
    """
    if origins is None:
        origins = [0.0] * 3
    if voxel_sizes is None:
        voxel_sizes = [1.0] * 3
    if ijk_min is None:
        ijk_min = [0, 0, 0]
    assert ijk_min is not None
    grid = fvdb.GridBatch.from_dense(
        dense_data.size(0),
        dense_data.size()[1:4],
        ijk_min=ijk_min,
        voxel_sizes=voxel_sizes,
        origins=origins,
        device=dense_data.device,
    )
    # Note: this would map dense_feature[0, 0, 0] to grid[ijk_min]
    data = grid.inject_from_dense_cmajor(dense_data.contiguous(), dense_origins=ijk_min)
    return fVDBTensor(grid, data)
