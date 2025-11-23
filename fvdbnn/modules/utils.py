# -*- coding:utf-8 -*-
###
# File: utils.py
# Created Date: Saturday, November 22nd 2025, 12:30:51 am
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

from torch.profiler import record_function
from .vdbtensor import fVDBTensor


def fvnn_module(module):
    # Register class as a module in fvdb.nn
    old_forward = module.forward

    def _forward(self, *args, **kwargs):
        with record_function(repr(self)):
            return old_forward(self, *args, **kwargs)

    module.forward = _forward
    return module


@fvnn_module
class ElementwiseMixin:
    def forward(self, input: fVDBTensor) -> fVDBTensor:
        assert isinstance(input, fVDBTensor), "Input should have type fVDBTensor"
        res = super().forward(input.data.jdata)  # type: ignore
        spatial_cache = input.spatial_cache
        return fVDBTensor(input.grid, input.data.jagged_like(res), spatial_cache)
