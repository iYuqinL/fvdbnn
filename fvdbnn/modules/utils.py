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
from typing import Union
from torch.profiler import record_function
import torch
import fvdb
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
    def forward(
        self, 
        indata: Union[fVDBTensor, fvdb.JaggedTensor, torch.Tensor]
    ) -> Union[fVDBTensor, fvdb.JaggedTensor, torch.Tensor]:
        assert isinstance(indata, (fVDBTensor, fvdb.JaggedTensor, torch.Tensor)), (
            f"Input should have type fVDBTensor, fvdb.JaggedTensor and torch.Tensor, "
            f"but got {type(indata)}")

        realdata = indata if torch.is_tensor(indata) else indata.jdata
        res = super().forward(realdata)  # type: ignore

        if isinstance(indata, fVDBTensor):
            ret = fVDBTensor(
                indata.grid, indata.grid.jagged_like(res), 
                spatial_cache=indata.spatial_cache)
        elif isinstance(indata, fvdb.JaggedTensor):
            ret = indata.jagged_like(res)
        else:
            ret = res

        return ret
