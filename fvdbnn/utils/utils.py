# -*- coding:utf-8 -*-
###
# File: utils.py
# Created Date: Sunday, November 16th 2025, 12:40:40 am
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
import os
import gc

import torch
import fvdb


__all__ = ["get_nvml_handls", "get_gpu_compute_capacity", "smart_empty_cuda_cache",
           "safe_perbatch_jmin", "safe_perbatch_jmax"]


import atexit
try:
    import pynvml
    pynvml.nvmlInit()
    atexit.register(pynvml.nvmlShutdown)
    _nvml_handles = {}
except Exception:
    _nvml_handles = None

def get_nvml_handls(device: torch.device):
    if _nvml_handles is None:
        return None

    cuda_dev      = device.index  # e.g. 0
    visible_str   = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    # parse it and pick the physical index
    if visible_str is not None and visible_str != "":
        visible_ids = [int(x) for x in visible_str.split(",")]
        phys_index = visible_ids[cuda_dev]
    else:
        phys_index = cuda_dev

    if phys_index in _nvml_handles:
        return _nvml_handles[phys_index]

    _nvml_handles[phys_index] = pynvml.nvmlDeviceGetHandleByIndex(phys_index)

    return _nvml_handles[phys_index]

def smart_empty_cuda_cache(
        threshold=3.0, device=torch.device("cuda:0"), silent=True):
    """
    Args
    ----
    threshold (float): free cuda memory threshold (GiB).
    """
    _nvml_handle = get_nvml_handls(device)
    if _nvml_handle is None:
        return
    info = pynvml.nvmlDeviceGetMemoryInfo(_nvml_handle)
    free_mem_GB = info.free / (1024**3)
    if not silent:
        print(f"free gpu memory is: {free_mem_GB} GiB")
    if info.free < threshold * 1024**3:
        gc.collect()
        torch.cuda.empty_cache()

def get_gpu_compute_capacity(device: torch.device):
    idx = device.index
    props = torch.cuda.get_device_properties(idx)
    major, minor = props.major, props.minor
    return major, minor


# --- safe per-batch min (handle empty batches) ---
def safe_perbatch_jmin(jtensor: fvdb.JaggedTensor, dim: int = 0) -> fvdb.JaggedTensor:
    assert dim == 0, f"dim must be 0, got {dim}"
    min_jtensor_list = []
    for bidx in range(jtensor.num_tensors):
        jtensor_i_jdata = jtensor[bidx].jdata  # (ni, 3)
        if jtensor_i_jdata.numel() > 0:
            min_jtensor_list.append(jtensor_i_jdata.min(dim=dim)[0].unsqueeze(0))
        else:
            min_jtensor_list.append(torch.zeros(
                1, *(jtensor_i_jdata.shape[1:]),
                dtype=jtensor_i_jdata.dtype, device=jtensor_i_jdata.device))
    min_jtensor = fvdb.JaggedTensor(min_jtensor_list)
    return min_jtensor


# --- safe per-batch max (handle empty batches) ---
def safe_perbatch_jmax(jtensor: fvdb.JaggedTensor, dim: int = 0) -> fvdb.JaggedTensor:
    assert dim == 0, f"dim must be 0, got {dim}"
    max_jtensor_list = []
    for bidx in range(jtensor.num_tensors):
        jtensor_i_jdata = jtensor[bidx].jdata  # (ni, 3)
        if jtensor_i_jdata.numel() > 0:
            max_jtensor_list.append(jtensor_i_jdata.max(dim=dim)[0].unsqueeze(0))
        else:
            max_jtensor_list.append(torch.zeros(
                1, *(jtensor_i_jdata.shape[1:]),
                dtype=jtensor_i_jdata.dtype, device=jtensor_i_jdata.device))
    max_jtensor = fvdb.JaggedTensor(max_jtensor_list)
    return max_jtensor
