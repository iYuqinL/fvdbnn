# -*- coding:utf-8 -*-
###
# File: attn.py
# Created Date: Saturday, November 15th 2025, 3:00:21 pm
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

from typing import Union, Tuple
import torch

import fvdb
from ..modules.vdbtensor import fVDBTensor
from ..utils.utils import safe_perbatch_jmin, safe_perbatch_jmax

import flash_attn  # flashatt v2


def sdpa_flash_fvdb(
        q: fvdb.JaggedTensor,
        k: fvdb.JaggedTensor = None,
        v: fvdb.JaggedTensor = None,
        num_heads: int = 1,
        attn_drop_p: float = 0.0,
        is_causal: bool = False):
    """
    Args
    -----
    q (JaggedTensor): lshape (n1, n2, ...); 
    if k and v is not None eshape is (edim, ), else (3, edim).

    k (JaggedTensor): lshape (n1, n2, ...), eshape (edim, )

    v (JaggedTensor): lshape (n1, n2, ...), eshape (edim, )

    num_heads (int): num attention heads, default is 1.

    attn_drop_p (float): attention dropout prob, deafult is 0.

    is_causal (bool): if it is causal attention.

    Returns
    -------
    atten_v (JaggedTensor): lshape (n1, n2, ...), eshape (edim).
    """
    B = q.num_tensors
    qdata = q.jdata  # (M, qdim) or (M, 3, qdim)
    embed_dim = qdata.shape[-1]
    assert embed_dim % num_heads == 0
    head_dim = embed_dim // num_heads

    if k is None or v is None:
        assert k is None and v is None
        qkv = qdata.view(qdata.shape[0], 3, num_heads, head_dim)  # (M, 3, nH, d)
        cu_seqlens = q.joffsets.int()
        max_seqlen = max(list(q.lshape))
        attned_v = flash_attn.flash_attn_varlen_qkvpacked_func(
            qkv, cu_seqlens, max_seqlen, dropout_p=attn_drop_p, causal=is_causal)
    else:
        qdata = qdata.view(qdata.shape[0], num_heads, head_dim)  # (M, nH, d)
        kdata, vdata = k.jdata, v.jdata
        kdata = kdata.view(kdata.shape[0], num_heads, head_dim)
        vdata = vdata.view(vdata.shape[0], num_heads, head_dim)
        cu_seqlens_q = q.joffsets.int()
        max_seqlen_q = max(list(q.lshape))
        cu_seqlens_k = k.joffsets.int()
        max_seqlen_k = max(list(k.lshape))
        # (Mq, nH, d)
        attned_v = flash_attn.flash_attn_varlen_func(
            qdata, kdata, vdata,
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
            dropout_p=attn_drop_p, causal=is_causal)

    attned_v = attned_v.view(attned_v.shape[0], embed_dim)  # (M, D)
    attned_v = q.jagged_like(attned_v)
    return attned_v


def calc_attn_window_partition(
        grids: fvdb.GridBatch,
        window_size: Union[int, Tuple[int, ...]],
        window_shift: Union[int, Tuple[int, ...]],
        validate_partition_spatial: bool = False):
    """
    Calculate serialization and partitioning for grids.

    Args
    ----
    grids (GridBatch): fvdb grid batch.

    window_size (int): The attention window size.

    window_shift (int, Tuple[int, ...]): window shift steps.

    Returns
    -------
    grids_idxs_winsort (JaggedTensor): ([n1,n2,...], )
    grids indexs sorted by the contained window's index.

    grids_idxs_winargsort (JaggedTensor): ([n1,n2,...], )
    the indexs that reconver the order grids indexs 
    to match the origin grids jagged tensor.

    win_seq_lens (JaggedTensor): ([nw1,nw2,...], )
    each window's voxel seqence length.
    """
    device = grids.device

    window_shift = ((window_shift,) * 3
                    if isinstance(window_shift, int) else window_shift)
    window_size = ((window_size, ) * 3
                   if isinstance(window_size, int) else window_size)
    # JaggedTensor ([n1,n2,...], 3)
    ijks = grids.ijk.clone()
    # min_ijks: fvdb.JaggedTensor = ijks.jmin(dim=0)[0]  # ([1,1,...], 3)
    min_ijks = safe_perbatch_jmin(ijks, dim=0)  # ([1,1,...], 3)

    ijks_list = []
    for bidx in range(ijks.num_tensors):
        # if ijks[bidx] empty, ijks[bidx].jdata - min_ijks[bidx].jdata yields empty -> fine
        ijks_list.append(ijks[bidx].jdata - min_ijks[bidx].jdata)
    ijks = fvdb.JaggedTensor(ijks_list)

    index_dtype = grids.ijk.dtype
    # shift ijk coordinates
    ijks += torch.tensor(window_shift, device=ijks.device, dtype=ijks.dtype)
    ijks: fvdb.JaggedTensor = ijks
    # max_ijks: fvdb.JaggedTensor = ijks.jmax(dim=0)[0]    # ([1,1,...], 3)
    max_ijks = safe_perbatch_jmax(ijks, dim=0)  # ([1,1,...], 3)
    # ([1,1,...], 3)
    num_window: fvdb.JaggedTensor = (max_ijks + 1) / torch.tensor(
        window_size, device=ijks.device, dtype=ijks.dtype)
    empty_mask = torch.as_tensor(ijks.lshape, device=ijks.device, dtype=torch.bool) == 0
    num_window_jdata = num_window.jdata  # (B, 3)
    if torch.any(empty_mask):
        num_window_jdata[empty_mask, :] = 0
    num_window = num_window.jagged_like(
        torch.ceil(num_window_jdata).to(index_dtype))

    num_window_jdata = num_window.jdata  # (B, 3)
    # window_idx_offsets = torch.cat(
    #     (torch.ones_like(num_window_jdata[:, 0:1]), num_window_jdata[:, :2]), dim=1)
    # print(f"num_window_jdata: {num_window_jdata}")
    window_idx_offsets = torch.cat(
        (torch.ones_like(num_window_jdata[:, 0:1]),
         num_window_jdata[:, 2:3], num_window_jdata[:, 1:2]), dim=1)
    # print(f"window_idx_offsets: {window_idx_offsets}")
    # window indcies offsets which use to flatten the spatial window indices
    window_idx_offsets = torch.cumprod(window_idx_offsets, dim=1)[:, :3]  # (B, 3)
    if torch.any(empty_mask):
        window_idx_offsets[empty_mask, :] = 0
    # ::-1 is because i is high bit and k is low bit.
    window_idx_offsets = torch.flip(window_idx_offsets, dims=[1])
    # print(f"window_idx_offsets: {window_idx_offsets}")

    # voxels' corresponding window indices in spatial
    # ([n1,n2,...], 3)
    vox_win_idxs: fvdb.JaggedTensor = ijks // torch.tensor(
        window_size, device=ijks.device, dtype=ijks.dtype)

    # ([n1,n2,...],)
    grids_indexs = grids.ijk_to_index(grids.ijk, cumulative=True)
    # grids_indexs_not_global = (
    #     grids_indexs.jdata.max().item() < grids.total_voxels - 1)
    grids_indexs_joffsets = grids_indexs.joffsets
    grids_idxs_winargsort = torch.empty_like(grids_indexs.jdata)
    grids_idxs_winsort, win_seq_lens = [], []
    for bidx in range(grids.grid_count):
        jsidx, jeidx = grids_indexs_joffsets[bidx], grids_indexs_joffsets[bidx+1]
        vox_win_idxs_i = vox_win_idxs[bidx].jdata  # (nx, 3), nx can be 0 for empty
        fl_vox_win_idxs_i = vox_win_idxs_i * window_idx_offsets[bidx:bidx+1]
        fl_vox_win_idxs_i = fl_vox_win_idxs_i.sum(dim=1)  # (nx, )
        # here, fl_vox_win_idxs_i should have many same values
        # for that many voxels are contained in one window,
        # so argsort will sort the same values into adjacent positions,
        # which means the voxels in the same window will place in adjacent positions.
        window_argidxs = torch.argsort(fl_vox_win_idxs_i)  # (nx, )

        grids_i_indexs = grids_indexs[bidx].jdata  # (nx, )
        # if grids_indexs_not_global:
        #      grids_i_indexs += jsidx
        grids_idxs_winsort_i = grids_i_indexs[window_argidxs]  # (nx, )
        # consider a 1D tensor x = [0, 1, 2, 3, 4, 5, 6, 7, 8],
        # and grids_idxs_winsort_i is [0, 1, 3, 4, 2, 5, 6, 7, 8],
        # the x[grids_idxs_winsort_i] will be [0, 1, 3, 4, 2, 5, 6, 7, 8],
        # and here grids_idxs_winargsort will be [0, 1, 4, 2, 3, 5, 6, 7, 8]
        # so x[grids_idxs_winsort_i][grids_idxs_winargsort]
        # will be [0, 1, 2, 3, 4, 5, 6, 7, 8], which recover the correct order.
        grids_idxs_winargsort[grids_idxs_winsort_i] = (
            torch.arange(grids_idxs_winsort_i.shape[0],
                         device=device, dtype=grids_idxs_winargsort.dtype) + jsidx)

        grids_idxs_winsort.append(grids_idxs_winsort_i)

        win_seq_lens_i = torch.bincount(fl_vox_win_idxs_i)  # (nWx, )
        seq_mask_i = win_seq_lens_i != 0
        win_seq_lens_i = win_seq_lens_i[seq_mask_i]
        win_seq_lens.append(win_seq_lens_i)

    grids_idxs_winsort = fvdb.JaggedTensor(grids_idxs_winsort)
    grids_idxs_winargsort = grids_indexs.jagged_like(grids_idxs_winargsort)
    win_seq_lens = fvdb.JaggedTensor(win_seq_lens)

    # validate partition
    if validate_partition_spatial:
        # all windows' ijk diff should less than window_size
        cu_seqlens = torch.cumsum(
            torch.cat([torch.zeros_like(win_seq_lens.jdata[0:1]),
                       win_seq_lens.jdata], dim=0), dim=0)
        for widx in range(win_seq_lens.jdata.shape[0]):
            wsidx, weidx = cu_seqlens[widx], cu_seqlens[widx+1]
            win_idxs = grids_idxs_winsort.jdata[wsidx:weidx]
            win_ijks = grids.ijk.jdata[win_idxs]
            win_ijks_max_diff = win_ijks.max(dim=0)[0] - win_ijks.min(dim=0)[0]
            if not torch.all(win_ijks_max_diff <= torch.tensor(
                window_size, device=device, dtype=win_ijks_max_diff.dtype)):
                print(f"window {widx} ijk diff max: {win_ijks_max_diff}")
                print(f"window {widx} ijk: {win_ijks}")
                raise ValueError(f"window {widx} ijk diff max: {win_ijks_max_diff}")
        print(f"window partition validate done")

    return grids_idxs_winsort, grids_idxs_winargsort, win_seq_lens


def sparse_window_flash_attn(
        query: fVDBTensor,
        key: fvdb.JaggedTensor = None,
        value: fvdb.JaggedTensor = None,
        window_size: int = 8,
        window_shift: int = 0,
        attn_drop_p: float = 0.0,
        is_causal: bool = False):
    qgrid = query.grid
    winattn_indexs_key = f"winindexs_{window_size}-{window_shift}"
    winindexs_cache = query.get_spatial_cache(winattn_indexs_key)
    if winindexs_cache is not None:
        grids_idxs_winsort, grids_idxs_winargsort, win_seq_lens = winindexs_cache
    else:
        winindexs_cache = (
            calc_attn_window_partition(qgrid, window_size, window_shift))
        query.register_spatial_cache(winattn_indexs_key, winindexs_cache)
        grids_idxs_winsort, grids_idxs_winargsort, win_seq_lens = winindexs_cache

    if key is None and value is None:
        qkv = query.data.jdata  # (M, 3, nH, d)
        qkv = qkv[grids_idxs_winsort.jdata]
        cu_seqlens = torch.cat(
            (torch.tensor([0]).type_as(win_seq_lens.jdata),
             torch.cumsum(win_seq_lens.jdata, dim=0)), dim=0).int()
        max_seqlen = win_seq_lens.jdata.max().item()
        # (M, nH, d)
        attned_v = flash_attn.flash_attn_varlen_qkvpacked_func(
            qkv, cu_seqlens, max_seqlen,
            dropout_p=attn_drop_p, causal=is_causal)
    else:
        # (M, nH, d)
        q, k, v = query.data.jdata, key.jdata, value.jdata
        assert q.shape[0] == k.shape[0] == v.shape[0]
        q = q[grids_idxs_winsort.jdata]
        k = k[grids_idxs_winsort.jdata]
        v = v[grids_idxs_winsort.jdata]

        cu_seqlens_q = torch.cat(
            (torch.tensor([0]).type_as(win_seq_lens.jdata),
             torch.cumsum(win_seq_lens.jdata, dim=0)), dim=0).int()
        max_seqlen_q = win_seq_lens.jdata.max().item()
        cu_seqlens_k = cu_seqlens_q.clone().int()
        max_seqlen_k = max_seqlen_q
        # (Mq, nH, d)
        attned_v = flash_attn.flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
            dropout_p=attn_drop_p, causal=is_causal)

    attned_v = attned_v[grids_idxs_winargsort.jdata]  # (M, D)
    attned_v = qgrid.jagged_like(attned_v)
    return attned_v
