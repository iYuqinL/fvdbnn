# -*- coding:utf-8 -*-
###
# File: attn.py
# Created Date: Sunday, November 16th 2025, 12:07:44 am
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
import numpy as np
import torch
import torch.nn as nn
import flash_attn
from fvdb import JaggedTensor, GridBatch

from torch.utils.checkpoint import checkpoint as gradient_checkpoint

from ..modules.vdbtensor import fVDBTensor
from ..functional.attn import calc_attn_window_partition
from ..modules.linear import Cast2IntypeLinear
from ..modules.normalize import MultiHeadRMSNorm
from ..posenc import RotaryPositionEncoding
from ..utils.utils import get_gpu_compute_capacity


__all__ = ["MultiheadFlashAttnFVDB", "MultiheadFlashAttnSWinFVDB"]


class MultiheadAttnFVDBBase(nn.Module):
    """
    Args
    ----
    emb_dim (int): Dimension of the input embeddings.

    num_heads (int): Number of attention heads.

    bias (bool): Whether to include bias terms in the projection layers. Default is True.

    kv_bias (bool): Whether to apply separate bias terms to the key and value projections.
    Default is False.

    qk_rmsnorm (bool): Whether to apply RMSNorm to the query and key projections.
    Default is False.

    kdim (int): Dimension of the key projections.
    Defaults to None, meaning it will use `emb_dim`.

    vdim (int): Dimension of the value projections.
    Defaults to None, meaning it will use `emb_dim`.

    sure_cross_attn (bool): Whether to enable cross-attention. Default is False.

    attn_drop (float): Dropout rate applied to the attention weights. Default is 0.0.

    proj_drop (float): Dropout rate applied after the output projection. Default is 0.0.

    device (torch.device): Device on which the tensors will be allocated. Default is None.

    dtype (torch.dtype): Data type used for the tensors. Default is None.
    """

    def __init__(self,
                 emb_dim, num_heads,
                 bias=True, kv_bias=False, qk_rmsnorm=False,
                 kdim=None, vdim=None, sure_cross_attn=False,
                 kv_is_same_token: bool = False,
                 attn_drop=0.0, proj_drop=0.0,
                 use_rope=False,
                 checkpointing=False,
                 device=None, dtype=None):

        nnkwargs = {'device': device, 'dtype': dtype}
        super(MultiheadAttnFVDBBase, self).__init__()
        gpu_cc_major, gpu_cc_minor = get_gpu_compute_capacity(
            torch.device(f"cuda:{torch.cuda.current_device()}"))
        assert gpu_cc_major >= 8, (
            "Please use flashattn v2 attention on GPUs with compute capability >= 8.0.")

        self.checkpointing = checkpointing

        self.emb_dim = emb_dim
        self.kdim = kdim if kdim is not None else emb_dim
        self.vdim = vdim if vdim is not None else emb_dim
        if kv_is_same_token:
            self.vdim = self.kdim

        self._qkv_same_embed_dim = self.kdim == emb_dim and self.vdim == emb_dim
        self.sure_cross_attn = sure_cross_attn
        self.kv_is_same_token = kv_is_same_token

        self.num_heads = num_heads
        self.is_qk_rmsnorm = qk_rmsnorm
        self.attn_drop_p = attn_drop
        self.proj_drop_p = proj_drop
        self.use_rope = use_rope

        self.head_dim = emb_dim // num_heads
        assert self.head_dim*num_heads == self.emb_dim, (
            "emb_dim must be divisible by num_heads")

        self.qk_scale = self.head_dim**-0.5

        if (self._qkv_same_embed_dim is False) or self.sure_cross_attn:
            self.to_q = Cast2IntypeLinear(emb_dim, emb_dim, bias=bias, **nnkwargs)
            if not self.kv_is_same_token:
                self.to_k = Cast2IntypeLinear(self.kdim, emb_dim, bias=kv_bias, **nnkwargs)
                self.to_v = Cast2IntypeLinear(self.vdim, emb_dim, bias=kv_bias, **nnkwargs)
            else:
                self.to_kv = Cast2IntypeLinear(self.kdim, emb_dim*2, bias=kv_bias, **nnkwargs)
        else:
            self.to_qkv = Cast2IntypeLinear(emb_dim, 3*emb_dim, bias=bias, **nnkwargs)

        if self.use_rope:
            self.rope = RotaryPositionEncoding(self.head_dim, spatial_dim=3)

        if self.is_qk_rmsnorm:
            self.q_rmsnorm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rmsnorm = MultiHeadRMSNorm(self.head_dim, num_heads)

        self.attn_drop = nn.Dropout(attn_drop)

        self.out_proj = Cast2IntypeLinear(emb_dim, emb_dim, bias=bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self._reset_parameters()

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"forward method not implemented")

    def _rope(self,
              q: torch.Tensor,
              k: torch.Tensor,
              qgrid: GridBatch,
              kvgrid: GridBatch = None,
              kvindices_scale: float = 1.0,
              kvindices_bias: float = 0.0):
        """
        Args
        ----
        q (tensor): (M, *, nH, d)

        k (tensor): (M, *, nH, d)
        """
        indices = qgrid.ijk.jdata  # (M, 3)
        indices = indices.view([indices.shape[0]] + [1]*(len(q.shape)-2) + [3])
        indices = indices.expand(list(q.shape[:-1]) + [3])
        if kvgrid is None:
            q, k = self.rope(q, k, indices=indices)
        else:
            q = self.rope(q, indices=qgrid.ijk.jdata)
            kindices = kvgrid.ijk.jdata
            kindices = kindices * kvindices_scale + kvindices_bias
            kindices = kindices.view([kindices.shape[0]] + [1]*(len(k.shape)-2) + [3])
            kindices = kindices.expand(list(k.shape[:-1]) + [3])
            k = self.rope(k, indices=kindices)
        return q, k

    def _reset_parameters(self):
        from torch.nn.init import xavier_uniform_, constant_
        if self._qkv_same_embed_dim and (not self.sure_cross_attn):
            xavier_uniform_(self.to_qkv.weight)
            if hasattr(self.to_qkv, "bias") and self.to_qkv.bias is not None:
                constant_(self.to_qkv.bias, 0.)
        elif not self.kv_is_same_token:
            xavier_uniform_(self.to_q.weight)
            xavier_uniform_(self.to_k.weight)
            xavier_uniform_(self.to_v.weight)
            if hasattr(self.to_q, "bias") and self.to_q.bias is not None:
                constant_(self.to_q.bias, 0.)
            if hasattr(self.to_k, "bias") and self.to_k.bias is not None:
                constant_(self.to_k.bias, 0.)
            if hasattr(self.to_v, "bias") and self.to_v.bias is not None:
                constant_(self.to_v.bias, 0.)
        else:
            xavier_uniform_(self.to_q.weight)
            xavier_uniform_(self.to_kv.weight)
            if hasattr(self.to_q, "bias") and self.to_q.bias is not None:
                constant_(self.to_q.bias, 0.)
            if hasattr(self.to_kv, "bias") and self.to_kv.bias is not None:
                constant_(self.to_kv.bias, 0.)

        xavier_uniform_(self.out_proj.weight)
        if hasattr(self.out_proj, "bias") and self.out_proj.bias is not None:
            constant_(self.out_proj.bias, 0.)


class MultiheadFlashAttnFVDB(MultiheadAttnFVDBBase):
    def __init__(self, emb_dim, num_heads,
                 bias=True, kv_bias=False, qk_rmsnorm=False,
                 kdim=None, vdim=None, sure_cross_attn=False,
                 kv_is_same_token: bool = False,
                 attn_drop=0.0, proj_drop=0.0,
                 use_rope=False,
                 checkpointing=False,
                 device=None, dtype=None):

        super(MultiheadFlashAttnFVDB, self).__init__(
            emb_dim, num_heads, bias=bias, kv_bias=kv_bias,
            qk_rmsnorm=qk_rmsnorm, kdim=kdim, vdim=vdim,
            sure_cross_attn=sure_cross_attn,
            kv_is_same_token=kv_is_same_token,
            attn_drop=attn_drop, proj_drop=proj_drop,
            use_rope=use_rope, checkpointing=checkpointing,
            device=device, dtype=dtype)

    def forward(self,
                query: fVDBTensor,
                key: Union[fVDBTensor, JaggedTensor, torch.Tensor] = None,
                value: Union[fVDBTensor, JaggedTensor, torch.Tensor] = None,
                is_causal: bool = False):
        """
        Args
        ----
        query (JaggedTensor): Tensor of shape (B, L*, Dq),
        where L* is the variable sequence length.

        key (JaggedTensor, optional): Tensor of shape (B, L*, Dk).  
        Defaults to None, which indicates that self-attention is used.

        value (JaggedTensor, optional): Tensor of shape (B, L*, Dv).  
        Defaults to None, which indicates that self-attention is used.

        Returns
        -------
        atten_v (JaggedTensor): The attention-weighted values with shape (B, L*, D).
        """
        assert isinstance(query, fVDBTensor)
        qgrid, qdata = query.grid, query.data

        if key is not None and value is not None:
            if (isinstance(key, fVDBTensor) and
                    isinstance(value, fVDBTensor)):
                assert check_same_grid(key.grid, value.grid)
                kvgrid, kdata, vdata = key.grid, key.data, value.data
                kvgrid = None if check_same_grid(kvgrid, qgrid) else kvgrid
            elif (isinstance(key, JaggedTensor) and
                  isinstance(value, JaggedTensor)):
                kvgrid, kdata, vdata = None, key, value
            else:
                kvgrid = None
                kdata = JaggedTensor(key.unbind(dim=0))
                vdata = JaggedTensor(value.unbind(dim=0))
        elif key is not None and value is None:
            assert self.kv_is_same_token, (
                f"kv_is_same_token must True when key is not None and value is None")
            if isinstance(key, fVDBTensor):
                kvgrid, kdata, vdata = key.grid, key.data, None
            elif isinstance(key, JaggedTensor):
                kvgrid, kdata, vdata = None, key, None
            else:
                kvgrid = None
                kdata = JaggedTensor(key.unbind(dim=0))
                vdata = None
        else:
            assert key is None and value is None, (
                f"value must be None when key is None")
            kvgrid, kdata, vdata = None, None, None

        if self.checkpointing and self.training:
            attv = gradient_checkpoint(
                self._forward, qgrid, qdata, kvgrid, kdata, vdata,
                is_causal, use_reentrant=False)
        else:
            attv = self._forward(
                qgrid, qdata, kvgrid, kdata, vdata, is_causal)

        attv = fVDBTensor(qgrid, attv, spatial_cache=query.spatial_cache)

        return attv

    def _forward(self,
                 qgrid: GridBatch,
                 query: JaggedTensor,
                 kvgrid: GridBatch = None,
                 key: JaggedTensor = None,
                 value: JaggedTensor = None,
                 is_causal: bool = False,
                 kvindices_scale: float = 1.0,
                 kvindices_bias: float = 0.0):
        """
        Args
        ----
        qgrid (GridBatch): The FVDB grid batch for the queries.

        query (JaggedTensor): A jagged tensor with lshape [n1, n2, n3, ...]  
        and element shape (Dq,).

        kvgrid (GridBatch): The FVDB grid batch for the keys and values.

        key (JaggedTensor, optional): A jagged tensor with lshape [n1, n2, n3, ...]  
        and element shape (Dk,). Defaults to None, indicating self-attention is used.

        value (JaggedTensor, optional): A jagged tensor with lshape [n1, n2, n3, ...]  
        and element shape (Dv,). Defaults to None, indicating self-attention is used.

        Returns
        -------
        atten_v (JaggedTensor): The attention-weighted output values with shape (B, L*, D).
        """
        B = query.num_tensors
        # return attned_v
        if key is None and value is None:    # self-attention
            assert self.sure_cross_attn is False, (
                "key and value are required for cross-attention.")
            qjdata: torch.Tensor = query.jdata   # torch.Tensor (M, D)
            # qjidxs = query.jidx
            qkv: torch.Tensor = self.to_qkv(qjdata)  # (M, 3*D)
            # (M, 3, nH, d)
            qkv = qkv.view(qkv.shape[0], 3, self.num_heads, self.head_dim)

            if self.use_rope:
                q, k, v = qkv.unbind(dim=1)  # (M, nH, d)
                q, k = self._rope(
                    q, k, qgrid, kvgrid,
                    kvindices_scale=kvindices_scale,
                    kvindices_bias=kvindices_bias)
                qkv = torch.stack((q, k, v), dim=1).contiguous()  # (M, 3, nH, d)

            if self.is_qk_rmsnorm:
                q, k, v = torch.chunk(qkv, 3, dim=1)  # (M, 1, nH, d)
                q = self.q_rmsnorm(q.contiguous())
                k = self.k_rmsnorm(k.contiguous())
                qkv = torch.cat([q, k, v], dim=1).contiguous()

            # to here, qkv is (M, 3, nH, d), M is total totens from all batch
            n_elems_per_inst = torch.as_tensor(query.lshape)
            all_batch_have_same_seqlen = (
                torch.all((n_elems_per_inst - n_elems_per_inst[0]) == 0))
            if all_batch_have_same_seqlen:
                # use flash_attn_qkvpacked_func for higher efficiency
                qkv = qkv.view(query.num_tensors, n_elems_per_inst[0],
                               3, self.num_heads, self.head_dim)
                attned_v = flash_attn.flash_attn_qkvpacked_func(
                    qkv, dropout_p=self.attn_drop_p, causal=is_causal)
                attned_v = attned_v.view(query.num_tensors*n_elems_per_inst[0],
                                         self.num_heads, self.head_dim)
            else:
                cu_seqlens = query.joffsets.int()
                max_seqlen = max(list(query.lshape))
                # (M, nH, d)
                attned_v = flash_attn.flash_attn_varlen_qkvpacked_func(
                    qkv, cu_seqlens, max_seqlen,
                    dropout_p=self.attn_drop_p, causal=is_causal)
        elif key is not None and value is None:
            assert self.kv_is_same_token, (
                "kv_is_same_token must True when key is not None.")
            qjdata = query.jdata   # torch.Tensor (M, D)
            kvjdata = key.jdata
            # (M, nH, d)
            q = self.to_q(qjdata).view(-1, self.num_heads, self.head_dim)
            kv = self.to_kv(kvjdata)
            kv = kv.view(kv.shape[0], 2, self.num_heads, self.head_dim)

            if self.use_rope:
                k, v = kv.unbind(dim=1)  # (M, nH, d)
                q, k = self._rope(
                    q, k, qgrid, kvgrid,
                    kvindices_scale=kvindices_scale,
                    kvindices_bias=kvindices_bias)
                kv = torch.stack((k, v), dim=1).contiguous()  # (M, 2, nH, d)

            if self.is_qk_rmsnorm:
                k, v = torch.chunk(kv, 2, dim=1)  # (M, 1, nH, d)
                q = self.q_rmsnorm(q.contiguous())
                k = self.k_rmsnorm(k.contiguous())
                kv = torch.cat([k, v], dim=1).contiguous()

            # to here, q is (M, nH, d);
            # kv is (M, 2, nH, d), M is total totens from all batch
            n_elems_per_inst_q = torch.as_tensor(query.lshape)
            n_elems_per_inst_k = torch.as_tensor(key.lshape)
            all_batch_have_same_seqlen = (
                torch.all((n_elems_per_inst_q - n_elems_per_inst_q[0]) == 0) and
                torch.all((n_elems_per_inst_k - n_elems_per_inst_k[0]) == 0))
            if all_batch_have_same_seqlen:
                # use flash_attn_qkvpacked_func for higher efficiency
                q = q.view(query.num_tensors, n_elems_per_inst_q[0],
                           self.num_heads, self.head_dim)
                kv = kv.view(key.num_tensors, n_elems_per_inst_k[0],
                             2, self.num_heads, self.head_dim)
                attned_v = flash_attn.flash_attn_kvpacked_func(
                    q, kv, dropout_p=self.attn_drop_p, causal=is_causal)
                attned_v = attned_v.view(query.num_tensors*n_elems_per_inst_q[0],
                                         self.num_heads, self.head_dim)
            else:
                cu_seqlens_q = query.joffsets.int()
                max_seqlen_q = max(list(query.lshape))
                cu_seqlens_k = key.joffsets.int()
                max_seqlen_k = max(list(key.lshape))
                # (M, nH, d)
                attned_v = flash_attn.flash_attn_varlen_kvpacked_func(
                    q, kv,
                    cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
                    dropout_p=self.attn_drop_p, causal=is_causal)
        else:
            assert not self.kv_is_same_token, (
                "kv_is_same_token must False when key and value are not None.")
            # (M, Dq), (M, Dk), (M, Dv)
            qjdata, kjdata, vjdata = query.jdata, key.jdata, value.jdata
            # (M, nH, d)
            q = self.to_q(qjdata).view(-1, self.num_heads, self.head_dim)
            k = self.to_k(kjdata).view(-1, self.num_heads, self.head_dim)
            v = self.to_v(vjdata).view(-1, self.num_heads, self.head_dim)

            if self.use_rope:
                q, k = self._rope(
                    q, k, qgrid, kvgrid,
                    kvindices_scale=kvindices_scale, kvindices_bias=kvindices_bias)

            if self.is_qk_rmsnorm:
                q = self.q_rmsnorm(q)
                k = self.k_rmsnorm(k)

            n_elems_per_inst_q = torch.as_tensor(query.lshape)
            n_elems_per_inst_k = torch.as_tensor(key.lshape)
            n_elems_per_inst_v = torch.as_tensor(value.lshape)
            assert torch.all(n_elems_per_inst_k == n_elems_per_inst_v), (
                "key and value must have the same sequence length.")
            all_batch_have_same_seqlen = (
                torch.all((n_elems_per_inst_q - n_elems_per_inst_q[0]) == 0) and
                torch.all((n_elems_per_inst_k - n_elems_per_inst_k[0]) == 0))
            if all_batch_have_same_seqlen:
                q = q.view(query.num_tensors, n_elems_per_inst_q[0],
                           self.num_heads, self.head_dim)
                k = k.view(key.num_tensors, n_elems_per_inst_k[0],
                           self.num_heads, self.head_dim)
                v = v.view(value.num_tensors, n_elems_per_inst_k[0],
                           self.num_heads, self.head_dim)
                attned_v = flash_attn.flash_attn_func(
                    q, k, v, dropout_p=self.attn_drop_p, causal=is_causal)
                attned_v = attned_v.view(query.num_tensors*n_elems_per_inst_q[0],
                                         self.num_heads, self.head_dim)
            else:
                cu_seqlens_q = query.joffsets.int()
                max_seqlen_q = max(list(query.lshape))
                cu_seqlens_k = key.joffsets.int()
                max_seqlen_k = max(list(key.lshape))
                # (Mq, nH, d)
                attned_v = flash_attn.flash_attn_varlen_func(
                    q, k, v,
                    cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
                    dropout_p=self.attn_drop_p, causal=is_causal)

        attned_v = attned_v.view(attned_v.shape[0], self.emb_dim)  # (M, D)
        attned_v = self.out_proj(attned_v)
        attned_v = self.proj_drop(attned_v)

        attned_v = qgrid.jagged_like(attned_v)
        return attned_v


class MultiheadFlashAttnSWinFVDB(MultiheadAttnFVDBBase):
    def __init__(self, emb_dim, num_heads,
                 window_size: Union[int, Tuple[int, ...]],
                 window_shift: Union[int, Tuple[int, ...]],
                 bias=True, kv_bias=False, qk_rmsnorm=False,
                 kdim=None, vdim=None, sure_cross_attn=False,
                 attn_drop=0.0, proj_drop=0.0,
                 use_rope=False,
                 checkpointing=False,
                 device=None, dtype=None):

        super(MultiheadFlashAttnSWinFVDB, self).__init__(
            emb_dim, num_heads, bias=bias, kv_bias=kv_bias,
            qk_rmsnorm=qk_rmsnorm, kdim=kdim, vdim=vdim,
            sure_cross_attn=sure_cross_attn,
            attn_drop=attn_drop, proj_drop=proj_drop,
            use_rope=use_rope, checkpointing=checkpointing,
            device=device, dtype=dtype)

        self.window_size = window_size
        self.window_shift = window_shift

        window_size = ((window_size, ) * 3
                       if isinstance(window_size, int) else window_size)
        assert len(window_size) == 3, "window_size must be 3-dim"
        self.max_window_seq_len = np.prod(window_size)

    def forward(self, query: fVDBTensor,
                key: fVDBTensor = None,
                value: fVDBTensor = None,
                is_causal: bool = False):
        """
        Args
        ----
        query (fVDBTensor): A tensor with  lshape [n1, n2, n3, ...]  
        and element shape (Dq,).

        key (fVDBTensor, optional): A tensor with lshape [n1, n2, n3, ...]  
        and element shape (Dk,). Defaults to None, indicating self-attention is applied.

        value (fVDBTensor, optional): A tensor with lshape [n1, n2, n3, ...]  
        and element shape (Dv,). Defaults to None, indicating self-attention is applied.

        Returns
        -------
        atten_v (fVDBTensor): The attention-weighted output tensor with  
        lshape [n1, n2, n3, ...] and element shape (Dv,).
        """
        assert isinstance(query, fVDBTensor)
        assert key is None or isinstance(key, fVDBTensor)
        assert value is None or isinstance(value, fVDBTensor)
        if key is not None or value is not None:
            # assert key is not None and value is not None
            assert check_same_grid(query.grid, key.grid)
            if value is not None:
                assert check_same_grid(query.grid, value.grid)
        if key is None:
            assert value is None, "value must be None when key is None"

        grids = query.grid
        winattn_indexs_key = f"winindexs_{self.window_size}-{self.window_shift}"
        winindexs_cache = query.get_spatial_cache(winattn_indexs_key)
        if winindexs_cache is not None:
            grids_idxs_winsort, grids_idxs_winargsort, win_seq_lens = winindexs_cache
        else:
            winindexs_cache = (
                calc_attn_window_partition(grids, self.window_size, self.window_shift))
            query.register_spatial_cache(winattn_indexs_key, winindexs_cache)
            grids_idxs_winsort, grids_idxs_winargsort, win_seq_lens = winindexs_cache
        spatial_cache = query.spatial_cache

        query = query.data
        kvgrid = None
        key = key.data if key is not None else None
        value = value.data if value is not None else None

        if self.checkpointing and self.training:
            attv = gradient_checkpoint(
                self._forward, grids, query, kvgrid, key, value,
                grids_idxs_winsort, grids_idxs_winargsort, win_seq_lens,
                is_causal, use_reentrant=False)
        else:
            attv = self._forward(
                grids, query, kvgrid, key, value,
                grids_idxs_winsort, grids_idxs_winargsort, win_seq_lens,
                is_causal)

        attv = fVDBTensor(grids, attv, spatial_cache=spatial_cache)
        return attv

    def _forward(self,
                 qgrid: GridBatch,
                 query: JaggedTensor,
                 kvgrid: GridBatch = None,
                 key: JaggedTensor = None,
                 value: JaggedTensor = None,
                 grids_idxs_winsort: JaggedTensor = None,
                 grids_idxs_winargsort: JaggedTensor = None,
                 win_seq_lens: JaggedTensor = None,
                 is_causal: bool = False):
        """
        Args
        ----
        query (JaggedTensor): A jagged tensor with lshape [n1, n2, n3, ...]  
        and element shape (Dq,).

        key (JaggedTensor, optional): A jagged tensor with lshape [n1, n2, n3, ...]  
        and element shape (Dk,). Defaults to None, indicating self-attention is applied.

        value (JaggedTensor, optional): A jagged tensor with lshape [n1, n2, n3, ...]  
        and element shape (Dv,). Defaults to None, indicating self-attention is applied.

        grids_idxs_winsort (JaggedTensor): A jagged tensor with lshape [n1, n2, ...]  
        containing grid indices sorted by the index of the window they belong to.

        grids_idxs_winargsort (JaggedTensor): A jagged tensor with lshape [n1, n2, ...]  
        containing the indices used to recover the original order of the grid  
        jagged tensor from the window-sorted version.

        win_seq_lens (JaggedTensor): A jagged tensor with lshape [nw1, nw2, ...]  
        representing the voxel sequence length of each window.

        Returns
        -------
        atten_v (JaggedTensor): The attention-weighted output values with  
        lshape [n1, n2, n3, ...] and element shape (Dv,).
        """
        B = query.num_tensors

        if key is None and value is None:    # self-attention
            assert self.sure_cross_attn is False, (
                "key and value are required for cross-attention.")
            qjdata: torch.Tensor = query.jdata   # torch.Tensor (M, D)
            qkv: torch.Tensor = self.to_qkv(qjdata)  # (M, 3*D)
            # (M, 3, nH, d)
            qkv = qkv.view(qkv.shape[0], 3, self.num_heads, self.head_dim)

            if self.use_rope:
                q, k, v = qkv.unbind(dim=1)  # (M, nH, d)
                q, k = self._rope(q, k, qgrid, kvgrid)
                qkv = torch.stack((q, k, v), dim=1).contiguous()

            if self.is_qk_rmsnorm:
                q, k, v = torch.chunk(qkv, 3, dim=1)  # (M, 1, nH, d)
                q = self.q_rmsnorm(q.contiguous())
                k = self.k_rmsnorm(k.contiguous())
                qkv = torch.cat([q, k, v], dim=1).contiguous()

            qkv = qkv[grids_idxs_winsort.jdata]     # (M, 3*D)
            # to here, qkv is (M, 3, nH, d), M is total totens from all batch
            cu_seqlens = torch.cat(
                (torch.tensor([0]).type_as(win_seq_lens.jdata),
                 torch.cumsum(win_seq_lens.jdata, dim=0)), dim=0).int()
            max_seqlen = win_seq_lens.jdata.max().item()
            assert max_seqlen <= self.max_window_seq_len, (
                f"max_seqlen ({max_seqlen}) must be <= "
                f"max_window_seq_len ({self.max_window_seq_len})")
            # (M, nH, d)
            attned_v = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv, cu_seqlens, max_seqlen,
                dropout_p=self.attn_drop_p, causal=is_causal)

        elif key is not None and value is None:
            assert self.kv_is_same_token, (
                "kv_is_same_token must True when key is not None.")
            # (M, Dq), (M, Dk), (M, Dv)
            qjdata, kvjdata = query.jdata, key.jdata
            # (M, nH, d)
            q = self.to_q(qjdata).view(-1, self.num_heads, self.head_dim)
            kv = self.to_kv(kvjdata)
            kv = kv.view(kv.shape[0], 2, self.num_heads, self.head_dim)

            if self.use_rope:
                k, v = kv.unbind(dim=1)  # (M, nH, d)
                q, k = self._rope(q, k, qgrid, kvgrid)
                kv = torch.stack((k, v), dim=1).contiguous()  # (M, 2, nH, d)

            if self.is_qk_rmsnorm:
                k, v = torch.chunk(kv, 2, dim=1)  # (M, 1, nH, d)
                q = self.q_rmsnorm(q.contiguous())
                k = self.k_rmsnorm(k.contiguous())
                kv = torch.cat([k, v], dim=1).contiguous()

            q = q[grids_idxs_winsort.jdata]
            kv = kv[grids_idxs_winsort.jdata]

            cu_seqlens_q = torch.cat(
                (torch.tensor([0]).type_as(win_seq_lens.jdata),
                    torch.cumsum(win_seq_lens.jdata, dim=0)), dim=0).int()
            max_seqlen_q = win_seq_lens.jdata.max().item()
            cu_seqlens_k = cu_seqlens_q.clone().int()
            max_seqlen_k = max_seqlen_q
            assert max_seqlen_k <= self.max_window_seq_len, (
                f"max_seqlen_k ({max_seqlen_k}) must be <= "
                f"max_window_seq_len ({self.max_window_seq_len})")

            # (Mq, nH, d)
            attned_v = flash_attn.flash_attn_varlen_kvpacked_func(
                q, kv,
                cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
                dropout_p=self.attn_drop_p, causal=is_causal)

        else:
            assert not self.kv_is_same_token, (
                "kv_is_same_token must False when key and value are not None.")
            # (M, Dq), (M, Dk), (M, Dv)
            qjdata, kjdata, vjdata = query.jdata, key.jdata, value.jdata
            # (M, nH, d)
            q = self.to_q(qjdata).view(-1, self.num_heads, self.head_dim)
            k = self.to_k(kjdata).view(-1, self.num_heads, self.head_dim)
            v = self.to_v(vjdata).view(-1, self.num_heads, self.head_dim)

            if self.use_rope:
                q, k = self._rope(q, k, qgrid, kvgrid)

            if self.is_qk_rmsnorm:
                q = self.q_rmsnorm(q)
                k = self.k_rmsnorm(k)

            q = q[grids_idxs_winsort.jdata]
            k = k[grids_idxs_winsort.jdata]
            v = v[grids_idxs_winsort.jdata]

            cu_seqlens_q = torch.cat(
                (torch.tensor([0]).type_as(win_seq_lens.jdata),
                    torch.cumsum(win_seq_lens.jdata, dim=0)), dim=0).int()
            max_seqlen_q = win_seq_lens.jdata.max().item()
            cu_seqlens_k = cu_seqlens_q.clone().int()
            max_seqlen_k = max_seqlen_q
            assert max_seqlen_k <= self.max_window_seq_len, (
                f"max_seqlen_k ({max_seqlen_k}) must be <= "
                f"max_window_seq_len ({self.max_window_seq_len})")
            # (Mq, nH, d)
            attned_v = flash_attn.flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
                dropout_p=self.attn_drop_p, causal=is_causal)

        attned_v = attned_v.view(attned_v.shape[0], self.emb_dim)  # (M, D)
        attned_v = self.out_proj(attned_v)
        attned_v = self.proj_drop(attned_v)

        attned_v = attned_v[grids_idxs_winargsort.jdata]  # (M, D)

        attned_v = qgrid.jagged_like(attned_v)
        return attned_v


def check_same_grid(grid1: GridBatch, grid2: GridBatch, roughly=True):
    if (grid1.total_voxels != grid2.total_voxels):
        return False
    if grid1.grid_count != grid2.grid_count:
        return False
    if not torch.allclose(grid1.voxel_sizes, grid2.voxel_sizes):
        return False
    if not torch.allclose(grid1.origins, grid2.origins):
        return False
    if roughly:
        return True

    if not torch.all(grid1.jidx == grid2.jidx):
        return False
    if not torch.all(grid1.joffsets == grid2.joffsets):
        return False
    if not torch.all(grid1.ijk.jdata == grid2.ijk.jdata):
        return False

    return True
