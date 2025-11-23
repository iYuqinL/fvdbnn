# -*- coding:utf-8 -*-
###
# File: transformer.py
# Created Date: Sunday, November 16th 2025, 10:54:32 am
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
from typing import Union, List, Literal
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

import fvdb

from ..modules.activation import GELUFVDB
from ..layers.attn import MultiheadFlashAttnFVDB
from ..layers.attn import MultiheadFlashAttnSWinFVDB
from ..layers.ffn import FeedForwardNetworkFVDB
from ..modules.normalize import LayerNorm32FVDB

from ..modules.vdbtensor import fVDBTensor


class SelfAttnTransformerBlockFVDB(nn.Module):
    def __init__(self,
                 emb_dim: int,
                 num_heads: int,
                 bias: bool = False,
                 kv_bias: bool = False,
                 qk_rmsnorm: bool = False,
                 ln_affine: Union[bool, List[bool]] = False,
                 attn_dropout: float = 0.0,
                 attn_outproj_dropout: float = 0.0,
                 attn_mode: Literal["full", "swin"] = "full",
                 attn_window_size: int = None,
                 attn_window_shift: int = None,
                 ffn_ratio: float = 4.0,
                 ffn_outdim: int = None,
                 ffn_bias: bool = False,
                 ffn_dropout: float = 0.0,
                 ffn_activation: nn.Module = lambda: GELUFVDB(approximate="tanh"),
                 use_rope: bool = False,
                 checkpointing: bool = False,
                 device: torch.device = None,
                 dtype: torch.dtype = None,
                 eps: float = 1e-6,
                 **kwargs):
        super(SelfAttnTransformerBlockFVDB, self).__init__(**kwargs)
        self.checkpointing = checkpointing
        self.attn_mode = attn_mode
        self.attn_window_size = attn_window_size
        self.attn_window_shift = attn_window_shift

        if attn_mode == "swin":
            assert (attn_window_size is not None and attn_window_shift is not None), (
                "attn_window_size and attn_window_shift must be specified for swin attn")

        if not isinstance(ln_affine, (tuple, list)):
            assert isinstance(ln_affine, bool), "ln_affine must be bool or ist of bool"
            ln_affine = [ln_affine] * 3
        assert len(ln_affine) >= 2, (
            f"ln_affine must have at least 2 elements, got {ln_affine}")
        self.norm1 = LayerNorm32FVDB(emb_dim, elementwise_affine=ln_affine[0], eps=eps)
        if attn_mode == "full":
            self.attn = MultiheadFlashAttnFVDB(
                emb_dim,
                num_heads=num_heads,
                bias=bias,
                kv_bias=kv_bias,
                qk_rmsnorm=qk_rmsnorm,
                attn_drop=attn_dropout,
                proj_drop=attn_outproj_dropout,
                use_rope=use_rope,
                device=device,
                dtype=dtype
            )
        elif attn_mode == "swin":
            self.attn = MultiheadFlashAttnSWinFVDB(
                emb_dim,
                num_heads=num_heads,
                window_size=attn_window_size,
                window_shift=attn_window_shift,
                bias=bias,
                kv_bias=kv_bias,
                qk_rmsnorm=qk_rmsnorm,
                attn_drop=attn_dropout,
                proj_drop=attn_outproj_dropout,
                use_rope=use_rope,
                device=device,
                dtype=dtype
            )
        else:
            raise ValueError(f"attn_mode must be 'full' or 'swin', got {attn_mode}")

        self.norm2 = LayerNorm32FVDB(emb_dim, elementwise_affine=ln_affine[1], eps=eps)
        self.mlp = FeedForwardNetworkFVDB(
            in_features=emb_dim,
            hidden_ratio=ffn_ratio,
            bias=ffn_bias,
            out_features=ffn_outdim,
            activation=ffn_activation,
            dropout=ffn_dropout,
            device=device,
            dtype=dtype
        )

    def forward(self, x: fVDBTensor) -> fVDBTensor:
        grid, data, spatial_cache = x.grid, x.data, x.spatial_cache
        if self.checkpointing and self.training:
            grid, data, spatial_cache = gradient_checkpoint(
                self._forward, grid, data, spatial_cache, use_reentrant=False)
        else:
            grid, data, spatial_cache = self._forward(grid, data, spatial_cache)

        x = fVDBTensor(grid, data, spatial_cache)
        return x

    def _forward(self,
                 grid: fvdb.GridBatch,
                 data: fvdb.JaggedTensor,
                 spatial_cache: dict = None):
        x = fVDBTensor(grid, data, spatial_cache)

        h = self.norm1(x)
        h = self.attn(h)
        x = x + h

        h = self.norm2(x)
        h = self.mlp(h)
        x: fVDBTensor = x + h

        grid, data, spatial_cache = x.grid, x.data, x.spatial_cache
        return grid, data, spatial_cache


class ModulateSelfAttnTransformerBlockFVDB(nn.Module):
    def __init__(self,
                 emb_dim: int,
                 num_heads: int,
                 bias: bool = False,
                 kv_bias: bool = False,
                 qk_rmsnorm: bool = False,
                 ln_affine: Union[bool, List[bool]] = False,
                 attn_dropout: float = 0.0,
                 attn_outproj_dropout: float = 0.0,
                 attn_mode: Literal["full", "swin"] = "full",
                 attn_window_size: int = None,
                 attn_window_shift: int = None,
                 ffn_ratio: float = 4.0,
                 ffn_outdim: int = None,
                 ffn_bias: bool = False,
                 ffn_dropout: float = 0.0,
                 ffn_activation: nn.Module = lambda: GELUFVDB(approximate="tanh"),
                 use_rope: bool = False,
                 share_modulate: bool = False,
                 checkpointing: bool = False,
                 device: torch.device = None,
                 dtype: torch.dtype = None,
                 eps: float = 1e-6,
                 **kwargs):
        super(ModulateSelfAttnTransformerBlockFVDB, self).__init__(**kwargs)
        self.checkpointing = checkpointing
        self.share_modulate = share_modulate
        self.attn_mode = attn_mode
        self.attn_window_size = attn_window_size
        self.attn_window_shift = attn_window_shift

        if attn_mode == "swin":
            assert (attn_window_size is not None and attn_window_shift is not None), (
                "attn_window_size and attn_window_shift must be specified for swin attn")

        if not isinstance(ln_affine, (tuple, list)):
            assert isinstance(ln_affine, bool), "ln_affine must be bool or ist of bool"
            ln_affine = [ln_affine] * 3
        assert len(ln_affine) >= 2, (
            f"ln_affine must have at least 2 elements, got {ln_affine}")

        self.norm1 = LayerNorm32FVDB(emb_dim, elementwise_affine=ln_affine[0], eps=eps)
        if attn_mode == "full":
            self.attn = MultiheadFlashAttnFVDB(
                emb_dim,
                num_heads=num_heads,
                bias=bias,
                kv_bias=kv_bias,
                qk_rmsnorm=qk_rmsnorm,
                attn_drop=attn_dropout,
                proj_drop=attn_outproj_dropout,
                use_rope=use_rope,
                device=device,
                dtype=dtype
            )
        elif attn_mode == "swin":
            self.attn = MultiheadFlashAttnSWinFVDB(
                emb_dim,
                num_heads=num_heads,
                window_size=attn_window_size,
                window_shift=attn_window_shift,
                bias=bias,
                kv_bias=kv_bias,
                qk_rmsnorm=qk_rmsnorm,
                attn_drop=attn_dropout,
                proj_drop=attn_outproj_dropout,
                use_rope=use_rope,
                device=device,
                dtype=dtype
            )
        else:
            raise ValueError(f"attn_mode must be 'full' or 'swin', got {attn_mode}")

        self.norm2 = LayerNorm32FVDB(emb_dim, elementwise_affine=ln_affine[1], eps=eps)
        self.mlp = FeedForwardNetworkFVDB(
            in_features=emb_dim,
            hidden_ratio=ffn_ratio,
            bias=ffn_bias,
            out_features=ffn_outdim,
            activation=ffn_activation,
            dropout=ffn_dropout,
            device=device,
            dtype=dtype
        )

        if not self.share_modulate:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(emb_dim, 6 * emb_dim, bias=True)
            )

    def forward(self, x: fVDBTensor, mod: torch.Tensor,
                mod_shape_has_been_processed: bool = False) -> fVDBTensor:
        grid, data, spatial_cache = x.grid, x.data, x.spatial_cache
        if self.checkpointing and self.training:
            grid, data, spatial_cache = gradient_checkpoint(
                self._forward, grid, data, mod, spatial_cache,
                mod_shape_has_been_processed, use_reentrant=False)
        else:
            grid, data, spatial_cache = self._forward(
                grid, data, mod, spatial_cache,
                mod_shape_has_been_processed=mod_shape_has_been_processed)

        x = fVDBTensor(grid, data, spatial_cache)
        return x

    def _forward(self, grid: fvdb.GridBatch,
                 data: fvdb.JaggedTensor,
                 mod: torch.Tensor,
                 spatial_cache: dict = None,
                 mod_shape_has_been_processed: bool = False):
        if not self.share_modulate:
            mod = self.adaLN_modulation(mod)  # (B, 6*emb_dim) or ([n1,n2,...], 6*emb_dim)

        num_elems_inst = torch.as_tensor(data.lshape)
        all_inst_same_seqlen = (
            torch.all((num_elems_inst - num_elems_inst[0]) == 0))
        if not mod_shape_has_been_processed and not all_inst_same_seqlen:
            joffsets = data.joffsets
            counts = (joffsets[1:] - joffsets[:-1])
            counts = counts.to(dtype=torch.long, device=mod.device)
            # (B, C) --> (N, C) where N = sum(counts)
            mod_ = mod.repeat_interleave(counts, dim=0)
            mod = mod_

        (shift_msa, scale_msa, gate_msa,
         shift_mlp, scale_mlp, gate_mlp) = mod.chunk(6, dim=-1)

        x = fVDBTensor(grid, data, spatial_cache)
        h = self.norm1(x)
        h = modulate_scale_shift(
            h, scale_msa, shift_msa, all_inst_same_seqlen, num_elems_inst,
            force_directly_modulate=mod_shape_has_been_processed)
        h = self.attn(h)
        h = modulate_gate(
            h, gate_msa, all_inst_same_seqlen, num_elems_inst,
            force_directly_modulate=mod_shape_has_been_processed)

        x = x + h

        h = self.norm2(x)
        h = modulate_scale_shift(
            h, scale_mlp, shift_mlp, all_inst_same_seqlen, num_elems_inst,
            force_directly_modulate=mod_shape_has_been_processed)
        h = self.mlp(h)
        h = modulate_gate(
            h, gate_mlp, all_inst_same_seqlen, num_elems_inst,
            force_directly_modulate=mod_shape_has_been_processed)
        x = x + h

        return x.grid, x.data, x.spatial_cache


class ModulateCrossAttnTransformerBlockFVDB(nn.Module):
    def __init__(self,
                 emb_dim: int,
                 context_emb_dim: int,
                 num_heads: int,
                 bias: bool = True,
                 kv_bias: bool = True,
                 qk_rmsnorm: bool = False,
                 qk_rmsnorm_cross: bool = False,
                 ln_affine: Union[bool, list[bool]] = False,
                 attn_dropout: float = 0.0,
                 attn_outproj_dropout: float = 0.0,
                 self_attn_mode: Literal["full", "swin"] = "full",
                 self_attn_window_size: int = None,
                 self_attn_window_shift: int = None,
                 cross_attn_mode: Literal["full", "swin"] = "full",
                 cross_attn_window_size: int = None,
                 cross_attn_window_shift: int = None,
                 ffn_ratio: float = 4.0,
                 ffn_outdim: int = None,
                 ffn_bias: bool = True,
                 ffn_dropout: float = 0.0,
                 ffn_activation: nn.Module = lambda: GELUFVDB(approximate="tanh"),
                 use_rope: bool = False,
                 share_modulate: bool = False,
                 checkpointing: bool = False,
                 device: torch.device = None,
                 dtype: torch.dtype = None,
                 eps: float = 1e-6,
                 **kwargs):
        super(ModulateCrossAttnTransformerBlockFVDB, self).__init__(**kwargs)
        self.checkpointing = checkpointing
        self.share_modulate = share_modulate
        self.self_attn_mode = self_attn_mode
        self.cross_attn_mode = cross_attn_mode
        self.self_attn_window_size = self_attn_window_size
        self.self_attn_window_shift = self_attn_window_shift
        self.cross_attn_window_size = cross_attn_window_size
        self.cross_attn_window_shift = cross_attn_window_shift

        if not isinstance(ln_affine, (list, tuple)):
            ln_affine = [ln_affine] * 3
        assert len(ln_affine) >= 3, (
            f"ln_affine must have at least 3 elements, got {ln_affine}")

        self.norm1 = LayerNorm32FVDB(emb_dim, elementwise_affine=ln_affine[0], eps=eps)
        if self_attn_mode == "full":
            self.self_attn = MultiheadFlashAttnFVDB(
                emb_dim,
                num_heads=num_heads,
                bias=bias,
                kv_bias=kv_bias,
                qk_rmsnorm=qk_rmsnorm,
                attn_drop=attn_dropout,
                proj_drop=attn_outproj_dropout,
                use_rope=use_rope,
                device=device,
                dtype=dtype
            )
        elif self_attn_mode == "swin":
            self.self_attn = MultiheadFlashAttnSWinFVDB(
                emb_dim,
                num_heads=num_heads,
                window_size=self_attn_window_size,
                window_shift=self_attn_window_shift,
                bias=bias,
                kv_bias=kv_bias,
                qk_rmsnorm=qk_rmsnorm,
                attn_drop=attn_dropout,
                proj_drop=attn_outproj_dropout,
                use_rope=use_rope,
                device=device,
                dtype=dtype
            )
        else:
            raise ValueError(
                f"self_attn_mode must be 'full' or 'swin', got {self_attn_mode}")

        self.norm2 = LayerNorm32FVDB(emb_dim, elementwise_affine=ln_affine[1], eps=eps)
        assert cross_attn_mode in ["full"], (
            f"cross_attn_mode only support 'full' now, got {cross_attn_mode}")
        self.cross_attn = MultiheadFlashAttnFVDB(
            emb_dim,
            num_heads=num_heads,
            bias=bias,
            kv_bias=kv_bias,
            kdim=context_emb_dim,
            vdim=context_emb_dim,
            sure_cross_attn=True,
            kv_is_same_token=True,
            qk_rmsnorm=qk_rmsnorm_cross,
            attn_drop=attn_dropout,
            proj_drop=attn_outproj_dropout,
            use_rope=use_rope,
            device=device,
            dtype=dtype
        )

        self.norm3 = LayerNorm32FVDB(emb_dim, elementwise_affine=ln_affine[2], eps=eps)
        self.mlp = FeedForwardNetworkFVDB(
            emb_dim,
            hidden_ratio=ffn_ratio,
            bias=ffn_bias,
            out_features=ffn_outdim,
            dropout=ffn_dropout,
            activation=ffn_activation,
            device=device,
            dtype=dtype
        )

        if not self.share_modulate:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(emb_dim, 6 * emb_dim, bias=True)
            )

    def forward(self,
                x: fVDBTensor,
                mod: torch.Tensor,
                context: Union[fVDBTensor, fvdb.JaggedTensor, torch.Tensor],
                mod_shape_has_been_processed: bool = False):
        grid, data, spatial_cache = x.grid, x.data, x.spatial_cache
        if self.checkpointing and self.training:
            grid, data, spatial_cache = gradient_checkpoint(
                self._forward, grid, data, mod, context, spatial_cache,
                mod_shape_has_been_processed, use_reentrant=False)
        else:
            grid, data, spatial_cache = self._forward(
                grid, data, mod, context, spatial_cache, mod_shape_has_been_processed)

        x = fVDBTensor(grid, data, spatial_cache)
        return x

    def _forward(self,
                 grid: fvdb.GridBatch,
                 data: fvdb.JaggedTensor,
                 mod: torch.Tensor,
                 context: Union[fVDBTensor, fvdb.JaggedTensor, torch.Tensor],
                 spatial_cache: dict = None,
                 mod_shape_has_been_processed: bool = False):
        if not self.share_modulate:
            mod = self.adaLN_modulation(mod)  # (B, 6*emb_dim) or ([n1,n2,..], 6*emb_dim)

        num_elems_inst = torch.as_tensor(data.lshape)
        all_inst_same_seqlen = (
            torch.all((num_elems_inst - num_elems_inst[0]) == 0))
        if not mod_shape_has_been_processed and not all_inst_same_seqlen:
            joffsets = data.joffsets
            counts = (joffsets[1:] - joffsets[:-1])
            counts = counts.to(dtype=torch.long, device=mod.device)
            # (B, C) --> (N, C) where N = sum(counts)
            mod_ = mod.repeat_interleave(counts, dim=0)
            mod = mod_

        (shift_msa, scale_msa, gate_msa,
         shift_mlp, scale_mlp, gate_mlp) = mod.chunk(6, dim=-1)

        x = fVDBTensor(grid, data, spatial_cache)
        h = self.norm1(x)
        h = modulate_scale_shift(
            h, scale_msa, shift_msa, all_inst_same_seqlen, num_elems_inst,
            force_directly_modulate=mod_shape_has_been_processed)
        h = self.self_attn(h)
        h = modulate_gate(
            h, gate_msa, all_inst_same_seqlen, num_elems_inst,
            force_directly_modulate=mod_shape_has_been_processed)

        x = x + h

        h = self.norm2(x)
        h = self.cross_attn(h, context)   # q: h, kv: context

        x = x + h

        h = self.norm3(x)
        h = modulate_scale_shift(
            h, scale_mlp, shift_mlp, all_inst_same_seqlen, num_elems_inst,
            force_directly_modulate=mod_shape_has_been_processed)
        h = self.mlp(h)
        h = modulate_gate(
            h, gate_mlp, all_inst_same_seqlen, num_elems_inst,
            force_directly_modulate=mod_shape_has_been_processed)

        x = x + h

        return x.grid, x.data, x.spatial_cache


def modulate_scale_shift(
        h: Union[fVDBTensor, fvdb.JaggedTensor],
        scale: torch.Tensor,
        shift: torch.Tensor,
        all_inst_same_seqlen: bool = None,
        num_elems_inst: torch.Tensor = None,
        force_directly_modulate: bool = False):
    if force_directly_modulate:
        return h * (scale + 1) + shift

    if num_elems_inst is None:
        num_elems_inst = torch.as_tensor(h.lshape)
    if all_inst_same_seqlen is None:
        all_inst_same_seqlen = (
            torch.all((num_elems_inst - num_elems_inst[0]) == 0))

    if all_inst_same_seqlen:
        hjdata = h.jdata.view(h.num_tensors, num_elems_inst[0], *h.eshape)
        hjdata = hjdata * (scale.unsqueeze(1) + 1) + shift.unsqueeze(1)
        hdata = h.grid.jagged_like(hjdata.reshape(-1, *h.eshape))
        h = fVDBTensor(h.grid, hdata, h.spatial_cache)
    else:
        # for this case, scale & shift must be repeat interleaved
        h = h * (scale + 1) + shift

    return h


def modulate_gate(
        h: Union[fVDBTensor, fvdb.JaggedTensor],
        gate: torch.Tensor,
        all_inst_same_seqlen: bool = None,
        num_elems_inst: torch.Tensor = None,
        force_directly_modulate: bool = False):
    if force_directly_modulate:
        return h * gate

    if num_elems_inst is None:
        num_elems_inst = torch.as_tensor(h.lshape)
    if all_inst_same_seqlen is None:
        all_inst_same_seqlen = (
            torch.all((num_elems_inst - num_elems_inst[0]) == 0))

    if all_inst_same_seqlen:
        hjdata = h.jdata.view(h.num_tensors, num_elems_inst[0], *h.eshape)
        hjdata = hjdata * gate.unsqueeze(1)
        hdata = h.grid.jagged_like(hjdata.reshape(-1, *h.eshape))
        h = fVDBTensor(h.grid, hdata, h.spatial_cache)
    else:
        # for this case, gate must be repeat interleaved
        h = h * gate

    return h
