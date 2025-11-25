# -*- coding:utf-8 -*-
###
# File: ffn.py
# Created Date: Sunday, November 16th 2025, 1:35:03 am
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
import torch.nn as nn
import torch.nn.functional as F

from ..modules.vdbtensor import fVDBTensor
from ..modules.activation import GELUFVDB
from ..modules.linear import LinearFVDB
from ..modules.dropout import DropoutFVDB


__all__ = ["FeedForwardNetworkFVDB"]


class FeedForwardNetworkFVDB(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int = None,
                 hidden_ratio: float = None,
                 hidden_features: int = None,
                 bias: bool = True,
                 activation: nn.Module = lambda: GELUFVDB(approximate="tanh"),
                 dropout: float = 0.0,
                 device: torch.device = None,
                 dtype: torch.dtype = None):
        """
        FeedForwardNetworkFVDB is a module that implements the feedforward network
        in the Transformer model. It consists of two linear layers with a
        non-linear activation function in between.

        Args
        -----
        in_features (int): The number of input features.

        out_features (int, optional): The number of output features. Defaults to None.

        hidden_ratio (float, optional): The ratio of the hidden features to
        the input features. Defaults to None.

        hidden_features (int, optional): The number of hidden features. Defaults to None.

        bias (bool, optional): Whether to use bias in the linear layers. Defaults to True.

        activation (nn.Module, optional): The activation function to use.
        Defaults to GELU(approximate="tanh").

        dropout (float, optional): The dropout probability. Defaults to 0.0.
        """
        nnkwargs = {"device": device, "dtype": dtype}
        super(FeedForwardNetworkFVDB, self).__init__()

        out_features = out_features if out_features is not None else in_features
        if hidden_features is None and hidden_ratio is None:
            hidden_features = in_features
        elif hidden_features is None:
            hidden_features = int(in_features * hidden_ratio)
        assert isinstance(hidden_features, int), "hidden_features must be an integer"

        self.mlp = nn.Sequential(
            LinearFVDB(in_features, hidden_features, bias=bias, **nnkwargs),
            activation(),
            LinearFVDB(hidden_features, out_features, bias=bias, **nnkwargs),
            DropoutFVDB(dropout)
        )

    def forward(self, x: fVDBTensor) -> fVDBTensor:
        """
        Forward pass of the FeedForwardNetworkFVDB.

        Args
        -----
        x (fVDBTensor): The input tensor.

        Returns
        -------
        fVDBTensor: The output tensor.
        """
        x = self.mlp(x)
        return x
