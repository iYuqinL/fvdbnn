# -*- coding:utf-8 -*-
###
# File: dropout.py
# Created Date: Saturday, November 22nd 2025, 12:34:26 am
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

import torch.nn as nn
from .utils import ElementwiseMixin


__all__ = ["DropoutFVDB"]


class DropoutFVDB(ElementwiseMixin, nn.Dropout):
    r"""
    During training, randomly zeroes some of the elements of
    the input tensor with probability :attr:`p` using samples from a Bernoulli distribution.
    The elements to zero are randomized on every forward call.
    """
