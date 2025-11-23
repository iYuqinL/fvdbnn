# -*- coding:utf-8 -*-
###
# File: activation.py
# Created Date: Saturday, November 22nd 2025, 12:33:36 am
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


__all__ = ["ELUFVDB", "CELUFVDB", "GELUFVDB", "ReLUFVDB", 
           "LeakyReLUFVDB", "SELUFVDB",
           "SiLUFVDB", "TanhFVDB", "SigmoidFVDB"]


class ELUFVDB(ElementwiseMixin, nn.ELU):
    r"""
    Applies the Exponential Linear Unit function element-wise:
    .. math::
    \text{ELU}(x) = \begin{cases}
    x, & \text{ if } x > 0\\
    \alpha * (\exp(x) - 1), & \text{ if } x \leq 0
    \end{cases}
    """


class CELUFVDB(ElementwiseMixin, nn.CELU):
    r"""
    Applies the CELU function element-wise.

    .. math::
        \text{CELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x/\alpha) - 1))
    """


class GELUFVDB(ElementwiseMixin, nn.GELU):
    r"""
    Applies the Gaussian Error Linear Units function.

    .. math:: \text{GELU}(x) = x * \Phi(x)

    where :math:`\Phi(x)` is the Cumulative Distribution Function
    for Gaussian Distribution.
    """


class ReLUFVDB(ElementwiseMixin, nn.ReLU):
    r"""
    Applies the rectified linear unit function element-wise:
    :math:`\text{ReLU}(x) = (x)^+ = \max(0, x)`
    """


class LeakyReLUFVDB(ElementwiseMixin, nn.LeakyReLU):
    r"""
    Applies the element-wise function:
    :math:`\text{LeakyReLU}(x) = \max(0, x) + \text{negative\_slope} * \min(0, x)`
    """


class SELUFVDB(ElementwiseMixin, nn.SELU):
    r"""
    Applies element-wise, :math:`\text{SELU}(x) = \lambda \left\{
    \begin{array}{lr}
    x, & \text{if } x > 0 \\
    \text{negative\_slope} \times e^x - \text{negative\_slope}, & \text{otherwise }
    \end{array}
    \right.`
    """


class SiLUFVDB(ElementwiseMixin, nn.SiLU):
    r"""
    Applies element-wise, :math:`\text{SiLU}(x) = x * \sigma(x)`,
    where :math:`\sigma(x)` is the sigmoid function.
    """


class TanhFVDB(ElementwiseMixin, nn.Tanh):
    r"""
    Applies element-wise,
    :math:`\text{Tanh}(x) = \tanh(x) = \frac{e^x - e^{-x}} {e^x + e^{-x}}`
    """


class SigmoidFVDB(ElementwiseMixin, nn.Sigmoid):
    r"""
    Applies element-wise, :math:`\text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}`
    """
