#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2021/7/23 5:01
@File    : set_network.py
@Software: PyCharm
@Desc    : 设置torch网络。
"""
from torch import nn

__all__ = ['set_bn_momentum']


def set_bn_momentum(model: nn.Module, momentum=0.1):
    """设置网络BN层的动量。

    Args:
        model: torch模型。
        momentum: 动量。
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum
