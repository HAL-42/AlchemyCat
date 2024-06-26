#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2022/4/13 21:15
@File    : module_init_fns.py
@Software: PyCharm
@Desc    : 
"""
from torch import nn

__all__ = ['init_conv_kaiming_bn_identity', 'set_bn_momentum']


def init_conv_kaiming_bn_identity(module: nn.Module) -> nn.Module:
    for m in module.modules():
        if isinstance(m, (nn.modules.conv._ConvNd, nn.Linear)):  # _ConvNd下辖1、2、3D正反卷积，拥有weight和bias两个参数。
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.modules.batchnorm._NormBase):
            # _NormBase下辖1、2、3D及同步批归一化，拥有weight和bias两个参数，running mean、var等buffer。
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    return module


def set_bn_momentum(model: nn.Module, momentum: float=0.1):
    """设置网络BN层的动量。

    Args:
        model: torch模型。
        momentum: 动量。
    """
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.momentum = momentum
