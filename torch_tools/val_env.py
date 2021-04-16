#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: val_env.py
@time: 2020/3/29 3:17
@desc:
"""
import torch
import torch.nn as nn

__all__ = ['ValidationEnv']


class ValidationEnv(object):
    """Context manager for validate model"""
    def __init__(self, model: nn.Module, change_benchmark: bool=False):
        """Context manager for validate model

        Args:
            model: model to be validated
            change_benchmark: If True, will inverse torch.backends.cudnn.benchmark before and after context.
                (Default: False)
        """
        self.model = model
        self.change_benchmark = change_benchmark

    def __enter__(self) -> nn.Module:
        torch.set_grad_enabled(False)
        self.model.eval()
        if self.change_benchmark:
            torch.backends.cudnn.benchmark = not torch.backends.cudnn.benchmark
        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.set_grad_enabled(True)
        self.model.train()
        if self.change_benchmark:
            torch.backends.cudnn.benchmark = not torch.backends.cudnn.benchmark
