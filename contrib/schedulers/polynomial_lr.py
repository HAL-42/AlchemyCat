#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: polynomial_lr.py
@time: 2020/2/25 11:57
@desc:
"""
from typing import Any, Union

import torch
from torch.optim.lr_scheduler import _LRScheduler

from alchemy_cat.py_tools import indent


class PolynomialLR(_LRScheduler):
    def __init__(self, optimizer: Any, step_size: int, max_iter: int,
                 power: Union[int, float], last_epoch: int=-1):
        self.step_size = step_size
        self.max_iter = max_iter
        self.power = power
        super(PolynomialLR, self).__init__(optimizer, last_epoch)

    def _polynomial_decay(self, lr):
        return lr * (1 - float(self.last_epoch) / self.max_iter) ** self.power

    def get_lr(self):
        if (
            (self.last_epoch == 0)
            or (self.last_epoch % self.step_size != 0)
            or (self.last_epoch > self.max_iter)
        ):
            return [group["lr"] for group in self.optimizer.param_groups]
        return [self._polynomial_decay(lr) for lr in self.base_lrs]

    def __repr__(self):
        return f"PolynomialLR <{self.__class__}>: \n" \
               + indent(f"step_size: {self.step_size}") + '\n' \
               + indent(f"max_iter: {self.max_iter}") + '\n' \
               + indent(f"power: {self.power}")
