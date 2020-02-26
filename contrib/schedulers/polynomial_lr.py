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
import torch
from torch.optim.lr_scheduler import _LRScheduler

from typing import Union


class PolynomialLR(_LRScheduler):
    def __init__(self, optimizer: torch.optim.optimizer.Optimizer, step_size: int, iter_max: int,
                 power: Union[int, float], last_epoch: int=-1):
        self.step_size = step_size
        self.iter_max = iter_max
        self.power = power
        super(PolynomialLR, self).__init__(optimizer, last_epoch)

    def _polynomial_decay(self, lr):
        return lr * (1 - float(self.last_epoch) / self.iter_max) ** self.power

    def get_lr(self):
        if (
            (self.last_epoch == 0)
            or (self.last_epoch % self.step_size != 0)
            or (self.last_epoch > self.iter_max)
        ):
            return [group["lr"] for group in self.optimizer.param_groups]
        return [self._polynomial_decay(lr) for lr in self.base_lrs]
