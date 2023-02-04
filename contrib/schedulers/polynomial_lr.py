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

import warnings

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

from alchemy_cat.py_tools import indent

from .warnup_lr import WarmUpLR

__all__ = ['PolynomialLR', 'WarmPolynomialLR']


class PolynomialLR(_LRScheduler):
    def __init__(self, optimizer: Any, step_size: int, max_iter: int,
                 power: Union[int, float], last_epoch: int=-1):
        DeprecationWarning(f"{type(self)} is deprecated, use {WarmPolynomialLR} instead. ")
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


class WarmPolynomialLR(WarmUpLR):
    def __init__(self, optimizer: Optimizer, step_size: int=1, max_iter: int=10000,
                 power: int | float=.9, last_epoch: int=-1, min_lr: float=.0, end_lr_factors: float | list[float]=.0,
                 warm_up_iter: int=0, warm_up_min_lr: float=.0):
        if step_size != 1:
            DeprecationWarning(f"{step_size=} should be 1. ")

        assert 0 <= warm_up_iter < max_iter

        self.max_iter = max_iter
        self.power = power
        self.min_lr = min_lr
        self.end_lr_factors = (end_lr_factors if isinstance(end_lr_factors, list)
                               else [end_lr_factors] * len(optimizer.param_groups))
        super().__init__(optimizer, last_epoch, warm_up_iter, warm_up_min_lr)

    def _polynomial_decay(self, lr, end_lr_factor):
        return (lr - end_lr_factor * lr) * (1 - (self.last_epoch - self.warm_up_iter) /
                                            (self.max_iter - self.warm_up_iter)) ** self.power + end_lr_factor * lr

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        assert -1 <= self.last_epoch <= self.max_iter  # 该区间能给出合法学习率。

        if self.last_epoch < self.warm_up_iter:
            return WarmUpLR.get_lr(self)
        else:
            return [max(self._polynomial_decay(lr, end_lr_factor), self.min_lr)
                    for lr, end_lr_factor in zip(self.base_lrs, self.end_lr_factors, strict=True)]

    def __repr__(self):
        return f"PolynomialLR <{self.__class__}>: \n" \
               f"  max_iter: {self.max_iter}\n" \
               f"  power: {self.power}\n" \
               f"  min_lr: {self.min_lr}\n" \
               f"  end_lr: {self.end_lr_factors}\n" \
               f"  warm_up_iter: {self.warm_up_iter}\n" \
               f"  warm_up_min_lr: {self.warm_up_min_lr}"
