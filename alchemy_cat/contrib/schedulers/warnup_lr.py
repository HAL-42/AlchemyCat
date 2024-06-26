#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/2/4 16:56
@File    : warnup_lr.py
@Software: PyCharm
@Desc    : 
"""
import warnings

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

__all__ = ['WarmUpLR']


class WarmUpLR(_LRScheduler):

    def __init__(self, optimizer: Optimizer, last_epoch: int=-1, warm_up_iter: int=0, warm_up_min_lr: float=.0):
        self.warm_up_iter = warm_up_iter
        self.warm_up_min_lr = warm_up_min_lr
        super().__init__(optimizer, last_epoch)

    def _warm_up_ascend(self, lr: float):
        return lr * ((self.last_epoch + 1) / (self.warm_up_iter + 1))

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        assert -1 <= self.last_epoch <= self.warm_up_iter  # 该区间能给出合法学习率。
        return [max(self._warm_up_ascend(lr), self.warm_up_min_lr) for lr in self.base_lrs]
