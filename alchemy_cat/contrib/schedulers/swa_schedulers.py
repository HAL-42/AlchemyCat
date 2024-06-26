#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/2/4 22:11
@File    : swa_schedulers.py
@Software: PyCharm
@Desc    : 
"""
from typing import Callable

import warnings
import math

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

__all__ = ['SWACycleLR', 'SWAAnnealLR']


def _linear_trans(t: float) -> float:  # 0-1等射到0-1。
    return t


def _cos_trans(t: float) -> float:  # 0-1cos变换到0-1。
    return (1 - math.cos(math.pi * t)) / 2


def _lrs_mul_factor(lrs: list[float], lr_factor: float | list[float]) -> list[float]:
    if isinstance(lr_factor, float):
        lr_factor = [lr_factor] * len(lrs)
    return [l * f for l, f in zip(lrs, lr_factor, strict=True)]


def _descend_lr_by_t(lrs_start: list[float], lrs_end: list[float], t: float) -> list[float]:
    return [lr_start * (1 - t) + lr_end * t for lr_start, lr_end in zip(lrs_start, lrs_end, strict=True)]


class SWACycleLR(_LRScheduler):
    def __init__(self, optimizer: Optimizer,
                 period_iters: int,
                 lr_start_factor: float | list[float]= 1., lr_end_factor: float | list[float]=0.,
                 start_factor_rel_base: bool=False, end_factor_rel_base: bool=False,
                 anneal_func: Callable[[float], float] | str='linear',
                 last_epoch: int=-1, **_):
        assert period_iters >= 2
        self.period_iters = period_iters

        self.lr_start_factor = lr_start_factor
        self.lr_end_factor = lr_end_factor
        self.start_factor_rel_base = start_factor_rel_base
        self.end_factor_rel_base = end_factor_rel_base

        self.lrs_start: list[float] | None = None
        self.lrs_end: list[float] | None = None

        match anneal_func:
            case 'linear':
                self.anneal_func = _linear_trans
            case 'cos':
                self.anneal_func = _cos_trans
            case _:
                assert callable(anneal_func)
                self.anneal_func = anneal_func

        super(SWACycleLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        cur_lrs = [group["lr"] for group in self.optimizer.param_groups]
        if self.last_epoch == 0:  # * 初始化时不动如山。
            return cur_lrs
        if self.last_epoch == 1:  # * 接手时确定学习率起始点。
            self.lrs_start = _lrs_mul_factor(self.base_lrs if self.start_factor_rel_base else cur_lrs,
                                             self.lr_start_factor)
            self.lrs_end = _lrs_mul_factor(self.base_lrs if self.end_factor_rel_base else self.lrs_start,
                                           self.lr_end_factor)
        t = self.anneal_func(((self.last_epoch - 1) % self.period_iters) / self.period_iters)
        return _descend_lr_by_t(self.lrs_start, self.lrs_end, t)

    def __repr__(self):
        return f"SWACycleLR <{self.__class__}>: \n" \
               f"   period_iters: {self.period_iters}\n" \
               f"   lr_start_factor: {self.lr_start_factor}\n" \
               f"   lr_end_factor: {self.lr_end_factor}\n" \
               f"   lrs_start: {self.lrs_start}\n" \
               f"   lrs_end: {self.lrs_end}"


class SWAAnnealLR(_LRScheduler):
    def __init__(self, optimizer: Optimizer,
                 anneal_iters: int,
                 lr_start_factor: float | list[float] = 1., lr_end_factor: float | list[float] = 0.,
                 start_factor_rel_base: bool=False, end_factor_rel_base: bool=False,
                 anneal_func: Callable[[float], float] | str = 'linear',
                 last_epoch: int = -1, **_):
        assert anneal_iters >= 1
        self.anneal_iters = anneal_iters

        self.lr_start_factor = lr_start_factor
        self.lr_end_factor = lr_end_factor
        self.start_factor_rel_base = start_factor_rel_base
        self.end_factor_rel_base = end_factor_rel_base

        self.lrs_start: list[float] | None = None
        self.lrs_end: list[float] | None = None

        match anneal_func:
            case 'linear':
                self.anneal_func = _linear_trans
            case 'cos':
                self.anneal_func = _cos_trans
            case _:
                assert callable(anneal_func)
                self.anneal_func = anneal_func

        super(SWAAnnealLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        cur_lrs = [group["lr"] for group in self.optimizer.param_groups]
        if self.last_epoch == 0:  # * 初始化时不动如山。
            return cur_lrs
        if self.last_epoch == 1:  # * 接手时确定学习率起始点。
            self.lrs_start = _lrs_mul_factor(self.base_lrs if self.start_factor_rel_base else cur_lrs,
                                             self.lr_start_factor)
            self.lrs_end = _lrs_mul_factor(self.base_lrs if self.end_factor_rel_base else self.lrs_start,
                                           self.lr_end_factor)
        t = self.anneal_func(min((self.last_epoch - 1) / self.anneal_iters, 1.))
        return _descend_lr_by_t(self.lrs_start, self.lrs_end, t)

    def __repr__(self):
        return f"SWAAnnealLR <{self.__class__}>: \n" \
               f"   period_iters: {self.anneal_iters}\n" \
               f"   lr_start_factor: {self.lr_start_factor}\n" \
               f"   lr_end_factor: {self.lr_end_factor}\n" \
               f"   lrs_start: {self.lrs_start}\n" \
               f"   lrs_end: {self.lrs_end}"
