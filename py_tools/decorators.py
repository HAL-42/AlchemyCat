#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2021/11/8 21:16
@File    : decorators.py
@Software: PyCharm
@Desc    : 
"""
from warnings import warn

from .timer import Timer
from .color_print import *

__all__ = ['deprecated', 'func_runtime_timer']


def deprecated(func):
    """Show deprecation warning of a function."""

    def wrapper(*args, **kwargs):
        warn('Func {} is now deprecated!'.format(func.__name__))
        return wrapper(*args, **kwargs)

    return wrapper


def func_runtime_timer(offset: int = 40, show_interval: int = 10):
    def decorator(func):
        cons = []

        def func_wrapper(*args, **kwargs):
            space = offset * " "
            nonlocal cons
            with Timer(unit='ms', precision=2, name=f'{func.__name__}耗时') as t:
                result = func(*args, **kwargs)
            bprint(f"{space}| {t}")
            cons.append(t.total)
            if cons and len(cons) % show_interval == 0:
                bprint(f"{space}| -------------------------------------------------------\n"
                       f"{space}| {func.__name__}已经执行{len(cons)}次。\n"
                       f"{space}| 平均耗时{round(sum(cons) / len(cons) * 1000, 2)}ms，\n"
                       f"{space}| 最大耗时{round(max(cons) * 1000, 2)}ms，\n"
                       f"{space}| 最小耗时{round(min(cons) * 1000, 2)}ms。\n"
                       f"{space}| -------------------------------------------------------")
            return result

        return func_wrapper

    return decorator
