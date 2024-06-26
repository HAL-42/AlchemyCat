#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2021/11/25 20:50
@File    : cfg.py
@Software: PyCharm
@Desc    : 
"""
import numpy as np

from alchemy_cat.py_tools import Cfg2Tune, Param2Tune


def func_a(goo):
    return goo


def func_b(goo):
    return func_a(goo)


def func_c(goo):
    np.array(goo)


def func_d():
    return func_d


cfg = config = Cfg2Tune()

cfg.rslt_dir = 'def_inside_config'

# * 文件内定义的函数。
cfg.foo0.foo1.a = Param2Tune([func_a, func_b, func_c, func_d])
# * Param on diff Cfg2Tune
cfg.foo0.fix = Param2Tune(['foo.bar'])
# * Param on root Cfg2Tune
cfg.c = Param2Tune([False, True])
# * 静态值，等于文件内定义的函数。
cfg.foo0.a_max = func_a
cfg.foo0.a_min = (func_c, func_d)
