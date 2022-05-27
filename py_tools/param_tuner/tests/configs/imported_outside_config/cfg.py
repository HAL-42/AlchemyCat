#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2021/11/25 20:18
@File    : cfg.py
@Software: PyCharm
@Desc    : 
"""
from numpy import sum, min, max
from alchemy_cat.py_tools.param_tuner.tests.configs.config_lib import func_a, func_b, func_c, func_d

from alchemy_cat.py_tools import Cfg2Tune, Param2Tune


cfg = config = Cfg2Tune()

cfg.rslt_dir = 'imported_outside_config'

# * 外部库中函数。
cfg.foo0.foo1.a = Param2Tune([sum, min, max])
# * 当前项目中函数。
cfg.foo0.foo1.b = Param2Tune([func_a, func_b, func_c, func_d])
# * Param on diff Cfg2Tune
cfg.foo0.fix = Param2Tune(['foo.bar'])
# * Param on root Cfg2Tune
cfg.c = Param2Tune([False, True])
# * 静态值，等于外部库函数或项目中函数。
cfg.foo0.a_max = sum
cfg.foo0.a_min = (func_c, func_d)
