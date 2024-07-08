#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2021/6/22 22:29
@File    : cfg.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Cfg2Tune, Param2Tune, ParamLazy

cfg = config = Cfg2Tune()

cfg.rslt_dir = 'param_lazy_config'

# * Param on same Cfg2Tune
cfg.foo0.foo1.a = Param2Tune([0, 1, 2])

# * Param Lazy
cfg.foo0.lazy0 = ParamLazy(lambda c: c.foo0.foo1.a + 10)
cfg.lazy1 = ParamLazy(lambda c: c.foo0.foo1.a - c.foo0.a_max)

# * Static Value
cfg.foo0.a_max = 2
