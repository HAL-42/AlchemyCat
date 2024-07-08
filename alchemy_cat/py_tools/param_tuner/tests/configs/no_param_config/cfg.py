#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2021/5/6 22:09
@File    : cfg.py.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Cfg2Tune

cfg = config = Cfg2Tune()

cfg.rslt_dir = 'no_param_config'

# * Param on same Cfg2Tune
cfg.foo0.foo1.a = 1
cfg.foo0.foo1.b = .1
# * Param on diff Cfg2Tune
cfg.foo0.fix = '1'
# * Param on root Cfg2Tune
cfg.c = True
# * Static Value
cfg.foo0.a_max = 2
