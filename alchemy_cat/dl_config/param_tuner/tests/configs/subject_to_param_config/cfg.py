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
from alchemy_cat.dl_config import Cfg2Tune, Param2Tune

cfg = config = Cfg2Tune()

cfg.rslt_dir = 'subject_to_param_config'

# * Param on same Cfg2Tune
cfg.foo0.foo1.a = Param2Tune([0, 1, 2])
cfg.foo0.foo1.b = Param2Tune([.1, .2], lambda x: int(10 * x) != cfg.foo0.foo1.a.cur_val)
# * Param on root Cfg2Tune
cfg.c = Param2Tune([False, True], subject_to=lambda x: x == (10 * cfg.foo0.foo1.b.cur_val > cfg.foo0.foo1.a.cur_val))
# * Static Value
cfg.foo0.a_max = 2
