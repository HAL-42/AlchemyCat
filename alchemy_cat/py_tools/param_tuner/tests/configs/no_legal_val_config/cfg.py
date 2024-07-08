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
from alchemy_cat.py_tools import Cfg2Tune, Param2Tune

cfg = config = Cfg2Tune()

cfg.rslt_dir = 'no_legal_val_config'

# * Param on same Cfg2Tune
cfg.foo0.foo1.a = Param2Tune([0, 1, 2])


def b_subject_to(opt_val):
    if cfg.foo0.foo1.a.cur_val > 1:
        return opt_val > 1
    return True


cfg.foo0.foo1.b = Param2Tune([.1, .2], b_subject_to)
# * Param on root Cfg2Tune


def c_subject_to(_):
    if cfg.foo0.foo1.b.cur_val == .2:
        return False
    return True


cfg.c = Param2Tune([False, True], c_subject_to)
# * Static Value
cfg.foo0.a_max = 2
