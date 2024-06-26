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
from alchemy_cat.py_tools import Cfg2Tune, Param2Tune, ItemLazy

cfg = config = Cfg2Tune()

cfg.rslt_dir = 'itemlazy_config'


def b_get1(c):
    return c.foo0.foo1.a * 2


def b_get2(c):
    return int(c.c)


def lazy_get(c):
    return c.foo0.a_max['b'] * c.foo0.foo1.a


# * Param on same Cfg2Tune
cfg.foo0.foo1.a = Param2Tune([0, 1, 2])
cfg.foo0.foo1.b = Param2Tune([ItemLazy(b_get1, rel=False), ItemLazy(b_get2, rel=False)])
# * Param on diff Cfg2Tune
cfg.foo0.fix = Param2Tune(['0'])
# * Param on root Cfg2Tune
cfg.c = Param2Tune([False, True])
# * Static Value
cfg.foo0.a_max = {'a': 1, 'b': 2}
# * Item Lazy
cfg.foo1.lazy = ItemLazy(lazy_get)
