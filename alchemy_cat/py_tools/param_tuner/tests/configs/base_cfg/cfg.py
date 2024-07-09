#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2022/5/25 21:38
@File    : cfg.py
@Software: PyCharm
@Desc    : 
"""
try:
    from addict import Dict
except ImportError:
    Dict = dict

from alchemy_cat.py_tools import Config


cfg = config = Config()

# * 二级子树。
cfg.foo0.foo1.a = 1
cfg.foo0.foo1.b = {'a': 2, 'b': 3}
# * 一级子树。
cfg.foo2.fix = 4
cfg.foo2.a_max = Dict({'c': 5, 'd': 6})
# * 叶子。
cfg.c = 7
cfg.d = Dict({'e': 8, 'f': 9})
