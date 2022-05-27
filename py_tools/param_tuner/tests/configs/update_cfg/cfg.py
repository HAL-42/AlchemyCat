#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2022/5/25 21:38
@File    : cfg.py
@Software: PyCharm
@Desc    : 测试人有树、叶我无、人有树我有叶，人有叶我有树，人有树我有树。
"""
from alchemy_cat.py_tools import Config


cfg = config = Config()

# * 二级子树。
# 人有树我有树。
cfg.foo0.foo1.b = {'aa': 2, 'bb': 3}  # 人有叶我有叶。
cfg.foo0.foo3.l = 'hh'  # 人有树我无。
cfg.foo0.i = 6.8  # 人有叶我无。
# * 一级子树。
# 人有叶我有树。
cfg.foo2 = ('c', 'd')
# * 叶子。
cfg.cc = 7  # 人有叶我无。
cfg.d = Config({'ee': 8, 'ff': 9})  # 人有树我有叶。
cfg.e.f = {10, 'gg'}  # 人有树我无。
cfg.e.g = 11
