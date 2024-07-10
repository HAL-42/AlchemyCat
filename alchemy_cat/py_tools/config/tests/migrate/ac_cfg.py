#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2024/7/10 18:05
@File    : ac_cfg.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Config

cfg = Config()

cfg.rand_seed = 0
cfg.X_method.method = 'L2'
cfg.lp.ini.k = 50
cfg.lp.ini.cg_guess = False
cfg.some.lst = [1, 2, 3]
cfg.some.lst_dic = [{'a': 1}, {'b': 2}]
