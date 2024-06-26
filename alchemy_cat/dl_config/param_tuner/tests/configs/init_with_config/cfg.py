#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2022/5/25 21:49
@File    : cfg.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.dl_config import Cfg2Tune


cfg = config = Cfg2Tune('configs/base_cfg/cfg.py')

cfg.rslt_dir = 'init_with_config'
