#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/2/4 15:57
@File    : __init__.py.py
@Software: PyCharm
@Desc    : 
"""
from .auto_allocate_cuda import *
try:
    from .cuda_block import *
except ImportError:
    # 没有安装gpustat
    pass
