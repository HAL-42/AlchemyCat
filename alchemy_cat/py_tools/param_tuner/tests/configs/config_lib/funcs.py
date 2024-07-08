#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2021/11/25 20:51
@File    : funcs.py
@Software: PyCharm
@Desc    : 
"""
import numpy as np

__all__ = ["func_a", "func_b", "func_c", "func_d"]


def func_a(goo):
    return goo


def func_b(goo):
    return func_a(goo)


def func_c(goo):
    np.array(goo)


def func_d():
    return func_d
