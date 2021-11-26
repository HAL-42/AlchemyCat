#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2021/7/8 17:15
@File    : subplots_row_col.py
@Software: PyCharm
@Desc    : 
"""
from typing import Tuple
import math

__all__ = ["square", "row_all", "col_all", "rect"]


def square(num: int) -> Tuple[int, int]:
    col_num = math.ceil(math.sqrt(num))
    row_num = math.ceil(num / col_num)
    return (row_num, col_num)


def row_all(num: int) -> Tuple[int, int]:
    return (num, 1)


def col_all(num: int) -> Tuple[int, int]:
    return (1, num)


def rect(num: int, row_num: int=1) -> Tuple[int, int]:
    col_num = math.ceil(num / row_num)
    return (row_num, col_num)
