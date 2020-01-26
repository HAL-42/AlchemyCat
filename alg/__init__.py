#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: __init__.py
@time: 2019/12/7 23:58
@desc:
"""
from alchemy_cat.alg.cnn_align import find_nearest_even_size, find_nearest_odd_size
from alchemy_cat.alg.msc_flip_inference import msc_flip_inference

from alchemy_cat.py_tools import is_int

from collections import Iterable

from typing import Any

# Taken from python 3.5 docs
def _accumulate(iterable, fn=lambda x, y: x + y):
    'Return running totals'
    # _accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # _accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = fn(total, element)
        yield total


def size2HW(size: Any) -> tuple:
    """Convert size to H, W

    Args:
        size (Any): If size is int, return (size, size), else return (size[0], size[1])

    Returns: (H, W) converted from size
    """
    if is_int(size):
        return int(size), int(size)
    elif len(size) == 2 and is_int(size[0]) and is_int(size[1]):
        return int(size[0]), int(size[1])
    else:
        raise ValueError("size should be int or [int, int] or (int, int)")