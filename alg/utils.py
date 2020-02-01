#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: utils.py
@time: 2020/1/26 23:08
@desc:
"""
from collections.abc import Iterator

from typing import Any, Tuple, Iterable, Callable

from alchemy_cat.py_tools import is_int


__all__ = ['accumulate', 'size2HW']

# Taken from python 3.5 docs
def accumulate(iterable: Iterable, fn: Callable=lambda x, y: x + y):
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


def size2HW(size: Any) -> Tuple[int, int]:
    """Convert size to H, W

    Args:
        size (Any): If size is int, return (size, size), else return (size[0], size[1])

    Returns: (H, W) converted from size
    """
    if is_int(size):
        return int(size), int(size)
    if isinstance(size, Iterator):
        return size2HW(list(size))
    elif len(size) == 2 and is_int(size[0]) and is_int(size[1]):
        return int(size[0]), int(size[1])
    else:
        raise ValueError("size should be int or [int, int] or (int, int)")


