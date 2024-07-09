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

from typing import Any, Tuple, Iterable, Callable, Union

from alchemy_cat.py_tools.type import is_int, is_float, totuple


__all__ = ['accumulate', 'size2HW', 'color2scalar']


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
        if size < 0:
            raise ValueError(f"size{size} must >= 0")
        return int(size), int(size)
    if isinstance(size, Iterator):
        return size2HW(list(size))
    elif len(size) == 2 and is_int(size[0]) and is_int(size[1]):
        if size[0] < 0 or size[1] < 0:
            raise ValueError(f"H, W of size {size} must >= 0")
        return int(size[0]), int(size[1])
    else:
        raise ValueError("size should be int or [int, int] or (int, int)")


def color2scalar(val: Union[int, float, Iterable]) -> Tuple:
    """Convert color value to color scalar

    Args:
        val (Union[int, float, Iterable]): If val is int or float, return (val, val, val), if value is Iterable with 3
            element, return totuple(val), else raise error

    Returns: tuple color scalar
    """
    if is_int(val) or is_float(val):
        img_pad_scalar = (val,) * 3
    else:
        img_pad_scalar = totuple(val)
        if len(img_pad_scalar) != 3:
            raise ValueError(f"If img_pad_value {val} is scalar, it should be a 3-channel color scalar")

    return img_pad_scalar