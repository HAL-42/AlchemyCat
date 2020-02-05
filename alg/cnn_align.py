#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: cnn_align.py
@time: 2020/1/15 6:27
@desc:
"""
__all__ = ["find_nearest_even_size", "find_nearest_odd_size"]


def _check_input(size, min_n: int):
    size = int(size)
    min_n = int(min_n)
    if not isinstance(size, int) or not isinstance(min_n, int):
        raise ValueError(f"size={size}, min_n={min_n} should be able to be int()")
    if size <= 0:
        raise ValueError(f"size=f{size} should > 0")
    if min_n < 1:
        raise ValueError(f"min_n=f{min_n} should >= 1")
    return size, min_n


def find_nearest_odd_size(size, min_n: int=4, is_both_way: bool=False):
    """Find the nearest number to the size which can be writen as k * 2 ^ min_n + 1

    Args:
        size (int): The size of origin img/label
        is_both_way (bool): If True, return odd size nearest number in both way rather only the larger number

    Returns: nearest odd size larger than size if not is_both_way, else nearest odd size smaller or larger than size.
    """
    size, min_n = _check_input(size, min_n)

    base = 2 ** min_n
    k = size // base
    residual = size % base

    if residual == 1 and k >=1:
        return size
    elif not is_both_way:
        return (k * base + 1) if residual == 0 else ((k + 1) * base + 1)
    else:
        left_gap = (base -1) if (residual == 0) else (residual - 1)
        right_gap = 1 if (residual == 0) else (base + 1 - residual)
        return (size - left_gap) if (left_gap < right_gap and (size - left_gap) >= (base + 1)) else (size + right_gap)


def find_nearest_even_size(size, min_n: int=4, is_both_way: bool=False):
    """Find the nearest number to the size which can be writen as k * 2 ^ min_n

    Args:
        size (int): The size of origin img/label
        is_both_way (bool): If True, return even size nearest number in both way rather only the larger number

    Returns: nearest odd size larger than size if not is_both_way, else nearest even size smaller or larger than size.
    """
    size, min_n = _check_input(size, min_n)

    base = 2 ** min_n
    k = size // base
    residual = size % base

    if residual == 0 and k >= 1:
        return size
    elif not is_both_way:
        return (k + 1) * base
    else:
        left_gap = residual
        right_gap = base - residual
        return (size - left_gap) if (left_gap < right_gap and k >= 1) else (size + right_gap)


if __name__ == "__main__":
    print(find_nearest_odd_size(1), find_nearest_even_size(2))
    print(find_nearest_odd_size(2), find_nearest_even_size(1))
    print(find_nearest_odd_size(512), find_nearest_odd_size(512, is_both_way=True), find_nearest_even_size(512))
    print(find_nearest_odd_size(513 * 0.75), find_nearest_odd_size(518 * 0.75, is_both_way=True),
          find_nearest_even_size(512 * 0.75))
    print(find_nearest_odd_size(513 * 0.5), find_nearest_odd_size(518 * 0.5, is_both_way=True),
          find_nearest_even_size(512 * 0.5))
    print(find_nearest_odd_size(513 * 1.25), find_nearest_odd_size(518 * 1.25, is_both_way=True),
          find_nearest_even_size(512 * 1.25))
    print(find_nearest_odd_size(513 * 1.5), find_nearest_odd_size(518 * 1.5, is_both_way=True),
          find_nearest_even_size(512 * 1.5))
    print(find_nearest_odd_size(513 * 1.75), find_nearest_odd_size(518 * 1.75, is_both_way=True),
          find_nearest_even_size(512 * 1.75))


