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
def find_nearest_odd_size(size, min_n: int=4, is_both_way: bool=False):
    """Find the nearest number to the size which can be writen as k * 2 ^ min_n + 1

    Args:
        size (int): The size of origin img/label
        is_both_way (bool): If True, return odd size nearest number in both way rather only the larger number

    Returns: nearest odd size larger than size if not is_both_way, else nearest odd size smaller or larger than size.
    """
    size = int(size)
    min_n = int(min_n)
    if not isinstance(size, int) or not isinstance(min_n, int):
        raise ValueError(f"size={size}, min_n={min_n} should be able to be int()")
    if size <= 0:
        raise ValueError(f"size=f{size} should > 0")
    if min_n < 1:
        raise ValueError(f"min_n=f{min_n} should >= 1")
    # * Make size odd
    size = size if size % 2 else size + 1
    # * Find nearest odd size
    i = 0
    while ((size + 2 * i - 1) % (2 ** min_n)) or (size + 2 * i - 1 <= 0):
        if is_both_way:
            i = i * -1 if i > 0 else (i * -1) + 1
        else:
            i += 1
    return size + 2 * i


def find_nearest_even_size(size, min_n: int=4, is_both_way: bool=False):
    """Find the nearest number to the size which can be writen as k * 2 ^ min_n

    Args:
        size (int): The size of origin img/label
        is_both_way (bool): If True, return even size nearest number in both way rather only the larger number

    Returns: nearest odd size larger than size if not is_both_way, else nearest even size smaller or larger than size.
    """
    size = int(size)
    min_n = int(min_n)
    if not isinstance(size, int) or not isinstance(min_n, int):
        raise ValueError(f"size={size}, min_n={min_n} should be able to be int()")
    if size <= 0:
        raise ValueError(f"size=f{size} should > 0")
    if min_n < 1:
        raise ValueError(f"min_n=f{min_n} should >= 1")
    # * Make size odd
    size = size + 1 if size % 2 else size
    # * Find nearest odd size
    i = 0
    while (size + 2 * i) % (2 ** min_n) or (size + 2 * i <= 0):
        if is_both_way:
            i = i * -1 if i > 0 else (i * -1) + 1
        else:
            i += 1
    return size + 2 * i


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


