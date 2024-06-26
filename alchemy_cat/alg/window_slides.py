#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2022/10/13 21:18
@File    : window_slides.py
@Software: PyCharm
@Desc    : 
"""
__all__ = ['slide_n_windows_len_l']


def slide_n_windows_len_l(n: int, l: int, lst: list | tuple) -> list:
    """从lst中截取尽量不相交、距离远的n个长l的切片。

    Args:
        n: 窗口数。
        l: 窗口长。
        lst: list序列。

    Returns:

    """
    step = (len(lst) - l) // (n - 1)

    return [lst[i * step:i * step + l] for i in range(n)]
