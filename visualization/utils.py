#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: utils.py
@time: 2020/1/7 23:46
@desc:
"""
import numpy as np


def stack_figs(in_list: list) -> np.ndarray:
    """
    Args:
        in_list (list): A list of numpy array shaped (H, W, C) with length N

    Returns:
        np.ndarray: Stacked np array with shape (N, H, W, C)

    Arrays in list will be padded to the max height/width in the list. The padding value will be 0 and located at right
    and bottom.
    """
    max_h = 0
    max_w = 0
    for i, item in enumerate(in_list):
        if item.shape[0] > max_h:
            max_h = item.shape[0]
        if item.shape[1] > max_w:
            max_w = item.shape[1]
    out_arr = np.zeros((len(in_list), max_h, max_w, 3), dtype=in_list[0].dtype)
    for i, item in enumerate(in_list):
        pad_h = max_h - item.shape[0]
        pad_w = max_w - item.shape[1]
        out_arr[i] = np.pad(item, ((0,pad_h),(0,pad_w),(0,0)), mode='constant', constant_values=0)
    return out_arr


def HWC2CHW(arr: np.ndarray) -> np.ndarray:
    """
    Args:
        arr (np.ndarray): nparray with shape (..., H, W, C)

    Returns:
        (np.ndarray): nparray transposed to (..., C, H, W)
    """
    return arr.transpose(tuple(range(0, arr.ndim - 3)) + (-1, -3, -2))


def CHW2HWC(arr: np.ndarray) -> np.ndarray:
    """
    Args:
        arr (np.ndarray): nparray with shape (..., C, H, W)

    Returns:
        (np.ndarray): nparray transposed to (..., H, W, C)
    """
    return arr.transpose(tuple(range(0, arr.ndim - 3)) + (-2, -1 -3))


def RGB2BGR(arr):
    """
    Args:
        arr (Tensor, ndarray): arr with shape (..., C[RGB])

    Returns:
        arr with shape (..., C[BGR])
    """
    return arr[..., ::-1]


def BGR2RGB(arr):
    """
    Args:
        arr (Tensor, ndarray): arr with shape (..., C[BGR])

    Returns:
        arr with shape (..., C[RGB])
    """
    return arr[..., ::-1]