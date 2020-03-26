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
import torch

from typing import Union, Iterable

from alchemy_cat.data.plugins.augers import pad_img_label

__all__ = ["HWC2CHW", "CHW2HWC", "BGR2RGB", "RGB2BGR", "c1_to_cn"]


def stack_figs(in_list: list, img_pad_val: Union[int, float, Iterable] = (127, 140, 141),
               pad_location: Union[str, int]='right-bottom') -> np.ndarray:
    """Arrays in list will be padded to the max height/width in the list. The padding value will be 0 and located at right
    and bottom.

    Args:
        in_list (list): A list of numpy array shaped (H, W, C) with length N
        img_pad_val: (Union[int, float, Iterable]): If value is int or float, return (value, value, value),
            if value is Iterable with 3 element, return totuple(value), else raise error
        pad_location (Union[str, int]): Indicate pad location. Can be 'left-top'/0, 'right-top'/1, 'left-bottom'/2
            'right-bottom'/3, 'center'/4.

    Returns:
        np.ndarray: Stacked np array with shape (N, H, W, C)
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
        out_arr[i] = pad_img_label(item, pad_img_to=(max_h, max_w), img_pad_val=img_pad_val, pad_location=pad_location)
    return out_arr


def HWC2CHW(arr: np.ndarray) -> np.ndarray:
    """
    Args:
        arr (np.ndarray): ndarray with shape (..., H, W, C)

    Returns:
        (np.ndarray): ndarray transposed to (..., C, H, W)
    """
    return arr.transpose(tuple(range(0, arr.ndim - 3)) + (-1, -3, -2))


def CHW2HWC(arr: np.ndarray) -> np.ndarray:
    """
    Args:
        arr (np.ndarray): ndarray with shape (..., C, H, W)

    Returns:
        (np.ndarray): ndarray transposed to (..., H, W, C)
    """
    return arr.transpose(tuple(range(0, arr.ndim - 3)) + (-2, -1, -3))


def RGB2BGR(arr: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Args:
        arr (Tensor, ndarray): arr with shape (..., C[RGB])

    Returns:
        arr with shape (..., C[BGR])
    """
    return arr[..., ::-1]


def BGR2RGB(arr: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Args:
        arr (Tensor, ndarray): arr with shape (..., C[BGR])

    Returns:
        arr with shape (..., C[RGB])
    """
    return arr[..., ::-1]


def c1_to_cn(c1_array: np.ndarray, n: int=3):
    """Convert 1 channel img/imgs to n channel imgs by repeat in last dimension

    Args:
        c1_array: ndarray img/imgs
        n: Channel num of converted img

    Returns:
        Converted img with shape (c1_array.shape, n)
    """
    cn_array = np.zeros(c1_array.shape + (n, ), dtype=c1_array.dtype)
    for i in range(n):
        cn_array[..., i] = c1_array
    return cn_array
