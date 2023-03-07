#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/7 11:36
@File    : manip_ch.py
@Software: PyCharm
@Desc    : 
"""
import numpy as np
import torch

from typing import Union

__all__ = ["HWC2CHW", "CHW2HWC", "BGR2RGB", "RGB2BGR", "c1_to_cn"]


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
