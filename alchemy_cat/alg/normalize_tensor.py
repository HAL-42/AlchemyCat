#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: normalize_tensor.py
@time: 2020/3/29 22:37
@desc:
"""
import numpy as np
import torch

__all__ = ['min_max_norm', 'channel_min_max_norm']


def min_max_norm(arr: torch.Tensor | np.ndarray, dim: int | tuple, detach_min_max: bool=True,
                 thresh: float | None =None, eps: float=1e-7) -> torch.Tensor:
    """在指定维度min-max归一化输入张量。

    Args:
        arr: 输入张量。
        dim: 做归一化的维度。
        detach_min_max: 是否分离min和max。
        thresh: 若不为None，将归一化前张量中小于thresh的值置为thresh。
        eps: 防止除0错误的极小值。

    Returns:
        归一化后的张量。
    """
    # * 若为numpy数组，转为torch张量。
    if is_numpy := isinstance(arr, np.ndarray):
        arr = torch.from_numpy(arr)

    # * 应用阈值。
    if thresh is not None:
        arr = torch.maximum(arr, torch.tensor(thresh, device=arr.device, dtype=arr.dtype))

    # * 归一化。
    arr_min, arr_max = arr.amin(dim=dim, keepdim=True), arr.amax(dim=dim, keepdim=True)
    if detach_min_max:
        arr_min, arr_max = arr_min.detach(), arr_max.detach()
    arr_normed = (arr - arr_min) / (arr_max - arr_min + eps)  # 其实用max(max - min, 1e-7)更安全（已经归一化过的不变）。

    # * 若为numpy数组，转为numpy数组。
    if is_numpy:
        arr_normed = arr_normed.numpy()

    return arr_normed


def channel_min_max_norm(arr: torch.Tensor, eps: float=1e-5) -> torch.Tensor:
    """Min-Max normalization for (N, C, ...) tensor

    Args:
        arr: (N, C, ...) tensor
        eps: eps added to divider

    Returns:
        Min-Max normalized tensor in dim behind C
    """
    batch_arr = arr.view(arr.shape[0], arr.shape[1], -1)  # (N, C, cumprod(...))

    batch_min, _ = batch_arr.min(dim=-1, keepdim=True)
    batch_arr = batch_arr - batch_min  # Don't modify origin arr

    batch_max, _ = batch_arr.max(dim=-1, keepdim=True)
    batch_arr /= (batch_max + eps)  # Inplace Operation

    return batch_arr.view_as(arr)
