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
import torch

__all__ = ['channel_min_max_norm']


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
