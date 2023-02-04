#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2022/6/20 22:00
@File    : slide_inference.py
@Software: PyCharm
@Desc    : 滑动窗口法推理图片。
"""
from typing import Callable, Union, Iterable, Tuple, List

from math import ceil

import torch
import torch.nn.functional as F

from alchemy_cat.alg import size2HW, find_nearest_odd_size

__all__ = ['slide_inference', 'bisection']


def bisection(n: int):
    return n // 2, n // 2 + n % 2


def _pad_imgs(imgs: torch.Tensor, padded_h: int, padded_w: int) -> Tuple[torch.Tensor, List[int]]:
    pad_top, pad_bottom = bisection(max(padded_h - imgs.shape[2], 0))
    pad_left, pad_right = bisection(max(padded_w - imgs.shape[3], 0))

    return F.pad(imgs, [pad_left, pad_right, pad_top, pad_bottom], mode='constant', value=0.0), [pad_top, pad_bottom,
                                                                                                 pad_left, pad_right]


def slide_inference(imgs: torch.Tensor, model: Callable[[torch.Tensor], torch.Tensor], num_class: int,
                    window_sizes: Union[Iterable, int], strides: Union[Iterable, int],
                    pad: bool=False,
                    win_size_checker: Callable[[int], bool]=lambda x: find_nearest_odd_size(x, min_n=4) == x)\
        -> torch.Tensor:
    """滑动窗口法推理图片。

    Args:
        imgs: 待推理的(N, 3, H, W)图片。
        model: 推理函数，输入(N, 3, H, W)图片，输出得分。
        num_class: 模型logit输出通道数。
        window_sizes: 滑动窗口尺寸。若为可迭代对象，表(win_h, win_w)；若为int，则正方形窗口的边长。
        strides: 滑动窗口步长。若为可迭代对象，表(stride_h, stride_w)；若为int，表步长之于高宽上一致者。
        pad: 是否在推理前，将输入图片填充到滑动窗口尺寸。注意，若没有pad，当确保输入图片尺寸为模型支持的尺寸。
        win_size_checker: 检查最后真实切出的窗口，其尺寸是否符合要求。默认检查是否为n=4的奇尺寸。

    Returns:
        与输入imgs等大的(N, num_class, H, W)得分。
    """
    # * 解析输入参数。
    win_size_h, win_size_w = size2HW(window_sizes)
    stride_h, stride_w = size2HW(strides)

    # * 填充图片。
    if pad:
        padded_imgs, margin = _pad_imgs(imgs, win_size_h, win_size_w)
    else:
        padded_imgs, margin = imgs, [0] * 4

    # * 初始化得分和累加次数。
    logits = torch.zeros((padded_imgs.shape[0], num_class, padded_imgs.shape[2], padded_imgs.shape[3]),
                         dtype=torch.float32, device=padded_imgs.device)
    add_count = torch.zeros((padded_imgs.shape[2], padded_imgs.shape[3]),
                            dtype=torch.int32, device=padded_imgs.device)

    # * 计算滑动窗口。
    h_grids = ceil(max(padded_imgs.shape[2] - win_size_h, 0) / stride_h) + 1
    w_grids = ceil(max(padded_imgs.shape[3] - win_size_w, 0) / stride_w) + 1

    # * 逐窗口推理、累加。
    for h_idx in range(h_grids):
        y1 = h_idx * stride_h
        y2 = min(y1 + win_size_h, padded_imgs.shape[2])
        y1 = max(y2 - win_size_h, 0)
        for w_idx in range(w_grids):
            x1 = w_idx * stride_w
            x2 = min(x1 + win_size_w, padded_imgs.shape[3])
            x1 = max(x2 - win_size_w, 0)
            cropped_imgs = padded_imgs[:, :, y1:y2, x1:x2]

            # assert find_nearest_odd_size(cropped_imgs.shape[2], min_n=4) == cropped_imgs.shape[2]
            # assert find_nearest_odd_size(cropped_imgs.shape[3], min_n=4) == cropped_imgs.shape[3]
            assert win_size_checker(cropped_imgs.shape[2])
            assert win_size_checker(cropped_imgs.shape[3])

            cropped_logits = F.interpolate(model(cropped_imgs), size=(cropped_imgs.shape[2], cropped_imgs.shape[3]),
                                           mode='bilinear', align_corners=True)

            logits[:, :, y1:y2, x1:x2] += cropped_logits
            add_count[y1:y2, x1:x2] += 1

    # * 求平均得分。
    assert torch.all(add_count != 0)
    logits /= add_count

    # * 后处理得到与输入等大的得分图。
    return logits[:, :, margin[0]:(logits.shape[2] - margin[1]), margin[2]:(logits.shape[3] - margin[3])]
