#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: msc_flip_inference.py
@time: 2020/1/16 7:38
@desc:
"""
from collections import abc
from typing import Union, Callable, Iterable

import torch
import torch.nn.functional as F

from alchemy_cat.alg.utils import size2HW

__all__ = ['msc_flip_inference', 'flip_imgs', 'merge_flipped_probs']


def bisection(n: int):
    return n // 2, n // 2 + n % 2


def tensor_pad_imgs(imgs: torch.Tensor, size, value: float=0.) -> tuple[torch.Tensor, list[int]]:
    """中心填充imgs到size，若图片某个尺度大于size，则该方向不填充。

    Args:
        imgs: (N, C, H, W)图片张量。
        size: 要pad到的大小，size: int表示pad到方形size，(h: int, w: int)表示要填充到的高、宽。
        value: 要填充的值，默认为0。

    Returns:
        (填充后的张量，上、下、左、右的填充尺寸)。
    """
    padded_h, padded_w = size2HW(size)

    pad_top, pad_bottom = bisection(max(padded_h - imgs.shape[2], 0))
    pad_left, pad_right = bisection(max(padded_w - imgs.shape[3], 0))

    return F.pad(imgs, [pad_left, pad_right, pad_top, pad_bottom], mode='constant', value=value), [pad_top, pad_bottom,
                                                                                                   pad_left, pad_right]

# def _pad_imgs(imgs, size):
#     img_h, img_w = imgs.size(2), imgs.size(3)
#     padded_h, padded_w = size2HW(size)
#
#     pad_h = padded_h - img_h
#     pad_w = padded_w - img_w
#
#     if pad_h < 0 or pad_w < 0:
#         raise ValueError(f"pad_h={pad_h} or pad_w={pad_w} should equal or larger than 0")
#
#     return F.pad(imgs, [0, pad_w, 0, pad_h], mode='constant', value=0.0)


# def _pad_labels(labels, size, ignore_label):
#     label_h, label_w = labels.size(1), labels.size(2)
#     padded_h, padded_w = size2HW(size)
#
#     pad_h = padded_h - label_h
#     pad_w = padded_w - label_w
#
#     if pad_h < 0 or pad_w < 0:
#         raise ValueError(f"pad_h={pad_h} or pad_w={pad_w} should equal or larger than 0")
#
#     return F.pad(labels, [0, pad_w, 0, pad_h], mode='constant', value=ignore_label)

def flip_imgs(imgs):
    """Flip imgs and cat it N dim"""
    return torch.cat([imgs, torch.flip(imgs, [3])], dim=0)


def merge_flipped_probs(probs):
    """merge flipped probs by calculate the average of origin and flipped prob"""
    dual_logits = probs.view(2, probs.shape[0] // 2, *probs.shape[1:])
    dual_logits[1] = dual_logits[1].flip([3])
    return torch.mean(dual_logits, dim=0)


def msc_flip_inference(imgs: torch.Tensor, model: Callable[[torch.Tensor], torch.Tensor],
                       msc_factors: Iterable[Union[Iterable[float], float]],
                       is_flip: bool=True,
                       pad_imgs_to: Union[None, Iterable, int]=None,
                       pad_aligner: Union[Callable[[int], int], Iterable[Callable[[int], int]]]=lambda x: x,
                       msc_aligner: Union[Callable[[int], int], Iterable[Callable[[int], int]]]=lambda x: x,
                       align_corners: bool=False,
                       cuda_memory_saving: int=0,
                       softmax_norm: bool=True) \
        -> torch.Tensor:
    """MSC and flip inference

    Args:
        imgs: Imgs with shape (N, C, H, W)
        model: torch models
        msc_factors: Multi scale factors. Each factor can be 'factor_for_height_width' or
            '(height_factor, width_factor)'.
        is_flip: If true, the flipped imgs will be used for inference
        pad_imgs_to: If not None, the img will be padded to size(int or [int, int]) specified
        pad_aligner: The pad_size will be fixed by aligner(pad_size). If Iterable, then first and second aligners
            separately used to align H and W.
        msc_aligner: The scaled_size calculated by scale_factor * img_size will be fixed by aligner(scaled_size). If
            Iterable, then first and second aligners separately used to align H and W.
        align_corners: If True, use `align_corners` parameter for torch.nn.functional.interpolate.
        cuda_memory_saving: int before 0-2. The larger less_cuda_memory set, the less gpu memory used by function. May
            loss some performance. (Default: 0)
        softmax_norm: If Ture, the output logit will be normalized by softmax then fusion between different scales. Else
            the output logit will be implemented as prob and directly fusion and output.

    Returns: (N, C, H, W) probs of image predicts
    """
    origin_h, origin_w = imgs.shape[-2:]

    # * For saving more gpu memory, move imgs to cpu.
    if cuda_memory_saving > 1:
        imgs = imgs.cpu()

    # * Process pad aligner
    if isinstance(pad_aligner, abc.Callable):
        pad_aligner_h, pad_aligner_w = pad_aligner, pad_aligner
    elif isinstance(pad_aligner, abc.Iterable):
        pad_aligner_h, pad_aligner_w = pad_aligner
    else:
        raise ValueError(f"pad_aligner {pad_aligner} must be Callable or Iterable[Callable]")

    # * Process scale aligner
    if isinstance(msc_aligner, abc.Callable):
        msc_aligner_h, msc_aligner_w = msc_aligner, msc_aligner
    elif isinstance(msc_aligner, abc.Iterable):
        msc_aligner_h, msc_aligner_w = msc_aligner
    else:
        raise ValueError(f"msc_aligner {msc_aligner} must be Callable or Iterable[Callable]")

    # * Get probs from each scale
    msc_probs = []
    for msc_factor in msc_factors:
        # * Process msc factor
        if isinstance(msc_factor, float):
            msc_factor_h, msc_factor_w = msc_factor, msc_factor
        elif isinstance(msc_factor, abc.Iterable):
            msc_factor_h, msc_factor_w = msc_factor
        else:
            raise ValueError(f"msc_factor {msc_factor} in msc_factors {msc_factors} must be float or Iterable[float]")

        # * Get scaled size
        scaled_h, scaled_w = round(origin_h * msc_factor_h), round(origin_w * msc_factor_w)
        scaled_h, scaled_w = msc_aligner_h(scaled_h), msc_aligner_w(scaled_w)

        batch_X = F.interpolate(imgs, size=(scaled_h, scaled_w), mode='bilinear', align_corners=align_corners)
        if is_flip:
            batch_X = flip_imgs(batch_X)

        # * pad img to align(pad_img_to) size
        pad_imgs_to_h, pad_imgs_to_w = size2HW(pad_imgs_to) if pad_imgs_to is not None else (scaled_h, scaled_w)
        pad_align_size = (pad_aligner_h(pad_imgs_to_h), pad_aligner_w(pad_imgs_to_w))
        padded_batch_X, margin = tensor_pad_imgs(batch_X, pad_align_size)
        padded_h, padded_w = padded_batch_X.shape[-2:]

        padded_logits = model(padded_batch_X.cuda())

        if cuda_memory_saving > 0:
            padded_logits = padded_logits.cpu()

        if margin == [0, 0, 0, 0]:
            logits = F.interpolate(padded_logits, size=(origin_h, origin_w),
                                   mode='bilinear', align_corners=align_corners)
        else:
            logits = F.interpolate(padded_logits, size=(padded_h, padded_w),
                                   mode='bilinear', align_corners=align_corners)
            logits = logits[:, :, margin[0]:(logits.shape[2] - margin[1]), margin[2]:(logits.shape[3] - margin[3])]
            logits = F.interpolate(logits, size=(origin_h, origin_w),
                                   mode='bilinear', align_corners=align_corners)
        probs = F.softmax(logits, dim=1) if softmax_norm else logits
        if is_flip:
            probs = merge_flipped_probs(probs)

        msc_probs.append(probs)
    # * Get probs with padded size
    probs = torch.stack(msc_probs, dim=0).mean(dim=0)

    # * return with size of origin img
    return probs
