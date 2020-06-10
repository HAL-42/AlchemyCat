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
import torch
import torch.nn.functional as F

from typing import Union, Callable, Iterable
from collections import abc

from alchemy_cat.alg.utils import size2HW


def _pad_imgs(imgs, size):
    img_h, img_w = imgs.size(2), imgs.size(3)
    padded_h, padded_w = size2HW(size)

    pad_h = padded_h - img_h
    pad_w = padded_w - img_w

    if pad_h < 0 or pad_w < 0:
        raise ValueError(f"pad_h={pad_h} or pad_w={pad_w} should equal or larger than 0")

    return F.pad(imgs, [0, pad_w, 0, pad_h], mode='constant', value=0.0)


def _pad_labels(labels, size, ignore_label):
    label_h, label_w = labels.size(1), labels.size(2)
    padded_h, padded_w = size2HW(size)

    pad_h = padded_h - label_h
    pad_w = padded_w - label_w

    if pad_h < 0 or pad_w < 0:
        raise ValueError(f"pad_h={pad_h} or pad_w={pad_w} should equal or larger than 0")

    return F.pad(labels, [0, pad_w, 0, pad_h], mode='constant', value=ignore_label)


def msc_flip_inference(imgs: torch.Tensor, model: Callable[[torch.Tensor], torch.Tensor],
                       msc_factors: Iterable[Union[Iterable[float], float]],
                       is_flip: bool=True,
                       pad_imgs_to: Union[None, Iterable, int]=None,
                       pad_aligner: Union[Callable[[int], int], Iterable[Callable[[int], int]]]=lambda x: x,
                       msc_aligner: Union[Callable[[int], int], Iterable[Callable[[int], int]]]=lambda x: x,
                       cuda_memory_saving: int=0) \
        -> torch.Tensor:
    """MSC and flip inference

    Args:
        imgs: Imgs with shape (N, C, H, W)
        model: torch models
        msc_factors: Multi scale factors. Each factor can be 'factor_for_height_width' or
            '(height_factor, width_factor)'.
        is_flip: If true, the flipped imgs will be used for inference
        pad_imgs_to: If not None, the img will be pad to size(int or [int, int]) specified
        pad_aligner: The pad_size will be fix by aligner(pad_size). If Iterable, then first and second aligners
            separately used to align H and W.
        msc_aligner: The scaled_size calculated by scale_factor * img_size will be fix by aligner(scaled_size). If
            Iterable, then first and second aligners separately used to align H and W.
        cuda_memory_saving: int before 0-2. The larger the less_cuda_memory, the less gpu memory used by function. May
            loss some performance. (Default: 0)

    Returns: (N, C, H, W) probs of image predicts
    """
    origin_h, origin_w = imgs.shape[-2:]

    # * For saving more gpu memory, move imgs to cpu.
    if cuda_memory_saving > 1:
        imgs = imgs.cpu()

    # * pad img to pad_img_to size
    if pad_imgs_to is not None:
        padded_imgs = _pad_imgs(imgs, pad_imgs_to)
    else:
        padded_imgs = imgs

    # * Process pad aligner
    if isinstance(pad_aligner, abc.Callable):
        pad_aligner_h, pad_aligner_w = pad_aligner, pad_aligner
    elif isinstance(pad_aligner, abc.Iterable):
        pad_aligner_h, pad_aligner_w = pad_aligner
    else:
        raise ValueError(f"pad_aligner {pad_aligner} must be Callable or Iterable[Callable]")

    # * pad img to align size
    pad_align_size = (pad_aligner_h(padded_imgs.shape[-2]), pad_aligner_w(padded_imgs.shape[-1]))
    padded_imgs = _pad_imgs(padded_imgs, pad_align_size)

    padded_h, padded_w = padded_imgs.shape[-2:]

    # * Process scale aligner
    if isinstance(msc_aligner, abc.Callable):
        msc_aligner_h, msc_aligner_w = msc_aligner, msc_aligner
    elif isinstance(msc_aligner, abc.Iterable):
        msc_aligner_h, msc_aligner_w = msc_aligner
    else:
        raise ValueError(f"msc_aligner {msc_aligner} must be Callable or Iterable[Callable]")

    def flip_imgs(imgs):
        """Flip imgs and cat it N dim"""
        return torch.cat([imgs, torch.flip(imgs, [3])], dim=0)

    def merge_flipped_probs(probs):
        """merge flipped probs by calculate the average of origin and flipped prob"""
        dual_logits = probs.view(2, probs.shape[0] // 2, *probs.shape[1:])
        dual_logits[1] = dual_logits[1].flip([3])
        return torch.mean(dual_logits, dim=0)

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
        scaled_h, scaled_w = int(padded_h * msc_factor_h), int(padded_w * msc_factor_w)
        scaled_h, scaled_w = msc_aligner_h(scaled_h), msc_aligner_w(scaled_w)

        batch_X = F.interpolate(padded_imgs, size=(scaled_h, scaled_w), mode='bilinear', align_corners=True)
        if is_flip:
            batch_X = flip_imgs(batch_X)

        scaled_logits = model(batch_X.cuda())

        if cuda_memory_saving > 0:
            scaled_logits = scaled_logits.cpu()

        logits = F.interpolate(scaled_logits, size=(padded_h, padded_w), mode='bilinear', align_corners=True)
        probs = F.softmax(logits, dim=1)
        if is_flip:
            probs = merge_flipped_probs(probs)

        msc_probs.append(probs)
    # * Get probs with padded size
    probs = torch.stack(msc_probs, dim=0).mean(dim=0)

    # * return with size of origin img
    return probs[:, :, :origin_h, :origin_w]
