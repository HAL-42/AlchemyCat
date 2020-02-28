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

from typing import Union, List, Tuple, Optional, Callable, Iterable

from alchemy_cat.alg.utils import size2HW


def _pad_imgs(imgs, size):
    img_h, img_w =imgs.size(2), imgs.size(3)
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
                       msc_factors: Union[List[int], Tuple[int]], is_flip: bool=True,
                       pad_imgs_to: Union[None, Iterable, int]=None, pad_aligner: Optional[Callable[[int], int]]=None,
                       msc_aligner: Optional[Callable[[int], int]]=None) -> torch.Tensor:
    """MSC and flip inference

    Args:
        imgs (torch.Tensor): imgs with shape (N, C, H, W)
        model (Callable): torch models
        msc_factors (Union[list[int], tuple[int]]): multi scale factors
        is_flip (bool): If true, the flipped imgs will be used for inference
        pad_imgs_to (Union[None, Iterable, int]): If not None, the img will be pad to size(int or [int, int]) specified
        pad_aligner (Callable): If not None, the pad size will be refine to pad_aligner(pad_size)
        msc_aligner (Callable): If not None, the scale size will be refine to msc_aligner(scale_size)

    Returns: Probs of image predicts
    """
    origin_h, origin_w = imgs.shape[-2:]

    # * pad img to pad_img_to size
    if pad_imgs_to is not None:
        padded_imgs = _pad_imgs(imgs, pad_imgs_to)
    else:
        padded_imgs = imgs

    # * pad img to align size
    if pad_aligner is not None:
        align_size = (pad_aligner(s) for s in padded_imgs.shape[-2:])
        padded_imgs = _pad_imgs(padded_imgs, align_size)

    padded_h, padded_w = padded_imgs.shape[-2:]

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
        scaled_h, scaled_w = (int(s * msc_factor) for s in (padded_h, padded_w))
        if msc_aligner is not None:
            scaled_h, scaled_w = msc_aligner(scaled_h), msc_aligner(scaled_w)

        batch_X = F.interpolate(padded_imgs, size=(scaled_h, scaled_w), mode='bilinear', align_corners=True)
        if is_flip:
            batch_X = flip_imgs(batch_X)

        scaled_logits = model(batch_X)
        logits = F.interpolate(scaled_logits, size=(padded_h, padded_w), mode='bilinear', align_corners=True)
        probs = F.softmax(logits, dim=1)
        if is_flip:
            probs = merge_flipped_probs(probs)

        msc_probs.append(probs)
    # * Get probs with padded size
    probs = torch.stack(msc_probs, dim=0).mean(dim=0)

    # * return with size of origin img
    return probs[:, :, :origin_h, :origin_w]
