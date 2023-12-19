#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/16 15:37
@File    : score.py
@Software: PyCharm
@Desc    : 
"""
import numpy as np
import torch
import torch.nn.functional as F
from alchemy_cat.alg import size2HW, resize_cam, resize_cam_cuda, min_max_norm

__all__ = ['cam2score', 'cam2score_cuda', 'cat_bg_score', 'cat_bg_score_cuda', 'idx2seed', 'idx2seed_cuda']


def cam2score(cam: np.ndarray, dsize=None, resize_first: bool=True) -> np.ndarray:
    h, w = size2HW(dsize) if dsize is not None else cam.shape[1:]

    if resize_first:
        score = resize_cam(cam, (h, w))
        score = min_max_norm(score, dim=(1, 2), thresh=0.)
    else:
        score = min_max_norm(cam, dim=(1, 2), thresh=0.)
        score = resize_cam(score, (h, w))

    return score


def cam2score_cuda(cam: torch.Tensor, dsize=None, resize_first: bool=True) -> torch.Tensor:
    h, w = size2HW(dsize) if dsize is not None else cam.shape[1:]

    if resize_first:
        score = resize_cam_cuda(cam, (h, w))
        score = min_max_norm(score, dim=(1, 2), thresh=0.)
    else:
        score = min_max_norm(cam, dim=(1, 2), thresh=0.)
        score = resize_cam_cuda(score, (h, w))

    return score


def cat_bg_score(fg_score: np.ndarray, bg_method: dict) -> np.ndarray:
    """计算背景得分并拼接。

    Args:
        fg_score: (C, H, W) 的得分。
        bg_method: 背景得分计算方法。

    Returns:
        (C+1, H, W) 的前背景得分。
    """
    match bg_method:
        case {'method': 'thresh', 'thresh': thresh}:
            bg_score = np.full_like(fg_score[0], thresh)[None, ...]
        case {'method': 'pow', 'pow': p}:
            bg_score = np.power(1 - np.max(fg_score, axis=0, keepdims=True), p)
        case {'method': 'no_bg'}:  # 无需添加bg，为兼容性而设。
            bg_score = np.empty((0, *fg_score.shape[1:]), dtype=fg_score.dtype)
        case {'method': 'alpha_bg', 'alpha': alpha}:  # bg存在，调整bg得分（做alpha变换）。
            fg_score = fg_score.copy()
            fg_score[0, ...] = fg_score[0, ...] ** alpha
            bg_score = np.empty((0, *fg_score.shape[1:]), dtype=fg_score.dtype)
        case _:
            raise ValueError(f'Unknown bg_method: {bg_method}')

    bg_fg_score = np.concatenate((bg_score, fg_score), axis=0)

    return bg_fg_score


def cat_bg_score_cuda(fg_score: torch.Tensor, bg_method: dict) -> torch.Tensor:
    """计算背景得分并拼接。

    Args:
        fg_score: (C, H, W) 的得分。
        bg_method: 背景得分计算方法。

    Returns:
        (C+1, H, W) 的前背景得分。
    """
    match bg_method:
        case {'method': 'thresh', 'thresh': thresh}:
            bg_score = torch.full_like(fg_score[0], thresh)[None, ...]
        case {'method': 'pow', 'pow': p}:
            bg_score = torch.pow(1 - torch.amax(fg_score, dim=0, keepdim=True), p)
        case {'method': 'no_bg'}:  # 无需添加bg，为兼容性而设。
            bg_score = torch.empty((0, *fg_score.shape[1:]), device=fg_score.device, dtype=fg_score.dtype)
        case {'method': 'alpha_bg', 'alpha': alpha}:  # bg存在，调整bg得分（做alpha变换）。
            fg_score = fg_score.clone()
            fg_score[0, ...] = fg_score[0, ...] ** alpha
            bg_score = torch.empty((0, *fg_score.shape[1:]), device=fg_score.device, dtype=fg_score.dtype)
        case _:
            raise ValueError(f'Unknown bg_method: {bg_method}')

    bg_fg_score = torch.cat((bg_score, fg_score), dim=0)

    return bg_fg_score


def idx2seed(idx: np.ndarray, fg_cls: np.ndarray) -> np.ndarray:
    """将索引张量寻址为对应的种子张量。

    Args:
        idx: (H, W) 的索引张量。
        fg_cls: (C,) 的前景类别张量。

    Returns:
        (H, W) 的种子张量。
    """
    cls_lb = np.pad(fg_cls + 1, (1, 0), mode='constant', constant_values=0)
    seed = cls_lb[idx]

    return seed


def idx2seed_cuda(idx: torch.Tensor, fg_cls: torch.Tensor) -> torch.Tensor:
    """将索引张量寻址为对应的种子张量。

    Args:
        idx: (H, W) 的索引张量。
        fg_cls: (C,) 的前景类别张量。

    Returns:
        (H, W) 的种子张量。
    """
    cls_lb = F.pad(fg_cls + 1, (1, 0), mode='constant', value=0)
    seed = cls_lb[idx]

    return seed
