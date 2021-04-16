#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: clamp_softmax.py
@time: 2020/3/28 23:02
@desc:
"""
import torch
import torch.nn.functional as F

__all__ = ['clamp_probs', 'clamp_softmax']


def clamp_probs(probs: torch.Tensor, eps: float=1e-5) -> torch.Tensor:
    """Clamp and normalize probs to make sure that min prob in probs won't be too small

    Args:
        probs: (N, C, H, W) probs
        eps: Min prob allowed when clamp probs

    Returns:
        Clamped and normalized probs with shape (N, C, H, W)
    """
    clamped_probs = torch.clamp(probs, eps, 1)
    normalized_probs = clamped_probs / torch.sum(clamped_probs, dim=1, keepdim=True)
    return normalized_probs


def clamp_softmax(score_map: torch.Tensor, eps: float=1e-5):
    """Calculate probs from score map, then clamp and normalize probs to make sure that min prob in probs won't be
    too small

    Args:
        score_map: (N, C, H, W) score map
        eps: Min prob allowed when clamp probs

    Returns:
        Clamped and normalized probs with shape (N, C, H, W)
    """
    return clamp_probs(F.softmax(score_map, 1), eps)
