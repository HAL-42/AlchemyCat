#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/30 20:18
@File    : mask.py
@Software: PyCharm
@Desc    : 
"""
import torch
import numpy as np

__all__ = ['thresh_mask']


def thresh_mask(data: dict, thresh: float, ignore_label: int=255) -> np.ndarray:
    """根据bg_fg_score的最大值，将小于阈值的seed置为ignore_label。

    Args:
        data: data['seed']为种子，data['bg_fg_score']为前背景得分。
        thresh: 小于阈值的种子点被置为ignore_label。
        ignore_label: 默认255。

    Returns:
        伪真值。
    """
    seed, fg_bg_score = data['seed'].copy(), data['bg_fg_score']  # (H, W), (P+1, H, W)

    max_score = torch.asarray(fg_bg_score, device='cuda').amax(dim=0)  # (H, W)
    ignore_mask = (max_score < thresh).cpu().numpy()

    seed[ignore_mask] = ignore_label

    return seed
