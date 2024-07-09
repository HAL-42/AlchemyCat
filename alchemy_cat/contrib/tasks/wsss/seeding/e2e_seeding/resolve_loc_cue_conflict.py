#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2021/4/17 14:47
@File    : resolve_loc_cue_conflict.py
@Software: PyCharm
@Desc    : 
"""
from typing import List
import numpy as np

from alchemy_cat.py_tools.type import is_intarr

__all__ = ['resolve_loc_cue_conflict_by_priority', 'resolve_loc_cue_conflict_by_area_order']


def resolve_loc_cue_conflict_by_priority(loc_cue: np.ndarray, priority: List[np.ndarray]) -> np.ndarray:
    """Solve conflict in candidate seed according to priority between channels.

    Args:
        loc_cue: (C, H, W) numpy array. Where (c, h, w) > 0 means channel c is one of the channel of the (h, w)
            pixel in sample n
        priority: Priority among channels. priority[index] = channels who prior to index'th channel

    Returns: (C, H, W) numpy array.
    """
    assert loc_cue.ndim == 3
    assert is_intarr(loc_cue)

    assert len(priority) == loc_cue.shape[0]

    one_hot_seed = loc_cue.copy()
    for ch_idx, prior_channels in enumerate(priority):
        one_hot_seed[ch_idx][np.sum(loc_cue[prior_channels, ...], axis=0) > 0] = 0

    assert np.all((np.sum(one_hot_seed, axis=0) <= 1) & np.sum(one_hot_seed, axis=0) >= 0)

    return one_hot_seed


def resolve_loc_cue_conflict_by_area_order(loc_cue_proposal, ignore_label, train_boat=False):
    """This function generate seed with priority strategy"""
    seed = np.ones(loc_cue_proposal.shape[:2], dtype=np.int64) * ignore_label

    # generate background seed
    seed[loc_cue_proposal[:, :, 0] == 1] = 0

    # generate foreground seed in the order of their area
    area = np.sum(loc_cue_proposal, axis=(0, 1))
    cls_order = np.argsort(area)[::-1]  # area in descending order
    for cls in cls_order:
        if area[cls] == 0:
            break
        seed[loc_cue_proposal[:, :, cls] == 1] = cls

    if train_boat:
        train_boat_ind = ((seed == 4) | (seed == 19)) & (loc_cue_proposal[:, :, 0] == 1)
        seed[train_boat_ind] = 0

    return seed
