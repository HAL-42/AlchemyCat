#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2021/5/23 19:52
@File    : solve_conflict.py
@Software: PyCharm
@Desc    : 
"""
from typing import Optional

import numpy as np

from .utils import cls_in_label2tag_always_bg

__all__ = ["solve_conflict_by_area", "solve_conflict_on_channel"]


def solve_conflict_by_area(loc_cue: np.ndarray, cls_in_label: np.ndarray, area: Optional[np.ndarray]=None,
                           ignore_label: int=255, train_boat: bool=True) -> np.ndarray:
    """Solve location cue's conflict by make class with small area to have higher priority.

    Args:
        loc_cue: (fore_class + 1, ...) location cue.
        cls_in_label: (class_num, ) image level label.
        area: The same shape with loc_cue, indicate location's area. If None, set all area to 1. (Default: None)
        ignore_label: int ignore label. (Default: 255)
        train_boat: If True, suppress train_boat's priority after background. (Default: True)

    Returns:
        (...) numeric pred.
    """
    pred = np.ones(loc_cue.shape[1:], dtype=np.int64) * ignore_label

    bg_loc_cue, fg_loc_cue = loc_cue[0], loc_cue[1:]
    fg_tag = cls_in_label2tag_always_bg(cls_in_label)[1:]

    # * Generate background pred
    pred[bg_loc_cue == 1] = 0

    # * Generate foreground seed in the order of their area
    # ** Compute fg area
    area = area if area is not None else np.ones(loc_cue.shape[1:], dtype=np.int64)
    fg_area = np.sum((fg_loc_cue * area).reshape(fg_loc_cue.shape[0], -1), axis=-1)
    # ** Plot cue on pred
    argsort_idx = np.argsort(fg_area)[::-1]  # area in descending order
    for idx in argsort_idx:
        if fg_area[idx] == 0:
            break
        pred[fg_loc_cue[idx] == 1] = fg_tag[idx]

    if train_boat:
        train_boat_ind = ((pred == 4) | (pred == 19)) & (bg_loc_cue == 1)
        pred[train_boat_ind] = 0

    return pred


def solve_conflict_on_channel(loc_cue: np.ndarray, cls_in_label: np.ndarray,
                              channelwise_pred: np.ndarray, ignore_label: int=255) -> np.ndarray:
    """Solve location cue's conflict by run channelwise prediction.

    Args:
        loc_cue: (fore_class + 1, ...) location cue.
        cls_in_label: (class_num, ) image level label.
        channelwise_pred: Channelwise prediction.
        ignore_label: int ignore label. (Default: 255)

    Returns:
        (...) numeric pred.
    """
    pred = np.ones(loc_cue.shape[1:], dtype=np.int64) * ignore_label

    loc_cue_sum = np.sum(loc_cue, axis=0)
    confident_loc = loc_cue_sum == 1
    conflict_loc = loc_cue_sum > 1

    tag = cls_in_label2tag_always_bg(cls_in_label)

    pred[confident_loc] = tag[np.argmax(loc_cue, axis=0), ][confident_loc]  # bg will have sum == 0
    pred[conflict_loc] = channelwise_pred[conflict_loc]

    return pred
