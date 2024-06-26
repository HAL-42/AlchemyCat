#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2021/5/23 19:52
@File    : loc_cue.py
@Software: PyCharm
@Desc    : 
"""
import numpy as np

__all__ = ["get_loc_cue_max_thresh", "get_loc_cue_contrib_thresh"]


def get_loc_cue_max_thresh(bg_fg_score: np.ndarray, thresh) -> np.ndarray:
    """Get localization cues according bg_fg_score.

    Args:
        bg_fg_score: (class_num, ...) score.
        thresh: hard threshold to extract cues.

    Returns:
        (class_num, ...) loc cue.
    """
    return (bg_fg_score > thresh).astype(np.uint8)


def get_loc_cue_contrib_thresh(bg_fg_score: np.ndarray, thresh) -> np.ndarray:
    """Get localization cues according bg_fg_score.

    Args:
        bg_fg_score: (class_num, ...) score.
        thresh: contrib threshold to extract cues.

    Returns:
        (class_num, ...) loc cue.
    """
    flat_score = bg_fg_score.reshape(bg_fg_score.shape[0], -1)
    loc_cue = np.zeros(shape=flat_score.shape, dtype=np.uint8)

    # * L1 Norm
    flat_score /= np.sum(flat_score, axis=1, keepdims=True)

    # * Sort
    bg_fg_argsort = np.argsort(flat_score, axis=1)[:, ::-1]
    bg_fg_sort = np.sort(flat_score, axis=1)[:, ::-1]
    bg_fg_cumsum = np.cumsum(bg_fg_sort, axis=1)

    # Set cue
    for bg_fg_idx in range(flat_score.shape[0]):
        cue_num = 0
        for score_cum_sum in bg_fg_cumsum[bg_fg_idx]:
            cue_num += 1
            if score_cum_sum > thresh:
                break

        loc_cue[bg_fg_idx, bg_fg_argsort[bg_fg_idx, :cue_num]] = 1

    return loc_cue.reshape(bg_fg_score.shape)
