#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2021/5/23 19:52
@File    : score2seed.py
@Software: PyCharm
@Desc    : 
"""
from typing import Optional, Callable

import numpy as np

from .utils import get_bg_fg_score, cls_in_label2tag_always_bg, complement_entropy_with_shift
from .loc_cue import get_loc_cue_max_thresh, get_loc_cue_contrib_thresh
from .solve_conflict import solve_conflict_by_area, solve_conflict_on_channel

__all__ = ["cw_pred_entropy_thresh", "sw_pred_max_thresh_solve_area", "sw_pred_contrib_thresh_solve_channel",
           "sw_pred_contrib_thresh_solve_area", "sw_pred_max_thresh_solve_channel"]


def cw_pred_entropy_thresh(score: np.ndarray, cls_in_label: np.ndarray, alpha: float = 6.,
                           thresh: float = .05, shift: float = .1, ignore_label: int = 255,
                           score_refiner: Callable[[np.ndarray], np.ndarray]=lambda x: x) -> np.ndarray:
    bg_fg_score = score_refiner(get_bg_fg_score(score, alpha))

    pred = np.ones(score.shape[1:], dtype=np.int64) * ignore_label

    comp_entropy = complement_entropy_with_shift(bg_fg_score[None, ...], shift)[0]
    confident_loc = comp_entropy > thresh

    tag = cls_in_label2tag_always_bg(cls_in_label)

    pred[confident_loc] = tag[np.argmax(bg_fg_score, axis=0), ][confident_loc]

    return pred


def sw_pred_max_thresh_solve_area(score: np.ndarray, cls_in_label: np.ndarray,
                                  thresh: float = .2, alpha: float = 6.,
                                  area: Optional[np.ndarray] = None, ignore_label: int = 255, train_boat: bool = True,
                                  score_refiner: Callable[[np.ndarray], np.ndarray]=lambda x: x) -> np.ndarray:
    bg_fg_score = score_refiner(get_bg_fg_score(score, alpha))
    loc_cue = get_loc_cue_max_thresh(bg_fg_score, thresh)
    return solve_conflict_by_area(loc_cue, cls_in_label, area, ignore_label, train_boat)


def sw_pred_max_thresh_solve_channel(score: np.ndarray, cls_in_label: np.ndarray,
                                     s_thresh: float = .2, s_alpha: float = 6.,
                                     c_alpha: float = 6., c_thresh: float = .05, shift: float = .1,
                                     ignore_label: int = 255,
                                     score_refiner: Callable[[np.ndarray], np.ndarray]=lambda x: x) -> np.ndarray:
    cw_pred = cw_pred_entropy_thresh(score, cls_in_label, c_alpha, c_thresh, shift, ignore_label, score_refiner)

    bg_fg_score = score_refiner(get_bg_fg_score(score, s_alpha))
    loc_cue = get_loc_cue_max_thresh(bg_fg_score, s_thresh)
    return solve_conflict_on_channel(loc_cue, cls_in_label, cw_pred, ignore_label)


def sw_pred_contrib_thresh_solve_area(score: np.ndarray, cls_in_label: np.ndarray,
                                      thresh: float = .8, alpha: float = 6.,
                                      area: Optional[np.ndarray] = None, ignore_label: int = 255,
                                      train_boat: bool = True,
                                      score_refiner: Callable[[np.ndarray], np.ndarray]=lambda x: x) -> np.ndarray:
    bg_fg_score = score_refiner(get_bg_fg_score(score, alpha))
    loc_cue = get_loc_cue_contrib_thresh(bg_fg_score, thresh)
    return solve_conflict_by_area(loc_cue, cls_in_label, area, ignore_label, train_boat)


def sw_pred_contrib_thresh_solve_channel(score: np.ndarray, cls_in_label: np.ndarray,
                                         s_thresh: float = .8, s_alpha: float = 6.,
                                         c_alpha: float = 6., c_thresh: float = .05, shift: float = .1,
                                         ignore_label: int = 255,
                                         score_refiner: Callable[[np.ndarray], np.ndarray]=lambda x: x) -> np.ndarray:
    cw_pred = cw_pred_entropy_thresh(score, cls_in_label, c_alpha, c_thresh, shift, ignore_label, score_refiner)

    bg_fg_score = score_refiner(get_bg_fg_score(score, s_alpha))
    loc_cue = get_loc_cue_contrib_thresh(bg_fg_score, s_thresh)
    return solve_conflict_on_channel(loc_cue, cls_in_label, cw_pred, ignore_label)
