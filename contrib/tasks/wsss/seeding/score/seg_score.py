#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2021/5/21 22:58
@File    : seg_score.py
@Software: PyCharm
@Desc    : 
"""
from typing import Tuple

import numpy as np

from .utils import gather_segment, cam_clip_norm_score

__all__ = ["get_seg_score_by_cam_mean"]


def get_seg_score_by_cam_mean(cam: np.ndarray, segment: np.ndarray, cls_in_label: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray]:
    fore_in_label = cls_in_label[1:]
    segment_cam, segment_area = gather_segment(cam[fore_in_label.astype(np.bool_)], segment)
    return cam_clip_norm_score(segment_cam), segment_area
