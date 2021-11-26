#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2021/6/16 22:49
@File    : pixel_score.py
@Software: PyCharm
@Desc    : 
"""
import numpy as np

from .utils import cam_clip_norm_score, get_fg_cam

__all__ = ["get_pixel_score_by_cam"]


def get_pixel_score_by_cam(cam: np.ndarray, cls_in_label: np.ndarray) -> np.ndarray:
    fg_cam = get_fg_cam(cam, cls_in_label)
    return cam_clip_norm_score(fg_cam)
