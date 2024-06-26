#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2021/5/20 22:57
@File    : utils.py
@Software: PyCharm
@Desc    : 
"""
from typing import Tuple, Callable

import numpy as np
import torch

from alchemy_cat.alg import channel_min_max_norm

__all__ = ["gather_segment", "scatter_segment", "cam_clip_norm_score", "get_fg_cam"]


def gather_segment(arr: np.ndarray, segment: np.ndarray,
                   gather_func: Callable[[np.ndarray], np.ndarray] = lambda vec: np.mean(vec, axis=1)) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Gather array's vector in each segment with gather func.

    Args:
        arr: (C, H, W) arr to be gathered.
        segment: (H, W) segment mask, each segment has it's unique int label.
        gather_func: Callable to gather (C, segment_area) arr in segment to a vector with length C.

    Returns:
        (C, segment_num) gathered segment vector, (segment_num, ) segment areas.
    """
    segment_indices = np.unique(segment)

    segment_vecs = []
    segment_areas = []

    for segment_idx in segment_indices:
        vec_in_segment = arr[:, segment == segment_idx]
        segment_areas.append(vec_in_segment.shape[1])
        segment_vecs.append(gather_func(vec_in_segment))

    return np.stack(segment_vecs, axis=1), np.array(segment_areas, dtype=np.int64)


def scatter_segment(segment_vec: np.ndarray, segment: np.ndarray) -> np.ndarray:
    """Scatter segment_vec to each segment.

    Args:
        segment_vec: (C, segment_num) segment vector.
        segment: (H, W) segment mask, each segment has it's unique int label.

    Returns:
        (C, H, W) arr scattered from segment_vec according to segment.
    """
    scattered_arr = np.zeros((segment_vec.shape[0], ) + segment.shape, dtype=segment_vec.dtype)

    segment_indices = np.unique(segment)

    for i, segment_idx in enumerate(segment_indices):
        scattered_arr[:, segment == segment_idx] = segment_vec[:, i][:, None]

    return scattered_arr


def cam_clip_norm_score(cam: np.ndarray) -> np.ndarray:
    clip_cam = np.clip(cam, .0, None)
    return channel_min_max_norm(torch.from_numpy(clip_cam[None, ...]), 1e-5).numpy()[0]


def get_fg_cam(cam: np.ndarray, cls_in_label: np.ndarray):
    fore_in_label = cls_in_label[1:]
    return cam[fore_in_label.astype(bool)]
