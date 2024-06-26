#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: cam_sal_to_seed.py
@time: 2020/3/27 1:10
@desc:
"""
import numpy as np
from alchemy_cat.contrib.tasks.wsss.seeding.e2e_seeding.resolve_loc_cue_conflict \
    import resolve_loc_cue_conflict_by_area_order

__all__ = ["cam_sal_to_seed"]


def cam_sal_to_seed(cam, sal, cls_in_label, cam_thresh, sal_thresh, ignore_label) -> np.ndarray:
    """Get localization cues with method in SEC paper

    Perform hard threshold for each foreground class

    Args:
        cam: (H, W, num_class - 1) cam
        sal: (H, W) saliency map
        cls_in_label: list of foreground classes
        cam_thresh: hard threshold to extract foreground class cues
        sal_thresh: hard threshold to extract background class cues
        ignore_label: ignore label in class cues

    Returns:
        (H, W) seed
    """
    loc_cue_proposal = np.zeros(shape=(cam.shape[0], cam.shape[1], cam.shape[2] + 1), dtype=np.int64)  # (H, W, num_class)
    for cls_idx in range(1, len(cls_in_label)):
        if cls_in_label[cls_idx] == 1:
            heat_map = cam[:, :, cls_idx - 1]
            loc_cue_proposal[:, :, cls_idx] = heat_map > cam_thresh * np.amax(heat_map)

    if cls_in_label[0] == 1:
        loc_cue_proposal[:, :, 0] = sal < sal_thresh

    # handle conflict seed
    seed = resolve_loc_cue_conflict_by_area_order(loc_cue_proposal, ignore_label, train_boat=True)

    return seed
