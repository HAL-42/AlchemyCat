#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: online_seed.py
@time: 2020/3/29 23:19
@desc:
"""
import numpy as np
import torch

from alchemy_cat.alg.normalize_tensor import channel_min_max_norm
from alchemy_cat.contrib.tasks.wsss.seeding.e2e_seeding.cam_sal_to_seed import cam_sal_to_seed

__all__ = ['get_online_seed_from_cam_sal_seg']


def get_online_seed_from_cam_sal_seg(cam: torch.Tensor, sal: torch.Tensor,
                                     seg_prob: torch.Tensor, cls_in_label: torch.Tensor,
                                     cam_thresh: float, sal_thresh: float, ignore_label: int) -> torch.Tensor:
    """Get online seed from CAM, saliency and segmentation prob output

    Args:
        cam: (N, num_class - 1, H, W) Class Activation Map
        sal: (N, H, W) Saliency
        seg_prob: (N, num_class, H, W) prob output of segmentation branch
        cls_in_label: (N, num_class) one hot label
        cam_thresh: cam lower bound
        sal_thresh: cam upper bound
        ignore_label: label ignored

    Returns:
        Online seeds
    """
    clamp_cam = torch.clamp(cam, min=0.0)

    norm_cam = channel_min_max_norm(clamp_cam)

    comb_cam = norm_cam + seg_prob[:, 1:, ...]

    seeds = []
    for ele_comb_cam, ele_sal, ele_cls_in_label in zip(comb_cam, sal, cls_in_label):
        np_comb_cam, np_sal, np_cls_in_label = ele_comb_cam.cpu().numpy().transpose(1, 2, 0), \
                                               ele_sal.cpu().numpy(), \
                                               ele_cls_in_label.cpu().numpy()

        loc_cue = cam_sal_to_seed(np_comb_cam, np_sal, np_cls_in_label, cam_thresh, sal_thresh, ignore_label)

        seed = np.zeros(seg_prob.shape[1:], dtype=np.int64)
        non_ignore_indices = np.nonzero(loc_cue != ignore_label)
        seed[loc_cue[non_ignore_indices], non_ignore_indices[0], non_ignore_indices[1]] = 1

        seeds.append(seed)

    np_seeds = np.stack(seeds, axis=0)
    return torch.from_numpy(np_seeds).long().cuda(seg_prob.device)
