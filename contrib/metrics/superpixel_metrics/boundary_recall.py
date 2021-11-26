#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2021/4/27 15:13
@File    : boundary_recall.py
@Software: PyCharm
@Desc    : 
"""
from typing import Optional

import numpy as np
from skimage.segmentation import find_boundaries
from skimage.morphology import disk, dilation, binary_dilation

from alchemy_cat.py_tools import Tracker, Statistic

__all__ = ["BoundaryRecall"]


class BoundaryRecall(Tracker):

    def __init__(self, ignore_label: Optional[int] = None, ignore_boundary_width: Optional[int] = None,
                 tolerance_radius: Optional[int] = 2):
        """Calculate Boundary Recall for superpixels

        Args:
            ignore_label: If boundary is marked with ignore label(like VOC non-aug), set ignore_label to eliminate
                ignore_label boundary.
            ignore_boundary_width: Boundary width of which marked by boundary.
            tolerance_radius: Max tolerance distance between superpixel boundary and gt boundary.
        """
        super().__init__()

        if ignore_label is not None:
            assert ignore_boundary_width is not None

        self.ignore_label = ignore_label
        self.ignore_boundary_width = ignore_boundary_width
        self.tolerance_radius = tolerance_radius

        self._fg_fg_recall_num = self._fg_bg_recall_num = 0
        self._fg_fg_gt_num = self._fg_bg_gt_num = 0

    def update(self, superpixel: np.ndarray, gt: np.ndarray):
        super().update(superpixel, gt)

        # * gt and superpixel should have same shape
        assert superpixel.shape == gt.shape

        # * Eliminate ignore label expressed boundary
        if self.ignore_label is not None:
            gt = self._eliminate_ignore_boundary_in_gt(gt)

        # * Get fg, bg mask
        fg_mask = gt > 0

        # * Get sp dilated boundary
        sp_boundary = find_boundaries(superpixel, connectivity=superpixel.ndim, mode='thick')
        dilated_sp_boundary = binary_dilation(sp_boundary, selem=disk(self.tolerance_radius)).astype(np.uint8)

        # * Get fg_fg_gt_boundary
        fg_fg_gt_boundary = find_boundaries(gt, connectivity=gt.ndim, mode='outer').astype(np.uint8)
        fg_fg_gt_boundary *= fg_mask

        # * Get fg_bg_gt_boundary
        fg_bg_gt_boundary = find_boundaries(gt, connectivity=gt.ndim, mode='thick').astype(np.uint8)
        fg_bg_gt_boundary -= fg_fg_gt_boundary

        # * update boundary recall
        self._fg_fg_recall_num += np.count_nonzero(dilated_sp_boundary * fg_fg_gt_boundary)
        self._fg_bg_recall_num += np.count_nonzero(dilated_sp_boundary * fg_bg_gt_boundary)

        # * update gt num
        self._fg_fg_gt_num += np.count_nonzero(fg_fg_gt_boundary)
        self._fg_bg_gt_num += np.count_nonzero(fg_bg_gt_boundary)

    def _eliminate_ignore_boundary_in_gt(self, gt_: np.ndarray):
        if np.any(gt_ == self.ignore_label):
            # * Set ignore to background
            gt = gt_.copy()
            gt[gt == self.ignore_label] = 0
            # * dilate with cross-shaped structuring element (connectivity=1) for ignore_boundary_width // 2 times.
            for _ in range(self.ignore_boundary_width // 2):
                gt = dilation(gt)
        else:
            gt = gt_
        assert not np.any(gt == self.ignore_label)
        return gt

    @Statistic.getter(importance=0)
    def fg_fg_boundary_recall(self):
        return self._fg_fg_recall_num / self._fg_fg_gt_num

    @Statistic.getter(importance=0)
    def fg_bg_boundary_recall(self):
        return self._fg_bg_recall_num / self._fg_bg_gt_num

    @Statistic.getter(importance=0)
    def boundary_recall(self):
        return (self._fg_fg_recall_num + self._fg_bg_recall_num) / (self._fg_fg_gt_num + self._fg_bg_gt_num)
