#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2021/4/27 15:13
@File    : undersegmentation_error.py
@Software: PyCharm
@Desc    : 
"""
from typing import Optional

import numpy as np

from alchemy_cat.py_tools import Tracker, Statistic

__all__ = ["UndersegmentationError"]


class UndersegmentationError(Tracker):

    def __init__(self, num_class: int, ignore_label: Optional[int]=255, label_thresh: float=0.2):
        """Compute undersegmentation error of bg, fg and total pixels. Compute average superpixels which intersect with
        fg region.

        Args:
            num_class: Number of classes.
            ignore_label: Ignore label in gt.
            label_thresh: If Superpixel's intersection area to gt > superpixel_area * label_thresh, this superpixel is
                considered to labeled with gt. One superpixel can have multiple gt label.
        """
        # ! Attention, ignore label was computed into bg errors.
        super().__init__()

        self.num_class = num_class

        self._cls_errors = [list() for _ in range(num_class - 1)]
        self._cls_fg_errors = [list() for _ in range(num_class - 1)]
        self._cls_bg_errors = [list() for _ in range(num_class - 1)]
        self._cls_sp_nums_inter_gt = [list() for _ in range(num_class - 1)]

        self.ignore_label = ignore_label
        self.label_thresh = label_thresh

    def update(self, superpixel: np.ndarray, gt: np.ndarray):
        super().update(superpixel, gt)

        # * superpixel should start from 0
        assert np.amin(superpixel) == 0
        # * gt and superpixel should have same shape
        assert superpixel.shape == gt.shape

        # * Get cls in label, drop background
        cls_in_label = np.unique(gt)
        if cls_in_label[0] == 0:
            cls_in_label = cls_in_label[1:]
        if self.ignore_label is not None and cls_in_label[-1] == self.ignore_label:
            cls_in_label = cls_in_label[:-1]

        # * Get one-hot gt & gt_area
        one_hot_gt = (cls_in_label[:, None, None] == gt[None, ...]).astype(np.uint8)
        gt_area = np.sum(one_hot_gt, axis=(1, 2))

        # * Get one-hot superpixel & sp_area
        sp_num = np.amax(superpixel) + 1

        one_hot_sp = np.zeros((sp_num, ) + superpixel.shape, dtype=np.uint8)
        pixel_indices = np.nonzero(np.ones_like(superpixel))
        one_hot_sp[superpixel.reshape(-1), pixel_indices[0], pixel_indices[1]] = 1

        sp_area = np.sum(one_hot_sp, axis=(1, 2))

        # * Get sp gt intersection num
        sp_gt_inter_num = np.sum(one_hot_sp[:, None, ...] * one_hot_gt[None, ...], axis=(2, 3))

        # * Get sp fg, bg and total underseg num
        sp_bg_underseg_num = sp_area - np.sum(sp_gt_inter_num, axis=1)
        sp_fg_underseg_num = (sp_area - sp_bg_underseg_num)[:, None] - sp_gt_inter_num
        sp_underseg_num = sp_area[:, None] - sp_gt_inter_num

        # * Compute thresh hold according to gt area & Get sp-gt label mask
        label_thresh_area = sp_area * self.label_thresh
        sp_gt_label_mask = (sp_gt_inter_num > label_thresh_area[:, None]).astype(np.uint8)

        # * Compute gt fg, bg and total underseg num
        gt_bg_underseg_error = np.sum(sp_bg_underseg_num[:, None] * sp_gt_label_mask, axis=0) / gt_area
        gt_fg_underseg_error = np.sum(sp_fg_underseg_num * sp_gt_label_mask, axis=0) / gt_area
        gt_underseg_error = np.sum(sp_underseg_num * sp_gt_label_mask, axis=0) / gt_area

        # * Compute superpixels num who intersect with gt
        sp_num_inter_gt = np.sum(sp_gt_label_mask, axis=0)

        # * Record metrics
        for cls, bg_underseg_error, fg_underseg_error, underseg_error, num_inter in zip(cls_in_label,
                                                                                        gt_bg_underseg_error,
                                                                                        gt_fg_underseg_error,
                                                                                        gt_underseg_error,
                                                                                        sp_num_inter_gt):
            self._cls_bg_errors[cls - 1].append(bg_underseg_error)
            self._cls_fg_errors[cls - 1].append(fg_underseg_error)
            self._cls_errors[cls - 1].append(underseg_error)
            self._cls_sp_nums_inter_gt[cls - 1].append(num_inter)

    @staticmethod
    def _average_intra_cls(lst: list):
        return [np.mean(cls_lst) for cls_lst in lst]

    @Statistic.getter(importance=0)
    def cls_fg_underseg_error(self):
        return self._average_intra_cls(self._cls_fg_errors)

    @Statistic.getter(importance=0)
    def cls_bg_underseg_error(self):
        return self._average_intra_cls(self._cls_bg_errors)

    @Statistic.getter(importance=0)
    def cls_underseg_error(self):
        return self._average_intra_cls(self._cls_errors)

    @Statistic.getter(importance=0)
    def cls_sp_num_inter_gt(self):
        return self._average_intra_cls(self._cls_sp_nums_inter_gt)

    @Statistic.getter(importance=0)
    def fg_underseg_error(self):
        return np.mean(self.cls_fg_underseg_error)

    @Statistic.getter(importance=0)
    def bg_underseg_error(self):
        return np.mean(self.cls_bg_underseg_error)

    @Statistic.getter(importance=0)
    def underseg_error(self):
        return np.mean(self.cls_underseg_error)

    @Statistic.getter(importance=0)
    def sp_num_inter_gt(self):
        return np.mean(self.cls_sp_num_inter_gt)
