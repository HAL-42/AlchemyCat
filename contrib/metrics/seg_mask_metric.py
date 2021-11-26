#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2021/5/24 16:25
@File    : seg_mask_metric.py
@Software: PyCharm
@Desc    : 
"""
import numpy as np

from alchemy_cat.contrib.metrics.conf_matrix_bases_metrics import SegmentationMetric
from alchemy_cat.py_tools import Statistic

__all__ = ["SegMaskMetric"]



class SegMaskMetric(SegmentationMetric):
    """SegMask Metric which implement ignore label in pred as a independent class."""

    @Statistic.getter(1)
    def macro_avg_precision(self):
        return np.nanmean(self.cls_precision[:-1])

    @Statistic.getter(1)
    def fg_precision(self):
        return np.nanmean(self.cls_precision[1:-1])

    @Statistic.getter(1)
    def fg_recall(self):
        return np.nanmean(self.cls_recall[1:])

    @Statistic.getter(1)
    def fg_mIoU(self):
        return np.mean(self.cls_IoU[1:][self.valid[1:]])

    @Statistic.getter(1)
    def bg_precision(self):
        return self.cls_precision[0]

    @Statistic.getter(1)
    def bg_recall(self):
        return self.cls_recall[0]

    @Statistic.getter(1)
    def bg_mIoU(self):
        return self.cls_IoU[0]
