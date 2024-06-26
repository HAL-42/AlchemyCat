#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: conf_matrix_bases_metrics.py
@time: 2020/2/26 23:16
@desc:
"""
import os
import traceback
from typing import Optional, Iterable

import numpy as np
import pandas as pd
from alchemy_cat.acplot import pretty_plot_confusion_matrix
from alchemy_cat.py_tools import Statistic, Tracker, rprint, quick_init
from matplotlib import pyplot as plt

__all__ = ['ClassificationMetric', 'SegmentationMetric']


class ClassificationMetric(Tracker):
    """Multi classification metric

    See statistics at https://en.wikipedia.org/wiki/Confusion_matrix
    """

    def __init__(self, class_num: Optional[int] = None, class_names: Optional[Iterable[str]]=None):
        # * Init attributes and save init_dict
        super(ClassificationMetric, self).__init__(quick_init(self, locals()))

        # * Check Input
        if class_names is None:
            if class_num is None:
                raise ValueError("class_num and class_names can not be None at the same time")
            else:
                class_names = [str(i) for i in range(class_num)]
        else:
            class_names = [str(name) for name in class_names]
            if class_num is not None and len(class_names) != class_num:
                raise ValueError(f"class_num = {class_num} should be equal to len(class_names) = {len(class_names)}")
            else:
                class_num = len(class_names)

        self.class_num, self.class_names = class_num, class_names

        self.conf_matrix = np.zeros((class_num, class_num), dtype=np.int64)
        self._last_conf_matrix = self.conf_matrix.copy()

    @staticmethod
    def get_conf_matrix(pred: np.ndarray, gt: np.ndarray, class_num: int) -> np.ndarray:
        """Get confusion matrix accroding to input pred and gt

        Function get pred and gt array from input, which mean predicting label and ground truth label respectively.
        The label can be class of the input image, or class of each pixel of the input image. As long as pred and gt
        have the same shape.

        Args:
            pred: Numpy array of pred label
            gt: Numpy array of ground truth label.
            class_num: number of classes

        Returns:
            confusion matrix calculated according to input pred and gt
        """
        if pred.shape != gt.shape:
            raise ValueError(f"pred's shape {pred.shape} should be equal to gt's shape {gt.shape}")

        mask = (gt >= 0) & (gt < class_num)
        conf_matrix = np.bincount(
            class_num * gt[mask].astype(np.int64) + pred[mask].astype(np.int64),
            minlength=class_num ** 2,
        ).reshape(class_num, class_num)
        return conf_matrix

    def update(self, pred: np.ndarray, gt: np.ndarray):
        """Update confusion matrix accroding to input pred and gt

        Function get pred and gt array from input, which mean predicting label and ground truth label respectively.
        The label can be class of the input image, or class of each pixel of the input image. As long as pred and gt
        have the same shape.

        Function will use get_conf_matrix to calculate the last_conf_matrix of current input then add it to
        self.conf_matrix

        Args:
            pred: Numpy array of pred label
            gt: Numpy array of ground truth label.
        """
        super(ClassificationMetric, self).update(pred, gt)

        self._last_conf_matrix = self.get_conf_matrix(pred, gt, self.class_num)
        self.conf_matrix += self._last_conf_matrix

    def _plot_conf_matrix(self, **kwargs) -> plt.Figure:
        df = pd.DataFrame(self.conf_matrix, index=self.class_names, columns=self.class_names)

        kwargs["figsize"] = [self.class_num * 2, self.class_num * 2] if "figsize" not in kwargs else kwargs["figsize"]
        return pretty_plot_confusion_matrix(df, pred_val_axis='x', **kwargs)

    def show_conf_matrix(self, **kwargs) -> plt.Figure:
        """Plot confusion matrix

        Args:
            **kwargs: kwargs for plt.figure()

        Returns:
            Shown figure
        """
        fig = self._plot_conf_matrix(**kwargs)
        fig.show()
        plt.show()

        return fig

    def save_metric(self, save_dir: str='.', importance: int=0, **kwargs):
        """Saving Metric with confusion matrix and statistics

        Args:
            save_dir: Dictionary where metrics saved
            importance: Only statistics' statistic.importance > importance will be saved
            **kwargs: keywords arguments for plotting and saving confusion matrix plot confusion_matrix.png
        """
        # * Save statistics
        self.save_statistics(save_dir, importance)

        # * Save confusion matrix
        conf_matrix_txt = os.path.join(save_dir, 'confusion_matrix.txt')
        np.savetxt(conf_matrix_txt, self.conf_matrix, fmt="%d")

        # * Save confusion matrix plot
        try:
            fig = self._plot_conf_matrix(**kwargs)
            fig.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
        except Exception:
            rprint(f"可视化混淆矩阵失败！")
            print(traceback.format_exc())

    @Statistic.getter(0)
    def samples_num(self):
        return self.conf_matrix.sum()

    N = samples_num

    @Statistic.getter(0)
    def condition_positive(self):
        return np.sum(self.conf_matrix, axis=1)

    @Statistic.getter(0)
    def condition_negative(self):
        return self.samples_num - self.condition_positive

    @Statistic.getter(0)
    def prediction_positive(self):
        return np.sum(self.conf_matrix, axis=0)

    @Statistic.getter(0)
    def prediction_negative(self):
        return self.samples_num - self.prediction_positive

    @Statistic.getter(0)
    def true_positive(self):
        return np.diag(self.conf_matrix)

    TP = true_positive

    @Statistic.getter(0)
    def false_negative(self):
        return self.condition_positive - self.true_positive

    FN = false_negative

    @Statistic.getter(0)
    def false_positive(self):
        return self.prediction_positive - self.true_positive

    FP = false_positive

    @Statistic.getter(0)
    def true_negative(self):
        return self.prediction_negative - self.false_negative

    TN = true_negative

    @Statistic.getter(1)
    def cls_weight(self):
        return self.condition_positive / self.N

    condition_weight = cls_weight

    @Statistic.getter(1)
    def prediction_weight(self):
        return self.prediction_positive / self.N

    @Statistic.getter(1)
    def cls_recall(self):
        return self.TP / self.condition_positive

    @Statistic.getter(1)
    def macro_avg_recall(self):
        return np.nanmean(self.cls_recall)

    @Statistic.getter(1)
    def weighted_avg_recall(self):
        return np.nansum(self.cls_weight * self.cls_recall)

    @Statistic.getter(1)
    def cls_precision(self):
        return self.TP / self.prediction_positive

    @Statistic.getter(1)
    def macro_avg_precision(self):
        return np.nanmean(self.cls_precision)

    @Statistic.getter(1)
    def weighted_avg_precision(self):
        return np.nansum(self.prediction_weight * self.cls_precision)

    @Statistic.getter(1)
    def cls_F1_score(self):
        return (2 * self.cls_precision * self.cls_recall) / (self.cls_precision + self.cls_recall)

    @Statistic.getter(2)
    def accuracy(self):
        return np.sum(self.TP) / self.N

    ACC = accuracy

    @Statistic.getter(2)
    def F1_score(self):
        return np.nanmean(self.cls_F1_score)

    F1 = F1_score

    @Statistic.getter(1)
    def weighted_F1_score(self):
        return np.nansum(self.cls_weight * self.cls_F1_score)


class SegmentationMetric(ClassificationMetric):

    @Statistic.getter(1)
    def cls_IoU(self):
        return self.TP / (self.FP + self.FN + self.TP)

    @Statistic.getter(1)
    def valid(self):
        return self.condition_positive > 0

    @Statistic.getter(2)
    def mIoU(self):
        return np.mean(self.cls_IoU[self.valid])

    @Statistic.getter(2)
    def weighted_mIoU(self):
        return np.sum(self.cls_weight[self.valid] * self.cls_IoU[self.valid])
