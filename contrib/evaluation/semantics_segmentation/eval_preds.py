#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: eval_preds.py
@time: 2020/4/6 0:01
@desc:
"""
from typing import Optional, Iterable, Callable, Union, Tuple
import os
from os import path as osp
import pickle
from collections import OrderedDict
from tqdm import tqdm

import cv2
import numpy as np

from alchemy_cat.contrib.metrics import SegmentationMetric
from alchemy_cat.py_tools import OneOffTracker

__all__ = ['eval_preds']


def eval_preds(class_num: int, class_names: Optional[Iterable[str]],
               preds_dir: str, preds_ignore_label: int,
               gts_dir: str, gts_ignore_label: int,
               result_dir: Optional[str]=None,
               pred_preprocess: Callable[[np.ndarray], np.ndarray]=lambda x: x,
               gt_preprocess: Callable[[np.ndarray], np.ndarray]=lambda x: x,
               importance: int=2,
               eval_individually: bool=True, take_pred_ignore_as_a_cls: bool=False) \
        -> Union[Tuple[SegmentationMetric, OrderedDict], SegmentationMetric]:

    """Evaluate predictions of semantic segmentation

    Args:
        class_num: Num of classes
        class_names: Name of classes
        preds_dir: Dictionary where preds stored
        preds_ignore_label: Ignore label for predictions
        gts_dir: Dictionary where ground truths stored
        gts_ignore_label: Ignore label for ground truths
        result_dir: If not None, eval result will be saved to result_dir. (Default: None)
        pred_preprocess: Preprocess function for prediction read in
        gt_preprocess: Preprocess function for ground truth read in
        importance: Segmentation Metric's importance filter. (Default: 2)
        eval_individually: If True, evaluate each sample. (Default: True)
        take_pred_ignore_as_a_cls: If True, the ignore label in preds will be seemed as a class. (Default: False)

    Returns:
        Segmentation Metric and metric result for each sample (If eval_individually is True)
    """
    assert preds_ignore_label >= class_num
    assert gts_ignore_label >= class_num

    if take_pred_ignore_as_a_cls:
        class_num += 1
        class_names += ['pred_ignore']

    metric = SegmentationMetric(class_num, class_names)
    if eval_individually:
        sample_metrics = OrderedDict()

    print("\n================================== Eval ==================================")
    pred_file_suffixes = os.listdir(preds_dir)
    for pred_file_suffix in tqdm(pred_file_suffixes,
                                 total=len(pred_file_suffixes),
                                 desc='eval progress', unit='sample', dynamic_ncols=True):
        # Set files
        id = pred_file_suffix.split('.')[0]
        pred_file = osp.join(preds_dir, pred_file_suffix)
        gt_file = osp.join(gts_dir, pred_file_suffix)

        # Read files
        pred = cv2.imread(pred_file, cv2.IMREAD_GRAYSCALE)
        assert pred is not None
        gt = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
        assert gt is not None

        assert pred.shape == gt.shape

        # Preprocess
        pred: np.ndarray = pred_preprocess(pred)
        gt: np.ndarray = gt_preprocess(gt)

        if take_pred_ignore_as_a_cls:
            pred[pred == preds_ignore_label] = class_num - 1
            if gts_ignore_label == class_num - 1:
                gt[gt == gts_ignore_label] = class_num

        # Evaluate
        metric.update(pred, gt)
        if eval_individually:
            with OneOffTracker(lambda: SegmentationMetric(class_num, class_names)) as individual_metric:
                individual_metric.update(pred, gt)
            sample_metrics[id] = individual_metric.statistics(importance)

    # Saving
    if result_dir is not None:
        metric.save_metric(result_dir, importance, dpi=400)
        with open(osp.join(result_dir, 'sample_statistics.pkl'), 'wb') as f:
            pickle.dump(sample_metrics, f)

    print("\n================================ Eval End ================================")

    if eval_individually:
        return metric, sample_metrics
    else:
        return metric