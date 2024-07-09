#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: test_tracker.py
@time: 2020/3/13 14:48
@desc:
"""
import pytest
import numpy as np
import copy
from sklearn.metrics import f1_score
from tqdm import tqdm
from colorama import Style, Fore

from alchemy_cat.contrib.voc import VOCAug, VOC_CLASSES
from alchemy_cat.data.plugins.augers import scale_img_label
from alchemy_cat.contrib.metrics import SegmentationMetric
from alchemy_cat.py_tools.tracker import Tracker, Statistic
from alchemy_cat.py_tools.type import dict_with_arr_is_eq

kScaleFactor = 0.125


def _get_conf_matrix(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    conf_matrix = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return conf_matrix


def scores(label_trues, label_preds, n_class):
    """Copied and modified from https://github.com/kazuto1011/deeplab-pytorch as Reference Metrics.
        Supposed to be correct"""
    conf_matrix = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        conf_matrix += _get_conf_matrix(lt.flatten(), lp.flatten(), n_class)

    condition_positive = np.sum(conf_matrix, axis=1)
    acc = np.diag(conf_matrix).sum() / conf_matrix.sum()
    cls_recall = np.diag(conf_matrix) / conf_matrix.sum(axis=1)
    macro_recall = np.nanmean(cls_recall)
    cls_precision = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
    macro_precision = np.nanmean(cls_precision)
    cls_F_score = (2 * cls_precision * cls_recall) / (cls_precision + cls_recall)
    F_score = np.nanmean(cls_F_score)

    cls_IoU = np.diag(conf_matrix) / (conf_matrix.sum(axis=1) + conf_matrix.sum(axis=0) - np.diag(conf_matrix))
    valid = conf_matrix.sum(axis=1) > 0  # added, if some class doesn't exit, then don't take it into count
    mean_IoU = np.nanmean(cls_IoU[valid])
    freq = conf_matrix.sum(axis=1) / conf_matrix.sum()
    freq_weighted_IoU = (freq[freq > 0] * cls_IoU[freq > 0]).sum()

    return {
        "Conf Matrix": conf_matrix,
        "Frequency Weighted IoU": freq_weighted_IoU,
        "Mean IoU": mean_IoU,
        "Class IoU": cls_IoU,
        "Accuracy": acc,
        "Macro Recall": macro_recall,
        "Class Recall": cls_recall,
        "Class Precision": cls_precision,
        "Macro Precision": macro_precision,
        "F Score": F_score,
        "Class Samples Num": condition_positive
    }


@pytest.fixture(scope='module')
def voc_dataset():
    return VOCAug(split='val')


@pytest.fixture(scope='module')
def metric_results():
    # * Tracker
    seg_metric = SegmentationMetric(class_names=VOC_CLASSES)

    # * For scores
    preds, gts = [], []

    for example in tqdm(VOCAug(split='val'), desc='Metric Progress', unit='samples', dynamic_ncols=True):
        _, img, label = example
        label = copy.deepcopy(label)
        img_shape = img.shape[:2]

        # * Scale-Recover label as pred
        scaled_img, scaled_label = scale_img_label(kScaleFactor, img, label)
        _, recovered_label = \
            scale_img_label(1, scaled_img, scaled_label, (lambda _: img_shape[0], lambda _: img_shape[1]))
        # label[recovered_label == 255] = 255
        recovered_label[(label != 255) & (recovered_label == 255)] = 0 # preds don't have ignore label

        # * Update tracker
        seg_metric.update(recovered_label, label)

        # * Update for score
        preds.append(recovered_label); gts.append(label)

    # * Calculate reference metrics
    ref_metrics = scores(gts, preds, len(VOC_CLASSES))

    return seg_metric, ref_metrics, preds, gts


@pytest.mark.tryfirst
def test_statistic_cache(metric_results):
    seg_metric, _, _, _ = metric_results
    seg_metric = copy.deepcopy(seg_metric)

    calculated_statistic = seg_metric.statistics()

    tracker_classes = [base_cls for base_cls in seg_metric.__class__.mro() if issubclass(base_cls, Tracker)]

    statistics = [statistic
            for cls in tracker_classes
            for statistic in cls.__dict__.values() if isinstance(statistic, Statistic)]

    for statistic in statistics:
        statistic.fget = None

    cached_statistic = seg_metric.statistics()

    # * Same length
    assert len(calculated_statistic) == len(cached_statistic)

    # * Save keys and values
    dict_with_arr_is_eq(calculated_statistic, cached_statistic)

    for cal_item, cached_item in zip(calculated_statistic.items(), cached_statistic.items()):
        cal_key, cal_value = cal_item
        cached_key, cached_val = cached_item

        assert cal_key == cached_key

        if isinstance(cal_value, np.ndarray):
            assert np.all(cal_value == cached_val)
        else:
            assert cal_value == cached_val


def test_statistics_calculation(metric_results):
    seg_metric, ref_metrics, preds, gts = metric_results

    assert np.all(seg_metric.conf_matrix == ref_metrics['Conf Matrix'])

    assert seg_metric.weighted_mIoU == ref_metrics['Frequency Weighted IoU']

    assert seg_metric.mIoU == ref_metrics['Mean IoU']

    assert np.all(seg_metric.cls_IoU == ref_metrics['Class IoU'])

    assert seg_metric.ACC == ref_metrics['Accuracy']

    assert seg_metric.macro_avg_recall == ref_metrics['Macro Recall']

    assert np.all(seg_metric.cls_recall == ref_metrics['Class Recall'])

    assert np.all(seg_metric.cls_precision == ref_metrics['Class Precision'])

    assert seg_metric.macro_avg_precision == ref_metrics['Macro Precision']

    assert seg_metric.F1_score == ref_metrics['F Score']

    assert np.all(seg_metric.condition_positive == ref_metrics['Class Samples Num'])

    preds, gts = np.concatenate(preds, axis=None), np.concatenate(gts, axis=None)
    mask = gts != 255
    preds, gts = preds[mask], gts[mask]

    # ORZ: It costs 1.5min to calculate f1_score. That's why we don't use ski-learn to calculate metric in alchemy_cat
    assert seg_metric.F1_score == f1_score(gts, preds, average='macro')

    assert seg_metric.weighted_F1_score == pytest.approx(f1_score(gts, preds, average='weighted'), rel=1e-6)


def test_update_count(metric_results, voc_dataset):
    seg_metric, _, _, _ = metric_results

    assert seg_metric.update_count == len(voc_dataset)


def test_reset(metric_results):
    seg_metric, _, _, _ = metric_results

    origin_init_dict = seg_metric.init_dict

    seg_metric.reset()

    assert seg_metric.update_count == 0

    assert len(seg_metric._statistic_status) == 0

    assert seg_metric.class_num == len(VOC_CLASSES)

    assert seg_metric.class_names == VOC_CLASSES

    assert np.all(seg_metric.conf_matrix == np.zeros((len(VOC_CLASSES), len(VOC_CLASSES)), dtype=np.int32))

    assert np.all(seg_metric._last_conf_matrix == np.zeros((len(VOC_CLASSES), len(VOC_CLASSES)), dtype=np.int32))

    assert dict_with_arr_is_eq(origin_init_dict, seg_metric.init_dict)


def test_show(metric_results):
    seg_metric, _, _, _ = metric_results
    seg_metric.show_conf_matrix()

    print(Fore.LIGHTRED_EX + "\n Need Visual judgment!" + Style.RESET_ALL)


def test_importance_filter(metric_results):
    seg_metric, _, _, _ = metric_results

    level0 = seg_metric.statistics(importance=0)
    for name, _ in level0.items():
        assert getattr(SegmentationMetric, name).importance >= 0

    level1 = seg_metric.statistics(importance=1)
    for name, _ in level1.items():
        assert getattr(SegmentationMetric, name).importance >= 1

    level2 = seg_metric.statistics(importance=2)
    for name, _ in level2.items():
        assert getattr(SegmentationMetric, name).importance >= 2


def test_statistic_order(metric_results):
    seg_metric, _, _, _ = metric_results

    keys = list(seg_metric.statistics().keys())

    for i in range(len(keys) - 1):
        for j in range(i + 1, len(keys)):
            assert getattr(SegmentationMetric, keys[i]).importance <= getattr(SegmentationMetric, keys[j]).importance
            if getattr(SegmentationMetric, keys[j]).importance == getattr(SegmentationMetric, keys[i]).importance:
                assert keys[i] < keys[j]


def test_save_metrics(metric_results):
    seg_metric, _, _, _ = metric_results

    seg_metric.save_metric('./Temp/imp0', importance=0, dpi=400)

    seg_metric.save_metric('./Temp/imp1', importance=1, dpi=400)

    seg_metric.save_metric('./Temp/imp2', importance=2, dpi=400)

    print(Fore.LIGHTRED_EX + "\n Need Visual judgment!" + Style.RESET_ALL)
