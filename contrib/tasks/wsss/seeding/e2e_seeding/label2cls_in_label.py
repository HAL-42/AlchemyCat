#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2021/5/16 18:11
@File    : label2cls_in_label.py
@Software: PyCharm
@Desc    : 
"""
import numpy as np

from alchemy_cat.contrib.voc import VOC_CLASSES, VOC

__all__ = ['label2cls_in_label']


def label2cls_in_label(label: np.ndarray, cls_num=len(VOC_CLASSES), ignore_label=VOC.ignore_label) -> np.ndarray:
    """将像素级标签转为图像级标签。默认按照VOC配置。

    Args:
        label: 像素级标签。
        cls_num: 类别数。
        ignore_label: “忽略”类别的编号。

    Returns:
        图像级标签。
    """
    return (np.bincount(label.ravel(), minlength=ignore_label + 1) != 0).astype(np.int64)[:cls_num]
