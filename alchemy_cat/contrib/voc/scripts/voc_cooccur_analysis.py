#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2021/6/22 15:07
@File    : voc_cooccur_analysis.py
@Software: PyCharm
@Desc    : 
"""
from typing import Optional

import numpy as np

from alchemy_cat.contrib.voc import VOCAug, VOC_CLASSES
from alchemy_cat.py_tools.type import tolist

kVOCSplit = 'train_aug'
kForeClsNum = len(VOC_CLASSES[1:])


def cls_in_label2tag(cls_in_label: np.ndarray):
    return np.nonzero(cls_in_label)[0]


def label2cls_in_label(label: np.ndarray, cls_num, ignore_label) -> np.ndarray:
    return (np.bincount(label.ravel(), minlength=ignore_label + 1) != 0).astype(np.int64)[:cls_num]


def get_permutations(elem_set: list, permutation_len: int, current_permutation: Optional[list]=None):
    assert permutation_len > 0
    assert elem_set

    if current_permutation is None:
        current_permutation = []

    for idx, elem in enumerate(elem_set):
        appended_permutation = current_permutation + [elem]
        if len(appended_permutation) == permutation_len:
            yield appended_permutation
        else:
            retain_set = elem_set[:]
            retain_set.pop(idx)
            yield from get_permutations(retain_set, permutation_len, appended_permutation)


def tag2cooccur_idx(tag: np.ndarray):
    """Get cooccur_idx from tag. e.g. tag = [1], cooccur_idx = (1, 1). tag = [0, 1, 5],
    cooccur_idx = ([0, 0, 1, 1, 5, 5], [1, 5, 0, 5, 0, 1]).

    Args:
        tag: img label tag.

    Returns:
        cooccur_idx
    """
    if len(tag) == 1:
        return tag[0], tag[0]
    else:
        return tuple(list(hw) for hw in zip(*get_permutations(tolist(tag), 2)))


if __name__ == "__main__":
    cooccur_matrix = np.zeros((kForeClsNum, kForeClsNum), dtype=np.int64)

    dt = VOCAug(split=kVOCSplit)

    for _, _, label in dt:
        fore_in_label = label2cls_in_label(label, kForeClsNum + 1, VOCAug.ignore_label)[1:]
        tag = cls_in_label2tag(fore_in_label)
        cooccur_idx = tag2cooccur_idx(tag)
        cooccur_matrix[cooccur_idx] += 1

    print(cooccur_matrix)
