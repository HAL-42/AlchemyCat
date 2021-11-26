#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2021/5/24 14:25
@File    : utils.py
@Software: PyCharm
@Desc    : 
"""
import numpy as np
from scipy.stats import entropy

__all__ = ["get_bg_fg_score", "cls_in_label2tag_always_bg", "cls_in_label2tag", "tag2cls_in_label",
           "complement_entropy_with_shift"]


def get_bg_fg_score(score: np.ndarray, alpha: float) -> np.ndarray:
    fg_score = score.reshape(score.shape[0], -1)
    bg_score = np.power(1 - np.amax(fg_score, axis=0, keepdims=True), alpha)
    bg_fg_score = np.concatenate((bg_score, fg_score), axis=0)
    return bg_fg_score.reshape((score.shape[0] + 1,) + score.shape[1:])


def cls_in_label2tag(cls_in_label: np.ndarray):
    return np.nonzero(cls_in_label)[0]


def cls_in_label2tag_always_bg(cls_in_label: np.ndarray):
    return np.insert(np.nonzero(cls_in_label[1:])[0] + 1, 0, 0, 0)


def tag2cls_in_label(tag: np.ndarray, num_cls: int):
    cls_in_label = np.zeros(num_cls, dtype=np.uint8)
    cls_in_label[tag] = 1
    return cls_in_label


def complement_entropy_with_shift(prob: np.ndarray, shift: float=0.1) -> np.ndarray:
    """Compute complement entropy with ( log2(cls_num) - entropy(prob) ) / log2(cls_num).

    Args:
        prob: (N, C, ...) prob. prob will be normalize if they don't sum to 1.
        shift: prob will be shifted by prob += shift. This will make only the diversity with the same magnitude as shift
            can influence the result.

    Returns:
        (N, ...) complement entropy.
    """
    return (np.log2(prob.shape[1]) - entropy(prob.swapaxes(0, 1) + shift, base=2)) / np.log2(prob.shape[1])
