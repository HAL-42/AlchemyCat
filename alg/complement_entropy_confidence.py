#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2021/4/17 15:51
@File    : complement_entropy_confidence.py
@Software: PyCharm
@Desc    : 
"""
import numpy as np
from scipy.stats import entropy

__all__ = ['complement_entropy_confidence']


def complement_entropy_confidence(prob: np.ndarray) -> np.ndarray:
    return (np.log2(prob.shape[1]) - entropy(prob.transpose((1, 0, 2, 3)), base=2)) / np.log2(prob.shape[1])
