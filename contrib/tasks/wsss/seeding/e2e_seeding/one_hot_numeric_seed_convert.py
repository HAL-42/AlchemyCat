#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2021/4/17 15:22
@File    : one_hot_numeric_seed_convert.py
@Software: PyCharm
@Desc    : 
"""
import numpy as np

__all__ = ['one_hot_seed_to_numeric_seed', 'numeric_seed_to_one_hot_seed']


def one_hot_seed_to_numeric_seed(one_hot_seed: np.ndarray, ignore_label: int=255) -> np.ndarray:
    """Convert one hot seed to numeric seed

    Args:
        one_hot_seed: (N, C, H, W) one hot seed where (n, c, h, w) = 1 indicates sample n's (h, w) pixel should
            be c class.
        ignore_label: numeric value of ignore label. (Default: 255)

    Returns:
        (N, H, W) numeric seed where (n, h, w) = c means sample n's (h, w) pixel is c class.
    """
    assert np.all((np.sum(one_hot_seed, axis=0) <= 1) & np.sum(one_hot_seed, axis=0) >= 0)

    numeric_seed = np.ones((one_hot_seed.shape[0], one_hot_seed.shape[2], one_hot_seed.shape[3]),
                           dtype=np.uint8) * ignore_label

    seed_loc = np.nonzero(one_hot_seed)
    numeric_seed[seed_loc[0], seed_loc[2], seed_loc[3]] = seed_loc[1]

    return numeric_seed


def numeric_seed_to_one_hot_seed(numeric_seed: np.ndarray, cls_num: int, ignore_label: int=255) -> np.ndarray:
    """Convert numeric seed to one-hot seed

    Args:
        numeric_seed: (N, H, W) numeric seed where (n, h, w) = c indicates sample n's (h, w) pixel should be c class.
        cls_num: number of classes.
        ignore_label: numeric value of ignore label. (Default: 255)

    Returns:
        (N, C, H, W) one-hot seed where (n, c, h, w) = 1 indicates sample n's (h, w) pixel should be c class.
    """
    one_hot_seed = np.zeros((numeric_seed.shape[0], cls_num, numeric_seed.shape[1], numeric_seed.shape[2]),
                            dtype=np.uint8)
    non_ignore_indices = np.nonzero(numeric_seed != ignore_label)
    one_hot_seed[non_ignore_indices[0], numeric_seed[non_ignore_indices],
                 non_ignore_indices[1], non_ignore_indices[2]] = 1
    return one_hot_seed
