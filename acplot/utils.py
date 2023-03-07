#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: utils.py
@time: 2020/1/7 23:46
@desc:
"""
import numpy as np

from typing import Union, Iterable

from alchemy_cat.data.plugins.augers import pad_img_label

__all__ = ['stack_figs']


def stack_figs(in_list: list, img_pad_val: Union[int, float, Iterable] = (127, 140, 141),
               pad_location: Union[str, int]='right-bottom') -> np.ndarray:
    """Arrays in list will be padded to the max height/width in the list. The padding value will be 0 and located at
    right and bottom.

    Args:
        in_list (list): A list of numpy array shaped (H, W, C) with length N
        img_pad_val: (Union[int, float, Iterable]): If value is int or float, return (value, value, value),
            if value is Iterable with 3 element, return totuple(value), else raise error
        pad_location (Union[str, int]): Indicate pad location. Can be 'left-top'/0, 'right-top'/1, 'left-bottom'/2
            'right-bottom'/3, 'center'/4.

    Returns:
        np.ndarray: Stacked np array with shape (N, H, W, C)
    """
    max_h = 0
    max_w = 0
    for i, item in enumerate(in_list):
        if item.shape[0] > max_h:
            max_h = item.shape[0]
        if item.shape[1] > max_w:
            max_w = item.shape[1]
    out_arr = np.zeros((len(in_list), max_h, max_w, 3), dtype=in_list[0].dtype)
    for i, item in enumerate(in_list):
        out_arr[i] = pad_img_label(item, pad_img_to=(max_h, max_w), img_pad_val=img_pad_val, pad_location=pad_location)
    return out_arr
