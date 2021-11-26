#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2021/7/22 18:17
@File    : sub_augers.py
@Software: PyCharm
@Desc    : 
"""
from typing import Union, Iterable

import numpy as np

from alchemy_cat.dag import Graph
from alchemy_cat.data.plugins.augers import RandCrop, pad_img_label, int_img2float32_img, centralize

__all__ = ['float_centralize', 'pad_crop']


def float_centralize(mean: Union[int, float, Iterable], std: Union[int, float, Iterable, None]=None,
                     scale: bool=False) -> Graph:
    sub_g = Graph(slim=True)

    if scale:
        mean = np.array(mean) / 255.
        std = np.array(std) / 255. if std is not None else None

    sub_g.add_node(int_img2float32_img, args=['img', {'scale': scale}], outputs=['float_img'])
    sub_g.add_node(centralize, args=['float_img', {'mean': mean}, {'std': std}],
                   outputs=['centralized_img'])

    return sub_g


def pad_crop(crop_size: Union[int, Iterable[int]]=321, ignore_label: int=255) -> Graph:
    sub_g = Graph(slim=True)

    sub_g.add_node(pad_img_label,
                   args=['img', 'label'],
                   kwargs={'pad_img_to': crop_size, 'pad_location': 'center', 'ignore_label': ignore_label},
                   outputs=['padded_img', 'padded_label'])

    rand_crop = RandCrop(crop_size)
    sub_g.add_node(rand_crop, args=['padded_img', 'padded_label'], outputs=['cropped_img', 'cropped_label'])

    return sub_g
