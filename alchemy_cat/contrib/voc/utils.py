#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: utils.py
@time: 2020/2/5 2:30
@desc:
"""
import numpy as np

__all__ = ["VOC_COLOR", "VOC_CLASSES", "color_map2label_map", "label_map2color_map", ]

VOC_COLOR = np.array([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                      [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                      [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                      [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                      [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                      [0, 64, 128]], dtype=np.uint8)

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

color_id2label = np.zeros(256 ** 3, dtype=np.uint8)
for i, color in enumerate(VOC_COLOR):
    color_id2label[(color[0] * 256 + color[1]) * 256 + color[2]] = i

label2color = np.ones((256, 3), dtype=np.uint8) * 255
label2color[:21, :] = VOC_COLOR


def color_map2label_map(color_map: np.ndarray) -> np.ndarray:
    """Convert VOC color map to label map

    Args:
        color_map (np.ndarray):
            Labels with shape (..., C)
    Returns (np.ndarray):
        Labels with shape (...)
    """
    color_id_map = ((color_map[..., 0] * 256 + color_map[..., 1]) * 256
                    + color_map[..., 2])
    return color_id2label[color_id_map]


def label_map2color_map(label_map: np.ndarray) -> np.ndarray:
    """Convert VOC label map to color map

    Args:
        label_map (np.ndarray):
            Labels with shape (...)
    Returns (np.ndarray):
        color_map with shape (..., C)
    """
    color_map = label_map[..., np.newaxis]
    color_map = np.concatenate((color_map, color_map, color_map), axis=-1)

    color_map[..., 0] = label2color[..., 0][color_map[..., 0]]
    color_map[..., 1] = label2color[..., 1][color_map[..., 1]]
    color_map[..., 2] = label2color[..., 2][color_map[..., 2]]

    return color_map
