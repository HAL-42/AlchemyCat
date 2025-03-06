#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2025/2/9 17:54
@File    : dt_constant.py
@Software: PyCharm
@Desc    :
"""
import typing as t

type_std_mean: t.TypeAlias = tuple[float, float, float]

OPENAI_DATASET_MEAN: t.Final[type_std_mean] = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD: t.Final[type_std_mean] = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN: t.Final[type_std_mean] = (0.485, 0.456, 0.406)
IMAGENET_STD: t.Final[type_std_mean] = (0.229, 0.224, 0.225)
INCEPTION_MEAN: t.Final[type_std_mean] = (0.5, 0.5, 0.5)
INCEPTION_STD: t.Final[type_std_mean] = (0.5, 0.5, 0.5)

DINO_PATCH_SIZE: t.Final[int] = 14
