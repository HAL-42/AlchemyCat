#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/11/26 21:54
@File    : typing.py
@Software: PyCharm
@Desc    : 
"""
import typing as t

import numpy as np
import torch

__all__ = ['SegDataset']


T_ret = t.TypeVar('T_ret')


class SegDataset(t.Protocol[T_ret]):
    class_names: t.Sequence
    class_num: int
    ignore_label: int
    image_ids: t.Sequence
    split: str

    def __getitem__(self, index: slice | list | np.ndarray | torch.Tensor | int) -> T_ret:
        ...

    def __add__(self, other):
        ...

    def get_item(self, index) -> T_ret:
        ...

    def get_by_img_id(self, img_id) -> T_ret:
        ...

    def __len__(self):
        ...

    def __repr__(self):
        ...

    def __iter__(self):
        ...
