#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/5 21:08
@File    : few_shot_dt.py
@Software: PyCharm
@Desc    : 
"""
from itertools import chain

import numpy as np
import torch
from alchemy_cat.data import Dataset

from .typing import SegDataset

__all__ = ['FewShotDt']


class FewShotDt(Dataset):
    def __init__(self, ori_dt: SegDataset, shot_num: int, seed: int, except_bg: bool=False):
        assert hasattr(ori_dt, 'get_by_img_id')
        assert hasattr(ori_dt, 'image_ids')
        assert hasattr(ori_dt, 'id2cls_labels')

        self.ori_dt = ori_dt
        self.shot_num = shot_num
        self.seed = seed
        self.except_bt = except_bg

        self.class_names = ori_dt.class_names
        self.class_num = ori_dt.class_num
        self.ignore_label = ori_dt.ignore_label

        self.cls_image_ids = {}
        self.image_ids = []

        self._sample_shots()

    def _sample_shots(self):
        # * 获得img_id - cls_label关系表。img_id来自原数据集。
        cls_labels = np.stack((self.ori_dt.id2cls_labels[img_id] for img_id in self.ori_dt.image_ids), axis=0)
        cls_labels = (cls_labels == 1).astype(np.int32)  # 过滤掉ignore=255标签，int32类型防止sum后溢出。
        cls_labels = torch.from_numpy(cls_labels)
        cls_num = cls_labels.shape[1]
        # * 对每个类别（可能要跳过背景）分别采样shot_num个样本。
        for cls in range(cls_num):
            # * （可选）跳过背景。
            if cls == 0 and self.except_bt:
                continue
            # * 在cls类别采样shot_num个img_id。
            labels = cls_labels[:, cls]
            shot_indices = torch.multinomial(labels.to(torch.float32), min(self.shot_num, labels.sum()),
                                             replacement=False,
                                             generator=torch.Generator('cpu').manual_seed(self.seed + cls))  # 随机隔离。
            self.cls_image_ids[cls] = [self.ori_dt.image_ids[idx] for idx in shot_indices]
            # * 防止重复采样。
            cls_labels[shot_indices, :] = 0
        # * 拼接得到image_ids。
        self.image_ids = list(chain(*self.cls_image_ids.values()))
        assert len(set(self.image_ids)) == len(self.image_ids)  # 确保无重复采样。

    def get_item(self, index: int):
        return self.ori_dt.get_by_img_id(self.image_ids[index])

    def __len__(self):
        return len(self.image_ids)
