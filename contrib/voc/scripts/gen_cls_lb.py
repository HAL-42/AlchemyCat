#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/5 14:25
@File    : gen_cls_lb.py
@Software: PyCharm
@Desc    : 
"""
import pickle

import numpy as np
from tqdm import tqdm

from alchemy_cat.contrib.voc import VOCAug, VOC_CLASSES


def lb2cls_lb(label: np.ndarray) -> np.ndarray:
    return (np.bincount(label.ravel(), minlength=VOCAug.ignore_label + 1) != 0).astype(np.uint8)[:len(VOC_CLASSES)]


# * 获取来自分割标签的图像标签。
seg_cls_labels = {}
no_bg_image_ids = []

for img_id, _, lb in tqdm(VOCAug('datasets', split='trainval_aug'), dynamic_ncols=True):
    seg_cls_labels[img_id] = lb2cls_lb(lb)

    if seg_cls_labels[img_id][0] == 0:
        no_bg_image_ids.append(img_id)

with open('datasets/VOC2012/third_party/seg_cls_labels.pkl', 'wb') as pkl_f:
    pickle.dump(seg_cls_labels, pkl_f)

with open('datasets/VOC2012/third_party/no_bg_image_ids.pkl', 'wb') as pkl_f:
    pickle.dump(no_bg_image_ids, pkl_f)

# * 获取来自检测标签的图像标签。
ref_det_cls_labels = np.load('datasets/voc_cls_labels.npy', allow_pickle=True).tolist()
det_cls_labels = {}

for int_img_id, ref_det_cls_lb in tqdm(ref_det_cls_labels.items(), dynamic_ncols=True):
    det_cls_labels[f'{(sid := str(int_img_id))[:4]}_{sid[4:]}'] = np.insert(ref_det_cls_lb.astype(np.uint8), 0, 1)

with open('datasets/VOC2012/third_party/det_cls_labels.pkl', 'wb') as pkl_f:
    pickle.dump(det_cls_labels, pkl_f)

# * 获取混合标签。
# seg_cls_lb与det_cls_lb对比发现：
# 1. 大约有818张图片，det_cls_lb中的困难物体没有标出。除此以外，基本相同。
# 2. bg=0的图片有10张，的确是没有背景。
difficult_image_ids = []
ignore_diff_cls_labels = {}
for img_id in det_cls_labels.keys():
    if np.any(diff_mask := seg_cls_labels[img_id][1:] != det_cls_labels[img_id][1:]):  # 背景遵循seg_cls_label。
        difficult_image_ids.append(img_id)

    ignore_diff_cls_lb = seg_cls_labels[img_id].copy()
    ignore_diff_cls_lb[1:][diff_mask] = 255
    ignore_diff_cls_labels[img_id] = ignore_diff_cls_lb

with open('datasets/VOC2012/third_party/difficult_image_ids.pkl', 'wb') as pkl_f:
    pickle.dump(difficult_image_ids, pkl_f)

with open('datasets/VOC2012/third_party/ignore_diff_cls_labels.pkl', 'wb') as pkl_f:
    pickle.dump(ignore_diff_cls_labels, pkl_f)
