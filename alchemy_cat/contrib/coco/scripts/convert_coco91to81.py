#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/6/20 19:10
@File    : convert_coco91to81.py
@Software: PyCharm
@Desc    : 
"""
import argparse
import os
import os.path as osp

import numpy as np
from PIL import Image
from alchemy_cat.contrib.coco import COCO
from alchemy_cat.data.plugins import arr2PIL
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--source', type=str)
parser.add_argument('-t', '--target', type=str)
parser.add_argument('--label_type', type=str, default='')
parser.add_argument('--split', type=str, default='train')
args = parser.parse_args()

dt = COCO('datasets', split=args.split, label_type=args.label_type, cls_labels_type=None)

os.makedirs(args.target, exist_ok=True)

for img_id in tqdm(dt.image_ids, dynamic_ncols=True, desc="处理", unit="张"):
    label = np.array(Image.open(osp.join(args.source, f'{img_id}.png')), dtype=np.uint8)
    converted_lb = COCO.coco91to81[label]
    arr2PIL(converted_lb).save(osp.join(args.target, f'{img_id}.png'))
