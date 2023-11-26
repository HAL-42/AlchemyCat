#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/2/22 20:19
@File    : voc上色.py
@Software: PyCharm
@Desc    : 
"""
import argparse
import os
import os.path as osp

from alchemy_cat.contrib.voc import VOCAug2
from alchemy_cat.data.plugins import arr2PIL
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--target', type=str)
parser.add_argument('-s', '--split', type=str)
parser.add_argument('-a', '--alpha', type=float, default=0.5)
args = parser.parse_args()

os.makedirs(args.target, exist_ok=True)

dt = VOCAug2('datasets', split=args.split)

for inp in tqdm(dt, dynamic_ncols=True, desc="处理", unit="张"):
    img_id, img, lb = inp['img_id'], inp['img'], inp['lb']
    color_lb = dt.label_map2color_map(lb)

    img_lb = (img * args.alpha + color_lb * (1 - args.alpha)).astype('uint8')

    arr2PIL(img_lb, order='RGB').save(osp.join(args.target, f'{img_id}.png'))
