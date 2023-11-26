#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/6/26 22:25
@File    : 转VOC上传.py
@Software: PyCharm
@Desc    : 
"""
import argparse
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--source', type=str)
parser.add_argument('-t', '--target', type=str)
args = parser.parse_args()

os.makedirs(args.target, exist_ok=True)

palette = []

for i in range(256):
    palette.extend((i, i, i))

palette[:3 * 21] = np.array([[0, 0, 0],
                             [128, 0, 0],
                             [0, 128, 0],
                             [128, 128, 0],
                             [0, 0, 128],
                             [128, 0, 128],
                             [0, 128, 128],
                             [128, 128, 128],
                             [64, 0, 0],
                             [192, 0, 0],
                             [64, 128, 0],
                             [192, 128, 0],
                             [64, 0, 128],
                             [192, 0, 128],
                             [64, 128, 128],
                             [192, 128, 128],
                             [0, 64, 0],
                             [128, 64, 0],
                             [0, 192, 0],
                             [128, 192, 0],
                             [0, 64, 128]], dtype='uint8').flatten()

for file in tqdm(os.listdir(args.source), desc="处理", unit="张", dynamic_ncols=True):
    im = Image.open(os.path.join(args.source, file))
    im = Image.fromarray(np.array(im).astype(np.uint8))
    im.putpalette(palette)
    im.save(os.path.join(args.target, file))
