#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2021/8/10 20:07
@File    : viz_voc_annotations.py
@Software: PyCharm
@Desc    :
"""
import argparse
import os
import os.path as osp

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

from alchemy_cat.py_tools import find_img_files, find_files_by_exts
from alchemy_cat.contrib.build_dt.voc_object_detection import load_voc_annotation


parser = argparse.ArgumentParser()
parser.add_argument('-r', '--root', type=str)
parser.add_argument('-l', '--label_source', default=1, type=int)
parser.add_argument('-s', '--show_viz', default=0, type=int)
args = parser.parse_args()

# * Colors
kColors = ['red', 'blue', 'darkorange', 'violet', 'cyan', 'green', 'black']

# * Create VOC Dir & Source Dir
kImgDir = osp.join(args.root, 'JPEGImages')
kAnnoDir = osp.join(args.root, 'Annotations')
kVizDir = osp.join(args.root, 'VizAnnotations')

os.makedirs(kVizDir, exist_ok=True)


# * Find all image, label, source files.
img_files = find_img_files(kImgDir)
anno_files = [osp.join(kAnnoDir, file) for file in sorted(list(find_files_by_exts(kAnnoDir, exts=['.xml'])))]

# * Prepare figure to show
fig: plt.Figure = plt.figure(dpi=150)

# * For each img, create corresponding xml annotations
for img_file, anno_file in tqdm(zip(img_files, anno_files),
                                desc='Processing', unit='imgs', dynamic_ncols=True,
                                total=len(img_files)):
    # * Get img id
    img_id = osp.splitext(osp.basename(img_file))[0]

    # * Get RGB img
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)[..., ::-1]

    # * Get annotation
    anno = load_voc_annotation(anno_file)

    # * Plot bottom img
    ax: plt.Axes = fig.add_subplot()
    ax.imshow(img)
    ax.axis("off")

    # * Plot bbox
    for i, obj in enumerate(anno['objects']):
        name = obj['name']
        x_min, y_min, x_max, y_max = obj['bndbox']
        ax.add_patch(patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                       fill=False, edgecolor=kColors[i % len(kColors)], lw=1))
        ax.text(x_min, (y_min-15), name, verticalalignment='top', color=kColors[i % len(kColors)], fontsize=5)

    # * Label source
    if args.label_source:
        ax.text(0, 0, f"{anno['source_database']} {anno['source_image']}",
                verticalalignment='top', color='darkred', fontsize=5, weight='bold')

    # * Show viz
    if args.show_viz:
        fig.show()

    # * Save viz
    fig.savefig(osp.join(kVizDir, f'{img_id}.png'), bbox_inches='tight', pad_inches=0)

    fig.clf()
