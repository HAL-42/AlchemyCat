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
import multiprocessing as mp
import os
import os.path as osp

import numpy as np
from PIL import Image
from alchemy_cat.contrib.coco import COCO
from alchemy_cat.contrib.evaluation.semantics_segmentation import eval_preds
from alchemy_cat.data.plugins import arr2PIL
from tqdm import tqdm


def 上色_label_file(label_file: str):
    if not label_file.endswith('.png'):
        return

    label = np.array(Image.open(osp.join(args.source, label_file)), dtype=np.uint8)
    color_label = dt.label_map2color_map(label)
    arr2PIL(color_label, order='RGB').save(osp.join(args.target, label_file))


def ignore2bg(lb: np.ndarray):
    lb[lb == 255] = 0
    return lb


parser = argparse.ArgumentParser()
parser.add_argument('-e', '--eval', type=int, default=0)
parser.add_argument('-c', '--colorize', type=int, default=0)
parser.add_argument('-s', '--source', type=str)
parser.add_argument('-t', '--target', type=str)
parser.add_argument('--label_type', type=str, default='')
parser.add_argument('--split', type=str, default='train')
args = parser.parse_args()

dt = COCO('datasets', split=args.split, label_type=args.label_type, cls_labels_type=None)

if args.colorize:
    os.makedirs(args.target, exist_ok=True)

    with mp.Pool(int(mp.cpu_count() * 0.2)) as p:
        for _ in tqdm(p.imap_unordered(上色_label_file, files := os.listdir(args.source), chunksize=10),
                      dynamic_ncols=True, desc="处理", unit="张", total=len(files)):
            pass

if args.eval:
    metric = eval_preds(class_num=dt.class_num,
                        class_names=dt.class_names,
                        preds_dir=str(args.source),
                        preds_ignore_label=dt.ignore_label,
                        gts_dir=dt.label_dir,
                        gts_ignore_label=dt.ignore_label,
                        pred_preprocess=ignore2bg,
                        result_dir=None,
                        importance=0,
                        eval_individually=False,
                        take_pred_ignore_as_a_cls=False)

    metric.print_statistics(0)
