#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/2/22 20:19
@File    : colorize_voc.py
@Software: PyCharm
@Desc    : 
"""
import typing as t

import argparse
import multiprocessing as mp
import os
import os.path as osp
from functools import partial

import numpy as np
from PIL import Image
from alchemy_cat.contrib.evaluation.semantics_segmentation import eval_preds
from alchemy_cat.contrib.voc import VOCAug2
from alchemy_cat.contrib.voc import label_map2color_map
from alchemy_cat.data.plugins import arr2PIL
from tqdm import tqdm


def colorize_label_file(label_file: str, source: str, target: str, l2c: t.Callable[[np.ndarray], np.ndarray]=None):
    if not label_file.endswith('.png'):
        return

    if l2c is None:
        l2c = label_map2color_map

    label = np.array(Image.open(osp.join(source, label_file)))
    color_label = l2c(label).astype(np.uint8)
    arr2PIL(color_label, order='RGB').save(osp.join(target, label_file))


def ignore2bg(lb: np.ndarray):
    lb[lb == 255] = 0
    return lb


def colorize_voc(source: str, target: str, num_workers: int=0, is_eval: bool=False,
                 l2c: t.Callable[[np.ndarray], np.ndarray]=None):
    os.makedirs(target, exist_ok=True)

    label_files = os.listdir(source)
    worker = partial(colorize_label_file, source=source, target=target, l2c=l2c)

    if num_workers > 0:
        with mp.Pool(num_workers) as p:
            for _ in tqdm(p.imap_unordered(worker, label_files, chunksize=10), total=len(label_files),
                          dynamic_ncols=True, desc="上色VOC", unit="张", miniters=len(label_files)//10):
                pass
    else:
        for _ in tqdm(map(worker, label_files), total=len(label_files),
                      dynamic_ncols=True, desc="上色VOC", unit="张", miniters=len(label_files)//10):
            pass

    if is_eval:
        dt = VOCAug2('datasets')

        metric = eval_preds(class_num=dt.class_num,
                            class_names=dt.class_names,
                            preds_dir=str(source),
                            preds_ignore_label=dt.ignore_label,
                            gts_dir=dt.label_dir,
                            gts_ignore_label=dt.ignore_label,
                            pred_preprocess=ignore2bg,
                            result_dir=None,
                            importance=0,
                            eval_individually=False,
                            take_pred_ignore_as_a_cls=False)

        metric.print_statistics(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--is_eval', type=int, default=0)
    parser.add_argument('-n', '--num_workers', type=int, default=1)
    parser.add_argument('-s', '--source', type=str)
    parser.add_argument('-t', '--target', type=str)
    _args = parser.parse_args()
    colorize_voc(source=_args.source, target=_args.target, num_workers=_args.num_workers, is_eval=bool(_args.is_eval))
