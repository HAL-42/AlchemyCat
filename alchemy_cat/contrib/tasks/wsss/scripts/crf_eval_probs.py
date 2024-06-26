#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/6/26 14:27
@File    : crf_eval_probs.py
@Software: PyCharm
@Desc    : 
"""
import argparse
import glob
import os
import os.path as osp

import numpy as np
from alchemy_cat.alg.dense_crf import ParDenseCRF
from alchemy_cat.contrib.evaluation.semantics_segmentation import eval_preds
from alchemy_cat.contrib.tasks.wsss.seeding2 import idx2seed
from alchemy_cat.contrib.voc import VOCAug2
from alchemy_cat.data.plugins import identical, arr2PIL
from alchemy_cat.dl_config import Config
from tqdm import tqdm


def loader(cam_dir: str, img_id: str, score_type: str='prob') -> (np.ndarray, np.ndarray):
    loaded = np.load(osp.join(cam_dir, f'{img_id}.npz'))
    prob = loaded[score_type]
    fg_cls = loaded['fg_cls']

    return prob, fg_cls


cfg = config = Config()

cfg.crf.ini.iter_max = 10
cfg.crf.ini.pos_w = 3
cfg.crf.ini.pos_xy_std = 1
cfg.crf.ini.bi_w = 4
cfg.crf.ini.bi_xy_std = 67
cfg.crf.ini.bi_rgb_std = 3
cfg.crf.ini.img_preprocess = identical  # RGB, uint8, (H, W, 3)图片。
cfg.crf.ini.align_corner = False
cfg.crf.ini.pool_size = 0

crf = ParDenseCRF(**cfg.crf.ini)


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--source', type=str)
parser.add_argument('-t', '--target', type=str)
parser.add_argument('--crf', type=int, default=0)
args = parser.parse_args()

os.makedirs(args.target, exist_ok=True)

dt = VOCAug2('datasets', split='val', rgb_img=True)

for prob_file in tqdm(glob.glob(osp.join(args.source, '*.npz')), desc="处理", unit="张", dynamic_ncols=True):
    img_id = osp.splitext(osp.basename(prob_file))[0]
    inp = dt.get_by_img_id(img_id)
    img, lb = inp.img, inp.lb

    prob, fg_cls = loader(args.source, img_id)

    if args.crf:
        prob = crf(img, prob)

    pred = idx2seed(np.argmax(prob, axis=0), fg_cls).astype(np.uint8)

    arr2PIL(pred).save(osp.join(args.target, f'{img_id}.png'))

metric = eval_preds(class_num=dt.class_num,
                    class_names=dt.class_names,
                    preds_dir=str(args.target),
                    preds_ignore_label=dt.ignore_label,
                    gts_dir=dt.label_dir,
                    gts_ignore_label=dt.ignore_label,
                    result_dir=None,
                    importance=0,
                    eval_individually=False,
                    take_pred_ignore_as_a_cls=False)

metric.print_statistics(0)
