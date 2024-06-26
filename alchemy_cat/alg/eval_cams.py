#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/13 16:20
@File    : eval_cams.py
@Software: PyCharm
@Desc    : 
"""
import glob
import multiprocessing as mp
import os
import pickle
import warnings
from collections import defaultdict
from functools import partial
from os import path as osp
from typing import Optional, Iterable, Callable, Tuple

import cv2
import numpy as np
from PIL import Image
from alchemy_cat.contrib.metrics import SegmentationMetric
from alchemy_cat.acplot import RGB2BGR
from alchemy_cat.data.plugins import identical
from alchemy_cat.py_tools import OneOffTracker
from alchemy_cat.dl_config import Config
from frozendict import frozendict as fd
from tqdm import tqdm

__all__ = ['eval_cams', 'search_and_eval']


def _eval_cams_core(cam_file_suffix: str,
                    cam2pred: Callable,
                    cam_dir: str, gt_dir: str, image_dir: str,
                    gt_preprocess: Callable[[np.ndarray], np.ndarray],
                    crf: Callable[[np.ndarray, np.ndarray], np.ndarray]
                    ) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    img_id = osp.splitext(cam_file_suffix)[0]
    cam_file = osp.join(cam_dir, cam_file_suffix)
    gt_file = osp.join(gt_dir, f'{img_id}.png')

    # Read files
    gt = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
    assert gt is not None
    gt: np.ndarray = gt_preprocess(gt)

    loaded = np.load(cam_file)
    cam, fg_cls = loaded['cam'].astype(np.float32), loaded['fg_cls'].astype(np.uint8)

    if crf is not None:
        dsize = np.asarray(Image.open(osp.join(image_dir, f'{img_id}.jpg')), dtype=np.uint8)
    else:
        dsize = gt.shape

    pred, fg_bg_score = cam2pred(cam, dsize, fg_cls, crf=crf)

    assert pred.shape == gt.shape

    return img_id, pred, fg_bg_score, fg_cls, gt


def eval_cams(class_num: int, class_names: Optional[Iterable[str]],
              cam_dir: str, preds_ignore_label: int,
              gt_dir: str, gts_ignore_label: int,
              cam2pred: Callable,
              image_dir: str | None=None,
              crf: Callable[[np.ndarray, np.ndarray], np.ndarray] | None=None,
              result_dir: Optional[str]=None,
              gt_preprocess: Callable[[np.ndarray], np.ndarray]=identical,
              importance: int=2,
              eval_individually: bool=True, take_pred_ignore_as_a_cls: bool=False,
              metric_cls: type=SegmentationMetric,
              pool_size=0,
              save_sample_data: bool=False) \
        -> (SegmentationMetric, defaultdict):

    """Evaluate predictions of semantic segmentation

    Args:
        class_num: Num of classes
        class_names: Name of classes
        cam_dir: Dictionary where cam stored
        preds_ignore_label: Ignore label for predictions
        gt_dir: Dictionary where ground truths stored
        image_dir: Dictionary where images stored. (Default: None)
        gts_ignore_label: Ignore label for ground truths
        result_dir: If not None, eval result will be saved to result_dir. (Default: None)
        cam2pred: Function transfer cam file to prediction.
        crf: Function to refine prediction. (Default: None)
        gt_preprocess: Preprocess function for ground truth read in
        importance: Segmentation Metric's importance filter. (Default: 2)
        eval_individually: If True, evaluate each sample. (Default: True)
        take_pred_ignore_as_a_cls: If True, the ignored label in preds will be seemed as a class. (Default: False)
        metric_cls: Use metric_cls(class_num, class_names) to eval preds. (Default: SegmentationMetric)
        pool_size: Pool size for multiprocessing. (Default: 0)
        save_sample_data: If True, sample data (pred, bg_fg_score) will be saved. (Default: False)

    Returns:
        Segmentation Metric and metric result for each sample (If eval_individually is True)
    """
    assert preds_ignore_label >= class_num
    assert gts_ignore_label >= class_num

    if take_pred_ignore_as_a_cls:
        class_num += 1
        class_names += ['pred_ignore']

    metric = metric_cls(class_num, class_names)

    sample_data = defaultdict(dict)

    # * 构造核心函数。
    core = partial(_eval_cams_core,
                   cam2pred=cam2pred,
                   cam_dir=cam_dir, gt_dir=gt_dir, image_dir=image_dir,
                   gt_preprocess=gt_preprocess,
                   crf=crf)

    print("\n================================== Eval ==================================")
    cam_file_suffixes = os.listdir(cam_dir)

    if pool_size > 0:
        p = mp.Pool(pool_size)
        it = p.imap_unordered(core, cam_file_suffixes, chunksize=10)
    else:
        it = map(core, cam_file_suffixes)

    for img_id, pred, bg_fg_score, fg_cls, gt in tqdm(it,
                                                      total=len(cam_file_suffixes), miniters=10,
                                                      desc='eval progress', unit='sample', dynamic_ncols=True):

        if take_pred_ignore_as_a_cls:
            pred[pred == preds_ignore_label] = class_num - 1
            if gts_ignore_label == class_num - 1:
                gt[gt == gts_ignore_label] = class_num

        # Evaluate
        metric.update(pred, gt)
        if eval_individually:
            with OneOffTracker(lambda: metric_cls(class_num, class_names)) as individual_metric:
                individual_metric.update(pred, gt)
            sample_data[img_id]['metric'] = individual_metric.statistics(importance)

        if save_sample_data:
            sample_data[img_id]['seed'] = pred  # 在外面叫seed。
            sample_data[img_id]['bg_fg_score'] = bg_fg_score
            sample_data[img_id]['fg_cls'] = fg_cls

    if pool_size > 0:
        p.close()
        p.join()

    # Saving
    if result_dir is not None:
        metric.save_metric(result_dir, importance, dpi=400)
        if eval_individually:
            with open(osp.join(result_dir, 'sample_statistics.pkl'), 'wb') as f:
                pickle.dump({i: d['metric'] for i, d in sample_data.items()}, f)

    print("\n================================ Eval End ================================")

    return metric, sample_data


def _save_sample_data(sample_data: defaultdict | str, save_dir: str, mask_cfg: Config):
    data_dir = osp.join(save_dir, 'data')
    mask_dir = osp.join(save_dir, 'mask')
    viz_mask_dir = osp.join(save_dir, 'viz_mask')

    mask_only = False
    if sample_data == 'saved':
        assert osp.isdir(data_dir)  # 检查是否真的保存了。
        if mask_cfg and (not osp.isdir(mask_dir)):  # 虽然存过了，但mask没存。
            mask_only = True
            data_files = glob.glob(osp.join(data_dir, '*.npz'))
            sample_data = {osp.splitext(osp.basename(f))[0]: np.load(f) for f in data_files}
        else:
            return

    os.makedirs(data_dir, exist_ok=True)
    if mask_cfg:
        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(viz_mask_dir, exist_ok=True)

    for img_id, data in sample_data.items():
        if not mask_only:
            np.savez(osp.join(data_dir, f'{img_id}.npz'),
                     seed=data['seed'], bg_fg_score=data['bg_fg_score'], fg_cls=data['fg_cls'])
        if mask_cfg:
            cv2.imwrite(osp.join(mask_dir, f'{img_id}.png'), mask := mask_cfg.cal(data).astype(np.uint8))
            cv2.imwrite(osp.join(viz_mask_dir, f'{img_id}.png'), RGB2BGR(mask_cfg.viz(mask)))


def search_and_eval(dt, cam_dir: str, seed_cfg: Config, rslt_dir: str, pool_size: int=0):
    """搜索不同bg_method，获取最优性能。

    Args:
        dt: 数据集，提供类别数、类别名、忽略标签、标签目录等信息。
        cam_dir: cam文件所在目录。
        rslt_dir: 结果的目录。
        seed_cfg: seed_cfg.cal(cam, gt.shape, fg_cls, seed_cfg.bg_methods[i], **seed_cfg.ini)将CAM转换为seed。
        pool_size: 并行计算的进程数。0表示不使用并行计算。 (Default: 0)

    Returns:
        None
    """
    os.makedirs(eval_dir := osp.join(rslt_dir, 'eval'), exist_ok=True)
    if seed_cfg.save is not None:
        assert seed_cfg.save in ['all', 'best']
        os.makedirs(seed_dir := osp.join(rslt_dir, 'seed'), exist_ok=True)

    # * 若已经有bg_method_metrics.pkl，则直接读取。
    if osp.isfile(bg_method_metrics_pkl := osp.join(eval_dir, 'bg_method_metrics.pkl')):
        with open(bg_method_metrics_pkl, 'rb') as f:
            bg_method_metrics = pickle.load(f)
        if seed_cfg.save is not None:
            warnings.warn(f"应当确保缓存的{bg_method_metrics_pkl}保存了对应的种子。")
    else:
        bg_method_metrics = {}

    # * 对各配置中的methods，计算其metric。
    best_bg_method, best_sample_data = None, None

    for bg_method in seed_cfg.bg_methods:
        bg_method = fd(dict(bg_method))  # 将dict转换为frozendict，以便作为字典的key。

        if bg_method in bg_method_metrics:
            metric, sample_data = bg_method_metrics[bg_method], 'saved'
        else:
            metric, sample_data = eval_cams(class_num=dt.class_num,
                                            class_names=dt.class_names,
                                            cam_dir=cam_dir,
                                            preds_ignore_label=255,
                                            gt_dir=dt.label_dir,
                                            gts_ignore_label=dt.ignore_label,
                                            cam2pred=partial(seed_cfg.cal,
                                                             bg_method=bg_method, **seed_cfg.ini),
                                            image_dir=dt.image_dir,
                                            crf=seed_cfg.crf,
                                            result_dir=None,
                                            importance=0,
                                            eval_individually=False,
                                            take_pred_ignore_as_a_cls=False,
                                            pool_size=pool_size,
                                            save_sample_data=seed_cfg.save is not None)

        print(f'Current mIoU: {metric.mIoU:.4f} (bg_method={bg_method})')
        bg_method_metrics[bg_method] = metric

        if (best_bg_method is None) or (metric.mIoU > bg_method_metrics[best_bg_method].mIoU):
            best_bg_method, best_sample_data = bg_method, sample_data

        if seed_cfg.save == 'all':
            _save_sample_data(sample_data, osp.join(seed_dir, f'{metric.mIoU:.2f}'), seed_cfg.mask)

    # * 保存method_metrics.pkl。
    with open(bg_method_metrics_pkl, 'wb') as f:
        pickle.dump(bg_method_metrics, f)

    # * 打印所有评价结果。
    for bg_method, metric in bg_method_metrics.items():
        print(f'mIoU: {metric.mIoU:.4f} (bg_method={bg_method})')
    print(f'Best mIoU: {bg_method_metrics[best_bg_method].mIoU:.4f} (bg_method={best_bg_method})')

    # * 遍历method_metrics字典，找到最好的metric。
    bg_method_metrics[best_bg_method].save_metric(eval_dir, importance=0, figsize=(24, 24))
    if seed_cfg.save == 'best':
        _save_sample_data(best_sample_data, osp.join(seed_dir, 'best'), seed_cfg.mask)
