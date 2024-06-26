#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2021/7/23 3:45
@File    : validate.py
@Software: PyCharm
@Desc    : 
"""
from tqdm import tqdm
import numpy as np

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from alchemy_cat.data import DataAuger
from alchemy_cat.contrib.voc import VOC_CLASSES
from alchemy_cat.contrib.metrics import SegmentationMetric
from alchemy_cat.alg import msc_flip_inference, find_nearest_odd_size

__all__ = ['validate_seg']


def validate_seg(model: nn.Module, val_auger: DataAuger, iteration: int, writer: SummaryWriter=None):
    """测试分割模型。

    Args:
        model: torch模型。
        val_auger: validation数据集的data_auger。
        writer: tensorboard的writer。
        iteration: 当前迭代次数。

    Returns:

    """
    print("\n================================== Validation ==================================")
    metric = SegmentationMetric(len(VOC_CLASSES), VOC_CLASSES)
    device = list(model.parameters())[0].device

    for _, img, label in tqdm(val_auger, total=len(val_auger),
                              desc="Validation Process", unit='samples', dynamic_ncols=True):
        bt_img = torch.from_numpy(img).to(device=device, dtype=torch.float32)[None, ...]
        score_map = msc_flip_inference(imgs=bt_img,
                                       model=model,
                                       msc_factors=[1.],
                                       is_flip=False,
                                       msc_aligner=lambda x: find_nearest_odd_size(x, min_n=3),
                                       softmax_norm=False
                                       ).cpu().numpy()[0]
        metric.update(np.argmax(score_map, axis=0), label)

    metric.print_statistics(importance=1)

    if writer is not None:
        writer.add_scalar('mIoU', metric.mIoU, iteration + 1)
        writer.add_scalar('Precision', metric.macro_avg_precision, iteration + 1)
        writer.add_scalar('Recall', metric.macro_avg_recall, iteration + 1)
        writer.add_scalar('Accuracy', metric.accuracy, iteration + 1)
    print("\n================================ Validation End ================================")
