#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2022/4/13 23:15
@File    : seg_criterion.py
@Software: PyCharm
@Desc    : 
"""
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

__all__ = ['UpsampleCE']


class UpsampleCE(nn.Module):

    def __init__(self, ignore_label: int=255, weight: Optional[torch.Tensor]=None):
        """给定ignore label和类别权重。前向时将logit上采样到label尺寸，计算交叉熵损失。"""
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=ignore_label, weight=weight)

    def forward(self, logit, label):
        if logit.shape[2:] != label.shape[1:]:
            upsample_logit = F.interpolate(logit, size=label.shape[1:], mode='bilinear', align_corners=True)
        else:
            upsample_logit = logit
        return self.criterion(upsample_logit, label)
