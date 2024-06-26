#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2022/8/2 16:28
@File    : softmax.py
@Software: PyCharm
@Desc    :
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['MaskedSoftmax']


class MaskedSoftmax(nn.Module):

    def __init__(self, dim: int=-1, epsilon: float=1e-5):
        super(MaskedSoftmax, self).__init__()
        self.dim = dim
        self.epsilon = epsilon

        self.softmax = nn.Softmax(dim=dim)

    def forward(self, sim_map: torch.Tensor, mask: torch.Tensor | None=None) -> torch.Tensor:
        """做有掩码的Softmax。

        Args:
            sim_map: logits。
            mask: 与logits同shape的掩码。

        Returns:
            带掩码的Softmax结果。
        """
        if mask is None:
            return F.softmax(sim_map, dim=self.dim)
        else:
            e_sim = torch.exp(sim_map - torch.max(sim_map, dim=self.dim, keepdim=True)[0])
            masked_e_sim = e_sim * mask
            masked_sums = masked_e_sim.sum(dim=self.dim, keepdim=True) + self.epsilon
            return masked_e_sim / masked_sums
