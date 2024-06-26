#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/22 21:18
@File    : cl_loss.py
@Software: PyCharm
@Desc    : 
"""
import torch
from torch import nn

__all__ = ['MultiLabelCLLoss']


class MultiLabelCLLoss(nn.Module):

    def __init__(self,
                 gamma: float | None = None,
                 reduce: str = 'pos_mean'):
        """将多标签分类的正类视作正样本，负类视作负样本，计算对比损失。

        Args:
            gamma: 相似度的放缩因子。
            reduce: 可以为：pos_mean，对每个正负样本对求平均；sample_mean，先对每个样本的正负样本对求平均，
                再对所有样本求平均。
        """
        super().__init__()
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, S: torch.Tensor, cls_lb: torch.Tensor) -> torch.Tensor:
        """将多标签分类的正类视作正样本，负类视作负样本，计算对比损失。

        Args:
            S: (N, G)的相似度图。
            cls_lb: (N, G)的类别标签。

        Returns:
            多标签对比损失。
        """
        # * 提前计算要用到的索引、数量。
        # ** 计算有效anchor的数量。有限anchor指的是至少有一个正样本和负样本的anchor。
        valid_anchor_mask = torch.any(cls_lb == 0, dim=1) & torch.any(cls_lb == 1, dim=1)
        valid_anchor_num = valid_anchor_mask.sum(dtype=torch.int32)
        if valid_anchor_num == 0:
            return S.mean() * 0
        # ** 计算正负样本掩码。
        neg_mask = (cls_lb == 0) * valid_anchor_mask[:, None]
        pos_mask = (cls_lb == 1) * valid_anchor_mask[:, None]
        # ** 计算正样本的锚序号。
        pi_a = torch.nonzero(pos_mask, as_tuple=True)[0]
        # ** 对每个正样本，计算与之共享anchor的正样本数量。
        pi_I = pos_mask.sum(dim=1, dtype=torch.int32)[pi_a]

        # * 计算CL损失。
        # ** 计算放缩后的相似度。
        S = self.gamma * S if self.gamma is not None else S
        # ** 计算所有相似度的指数。
        e_S = torch.exp(S)
        # * 计算Σje^Snj。
        Sigma_e_Sn_j = (e_S * neg_mask).sum(dim=1)
        # * 得到Spi, e^Spi。
        Spi = S[pos_mask]
        e_Spi = e_S[pos_mask]
        # * 得到每个正样本对应的Σje^Snj。
        pi_Sigma_e_Sn_j = Sigma_e_Sn_j[pi_a]

        # * 对每个正样本，计算对比损失。
        pi_cl_loss = -(Spi - torch.log(e_Spi + pi_Sigma_e_Sn_j))

        # * 按照reduction计算平均损失。
        match self.reduce:
            case 'pos_mean':
                return pi_cl_loss.mean()
            case 'sample_mean':
                pi_cl_loss = pi_cl_loss / pi_I
                return pi_cl_loss.sum() / valid_anchor_num
            case _:
                raise ValueError(f"不支持的{self.reduce=}。")
