#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: balanced_seed_loss.py
@time: 2020/3/28 22:54
@desc:
"""
from typing import Tuple

import torch
import torch.nn.functional as F

__all__ = ['BalancedSeedloss']


class BalancedSeedloss(object):
    def __init__(self, eps: float=1e-5):
        """Compute balanced seed loss

        Args:
            eps: Min prob allowed when clamp probs
        """
        self.eps = eps

    def clamp_softmax(self, score, dim=1):
        probs = torch.clamp(F.softmax(score, dim), self.eps, 1)
        probs = probs / torch.sum(probs, dim=dim, keepdim=True)
        return probs

    def __call__(self, score_map, one_hot_label) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute balanced seed loss

        Args:
            score_map: (N, C, H, W) score map
            one_hot_label: (N, C, H, W) one-hot label

        Returns:
            Balanced Seed Loss
        """
        assert not one_hot_label.requires_grad
        pi = one_hot_label.to(torch.float)

        assert not torch.any(torch.isinf(score_map))
        assert not torch.any(torch.isnan(score_map))

        log_qi = torch.log(self.clamp_softmax(score_map))

        assert not torch.any(torch.isnan(log_qi))

        log_fg_qi = log_qi[:, 1:, :, :]
        fg_pi = pi[:, 1:, :, :]
        fg_count = torch.sum(fg_pi, dim=(1, 2, 3)) + self.eps

        log_bg_qi = log_qi[:, 0:1, :, :]
        bg_pi = pi[:, 0:1, :, :]
        bg_count = torch.sum(bg_pi, dim=(1, 2, 3)) + self.eps

        fg_loss_ = torch.sum(fg_pi * log_fg_qi, dim=(1, 2, 3))
        fg_loss = -1 * torch.mean(fg_loss_ / fg_count)  # mean reduce on batch

        bg_loss_ = torch.sum(bg_pi * log_bg_qi, dim=(1, 2, 3))
        bg_loss = -1 * torch.mean(bg_loss_ / bg_count)  # mean reduce on batch

        total_loss = bg_loss + fg_loss
        assert not torch.any(torch.isnan(total_loss)), \
            "fg_loss: {} fg_count: {} bg_loss: {} bg_count: {}".format(fg_loss, fg_count, bg_loss, bg_count)

        return total_loss, bg_loss, fg_loss
