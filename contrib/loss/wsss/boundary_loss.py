#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: boundary_loss.py
@time: 2020/3/28 22:53
@desc:
"""
from typing import Tuple, Union

import numpy as np
import scipy.ndimage as nd
import torch
import torch.nn.functional as F
from torch.multiprocessing import Pool

from alchemy_cat.acplot.shuffle_ch import BGR2RGB

from alchemy_cat.alg.dense_crf import DenseCRF

__all__ = ['Boundaryloss']


class Boundaryloss(object):
    def __init__(self, dense_crf: DenseCRF, eps: float=1e-5, crf_num_workers: int=4):
        """Compute boundary loss

        Args:
            dense_crf: DenseCRF functor
            eps: Min prob allowed when clamp probs
            crf_num_workers: num of workers when crf parallel
        """
        self.dense_crf = dense_crf
        self.eps = eps
        self.crf_num_workers = crf_num_workers

        self.crf_pool = Pool(self.crf_num_workers)

    def crf(self, imgs, probs):
        np_imgs = BGR2RGB(imgs.cpu().numpy().astype(np.uint8).transpose(0, 2, 3, 1))  # (N, H, W, C)
        np_probs = probs.detach().cpu().numpy()  # (N, C, H, W)

        # Scaled imgs to probs shape
        scaled_imgs = nd.zoom(np_imgs,
                              (1.0, np_probs.shape[2] / np_imgs.shape[1], np_probs.shape[3] / np_imgs.shape[2], 1.0),
                              order=1)

        # CRF
        crf_probs = self.crf_pool.starmap(self.dense_crf, zip(scaled_imgs, np_probs))
        crf_prob = np.stack(crf_probs, axis=0)

        # Clamp smoothed probs
        # TODO: Can be removed?
        crf_prob[crf_prob < self.eps] = self.eps
        crf_prob = crf_prob / np.sum(crf_prob, axis=1, keepdims=True)

        # to Tensor
        return torch.from_numpy(crf_prob).float().cuda(probs.get_device())

    def clamp_softmax(self, score, dim=1):
        probs = torch.clamp(F.softmax(score, dim), self.eps, 1)
        probs = probs / torch.sum(probs, dim=dim, keepdim=True)
        return probs

    def __call__(self, images, score_map, out_prob=False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute the constrain-to-boundary loss

        Args:
            images: (N, 3, H, W) RGB img
            score_map: (N, C, H, W) score map
            out_prob: If true, return smoothed predict_probs. (Default: False)

        Returns:
            constrain-to-boundary loss
        """
        probs = self.clamp_softmax(score_map)
        smooth_probs = self.crf(images, probs)
        # Compute KL-Div
        # TODO: clamp is not needed?
        loss = torch.mean(torch.sum(smooth_probs * torch.log(torch.clamp(smooth_probs / probs, 0.05, 20)), dim=1))

        if out_prob:
            return loss, smooth_probs

        return loss

    def __del__(self):
        self.crf_pool.close()
        self.crf_pool.join()
