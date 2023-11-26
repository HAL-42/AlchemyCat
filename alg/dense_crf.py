#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: dense_crf.py
@time: 2020/3/29 0:05
@desc:
"""
from multiprocessing import Pool
from typing import Callable

import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
from PIL import Image
from alchemy_cat.data.plugins import scale_img_label

__all__ = ['DenseCRF', 'ParDenseCRF']


class DenseCRF(object):
    def __init__(self, iter_max, pos_w, pos_xy_std, bi_w, bi_xy_std, bi_rgb_std):
        self.iter_max = iter_max
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std

    def __call__(self, image, probmap):
        C, H, W = probmap.shape

        U = utils.unary_from_softmax(probmap)
        U = np.ascontiguousarray(U)

        image = np.ascontiguousarray(image)

        d = dcrf.DenseCRF2D(W, H, C)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
        d.addPairwiseBilateral(
            sxy=self.bi_xy_std, srgb=self.bi_rgb_std, rgbim=image, compat=self.bi_w
        )

        Q = d.inference(self.iter_max)
        Q = np.array(Q).reshape((C, H, W))

        return Q


class ParDenseCRF(DenseCRF):
    def __init__(self, iter_max, pos_w, pos_xy_std, bi_w, bi_xy_std, bi_rgb_std,
                 img_preprocess: Callable[[np.ndarray], np.ndarray]=lambda x: x, align_corner: bool=False,
                 pool_size: int=0):  # TODO PIL_MODE
        super().__init__(iter_max, pos_w, pos_xy_std, bi_w, bi_xy_std, bi_rgb_std)
        self.img_preprocess = img_preprocess
        self.align_corner = align_corner

        self.pool_size = pool_size
        self.crf_pool = Pool(self.pool_size) if self.pool_size > 0 else None

    def __call__(self, image: np.ndarray, probmap: np.ndarray) -> np.ndarray:
        """根据输入尺寸，决定是否并行运行。

        Args:
            image: (N, C, H, W)或(H, W, C)图片。
            probmap: (N, P+1, H, W)或(P+1, H, W)概率图。

        Returns:
            (N, P+1, H, W)或(P+1, H, W)优化后概率图。
        """
        if image.ndim == 3:
            return self.single_run(image, probmap)
        elif image.ndim == 4:
            return self.par_run(image, probmap)
        else:
            raise ValueError(f"Invalid image dim: {image.ndim}")

    def par_run(self, image: np.ndarray, probmap: np.ndarray) -> np.ndarray:
        """运行并行CRF。

        Args:
            image: (N, C, H, W)图片。
            probmap: (N, P+1, H, W)概率图。

        Returns:
            (N, P+1, H, W)优化后概率图。
        """
        image = image.transpose((0, 2, 3, 1))  # (N, H, W, C)

        if self.crf_pool is not None:
            return np.stack(self.crf_pool.starmap(self.single_run, zip(image, probmap)), axis=0)
        else:
            return np.stack([self.single_run(i, p) for i, p in zip(image, probmap)], axis=0)

    def single_run(self, image: np.ndarray, probmap: np.ndarray) -> np.ndarray:
        """运行单个CRF。

        Args:
            image: (H, W, C)图片。
            probmap: (P+1, H, W)概率图。

        Returns:
            (P+1, H, W)优化后概率图。
        """
        image = self.img_preprocess(image)
        if image.shape[:2] != (dsize := probmap.shape[1:]):
            image = scale_img_label(dsize, image, align_corner=self.align_corner, PIL_mode=Image.BICUBIC)
        return super().__call__(image, probmap)
