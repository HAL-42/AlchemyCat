#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: val_env.py
@time: 2020/3/29 3:17
@desc:
"""
import cv2
import torch
import torch.nn as nn
from alchemy_cat.py_tools.random import set_torch_rand_seed

from .rng_cacher import RNGCacher

__all__ = ['ValidationEnv']

class ValidationEnv(object):
    """Context manager for validate model"""
    def __init__(self, model: nn.Module,
                 change_benchmark: bool=False, seed: int=0, empty_cache: bool=True, cv2_num_threads: int=-1):
        """Context manager for validate model

        Args:
            model: model to be validated
            change_benchmark: If True, will inverse torch.backends.cudnn.benchmark before and after context.
                (Default: False)
            seed: Random seed for validation. (Default: 0)
            empty_cache: If True, will empty cuda cache before validation. (Default: True)
            cv2_num_threads: Number of threads used by opencv.
        """
        self.model = model

        self.change_benchmark = change_benchmark
        self.seed = seed
        self.empty_cache = empty_cache
        self.cv2_num_threads = cv2_num_threads

        self.out_grad_enabled: bool | None = None
        self.cacher = RNGCacher()
        self.out_model_training: bool | None = None
        self.out_cv2_num_threads: int | None = None

    def __enter__(self) -> nn.Module:
        # -* 尽量释放显存给推理。
        if self.empty_cache:
            torch.cuda.empty_cache()
        # -* 记录外部梯度可用性，关闭梯度。
        self.out_grad_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        # -* 记录外部随机性。
        self.cacher.cache()
        # -* 重设随机种子。
        set_torch_rand_seed(self.seed)
        # -* 记录外部模型是否可训，并设置为验证状态。
        self.out_model_training = self.model.training
        self.model.eval()
        # -* 改变benchmark。
        if self.change_benchmark:
            torch.backends.cudnn.benchmark = not torch.backends.cudnn.benchmark
        # -* 记录cv2线程数。
        self.out_cv2_num_threads = cv2.getNumThreads()
        cv2.setNumThreads(self.cv2_num_threads)
        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb):
        # -* 恢复cv2线程数。
        cv2.setNumThreads(self.out_cv2_num_threads)
        # * 恢复benchmark。
        if self.change_benchmark:
            torch.backends.cudnn.benchmark = not torch.backends.cudnn.benchmark
        # * 恢复外部模型可训性。
        self.model.train(self.out_model_training)
        # * 恢复外部随机状态。
        self.cacher.resume(purge=True)
        # * 恢复外部梯度可用性。
        torch.set_grad_enabled(self.out_grad_enabled)
