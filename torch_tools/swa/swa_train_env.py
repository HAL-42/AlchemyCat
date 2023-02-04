#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2022/9/14 15:24
@File    : swa_train_env.py
@Software: PyCharm
@Desc    : 
"""
import torch
import torch.nn as nn

from alchemy_cat.py_tools import set_torch_rand_seed

from ..rng_cacher import RNGCacher

__all__ = ['SWATrainEnv']


class SWATrainEnv(object):
    """创建SWA训练环境。"""
    def __init__(self, model: nn.Module, device: str | int | torch.device, seed: int | str,
                 norm_init: bool=True, cumulative_avg=True):
        """SWA训练环境。

        Args:
            model: SWA模型。
            device: SWA模型训练时所在设备。
            seed: SWA训练前设置的随机种子。
            norm_init: SWA模型训练前是否用标准初始化BN统计量。
            cumulative_avg: 为True，SWA模型训练时用累积法计算统计量，反之则动量更新统计量。
        """
        self.model = model
        self.device = device
        self.seed = seed
        self.norm_init = norm_init
        self.cumulative_avg = cumulative_avg

        self.out_grad_enabled: bool | None = None
        self.cacher = RNGCacher()
        self.out_model_device: bool | None = None
        self.out_model_training: bool | None = None
        self.momenta: dict[nn.Module, ...] = {}

    def __enter__(self) -> nn.Module:
        # * 记录外部梯度可用性，关闭梯度。
        self.out_grad_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        # * 记录外部随机性。
        self.cacher.cache()
        # * 重设随机种子。
        set_torch_rand_seed(self.seed)
        # * 记录外部模型所在设备，并迁移模型到指定设备。
        self.out_model_device = next(self.model.parameters()).device
        self.model.to(device=self.device)
        # * 记录外部模型是否可训，并设置为训练状态。
        self.out_model_training = self.model.training
        self.model.train()
        # * 记录模型动量。
        for module in self.model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                self.momenta[module] = module.momentum
                if self.norm_init:
                    module.running_mean = torch.zeros_like(module.running_mean)
                    module.running_var = torch.ones_like(module.running_var)
                if self.cumulative_avg:
                    module.momentum = None
                    module.num_batches_tracked *= 0
        assert len(self.momenta) > 0
        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb):
        # * 恢复BN层动量。
        for bn_module in self.momenta.keys():
            bn_module.momentum = self.momenta[bn_module]
        # * 恢复外部模型可训性。
        self.model.train(self.out_model_training)
        # * 模型迁移到在外部时的设备。
        self.model.to(device=self.out_model_device)
        # * 恢复外部随机状态。
        self.cacher.resume(purge=True)
        # * 恢复外部梯度可用性。
        torch.set_grad_enabled(self.out_grad_enabled)
