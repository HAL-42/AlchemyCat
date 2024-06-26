#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2022/5/31 17:32
@File    : rng_cacher.py
@Software: PyCharm
@Desc    : 
"""
from typing import Optional

import os
import os.path as osp

import torch
import torch.distributed as dist

__all__ = ['RNGCacher']


# TODO 增加numpy支持。
class RNGCacher(object):
    """缓存随机状态。"""

    def __init__(self):
        self._rng_state: Optional[torch.Tensor] = None
        self._cuda_rng_state: Optional[torch.Tensor] = None

    def cache(self):
        self._rng_state = torch.get_rng_state()
        self._cuda_rng_state = torch.cuda.get_rng_state()

    def resume(self, purge: bool=True):
        assert self._rng_state is not None
        assert self._cuda_rng_state is not None

        torch.set_rng_state(self._rng_state)
        torch.cuda.set_rng_state(self._cuda_rng_state)

        if purge:
            self._rng_state = None
            self._cuda_rng_state = None

    def __enter__(self):
        """退出后回到进入前的随机状态。"""
        self.cache()

    def __exit__(self, *args):
        self.resume(purge=True)

    def __iter__(self):
        """每次循环开始时，回到循环前随机状态。"""
        self.cache()
        while True:
            yield
            self.resume(purge=False)

    @staticmethod
    def rng_state_file(cache_dir):
        if dist.is_initialized():
            rng_state_file = osp.join(cache_dir, '.rng_state_log', f'rank{dist.get_rank()}_rng_state.pth')
        else:
            rng_state_file = osp.join(cache_dir, '.rng_state_log', 'rng_state.pth')
        return rng_state_file

    @classmethod
    def cache_to_file(cls, cache_dir: str):
        """将当前（rank）随机状态保存到文件。"""
        os.makedirs(osp.join(cache_dir, '.rng_state_log'), exist_ok=True)
        torch.save((torch.get_rng_state(), torch.cuda.get_rng_state()), cls.rng_state_file(cache_dir))

    @classmethod
    def resume_from_file(cls, cache_dir: str):
        """从文件中恢复（当前rank）的随机状态。"""
        rng_state, cuda_rng_state = torch.load(cls.rng_state_file(cache_dir))
        torch.set_rng_state(rng_state)
        torch.cuda.set_rng_state(cuda_rng_state)

    @classmethod
    def resume_except_cache_to_file(cls, cache_dir):
        """尝试从文件中恢复（当前rank）的随机状态，失败则将当前（rank）随机状态保存到文件。"""
        try:
            cls.resume_from_file(cache_dir)
        except Exception:
            print(f"无法从{cls.rng_state_file(cache_dir)}恢复随机状态, 转而保存当前随机状态。")
            cls.cache_to_file(cache_dir)
        else:
            print(f"从{cls.rng_state_file(cache_dir)}恢复随机状态。")
