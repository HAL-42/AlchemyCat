#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/11/26 22:25
@File    : to_cuda.py
@Software: PyCharm
@Desc    : 
"""
import torch

__all__ = ['dict_to_cuda']


def dict_to_cuda(inp: dict):
    for k, v in inp.items():
        if torch.is_tensor(v):
            inp[k] = v.to('cuda', non_blocking=True)
        if isinstance(v, dict):
            dict_to_cuda(v)
    return inp
