#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2022/4/30 17:26
@File    : dist_communication.py
@Software: PyCharm
@Desc    : 跨卡通信工具包。
"""
import torch
import torch.distributed as dist

__all__ = ['all_gather_tensor', 'all_gather_tensor_grad']


def all_gather_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """在多个rank上，将(N, ...)的张量收集到一起，拼接为(rank_num * N,...)的Tensor。梯度不会被保留。

    Args:
        tensor: (N, ...)张量。

    Returns:
        (rank_num * N,...)张量。
    """
    with torch.no_grad():  # 或冗余，强调是无梯度操作。
        tensor_list = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list, tensor)
    return torch.cat(tensor_list, dim=0)


def all_gather_tensor_grad(tensor: torch.Tensor) -> torch.Tensor:
    """在多个rank上，将(N, ...)的张量收集到一起，拼接为(rank_num * N,...)的Tensor。来自本rank张量的梯度会被保留。

    Args:
        tensor: (N, ...)张量。

    Returns:
        (rank_num * N, ...)张量，来自本rank张量的梯度会被保留。
    """
    with torch.no_grad():
        tensor_list = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list, tensor)
    tensor_list[dist.get_rank()] = tensor
    return torch.cat(tensor_list, dim=0)
