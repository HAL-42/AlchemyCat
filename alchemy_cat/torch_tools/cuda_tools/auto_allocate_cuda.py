#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2022/7/31 17:00
@File    : auto_allocate_cuda.py
@Software: PyCharm
@Desc    : 自动分配cuda设备。
"""
import os

__all__ = ['allocate_cuda_by_group_rank']

from .cuda_block import cuda_block_util_no_process_on


def allocate_cuda_by_group_rank(group_rank: int, group_cuda_num: int, block: bool=True,
                                verbosity: bool=True) -> tuple[list[int], dict[str, str]]:
    """依据任务组编号，分配cuda设备。

    Args:
        group_rank: 任务组编号。
        group_cuda_num: 任务需要设备数。
        block: 若为True，则阻塞直至分配到的GPU空闲。
        verbosity: 若为True，阻塞时打印心跳。

    Returns:
        分配给当前任务的cuda设备，以及指定了当前设备的环境变量。
    """
    # * 确定组数。
    cudas = [int(c) for c in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]  # 可用CUDA设备。
    cuda_num = len(cudas)  # 卡数。
    assert cuda_num % group_cuda_num == 0
    group_num = cuda_num // group_cuda_num  # 组数。

    # * 找到当前应当使用的CUDA设备。
    group_idx = group_rank % group_num  # 当前所在组。
    current_cudas = cudas[group_idx * group_cuda_num:(group_idx + 1) * group_cuda_num]  # 所在组拥有的CUDA设备。

    # * 等待当前CUDA设备空闲。
    if block:
        cuda_block_util_no_process_on(current_cudas, verbosity=verbosity)

    # * 获取当前指定了当前设备的环境变量。
    env_with_current_cuda = os.environ.copy()
    env_with_current_cuda['CUDA_VISIBLE_DEVICES'] = ','.join([str(c) for c in current_cudas])

    # * 返回分配的cuda设备。
    return current_cudas, env_with_current_cuda
