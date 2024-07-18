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

from .get_cuda import block_get_available_cuda, get_cudas, cudas2CUDA_VISIBLE_DEVICES

__all__ = ['allocate_cuda_by_group_rank']


def allocate_cuda_by_group_rank(group_rank: int,
                                group_cuda_num: int=None, group_num: int=None,
                                block: bool=True, verbosity: bool=True,
                                memory_need: float=-1., max_process: int=-1) -> tuple[list[int], dict[str, str]]:
    """Auto allocate cuda.

    Args:
        group_rank: Task group index.
        group_cuda_num: Number of devices required for the task.
        group_num: Number of tasks running simultaneously.
        block: If True, block allocated available GPU.
        verbosity: If True, print heartbeat while blocking.
        memory_need: Memory need for available GPU (MB). < -0.5 means need 95% GPU memory.
        max_process: Max process num on available GPU. < 0 means no limit.

    Returns:
        tuple[list of cuda indices, dict of environment variables with CUDA_VISIBLE_DEVICES set]
    """
    # -* 参数检查：group_cuda_num和group_num必须有且只有一个。
    assert (group_cuda_num is None) ^ (group_num is None), "group_cuda_num and group_num must have and only have one."

    # -* 若串行或不用卡，不做分配。
    if (group_num == 0) or (group_cuda_num == 0):
        return [], os.environ.copy()

    # -* 找到已有的CUDA设备。
    cudas = get_cudas()[0]
    cuda_num = len(cudas)  # 卡数。

    # -* 确定卡数或组数。
    if group_num is None:
        group_cuda_num: int
        group_num = cuda_num // group_cuda_num  # 卡数已知，算出组数。
    else:
        group_num: int
        group_cuda_num = cuda_num // group_num  # 组数已知，算出卡数。

    # -* 等待当前CUDA设备空闲。
    # -** 按照索引确定CUDA设备。
    group_idx = group_rank % group_num  # 当前所在组。
    current_cudas = cudas[group_idx * group_cuda_num:(group_idx + 1) * group_cuda_num]  # 所在组拥有的CUDA设备。
    # -** 动态分配。
    if block:
        current_cudas = block_get_available_cuda(cudas,
                                                 cuda_need=group_cuda_num,
                                                 memory_need=memory_need,
                                                 max_process=max_process,
                                                 verbosity=verbosity,
                                                 cudas_prefer=current_cudas)

    # -* 获取当前指定了当前设备的环境变量。
    env_with_current_cuda = os.environ.copy()
    env_with_current_cuda['CUDA_VISIBLE_DEVICES'] = cudas2CUDA_VISIBLE_DEVICES(current_cudas)

    # -* 返回分配的cuda设备。
    return current_cudas, env_with_current_cuda
