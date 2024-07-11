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

try:
    from .cuda_block import cuda_block_util_no_process_on  # noqa
    CUDA_BLOCK_AVAILABLE = True
except ImportError:
    CUDA_BLOCK_AVAILABLE = False

__all__ = ['allocate_cuda_by_group_rank']


def allocate_cuda_by_group_rank(group_rank: int,
                                group_cuda_num: int=None, group_num: int=None,
                                block: bool=True, verbosity: bool=True) -> tuple[list[int], dict[str, str]]:
    """依据任务组编号，分配cuda设备。

    Args:
        group_rank: 任务组编号。
        group_cuda_num: 任务需要设备数。
        group_num: 同时运行几个任务。
        block: 若为True，则阻塞直至分配到的GPU空闲。
        verbosity: 若为True，阻塞时打印心跳。

    Returns:
        分配给当前任务的cuda设备，以及指定了当前设备的环境变量。
    """
    # -* 参数检查：group_cuda_num和group_num必须有且只有一个。
    assert (group_cuda_num is None) ^ (group_num is None), "group_cuda_num and group_num must have and only have one."

    # -* 若没有`CUDA_VISIBLE_DEVICES`，则返回空。
    if ('CUDA_VISIBLE_DEVICES' not in os.environ) or (group_num == 0) or (group_cuda_num == 0):
        return [], os.environ.copy()

    # -* 找到已有的CUDA设备。
    cudas = [int(c) for c in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]  # 可用CUDA设备。
    cuda_num = len(cudas)  # 卡数。

    # -* 确定卡数或组数。
    if group_num is None:
        group_cuda_num: int
        group_num = cuda_num // group_cuda_num  # 卡数已知，算出组数。
    else:
        group_num: int
        group_cuda_num = cuda_num // group_num  # 组数已知，算出卡数。

    # -* 找到当前应当使用的CUDA设备。
    group_idx = group_rank % group_num  # 当前所在组。
    current_cudas = cudas[group_idx * group_cuda_num:(group_idx + 1) * group_cuda_num]  # 所在组拥有的CUDA设备。

    # -* 等待当前CUDA设备空闲。
    if CUDA_BLOCK_AVAILABLE and block:
        cuda_block_util_no_process_on(current_cudas, verbosity=verbosity)

    # -* 获取当前指定了当前设备的环境变量。
    env_with_current_cuda = os.environ.copy()
    env_with_current_cuda['CUDA_VISIBLE_DEVICES'] = ','.join([str(c) for c in current_cudas])

    # -* 返回分配的cuda设备。
    return current_cudas, env_with_current_cuda
