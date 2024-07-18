#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2022/7/7 14:56
@File    : cuda_block.py
@Software: PyCharm
@Desc    : 阻塞直至指定CUDA设备释放。
"""
import os
from typing import Union

from time import sleep

from gpustat import new_query, GPUStat

from alchemy_cat.py_tools import yprint, gprint

__all__ = ['parse_cudas', 'cudas2CUDA_VISIBLE_DEVICES', 'block_get_available_cuda', 'get_cudas']

kSleepSecs = 30


def parse_cudas(cudas: Union[list[Union[str, int]], str]) -> list[int]:
    if isinstance(cudas, str):
        cudas = [int(cuda) for cuda in cudas.split(',')]
    else:
        cudas = [int(cuda) for cuda in cudas]
    return cudas


def cudas2CUDA_VISIBLE_DEVICES(cudas: Union[list[Union[str, int]], str]) -> str:
    return ','.join([str(cuda) for cuda in parse_cudas(cudas)])


def block_get_available_cuda(cudas: Union[list[Union[str, int]], str],
                             cuda_need: int=-1, memory_need: float=-1., max_process: int=-1,
                             sleep_secs: int=kSleepSecs, verbosity: bool=True,
                             cudas_prefer: list[int]=None) -> list[int]:
    """Block the program util find enough available CUDA devices.

    Args:
        cudas: The list of CUDA devices can be represented as a list of strings (e.g., ['1', '2']),
            a list of integers (e.g., [0, 3]), or a string (e.g., '0,1,2').
        cuda_need: GPU need for available GPU. < 0 means need all GPU.
        memory_need: Memory need for available GPU (MB). < -0.5 means need 95% GPU memory.
        max_process: Max process num on available GPU. < 0 means no limit.
        sleep_secs: If block, sleep sleep_secs seconds and recheck.
        verbosity: Where print heartbeat.
        cudas_prefer: Prefer to use these cudas.

    Returns:
        Available CUDA devices.
    """
    # -* 获取常量。
    memory_total = new_query()[0].memory_total

    # -* 参数预处理。
    # -** cudas: list[int]
    cudas = parse_cudas(cudas)

    # -** 检查、解算gpu_need。此后 1 <= cuda_need <= len(cudas)。
    if cuda_need == 0:
        return []
    elif cuda_need > 0:
        assert len(cudas) >= cuda_need, f"Need {cuda_need} GPU, but only {len(cudas)} GPU available."
    else:
        cuda_need = len(cudas)

    # -** 检查、解算memory_need。此后 0 <= memory_need <= memory_total。
    assert memory_need <= memory_total, f"Memory need {memory_need}MB is larger than total memory {memory_total}MB."
    if memory_need < -0.5:  # 需要95%的显存。
        memory_need = memory_total * 0.95

    # -** 检查、解算max_process。此后 0 <= max_process。
    if max_process < 0:
        max_process = float('inf')

    # -* 遍历，直到有足够的空闲GPU。
    while True:
        gpu_states: list[GPUStat] = list(new_query())
        available_cudas: list[int] = []

        for c, gpu_state in ((cuda, gpu_states[cuda]) for cuda in cudas):
            if (gpu_state.memory_available >= memory_need) and (len(gpu_state.processes) <= max_process):
                available_cudas.append(c)

        if len(available_cudas) >= cuda_need:
            break
        else:
            yprint(f"[BLOCKING] {len(available_cudas)} cudas {available_cudas} in {cudas} available, "
                   f"waiting {sleep_secs}s. \n"
                   f"GPU need: {cuda_need}, Memory need: {memory_need:.1f}MB, Max process: {max_process}.")
            sleep(sleep_secs)

    if cudas_prefer is None:
        ret = available_cudas[:cuda_need]
    else:
        ret = set(cudas_prefer) & set(available_cudas)  # 找到prefer中可用的cuda。
        if len(ret) < cuda_need:  # 若可用cuda不足。
            ret = list(ret) + list(set(available_cudas) - ret)[:cuda_need-len(ret)]
        else:  # 若可用cuda足够。
            ret = list(ret)[:cuda_need]
    if verbosity:
        gprint(f"[SUCCESS] Get available cudas {ret}.")
    return ret


def get_cudas() -> tuple[list[int], str]:
    """Get CUDA_VISIBLE_DEVICES environment variable. If not exist, return all device.

    Returns:
        tuple[list of cuda index, CUDA_VISIBLE_DEVICES environment variable]
    """
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        cuda_env = os.environ['CUDA_VISIBLE_DEVICES']
        return parse_cudas(cuda_env), cuda_env
    else:
        cudas = [c.index for c in new_query()]
        return cudas, cudas2CUDA_VISIBLE_DEVICES(cudas)
