#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2022/9/7 15:05
@File    : avg_models.py
@Software: PyCharm
@Desc    : 
"""
from tqdm import tqdm

import torch

__all__ = ['avg_states']


def avg_states(state_files: list[str], device: torch.device= 'cpu', verbosity: bool=True) -> dict[str, torch.Tensor]:
    """多个模型参数取平均。

    Args:
        state_files: 模型文件。
        device: 模型加载到哪个设备上。
        verbosity: 若为True，打印合体进程条。

    Returns:
        平均后的模型参数。
    """
    avg_state: dict[str, torch.Tensor] | None = None
    for state_file in (tqdm(state_files, desc='合体', unit='个', dynamic_ncols=True) if verbosity else state_files):
        # * 载入模型。
        state: dict[str, torch.Tensor] = torch.load(state_file, map_location=device)
        # * 第一个模型，作为基础模型。
        if avg_state is None:
            avg_state = state
        # * 此后模型，检查和平均模型键值、参数相同，加到平均模型上。
        else:
            assert set(avg_state.keys()) == set(state.keys())
            for k in avg_state.keys():
                assert avg_state[k].dtype == state[k].dtype
                assert avg_state[k].shape == state[k].shape
                avg_state[k] += state[k]

    # * 模型参数取平均。
    for k, v in avg_state.items():
        avg_state[k] = v / len(state_files) if v.is_floating_point() else torch.div(v, len(state_files),
                                                                                    rounding_mode='floor')

    return avg_state
