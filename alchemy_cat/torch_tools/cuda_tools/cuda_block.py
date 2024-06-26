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
from time import sleep
import warnings

from gpustat import new_query

from alchemy_cat.py_tools import yprint, gprint

__all__ = ['cuda_block_util_no_process_on']

kSleepSecs = 30


def cuda_block_util_no_process_on(cudas: list[str | int] | str, sleep_secs: int=kSleepSecs, verbosity: bool=True):
    """阻塞程序，直到指定的cuda设备上没有任何进程。

    Args:
        cudas: cuda设备列表，可以用字符串列表（如['1', '2']）、整数列表（如[0, 3]）或字符串（如'0,1,2'）表示。
        sleep_secs: 若检测到有进程，间隔多少秒再做检测。
        verbosity: 是否打印阻塞信息。

    Returns:
            无。
    """
    if isinstance(cudas, str):
        cudas = [int(cuda) for cuda in cudas.split(',')]
    else:
        cudas = [int(cuda) for cuda in cudas]

    while True:
        gpu_states = new_query()

        for c, gpu_state in ((cuda, gpu_states[cuda]) for cuda in cudas):
            if len(gpu_state.processes) > 0:
                if verbosity:
                    yprint(f"检测到cuda:{c}上有进程{gpu_state.processes}占用，程序阻塞{sleep_secs}s后，再行检测...")
                break
        else:
            break

        sleep(sleep_secs)

    if verbosity:
        gprint(f"检测到所需CUDA设备{cudas}上已经没有进程占用，阻塞结束。")


if __name__ == "__main__":
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        warnings.warn("没有检测到环境变量CUDA_VISIBLE_DEVICES，阻塞无效，退出。")
        exit(0)

    cuda_block_util_no_process_on(os.environ['CUDA_VISIBLE_DEVICES'])
