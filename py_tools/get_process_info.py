#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: get_process_info.py
@time: 2020/6/3 0:00
@desc:
"""
from typing import Optional
import psutil

__all__ = ['get_process_info']


def get_process_info(pid: Optional[int]=None, verbosity=0) -> dict:
    """Return a dict with PID's process info.

    Args:
        pid: The PID of process. If None, use current process's PID. (Default: None)
        verbosity: Info verbosity 0-3. (Default: 0)

    Returns:
        Dict with process info
    """
    process_info = {}

    p = psutil.Process(pid)
    all_info = p.as_dict()

    process_info['exe'] = all_info['exe']
    process_info['cmdline'] = all_info['cmdline']
    process_info['cwd'] = all_info['cwd']
    if verbosity > 0:
        process_info['name'] = all_info['name']
        process_info['pid'] = all_info['pid']
        process_info['ppid'] = all_info['ppid']
        process_info['username'] = all_info['username']
        process_info['environ'] = all_info['environ']
    if verbosity > 1:
        process_info['cpu_percent'] = all_info['cpu_percent']
        process_info['nice'] = all_info['nice']
        process_info['memory_info'] = all_info['memory_info']
        process_info['open_files'] = all_info['open_files']
    if verbosity > 2:
        process_info = all_info

    return process_info
