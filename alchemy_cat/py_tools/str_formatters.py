#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: str_formatters.py
@time: 2020/2/6 11:43
@desc:
"""
from typing import Any
import time


__all__ = ['indent', 'get_local_time_str']


def indent(lines: Any, spaces: int=2) -> str:
    """Reformat input lines with spaces indent before each line

    Args:
        lines: lines to be added indents before. If not str, lines = str(lines)
        spaces: spaces in each indent. Default 2.

    Returns:
        Lines with indent
    """
    if not isinstance(lines, str):
       lines = str(lines)

    lines = lines.split('\n')
    indent = ' ' * spaces
    lines = [indent + line for line in lines]

    return '\n'.join(lines)


def get_local_time_str(for_file_name=False) -> str:
    """Return current time str in the format %Y-%m-%d %H:%M:%S"""
    if not for_file_name:
        cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    else:
        cur_time = time.strftime("%Y-%m-%d--%H-%M-%S", time.localtime())
    return cur_time
