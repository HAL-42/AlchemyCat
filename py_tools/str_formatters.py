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


__all__ = ['indent']


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