#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2022/5/10 13:51
@File    : parse_log.py
@Software: PyCharm
@Desc    : 
"""
from typing import Optional, Callable
from dataclasses import dataclass

import re

import numpy as np

__all__ = ['Field', 'find_fields_in_log']


@dataclass
class Field(object):
    """字段如何被查找和解析。"""
    pattern: Optional[str] = None
    parser: Callable = float
    val: Optional[list] = None


def find_fields_in_log(log_file: str, fields: dict, np_val: bool=False):
    """在日志中找到指定字段的值。

    Args:
        log_file: 日志文件。
        fields: 字段。
        np_val: 匹配到的字段是否要转为nparray，再赋值给字段的val。
    """
    # * 读取要解析的训练日志。
    with open(log_file, 'r', encoding='utf-8') as f:
        log = f.read()
    # * 循环匹配，获取字段。
    for field in fields.values():
        field.val = [field.parser(v) for v in re.findall(field.pattern, log)]
        if np_val:
            field.val = np.array(field.val)
