#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2021/6/18 17:36
@File    : utils.py
@Software: PyCharm
@Desc    : 
"""
from typing import Any
import os.path as osp

from ..type import is_int, is_float

__all__ = ['norm_param_name', 'name_param_val']

kLongestParamStr = 75


def norm_param_name(name: str) -> str:
    return name.replace('.', '·').replace(osp.sep, '-')  # value中的小数点用·替代，以免影响导入；/用-替代，以免影响路径。


def name_param_val(param_val: Any, longest_param_length: int=kLongestParamStr) -> str | int | float:
    if is_int(param_val) or is_float(param_val):  # int或float保持不变，适配pandas索引。
        return param_val

    param_val_str = str(param_val).splitlines()[0]
    if isinstance(param_val, dict) and '_param_val_name' in param_val:
        ret = str(param_val['_param_val_name'])
    elif callable(param_val) and hasattr(param_val, '__name__') and len(param_val.__name__) <= longest_param_length:
        ret = param_val.__name__
    elif len(param_val_str) <= longest_param_length:
        ret = param_val_str
    else:
        ret = str(hash(param_val))[:longest_param_length]

    ret = norm_param_name(ret)

    return ret
