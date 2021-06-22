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
from typing import Any, Tuple, Iterable, Union

from alchemy_cat.py_tools import is_int, is_float

__all__ = ["param_val2str", "param_vals2pd_idx"]

kLongestParamStr = 40


def param_val2str(param_val: Any, longest_param_length: int = kLongestParamStr) -> str:
    param_val_str = str(param_val).splitlines()[0]
    if callable(param_val) and len(param_val.__name__) <= longest_param_length:
        ret = param_val.__name__
    elif len(param_val_str) <= longest_param_length:
        ret = param_val_str
    else:
        ret = str(hash(param_val))[:longest_param_length]
    return ret


def param_vals2pd_idx(vals: Iterable[Any]) -> Tuple[Union[str, Any]]:
    pd_idx = []
    for val in vals:
        if is_int(val) or is_float(val):
            pd_idx.append(val)
        else:
            pd_idx.append(param_val2str(val))

    return tuple(pd_idx)
