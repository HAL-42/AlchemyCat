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

from ..type import is_int, is_float

__all__ = ["param_val2str", "param_vals2pd_idx"]

kLongestParamStr = 75


def param_val2str(param_val: Any, longest_param_length: int = kLongestParamStr) -> str:
    param_val_str = str(param_val).splitlines()[0].replace('.', '·')  # value中的小数点用·替代，以免影响导入。
    if isinstance(param_val, dict) and '_param_val_name' in param_val:
        ret = str(param_val['_param_val_name'])
    elif callable(param_val) and hasattr(param_val, '__name__') and len(param_val.__name__) <= longest_param_length:
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
