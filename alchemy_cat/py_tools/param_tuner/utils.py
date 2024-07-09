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
from typing import Any, Union, Final
import os.path as osp

__all__ = ['norm_param_name', 'name_param_val']

kLongestParamStr: Final[int] = 75


def norm_param_name(name: str) -> str:
    return name.replace('.', '·').replace(osp.sep, '-')  # value中的小数点用·替代，以免影响导入；/用-替代，以免影响路径。


def _is_int(elem: Any) -> bool:
    """拷贝自 alchemy_cat/py_tools/type.py，支持numpy、torch未安装情况下运行。"""
    try:
        import torch
        TORCH_AVAILABLE = True
    except ImportError:
        torch = None
        TORCH_AVAILABLE = False

    try:
        import numpy as np
        NUMPY_AVAILABLE = True
    except ImportError:
        np = None
        NUMPY_AVAILABLE = False

    if TORCH_AVAILABLE and isinstance(elem, torch.Tensor):
        if elem.ndim != 0:
            return False
        elif (not torch.is_floating_point(elem)) and (not torch.is_complex(elem)) and (elem.dtype is not torch.bool):
            return True
        else:
            return False

    if NUMPY_AVAILABLE and isinstance(elem, np.integer):
        return True

    if isinstance(elem, int) and (not isinstance(elem, bool)):
        return True

    return False


def name_param_val(param_val: Any, longest_param_length: int=kLongestParamStr) -> Union[str, int, float]:
    # if is_int(param_val) or is_float(param_val):  # int或float保持不变，适配pandas索引。
    if _is_int(param_val):  # NOTE 只有int保持不变，适配pandas索引。float中的小数点会影响导入，还是需要替换。
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
