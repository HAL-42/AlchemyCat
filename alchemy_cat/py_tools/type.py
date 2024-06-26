#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: type.py
@time: 2020/1/8 8:13
@desc:
"""
from typing import Any

import numpy as np
import torch


__all__ = ["is_intarr", "is_int", "is_floatarr", "is_float", "is_boolarr", "is_bool",
           "tolist", "totuple", "dict_with_arr_is_eq"]


def is_int(elem) -> bool:
    """
    Args:
        elem (UnKnown): element need to be judged. Note in this func, bool is not int.

    Returns:
        Is the elem is an int type(int, np.int*, torch.int*)
    """
    if isinstance(elem, torch.Tensor):
        if elem.ndim != 0:
            return False
        elif (not torch.is_floating_point(elem)) and (not torch.is_complex(elem)) and (elem.dtype is not torch.bool):
            return True
        else:
            return False

    if isinstance(elem, np.integer):
        return True

    if isinstance(elem, int) and (not isinstance(elem, bool)):
        return True

    return False


def is_intarr(arr) -> bool:
    """
    Args:
        arr (UnKnown): arr need to be judged. Note for torch, tensor(1) is both int and intarr.

    Returns:
        Is the arr is an int type(np.int*, torch.int*)
    """
    if (isinstance(arr, torch.Tensor) and (not torch.is_floating_point(arr)) and (not torch.is_complex(arr))
            and (arr.dtype is not torch.bool)
            and arr.ndim > 0):
        return True

    if isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.integer):
        return True

    return False


def is_float(elem) -> bool:
    """
    Args:
        elem (UnKnown): element need to be judged

    Returns:
        Is the elem is a float type(float, np.float*, torch.float*)
    """
    if isinstance(elem, torch.Tensor):
        if elem.ndim != 0:
            return False
        elif torch.is_floating_point(elem):
            return True
        else:
            return False

    if isinstance(elem, np.floating):
        return True

    if isinstance(elem, float):
        return True

    return False


def is_floatarr(arr) -> bool:
    """
    Args:
        arr (UnKnown): arr need to be judged.

    Returns:
        Is the arr is a float type(np.float*, torch.float*)
    """
    if isinstance(arr, torch.Tensor) and torch.is_floating_point(arr) and arr.ndim > 0:
        return True

    if isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.floating):
        return True

    return False


def is_bool(elem) -> bool:
    """
    Args:
        elem (UnKnown): element need to be judged

    Returns:
        Is the elem is a bool type(int, np.int*, torch.int*)
    """
    if isinstance(elem, torch.Tensor):
        if elem.ndim != 0:
            return False
        elif elem.dtype is torch.bool:
            return True
        else:
            return False

    if isinstance(elem, np.bool_):
        return True

    if isinstance(elem, bool):
        return True

    return False


def is_boolarr(arr) -> bool:
    """
    Args:
        arr (UnKnown): arr need to be judged.

    Returns:
        Is the arr is a bool type(np.float*, torch.float*)
    """
    if isinstance(arr, torch.Tensor) and (arr.dtype is torch.bool) and arr.ndim > 0:
        return True

    if isinstance(arr, np.ndarray) and (arr.dtype == np.bool_):
        return True

    return False


def tolist(arr: Any) -> list:
    """Return arr.tolist() for ndarray and tensor, other wise return list(arr)
        list(tensor) or list(ndarray) return a list with tensor or numpy scalar when arr with dim=1, which
        may cause confusing result for some function can't recognize tensor or scalar as python datatype.

        When the arr to be converted to list can be tensor/ndarray or other type both, this function can make sure
        the output list has python datatype element.
    Args:
        arr: arr need to be converted to list

    Returns:
        converted arr with type list
    """
    if isinstance(arr, torch.Tensor) or isinstance(arr, np.ndarray):
        return arr.tolist()
    else:
        return list(arr)


def totuple(arr: Any) -> tuple:
    """Return tuple(arr.tolist()) for ndarray and tensor, other wise return list(arr)
        tuple(tensor) or tuple(ndarray) return a tuple with tensor or numpy scalar when arr with dim=1, which
        may cause confusing result for some function can't recognize tensor or scalar as python datatype.

        When the arr to be converted to tuple can be tensor/ndarray or other type both, this function can make sure
        the output tuple has python datatype element.
    Args:
        arr: arr need to be converted to tuple

    Returns:
        converted arr with type tuple
    """
    if isinstance(arr, torch.Tensor) or isinstance(arr, np.ndarray):
        return tuple(arr.tolist())
    else:
        return tuple(arr)


def dict_with_arr_is_eq(dict1: dict, dict2: dict, request_order: bool=False, rtol: float=0.0, atol: float=0.0):
    """Compare to dict is equal when some dict's values is Tensor or ndarray

    Args:
        dict1: Fist dict to be compared.
        dict2: Second dict to be compared
        request_order: If True, orders of dict1 and dict2 will be compared. (Default: True)
        rtol: param rtol for np.isclose() when compare two ndarray values
        atol: param atol for np.isclose() when compare two ndarray values

    Returns:
        Is two dicts have keys and values
    """

    if len(dict1) != len(dict2):
        return False

    if request_order:
        if dict1.keys() != dict2.keys():
            return False

    for key1, val1 in dict1.items():
        if key1 not in dict2:
            return False

        val2 = dict2[key1]

        val1 = val1.detach().cpu().numpy() if isinstance(val1, torch.Tensor) else val1
        val2 = val2.detach().cpu().numpy() if isinstance(val2, torch.Tensor) else val2

        if type(val1) is not type(val2):
            return False

        if isinstance(val1, np.ndarray):
            if not np.all(np.isclose(val1, val2, rtol, atol)):
                return False
        else:
            if val1 != val2:
                return False

    return True
