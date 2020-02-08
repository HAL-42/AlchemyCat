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


__all__ = ["is_intarr", "is_int", "is_floatarr", "is_float", "tolist", "totuple"]


def is_int(elem) -> bool:
    """
    Args:
        elem (UnKnown): element need to be judged

    Returns:
        Is the elem is an int type(int, np.int*, torch.int*)
    """
    if isinstance(elem, int):
        return True

    if isinstance(elem, torch.Tensor):
        elem = elem.numpy()

    if not isinstance(elem, np.ndarray) and \
            (isinstance(elem, np.int0) or isinstance(elem, np.int8) or
             isinstance(elem, np.int16) or isinstance(elem, np.int32) or
             isinstance(elem, np.uint0) or isinstance(elem, np.uint8) or
             isinstance(elem, np.uint16) or isinstance(elem, np.uint32)):
            return True

    return False


def is_intarr(arr) -> bool:
    """
    Args:
        elem (UnKnown): arr need to be judged

    Returns:
        Is the arr is an int type(np.int*, torch.int*)
    """
    if isinstance(arr, torch.Tensor):
        arr = arr.numpy()

    if isinstance(arr, np.ndarray) and \
            (arr.dtype == np.int0 or arr.dtype == np.int8 or
             arr.dtype == np.int16 or arr.dtype == np.int32 or
             arr.dtype == np.uint0 or arr.dtype == np.uint8 or
             arr.dtype == np.uint16 or arr.dtype == np.uint32):
            return True

    return False


def is_float(elem) -> bool:
    """
    Args:
        elem (UnKnown): element need to be judged

    Returns:
        Is the elem is an int type(float, np.float*, torch.float*)
    """
    if isinstance(elem, float):
        return True

    if isinstance(elem, torch.Tensor):
        elem = elem.numpy()

    if not isinstance(elem, np.ndarray) and \
            (isinstance(elem, np.float16) or isinstance(elem, np.float32) or
             isinstance(elem, np.float64) or isinstance(elem, np.float)):
            return True

    return False


def is_floatarr(arr) -> bool:
    """
    Args:
        elem (UnKnown): arr need to be judged

    Returns:
        Is the arr is an int type(np.float*, torch.float*)
    """
    if isinstance(arr, torch.Tensor):
        arr = arr.numpy()

    if isinstance(arr, np.ndarray) and \
            (arr.dtype == np.float or arr.dtype == np.float32 or
             arr.dtype == np.float64 or arr.dtype == np.float16):
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