#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: random.py
@time: 2020/1/15 2:29
@desc:
"""
import hashlib
import random
from typing import Union

import numpy as np
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

__all__ = ['hash_str', 'set_numpy_rand_seed', 'set_py_rand_seed', 'set_torch_rand_seed', 'set_rand_seed',
           'set_rand_seed_according_torch']


def hash_str(s: str, dynamic: bool=False) -> int:
    """Hash a str to int

    Args:
        s (str): str to be hashed
        dynamic (bool): Whether to use dynamic hash. If True, hash will be different in different processes

    Returns:
        int: Hashed int
    """
    if not dynamic:
        return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16) % (2 ** 32)
    else:
        return hash(s) % (2 ** 32)


def set_numpy_rand_seed(seed: Union[int, str, np.ndarray]):
    """Set rand seed for numpy

    Args:
        seed (Union[int, str, np.ndarray]): int, str or np.ndarray seed, which will be hashed to get int seed
    """
    if isinstance(seed, str):
        seed = hash_str(seed, dynamic=False)
    elif isinstance(seed, int):
        seed = seed % (2 ** 32)
    elif not isinstance(seed, np.ndarray):
        raise ValueError(f"seed={seed} should be str, int or np.ndarray")

    np.random.seed(seed)


def set_torch_rand_seed(seed: Union[int, str]):
    """Set rand seed for torch on both cpu, cuda and cudnn

    Args:
        seed (Union[int, str]): int seed or str, which will be hashed to get int seed
    """
    if not TORCH_AVAILABLE:
        return
    if isinstance(seed, str):
        seed = hash_str(seed, dynamic=False)
    elif not isinstance(int(seed), int):
        raise ValueError(f"seed={seed} should be str or int")
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def set_py_rand_seed(seed: Union[int, str]):
    """Set rand seed for python

    Args:
        seed (Union[int, str]): int seed or str, which will be hashed to get int seed
    """
    if isinstance(seed, str):
        seed = hash_str(seed, dynamic=False)
    elif not isinstance(int(seed), int):
        raise ValueError(f"seed={seed} should be str or int")
    random.seed(int(seed))


def set_rand_seed(seed: Union[int, str]):
    """Set rand seed for numpy, torch(cpu, cuda, cudnn), python

    Args:
        seed (Union[int, str]): int seed or str, which will be hashed to get int seed
    """
    set_numpy_rand_seed(seed)
    set_py_rand_seed(seed)
    set_torch_rand_seed(seed)


def set_rand_seed_according_torch():
    """Set rand seed according to torch process's rand seed
    The rand seed of non-torch libs may duplicate in several dataloader worker processes.
    Use this function as dataloader's worker init function can solve this problem.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("torch is not available")
    seed = torch.initial_seed()
    set_py_rand_seed(seed)
    set_numpy_rand_seed(seed)
