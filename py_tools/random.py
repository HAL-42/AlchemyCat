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
import numpy as np
import torch
import random

from typing import Union

__all__ = ['set_numpy_rand_seed', 'set_py_rand_seed', 'set_torch_rand_seed', 'set_rand_seed',
           'set_rand_seed_according_torch']


def set_numpy_rand_seed(seed: Union[int, str]):
    """Set rand seed for numpy

    Args:
        seed (Union[int, str]): int seed or str, which will be hashed to get int seed
    """
    if isinstance(seed, str):
        seed = hash(seed)
    elif not isinstance(int(seed), int):
        raise ValueError(f"seed={seed} should be str or int")

    seed = seed % (2**32)
    np.random.seed(int(seed))


def set_torch_rand_seed(seed: Union[int, str]):
    """Set rand seed for torch on both cpu, cuda and cudnn

    Args:
        seed (Union[int, str]): int seed or str, which will be hashed to get int seed
    """
    if isinstance(seed, str):
        seed = hash(seed)
    elif not isinstance(int(seed), int):
        raise ValueError(f"seed={seed} should be str or int")
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True


def set_py_rand_seed(seed: Union[int, str]):
    """Set rand seed for python

    Args:
        seed (Union[int, str]): int seed or str, which will be hashed to get int seed
    """
    if isinstance(seed, str):
        seed = hash(seed)
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
    seed = torch.initial_seed()
    set_py_rand_seed(seed)
    set_numpy_rand_seed(seed)