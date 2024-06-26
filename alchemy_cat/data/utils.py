#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: utils.py
@time: 2020/3/20 2:07
@desc:
"""
from typing import Optional
import os
import lmdb
import pickle

__all__ = ['read_rand_seeds', 'write_rand_seeds']


def read_rand_seeds(rand_seed_log: str, idx: int, create: bool=False) -> Optional[dict]:
    """Read rand seeds from idx of rand_seed_log's

    Args:
        rand_seed_log: rand seed log file
        idx: Index of sample
        create: If True, lmdb database with path rand_seed_log will be created. (Default: False)

    Returns:
        If rand seeds exit, return rand seeds. Else return None.
    """
    if not create and not os.path.isdir(rand_seed_log):
        return None

    env = lmdb.open(rand_seed_log, meminit=False, readonly=True)

    with env.begin() as txn:
        rand_seeds = txn.get(str(idx).encode(), default=None)

    if rand_seeds is not None:
        rand_seeds = pickle.loads(rand_seeds)

    env.close()
    return rand_seeds


def write_rand_seeds(rand_seed_log: str, idx: int, rand_seeds: dict):
    """Write rand seeds to idx of rand_seed_log's

    Args:
        rand_seed_log: rand seed log file
        idx: Index of sample
        rand_seeds: rand seeds to be writen
    """
    env = lmdb.open(rand_seed_log, meminit=False)

    with env.begin(write=True) as txn:
        txn.put(str(idx).encode(), pickle.dumps(rand_seeds))

    env.close()
