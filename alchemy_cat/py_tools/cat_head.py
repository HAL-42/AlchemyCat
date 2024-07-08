#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2022/5/28 20:37
@File    : cat_head.py
@Software: PyCharm
@Desc    : 
"""
from typing import Optional, Union, TYPE_CHECKING

from dataclasses import dataclass

from .config import Config

__all__ = ['喵', 'meow']


@dataclass
class CatHead(object):
    """该类型实例「喵」可作为全局变量使用。"""
    cfg: Optional['Config'] = None
    is_debug: bool = False
    rand_seed_final: Union[str, int, None] = None


喵 = meow = CatHead()
