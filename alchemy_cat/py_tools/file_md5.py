#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/5/29 16:38
@File    : file_md5.py
@Software: PyCharm
@Desc    : 
"""
from typing import Union
import hashlib
from os import PathLike

__all__ = ['file_md5']


def file_md5(file: Union[str, PathLike]) -> str:
    """Get md5 of file"""

    with open(file, 'rb') as f:
        # * read contents of the file.
        data = f.read()
        # * pipe contents of the file through.
        md5_ret = hashlib.md5(data).hexdigest()

    return md5_ret
