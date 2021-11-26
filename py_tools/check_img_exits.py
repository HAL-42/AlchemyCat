#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2021/4/26 21:58
@File    : check_img_exits.py
@Software: PyCharm
@Desc    : 
"""
from imghdr import what
import os.path as osp

__all__ = ["check_img_exits"]


def check_img_exits(img_path: str) -> bool:
    """检查图像文件是否存在且完整。

    Args:
        img_path: 图像路径。

    Returns:
        返回图像是否存在。
    """
    if osp.isfile(img_path) and what(img_path) is not None:
        return True
    else:
        return False
