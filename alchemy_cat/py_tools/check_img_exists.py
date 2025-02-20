#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2021/4/26 21:58
@File    : check_img_exists.py
@Software: PyCharm
@Desc    : 
"""
import os
import os.path as osp
import typing as t

from PIL import Image

__all__ = ["check_img_exists"]


def check_img_exists(img_path: t.Union[str, os.PathLike]) -> bool:
    """检查图像文件是否存在且完整。

    Args:
        img_path: 图像路径。

    Returns:
        返回图像是否存在。
    """
    if not osp.exists(img_path):
        return False

    # 检查文件是否为有效的图像
    try:
        with Image.open(img_path) as img:
            img.verify()  # 验证图像文件是否完整
        return True
    except Exception:
        return False
