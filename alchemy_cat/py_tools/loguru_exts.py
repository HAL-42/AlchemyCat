#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2024/2/21 17:06
@File    : loguru_exts.py
@Software: PyCharm
@Desc    : 
"""
import sys

from loguru import logger

__all__ = ['init_loguru']


def init_loguru(**kwargs) -> type(logger):
    """删除此前所有loguru handlers并初始化一个新的、以stdout为输出的loguru logger。

    Args:
        **kwargs: 传递给loguru.add的参数。

    Returns:
        Logger: loguru logger。
    """
    # -* 配置loguru logger。
    config = {"handlers": [{"sink": sys.stdout,
                            "format": '<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | '
                                      '<level>{level: <8}</level> | '
                                      '<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - '
                                      '<level>{message}</level>',
                            "colorize": True,
                            **kwargs}]}
    logger.configure(**config)

    # -* warning以下message不加粗。
    logger.level('TRACE', color='<cyan>')
    logger.level("DEBUG", color='<blue>')
    logger.level("INFO", color='')
    logger.level('SUCCESS', color='<green>')

    return logger
