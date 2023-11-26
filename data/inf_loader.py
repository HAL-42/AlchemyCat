#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/9 21:49
@File    : inf_loader.py
@Software: PyCharm
@Desc    : 
"""
from torch.utils.data import DataLoader

__all__ = ['inf_loader']


def inf_loader(loader: DataLoader):
    """接受一个DataLoader对象作为参数，每次迭代都会返回下一个batch数据。

    Args:
        loader: 数据加载器。

    Returns:
        dataloader的下一个batch。
    """
    epoch_iter = iter(loader)
    while True:
        try:
            bt = next(epoch_iter)
        except StopIteration:
            epoch_iter = iter(loader)
            bt = next(epoch_iter)
        yield bt
