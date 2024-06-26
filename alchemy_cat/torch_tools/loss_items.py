#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/9 22:12
@File    : loss_items.py
@Software: PyCharm
@Desc    : 
"""
import torch

__all__ = ['cal_loss_items']


def cal_loss_items(loss_items: dict, inp, out) -> dict:
    # * 损失字典，增加总损失一项。
    losses = {'total_loss': 0}

    # * 遍历计算每一个损失项，并汇入总损失。
    for item_name, loss_item in loss_items.items():
        # ** 跳过空loss项（一般是config中被取消了）。
        if not loss_item:
            continue
        # ** 计算损失。
        loss = loss_item.cal(loss_item.cri, inp, out)
        if torch.is_tensor(loss):  # 总是假设返回了多项损失。
            loss = (loss,)
        # ** 获取损失项的名字和权重。
        if loss_item.names:  # 如果指定了names，提取之。
            names, weights = loss_item.names, loss_item.weights
        else:  # 若没有指定names，说明loss_item本身就是loss名字。
            names, weights = (item_name,), (loss_item.weights,)

        for ls, name, weight in zip(loss, names, weights, strict=True):
            losses[name] = ls
            losses['total_loss'] += losses[name] * weight

    return losses
