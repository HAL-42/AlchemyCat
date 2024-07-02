#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2024/7/2 14:53
@File    : utils.py
@Software: PyCharm
@Desc    : 
"""
import typing as t

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from alchemy_cat.dl_config import ADict

__all__ = ['eval_model']


class RetDict(t.TypedDict):
    test_loss: float
    acc: float


def eval_model(model: nn.Module) -> RetDict:
    test_loader = DataLoader(MNIST('/tmp/data',
                                   train=False,
                                   transform=T.Compose([T.Grayscale(3),
                                                        T.ToTensor(),
                                                        T.Normalize((0.1307,), (0.3081,)),])),
                             batch_size=512, shuffle=False, num_workers=2, drop_last=False)
    N = len(t.cast(t.Sized, test_loader.dataset))

    model.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to('cuda'), target.to('cuda')
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= N
    acc = 100. * correct / N

    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{N} ({acc:.0f}%)\n')

    model.train()

    ret: RetDict = ADict()
    ret.test_loss = test_loss
    ret.acc = acc

    return ret
