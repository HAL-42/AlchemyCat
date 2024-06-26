#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: average_value_tracker.py
@time: 2020/2/28 12:43
@desc: Modified from torchnet.meter.averagevaluemeter and torchnet.meter.movingaveragevaluemeter
"""
import math
import numpy as np
import torch

from alchemy_cat.py_tools import Tracker, Statistic

__all__ = ['AverageValueTracker', 'MovingAverageValueTracker']


class AverageValueTracker(Tracker):
    def __init__(self):
        super(AverageValueTracker, self).__init__()
        self._n = None
        self._sum = None
        self._var = None
        self._val = None
        self._mean =  None
        self._mean_old = None
        self._m_s = None
        self._std = None
        self._last = None

        self.reset()
        self._val = 0

    def _add(self, value, n=1):
        self._last = value

        self._val = value
        self._sum += value
        self._var += value * value
        self._n += n

        if self._n == 0:
            self._mean, self._std = np.nan, np.nan
        elif self._n == 1:
            self._mean = 0.0 + self._sum  # This is to force a copy in torch/numpy
            self._std = np.inf
            self._mean_old = self._mean
            self._m_s = 0.0
        else:
            self._mean = self._mean_old + (value - n * self._mean_old) / float(self._n)
            self._m_s += (value - self._mean_old) * (value - self._mean)
            self._mean_old = self._mean
            self._std = np.sqrt(self._m_s / (self._n - 1.0))

    def value(self):
        return self._mean, self._std

    def reset(self):
        self._n = 0
        self._sum = 0.0
        self._var = 0.0
        self._val = 0.0
        self._mean = np.nan
        self._mean_old = 0.0
        self._m_s = 0.0
        self._std = np.nan
        self._last = None

    @property
    def last(self):
        return self._last

    def update(self, value, n=1):
        super(AverageValueTracker, self).update(value, n)
        self._add(value, n)

    @Statistic
    def mean(self):
        return self.value()[0]

    @Statistic
    def std(self):
        return self.value()[1]


class MovingAverageValueTracker(Tracker):
    def __init__(self, window_size):
        super(MovingAverageValueTracker, self).__init__()
        self.window_size = window_size

        self._var = None
        self._n = None
        self._sum = None
        self._last = None
        self._value_queue = torch.zeros(window_size)

        self.reset()

    def reset(self):
        self._sum = 0.0
        self._n = 0
        self._var = 0.0
        self._last = None
        self._value_queue.fill_(0)

    def _add(self, value):
        self._last = value

        queue_id = (self._n % self.window_size)
        old_value = self._value_queue[queue_id]
        self._sum += value - old_value
        self._var += value * value - old_value * old_value
        self._value_queue[queue_id] = value
        self._n += 1

    @property
    def last(self):
        return self._last

    def value(self):
        n = min(self._n, self.window_size)
        mean = self._sum / max(1, n)
        std = math.sqrt(max((self._var - n * mean * mean) / max(1, n - 1), 0))
        return mean, std

    def update(self, value):
        super(MovingAverageValueTracker, self).update(value)
        self._add(value)

    @Statistic
    def mean(self):
        return self.value()[0]

    @Statistic
    def std(self):
        return self.value()[1]
