#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/2/4 22:12
@File    : scheduler_chainer.py
@Software: PyCharm
@Desc    : 
"""
import bisect

from torch.optim.lr_scheduler import _LRScheduler

__all__ = ['SchedulerChainer']


class SchedulerChainer(object):

    def __init__(self, schedulers: list[_LRScheduler], milestones: list[int | float]):
        """拼接多个调率器。

        Args:
            schedulers: 调率器列表。
            milestones: 调率器切换里程碑。
        """
        assert len(schedulers) > 0
        assert len(milestones) == len(schedulers) - 1

        self.schedulers = schedulers
        self.milestones = milestones + [float('inf')]

        self._iteration = -1

    @property
    def iteration(self) -> int:
        """返回当前迭代号（从0开始）。"""
        return self._iteration

    def step(self, iteration: int | None=None):
        if iteration is not None:
            self._iteration = iteration
        else:
            self._iteration += 1
        todo_iter = self.iteration + 1  # 调率器的作用迭代。

        if len(self.schedulers) == 1:  # 只有一个scheduler，迭代到底。
            self.schedulers[0].step()
        else:
            right_milestone_idx = bisect.bisect_left(self.milestones, todo_iter)
            self.schedulers[right_milestone_idx].step()
            if todo_iter == self.milestones[right_milestone_idx]:
                self.schedulers[right_milestone_idx + 1].step()
