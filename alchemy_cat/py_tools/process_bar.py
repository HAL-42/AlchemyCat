#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: my_process_bar.py
@time: 2019/7/30 2:29
@desc:
"""
import time
import numpy as np


class ProcessBar(object):

    def __init__(self, max_value, start_value=0, percent_step=0.01, lines_num=50):
        self.start_time = time.time()
        self.max_value = max_value
        self.start_value = start_value
        self.percent_step = percent_step
        self.lines_num = lines_num

        self.finished = False
        self.closed = False

        self.current_value = self.start_value
        self.current_percent = self.current_value / self.max_value
        self.current_time = self.start_time

        self.show_str = ''
        self._PrintBar()

    def _PrintBar(self):
        show_lines_num = int(np.round(self.lines_num * self.current_percent))
        lines = '|' * show_lines_num + ' ' * (self.lines_num - show_lines_num)
        self.show_str = "{percent:2d}%[{lines}]{delta_time:.2f}s".\
            format(**{'percent': int(np.round(self.current_percent * 100)), 'lines': lines,
                        'delta_time': time.time() - self.start_time})
        print(self.show_str, end='\r')

    def UpdateBar(self, current_value):
        if self.finished:
            return False
        if (current_value - self.current_value) / self.max_value < self.percent_step\
                and current_value != self.max_value:
            return False
        self.current_value = current_value
        self.current_percent = self.current_value / self.max_value
        self.current_time = time.time()
        self._PrintBar()
        if current_value == self.max_value:
            self.Close()
            self.finished = True
        return True

    def SkipMsg(self, msg: str, logger=None):
        print(' ' * (len(self.show_str) + 1), end='\r')
        if logger:
            logger.info(msg)
        else:
            print(msg)
        self._PrintBar()

    def Close(self, msg=None, logger=None):
        if not self.closed:
            print('')
            if isinstance(msg, str):
                if logger:
                    logger.info(msg)
                else:
                    print(msg)


if __name__ == "__main__":
    N = 1000
    process_bar = ProcessBar(N - 1)
    for i in range(N):
        process_bar.UpdateBar(i)
        time.sleep(0.001)
    print('Test')

