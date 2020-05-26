#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: logger.py
@time: 2020/3/23 21:49
@desc:
"""
import sys
import os
import os.path as osp

class Logger(object):
    """After call Logger(outfile), the stdout will print to both stdout and out_file"""

    def __init__(self, out_file, real_time: bool=False):
        """After call Logger(outfile), the stdout will print to both stdout and out_file

        Args:
            out_file: File where record the stdout
            real_time: If True, log file will flush after every write() call. (Default: False)
        """
        self.terminal = sys.stdout
        os.makedirs(osp.dirname(out_file), exist_ok=True)
        self.log = open(out_file, "w")
        sys.stdout = self
        self.real_time = real_time

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        if self.real_time:
            self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def __del__(self):
        self.flush()
        self.log.close()
