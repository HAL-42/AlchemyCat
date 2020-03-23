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

class Logger(object):
    """After call Logger(outfile), the stdout will print to both stdout and out_file"""

    def __init__(self, out_file):
        """After call Logger(outfile), the stdout will print to both stdout and out_file

        Args:
            out_file: File where record the stdout
        """
        self.terminal = sys.stdout
        self.log = open(out_file, "w")
        sys.stdout = self

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
