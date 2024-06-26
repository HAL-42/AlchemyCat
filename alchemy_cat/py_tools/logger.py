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

__all__ = ['Logger']


class _StreamLogger(object):

    def __init__(self, stream_name: str, log, real_time: bool=False, silence: bool=False):
        if stream_name == 'stdout':
            self.stream = sys.stdout
            sys.stdout = self
        elif stream_name == 'stderr':
            self.stream = sys.stderr
            sys.stderr = self
        else:
            raise ValueError(f"stream_name = {stream_name} should be 'stdout' or 'stderr'.")

        self.log = log

        self.real_time = real_time
        self.silence = silence

    def write(self, message):
        if not self.silence:
            self.stream.write(message)

        self.log.write(message)

        if self.real_time:
            self.log.flush()

    def flush(self):
        self.stream.flush()
        self.log.flush()


class Logger(object):
    """After call Logger(outfile), the stdout will print to both stdout and out_file"""

    def __init__(self, out_file, real_time: bool=False, silence: bool=False):
        """After call Logger(outfile), the stdout will print to both stdout and out_file

        Args:
            out_file: File where record the stdout
            real_time: If True, log file will flush after every write() call. (Default: False)
            silence: If True, output to terminal will be suppressed. (Default: False)
        """
        sys.alchemy_cat_logger = self  # Make Logger won't be deleted.

        os.makedirs(osp.dirname(out_file), exist_ok=True)
        self.log = open(out_file, "w", encoding="utf-8")

        self.real_time = real_time
        self.silence = silence

        self.stdout_logger = _StreamLogger('stdout', self.log, self.real_time, self.silence)
        self.stderr_logger = _StreamLogger('stderr', self.log, self.real_time, self.silence)

    def __del__(self):
        self.stdout_logger.flush()
        self.stderr_logger.flush()
        self.log.close()
