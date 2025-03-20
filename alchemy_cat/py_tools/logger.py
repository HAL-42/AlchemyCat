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

        self.stream_name = stream_name
        self.real_time = real_time
        self.silence = silence

        self._is_reset = False

    def write(self, message):
        if not self.silence:
            self.stream.write(message)

        self.log.write(message)

        if self.real_time:
            self.log.flush()

    def flush(self):
        self.stream.flush()
        self.log.flush()

    def reset(self):
        if self._is_reset:
            return

        if self.stream_name == 'stdout':
            sys.stdout = self.stream
        elif self.stream_name == 'stderr':
            sys.stderr = self.stream
        else:
            raise ValueError(f"stream_name = {self.stream_name} should be 'stdout' or 'stderr'.")
        self.stream = None
        self.log = None

        self._is_reset = True

    def __getattr__(self, name):
        """Forward attribute access to the underlying stream for undefined attributes."""
        if self._is_reset:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}' (object has been reset)")
        if self.stream is None:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}' (stream is None)")

        # Let getattr raise AttributeError directly if attribute doesn't exist on stream
        return getattr(self.stream, name)


class Logger(object):
    """After call Logger(outfile), the stdout will print to both stdout and out_file"""

    def __init__(self, out_file, real_time: bool=False, silence: bool=False):
        """After call Logger(outfile), the stdout will print to both stdout and out_file

        Args:
            out_file: File where record the stdout
            real_time: If True, log file will flush after every write() call. (Default: False)
            silence: If True, output to terminal will be suppressed. (Default: False)
        """
        if isinstance((old_logger := getattr(sys, "alchemy_cat_logger", None)), Logger):
            old_logger.reset()
        sys.alchemy_cat_logger = self  # Make Logger won't be deleted.

        os.makedirs(osp.dirname(out_file), exist_ok=True)
        self.log = open(out_file, "w", encoding="utf-8")

        self.real_time = real_time
        self.silence = silence

        self.stdout_logger = _StreamLogger('stdout', self.log, self.real_time, self.silence)
        self.stderr_logger = _StreamLogger('stderr', self.log, self.real_time, self.silence)

        self._is_reset = False

    def reset(self):
        if self._is_reset:
            return

        self.stdout_logger.flush()
        self.stderr_logger.flush()

        self.stdout_logger.reset()
        self.stderr_logger.reset()

        self.log.close()

        if getattr(sys, "alchemy_cat_logger", None) is self:
            del sys.alchemy_cat_logger

        self._is_reset = True

    def __del__(self):
        self.reset()
