#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: timer.py
@time: 2020/1/11 1:40
@desc:
"""
import time
from timeit import default_timer

from typing import Optional

__all__ = ['Timer']


class Timer(object):
    __unitfactor = {'s': 1,
                    'ms': 1000,
                    'us': 1000000}

    def __init__(self, unit: Optional[str]=None, precision: int=4, name: str="Default Timer"):
        """Timer to record the time interval in program

        Args:
            unit: Timer's unit. Can be s, ms, us. If not given, then the program will adaptively choose a unit.
            precision: Timer's output precision. The total time's decimal part will be rounded according to
                this precision.
            name: Timer's name.
        """
        if unit is not None and unit not in Timer.__unitfactor:
            raise ValueError('Unsupported time unit.')
        self._start = 0
        self._end = 0
        self._total = None
        self._unit = unit
        self._precision = precision
        self.name = name
    
    @property
    def total(self):
        assert self._total is not None
        return self._total

    def start(self):
        self._start = default_timer()
        return self

    def __enter__(self):
        return self.start()

    def close(self):
        self._end = default_timer()
        self._total = (self._end - self._start)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        if self._total is None:
            print(f"Timer {self.name} survived for {self}")

    def __repr__(self):
        if self._total is None:
            total = default_timer() - self._start
        else:
            total = self._total

        if self._unit is None:
            if total > 300.:
                total_s = int(total)
                ret = f"{total_s // 3600}h:{(total_s % 3600) // 60:02}m:{total_s % 60:02}s" + \
                      f"{round(total - total_s, self._precision)}"[1:]
            else:
                global factor, unit
                for unit, factor in Timer.__unitfactor.items():
                    if total * factor > 1.:
                        break
                ret = f"{round(total * factor, self._precision)}{unit}"
        else:
            total *= Timer.__unitfactor[self._unit]
            total = round(total, self._precision)
            ret = f"{total}{self._unit}"

        return f"{self.name}: " + ret


if __name__ == "__main__":
    timer = Timer().start()
    timer.close()
    print(timer)

    with Timer(precision=5) as timer:
        tmp = 1.
        for i in range(10000000):
            if i % 1000000 == 0:
                print(timer)
            tmp *= float(i)
    print(timer)

    timer = Timer(unit='us').start()
    tmp = 1.
    for i in range(1000000):
        tmp *= float(i)
    timer.close()
    print(timer.name, timer)

    timer = Timer(name="deleted Timer")
    time.sleep(2)
    del timer

    def foo():
        timer = Timer(name="unclosed Timer")
        time.sleep(2)

    foo()
