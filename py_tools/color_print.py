# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: color_print.py
@time: 2021/9/13 1:27
@desc:
"""
from colorama import Fore, Style

__all__ = ['rprint', 'bprint', 'yprint', 'cprint', 'gprint']


def rprint(*args, **kwargs):
    print(Fore.RED, end='')
    print(*args, **kwargs)
    print(Style.RESET_ALL, end='')


def bprint(*args, **kwargs):
    print(Fore.BLUE, end='')
    print(*args, **kwargs)
    print(Style.RESET_ALL, end='')


def yprint(*args, **kwargs):
    print(Fore.YELLOW, end='')
    print(*args, **kwargs)
    print(Style.RESET_ALL, end='')


def cprint(*args, **kwargs):
    print(Fore.CYAN, end='')
    print(*args, **kwargs)
    print(Style.RESET_ALL, end='')


def gprint(*args, **kwargs):
    print(Fore.GREEN, end='')
    print(*args, **kwargs)
    print(Style.RESET_ALL, end='')
