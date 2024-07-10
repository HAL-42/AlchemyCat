#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2024/7/10 16:40
@File    : from_x_to_y.py
@Software: PyCharm
@Desc    : 
"""
import argparse

from alchemy_cat.py_tools import Config, gprint

parser = argparse.ArgumentParser()
parser.add_argument('--x', type=str, help='Source config path')
parser.add_argument('--y', type=str, help='Target config path')
parser.add_argument('--y_type', type=str, help='Target config type', default='alchemy-cat',
                    choices=['yaml', 'mmcv', 'alchemy-cat'])
args = parser.parse_args()

Config.from_x_to_y(args.x, args.y, y_type=args.y_type)
gprint(f"[SUCCESS] {args.x} ---------> {args.y} ({args.y_type})")
