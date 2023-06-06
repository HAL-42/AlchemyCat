#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/6/6 13:03
@File    : link_dirs.py
@Software: PyCharm
@Desc    : 
"""
import argparse
import os
import os.path as osp
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--src', help="source dir. ", type=str)
parser.add_argument('-t', '--target', help="target dir. ", type=str)
args = parser.parse_args()

if not osp.isdir(args.target):
    raise NotADirectoryError(f"{args.target} is not a directory. ")

os.makedirs(args.src, exist_ok=True)

print(f"Linking: \n"
      f"    src: {args.src} \n"
      f"    target: {args.target}")
for d in os.listdir(args.target):
    print(f"Linking {d}...")
    subprocess.run(['ln', '-s', osp.join(args.target, d), osp.join(args.src, d)])
