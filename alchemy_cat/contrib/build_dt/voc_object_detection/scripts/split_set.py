#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2021/11/8 21:26
@File    : split_set.py
@Software: PyCharm
@Desc    : 
"""
import argparse
import os
import os.path as osp
import glob
from math import ceil
import random

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--root', type=str)
parser.add_argument('-t', '--train_ratio', default=0.75, type=float)
parser.add_argument('-v', '--val_ratio', default=0.25, type=float)
args = parser.parse_args()

# * Rand Seed
kSeed = 0

# * Dir & files
kAnnoDir = osp.join(args.root, 'Annotations')

kTrainFile = osp.join(args.root, 'ImageSets', 'Main', 'train.txt')
kValFile = osp.join(args.root, 'ImageSets', 'Main', 'val.txt')
kTestFile = osp.join(args.root, 'ImageSets', 'Main', 'test.txt')
kTrainValFile = osp.join(args.root, 'ImageSets', 'Main', 'trainval.txt')

os.makedirs(osp.join(args.root, 'ImageSets', 'Main'), exist_ok=True)

# * Set Rand Seed
random.seed(kSeed)

# * Get names
names = [osp.splitext(osp.basename(file))[0] for file in glob.glob(osp.join(kAnnoDir, '*.xml'))]

# * Compute split size
train_num = ceil(len(names) * args.train_ratio)
assert train_num <= len(names)

val_num = ceil(len(names) * args.val_ratio)
if train_num + val_num > len(names):
    val_num = len(names) - train_num

test_num = len(names) - train_num - val_num

# * Shuffle names & get splits
random.shuffle(names)
train_names = names[:train_num]
val_names = names[train_num:train_num + val_num]
test_names = names[train_num + val_num:train_num + val_num + test_num]
trainval_names = train_names + val_names

# * Write names
with open(kTrainFile, 'w') as f:
    f.write('\n'.join(train_names) + '\n')

with open(kValFile, 'w') as f:
    f.write('\n'.join(val_names) + '\n')

with open(kTestFile, 'w') as f:
    f.write('\n'.join(test_names) + '\n')

with open(kTrainValFile, 'w') as f:
    f.write('\n'.join(trainval_names) + '\n')
