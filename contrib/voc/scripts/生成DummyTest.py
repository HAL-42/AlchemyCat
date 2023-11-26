#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/6/26 20:19
@File    : 生成DummyTest.py
@Software: PyCharm
@Desc    : 
"""
import os
import os.path as osp

from alchemy_cat.contrib.voc import VOCAug2
from alchemy_cat.data.plugins import arr2PIL
from tqdm import tqdm

os.makedirs(dummy_dir := 'temp/DummyTest', exist_ok=True)

dt = VOCAug2('datasets', split='test')

for inp in tqdm(dt, desc="生成DummyTest", unit="张", dynamic_ncols=True):
    img_id, dummy_lb = inp.img_id, inp.lb
    arr2PIL(dummy_lb).save(osp.join(dummy_dir, f'{img_id}.png'))
