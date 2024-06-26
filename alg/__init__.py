#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: __init__.py
@time: 2019/12/7 23:58
@desc:
"""
from .cnn_align import *
from .utils import *
from .msc_flip_inference import msc_flip_inference
from .clamp_softmax import *
# NOTE data初始化调用alg，alg的dense_crf调用data.plugin, data.plugin调用data.datasets，导致循环import。
# from alchemy_cat.alg.dense_crf import *
from .normalize_tensor import *
from .complement_entropy_confidence import *
from .slide_inference import *
from .window_slides import *
from .masked_softmax import *
from .resize import *
from .sampling import *
# NOTE 同densecrf情况，eval_cam的导入会触发data模块的循环导入。
# from .eval_cams import *
