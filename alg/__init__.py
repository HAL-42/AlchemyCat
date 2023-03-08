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
from alchemy_cat.alg.cnn_align import *
from alchemy_cat.alg.utils import *
from alchemy_cat.alg.msc_flip_inference import msc_flip_inference
from alchemy_cat.alg.clamp_softmax import *
from alchemy_cat.alg.dense_crf import *
from alchemy_cat.alg.normalize_tensor import *
from alchemy_cat.alg.complement_entropy_confidence import *
from .slide_inference import *
from .window_slides import *
from .masked_softmax import *
