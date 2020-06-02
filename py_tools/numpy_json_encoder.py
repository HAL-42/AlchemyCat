#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: numpy_json_encoder.py
@time: 2020/6/2 1:13
@desc:
"""
import json
import numpy as np

__all__ = ['NumpyArrayEncoder']


class NumpyArrayEncoder(json.JSONEncoder):
    """Json encoder for numpy array, which will convert numpy array to list first"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)
