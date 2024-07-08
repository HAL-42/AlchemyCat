#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2021/6/22 22:28
@File    : test_param_lazy.py
@Software: PyCharm
@Desc    : 
"""
import pytest
import os.path as osp

import sys
sys.path = ['', 'py_tools/param_tuner/tests'] + sys.path  # noqa: E402

from alchemy_cat.py_tools import Cfg2Tune


@pytest.fixture(scope='function')
def param_lazy_config():
    return Cfg2Tune.load_cfg2tune(osp.join('configs', 'param_lazy_config', 'cfg.py'))


def test_param_lazy_config(param_lazy_config):
    def get_att_val(cfg):
        return [cfg.foo0.foo1.a, cfg.foo0.lazy0, cfg.lazy1, cfg.foo0.a_max]

    wanted_att_val = [
        [0, 10, -2, 2],
        [1, 11, -1, 2],
        [2, 12, 0, 2],
    ]

    att_val = [get_att_val(cfg) for cfg in param_lazy_config.get_cfgs()]
    assert wanted_att_val == att_val
