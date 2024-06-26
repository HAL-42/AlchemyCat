#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2021/5/7 10:12
@File    : test_param2tune.py
@Software: PyCharm
@Desc    :
"""
import pytest

import sys
sys.path = ['', 'py_tools/param_tuner/tests'] + sys.path  # noqa: E402

from alchemy_cat.py_tools import Param2Tune


@pytest.fixture(scope="function")
def param():
    def param_subject_to(opt_val):
        return opt_val % 2 == 0
    return Param2Tune([0, 1, 2, 3], param_subject_to)


def test_reset(param):
    assert param._cur_val is None
    next(iter(param))
    assert param._cur_val == 0
    param.reset()
    assert param._cur_val is None


def test_none_cur_val(param):
    param.reset()
    with pytest.raises(RuntimeError, match="Check if the former param call later param"):
        _ = param.cur_val


def test_iter(param):
    iter1_param = iter(param)
    assert (0, 0) == next(iter1_param) == (param.cur_val, param.cur_val_name)
    assert (2, 2) == next(iter1_param) == (param.cur_val, param.cur_val_name)
    with pytest.raises(StopIteration):
        next(iter1_param)
    assert param._cur_val is None and param._cur_val_name is None

    iter2_param = iter(param)
    assert (0, 0) == next(iter2_param) == (param.cur_val, param.cur_val_name)
    assert (2, 2) == next(iter2_param) == (param.cur_val, param.cur_val_name)
    with pytest.raises(StopIteration):
        next(iter2_param)
    assert param._cur_val is None and param._cur_val_name is None


def test_subject_to(param):
    legal_val = list(param)
    assert legal_val == [(0, 0), (2, 2)]
