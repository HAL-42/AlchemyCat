#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2021/5/7 10:12
@File    : test_cfg2tune.py
@Software: PyCharm
@Desc    : 
"""
import pickle

import pytest
import os.path as osp
import shutil

from alchemy_cat.py_tools import Cfg2Tune


@pytest.fixture(scope="function")
def cfg_dir():
    cfg_dir = osp.join('..', '..', '..', 'Temp', 'param_tuner_test_cache')
    yield cfg_dir
    shutil.rmtree(cfg_dir)


@pytest.fixture(scope='function')
def easy_config():
    return Cfg2Tune.load_cfg2tune(osp.join('configs', 'easy_config', 'cfg.py'))


@pytest.fixture(scope='function')
def mutual_depend_config():
    return Cfg2Tune.load_cfg2tune(osp.join('configs', 'mutual_depend_config', 'cfg.py'))


@pytest.fixture(scope='function')
def no_legal_val_config():
    return Cfg2Tune.load_cfg2tune(osp.join('configs', 'no_legal_val_config', 'cfg.py'))


@pytest.fixture(scope='function')
def no_param_config():
    return Cfg2Tune.load_cfg2tune(osp.join('configs', 'no_param_config', 'cfg.py'))


@pytest.fixture(scope='function')
def run_time_mutual_depend_config():
    return Cfg2Tune.load_cfg2tune(osp.join('configs', 'run_time_mutual_depend_config', 'cfg.py'))


@pytest.fixture(scope='function')
def subject_to_param_config():
    return Cfg2Tune.load_cfg2tune(osp.join('configs', 'subject_to_param_config', 'cfg.py'))


@pytest.fixture(scope='function')
def subject_to_static_config():
    return Cfg2Tune.load_cfg2tune(osp.join('configs', 'subject_to_static_config', 'cfg.py'))


@pytest.fixture(scope='function')
def wrong_sequence_config():
    return Cfg2Tune.load_cfg2tune(osp.join('configs', 'wrong_sequence_config', 'cfg.py'))


def test_update(easy_config):
    with pytest.warns(UserWarning, match="can't recursively track Param2Tune in the dict updated."):
        easy_config.update(dict())


def test_root_child_params2tune(easy_config):
    def dfs_check_child(cfg: Cfg2Tune):
        for v in cfg.values():
            if isinstance(v, Cfg2Tune):
                assert len(object.__getattribute__(v, "_params2tune")) == 0
                dfs_check_child(v)

    assert len(object.__getattribute__(easy_config, "_params2tune")) == 4
    dfs_check_child(easy_config)


def test_dump_reload(easy_config, cfg_dir):
    cfgs = [cfg.to_dict() for cfg in easy_config.get_cfgs()]
    cfg_pkls = easy_config.dump_cfgs(cfg_dir)

    load_cfgs = []
    for cfg_pkl in cfg_pkls:
        with open(cfg_pkl, "rb") as pkl_f:
            load_cfgs.append(pickle.load(pkl_f))

    assert cfgs == load_cfgs


def test_easy_config(easy_config):
    def get_att_val(cfg):
        return [cfg.foo0.foo1.a, cfg.foo0.foo1.b, cfg.foo0.fix, cfg.c, cfg.foo0.a_max]

    wanted_att_val = [
        [0, .1, '0', False, 2],
        [0, .1, '0', True, 2],
        [0, .2, '0', False, 2],
        [0, .2, '0', True, 2],
        [1, .1, '0', False, 2],
        [1, .1, '0', True, 2],
        [1, .2, '0', False, 2],
        [1, .2, '0', True, 2],
        [2, .1, '0', False, 2],
        [2, .1, '0', True, 2],
        [2, .2, '0', False, 2],
        [2, .2, '0', True, 2],
    ]

    att_val = [get_att_val(cfg) for cfg in easy_config.get_cfgs()]
    assert wanted_att_val == att_val


def test_easy_config_twice(easy_config):
    def get_att_val(cfg):
        return [cfg.foo0.foo1.a, cfg.foo0.foo1.b, cfg.foo0.fix, cfg.c, cfg.foo0.a_max]

    wanted_att_val = [
        [0, .1, '0', False, 2],
        [0, .1, '0', True, 2],
        [0, .2, '0', False, 2],
        [0, .2, '0', True, 2],
        [1, .1, '0', False, 2],
        [1, .1, '0', True, 2],
        [1, .2, '0', False, 2],
        [1, .2, '0', True, 2],
        [2, .1, '0', False, 2],
        [2, .1, '0', True, 2],
        [2, .2, '0', False, 2],
        [2, .2, '0', True, 2],
    ]

    att_val = [get_att_val(cfg) for cfg in easy_config.get_cfgs()]
    assert wanted_att_val == att_val

    att_val = [get_att_val(cfg) for cfg in easy_config.get_cfgs()]
    assert wanted_att_val == att_val


def test_wrong_sequence_config(wrong_sequence_config):
    with pytest.raises(RuntimeError, match="Check if the former param call later param"):
        next(wrong_sequence_config.get_cfgs())


def test_mutual_depend_config(mutual_depend_config):
    with pytest.raises(RuntimeError, match="Check if the former param call later param"):
        next(mutual_depend_config.get_cfgs())


def test_run_time_mutual_depend_config(run_time_mutual_depend_config):
    cfgs = run_time_mutual_depend_config.get_cfgs()

    for _ in range(2):
        next(cfgs)

    with pytest.raises(RuntimeError, match="Check if the former param call later param"):
        next(cfgs)


def test_no_legal_val_config(no_legal_val_config):
    def get_att_val(cfg):
        return [cfg.foo0.foo1.a, cfg.foo0.foo1.b, cfg.c]

    wanted_att_val = [
        [0, .1, False],
        [0, .1, True],
        [1, .1, False],
        [1, .1, True],
    ]

    att_val = [get_att_val(cfg) for cfg in no_legal_val_config.get_cfgs()]
    assert wanted_att_val == att_val


def test_same_name_config():
    with pytest.raises(RuntimeError, match="for param2tune repeated."):
        Cfg2Tune.load_cfg2tune(osp.join('configs', 'same_name_config', 'cfg.py'))


def test_no_param_config(no_param_config):
    with pytest.raises(RuntimeError, match="must have more than 1 Param2Tune"):
        next(no_param_config.get_cfgs())


def test_subject_to_static_config(subject_to_static_config):
    def get_att_val(cfg):
        return [cfg.foo0.foo1.a, cfg.foo0.foo1.b, cfg.c]

    wanted_att_val = [
        [0, .1, False],
        [0, .1, True],
        [0, .2, False],
        [0, .2, True],
        [1, .1, False],
        [1, .1, True],
        [1, .2, False],
        [1, .2, True],
    ]

    att_val = [get_att_val(cfg) for cfg in subject_to_static_config.get_cfgs()]
    assert wanted_att_val == att_val


def test_subject_to_param_config(subject_to_param_config):
    def get_att_val(cfg):
        return [cfg.foo0.foo1.a, cfg.foo0.foo1.b, cfg.c]

    wanted_att_val = [
        [0, .1, True],
        [0, .2, True],
        [1, .2, True],
        [2, .1, False],
    ]

    att_val = [get_att_val(cfg) for cfg in subject_to_param_config.get_cfgs()]
    assert wanted_att_val == att_val
