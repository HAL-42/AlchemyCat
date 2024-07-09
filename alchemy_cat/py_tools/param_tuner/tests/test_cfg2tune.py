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
import os
import os.path as osp
import pickle
import shutil
import sys
from typing import Optional

import pytest
try:
    from addict import Dict
except ImportError:
    Dict = dict

sys.path = ['', 'py_tools/param_tuner/tests'] + sys.path  # noqa: E402

from alchemy_cat.py_tools import Cfg2Tune, open_config, ItemLazy, Config


@pytest.fixture(scope="function")
def cfg_dir():
    cfg_dir = osp.join('Temp', 'param_tuner_test_cache')
    yield cfg_dir
    shutil.rmtree(cfg_dir, ignore_errors=True)


@pytest.fixture(scope='function')
def easy_config():
    return Cfg2Tune.load_cfg2tune(osp.join('configs', 'easy_config', 'cfg.py'))


@pytest.fixture(scope='function')
def imported_outside_config():
    return Cfg2Tune.load_cfg2tune(osp.join('configs', 'imported_outside_config', 'cfg.py'))


@pytest.fixture(scope='function')
def def_inside_config():
    sys.path.insert(0, os.getcwd())
    return Cfg2Tune.load_cfg2tune(osp.join('configs', 'def_inside_config', 'cfg.py'))


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


@pytest.fixture(scope='function')
def itemlazy_config():
    return Cfg2Tune.load_cfg2tune(osp.join('configs', 'itemlazy_config', 'cfg.py'))


@pytest.fixture(scope='function')
def init_with_config_config():
    return Cfg2Tune.load_cfg2tune(osp.join('configs', 'init_with_config', 'cfg.py'))


@pytest.fixture(scope='function')
def init_with_cfg2tune_config():
    return Cfg2Tune.load_cfg2tune(osp.join('configs', 'init_with_cfg2tune', 'cfg.py'))


@pytest.fixture(scope='function')
def init_with_update_config():
    return Cfg2Tune.load_cfg2tune(osp.join('configs', 'init_with_update', 'cfg.py'))


def test_root_child_params2tune(easy_config):
    assert len(easy_config.ordered_params2tune) == 4


def test_dump_reload(easy_config, cfg_dir):
    cfgs = [cfg.to_dict() for cfg in easy_config.get_cfgs()]
    cfg_pkls, _ = easy_config.dump_cfgs(cfg_dir)

    load_cfgs = []
    for cfg_pkl in cfg_pkls:
        with open(cfg_pkl, "rb") as pkl_f:
            load_cfgs.append(pickle.load(pkl_f))

    assert cfgs == load_cfgs


def test_dump_reload_cfg2tune_with_function_imported_outside(imported_outside_config, cfg_dir):
    """测试生成的cfg_tuned中含有函数的情况。函数从外部库导入，或者在本项目中定义。"""
    cfgs = [cfg.to_dict() for cfg in imported_outside_config.get_cfgs()]
    cfg_pkls, _ = imported_outside_config.dump_cfgs(cfg_dir)

    load_cfgs = []
    for cfg_pkl in cfg_pkls:
        with open(cfg_pkl, "rb") as pkl_f:
            load_cfgs.append(pickle.load(pkl_f))

    assert cfgs == load_cfgs


def test_dump_reload_with_function_define_inside(def_inside_config, cfg_dir):
    """测试生成的cfg_tuned中含有函数的情况。函数在Cfg2Tune所在py文件内定义。"""
    cfgs = [cfg.to_dict() for cfg in def_inside_config.get_cfgs()]
    cfg_pkls, _ = def_inside_config.dump_cfgs(cfg_dir)

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
        Cfg2Tune.load_cfg2tune(osp.join('py_tools/param_tuner/tests', 'configs', 'same_name_config', 'cfg.py'))


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


def test_itemlazy_config(itemlazy_config, cfg_dir):
    def get_att_val(cfg):
        return [cfg.foo0.foo1.a, cfg.foo0.foo1.b, cfg.foo0.fix, cfg.c, cfg.foo0.a_max, cfg.foo1.lazy]

    wanted_att_val = [
        [0, 0, '0', False, {'a': 1, 'b': 2}, 0],
        [0, 0, '0', True, {'a': 1, 'b': 2}, 0],
        [0, 0, '0', False, {'a': 1, 'b': 2}, 0],
        [0, 1, '0', True, {'a': 1, 'b': 2}, 0],
        [1, 2, '0', False, {'a': 1, 'b': 2}, 2],
        [1, 2, '0', True, {'a': 1, 'b': 2}, 2],
        [1, 0, '0', False, {'a': 1, 'b': 2}, 2],
        [1, 1, '0', True, {'a': 1, 'b': 2}, 2],
        [2, 4, '0', False, {'a': 1, 'b': 2}, 4],
        [2, 4, '0', True, {'a': 1, 'b': 2}, 4],
        [2, 0, '0', False, {'a': 1, 'b': 2}, 4],
        [2, 1, '0', True, {'a': 1, 'b': 2}, 4],
    ]

    cfgs = [open_config(cfg_pkl)[0] for cfg_pkl in itemlazy_config.dump_cfgs(cfg_dir)[0]]

    att_val = [get_att_val(ItemLazy.compute_item_lazy(cfg)) for cfg in cfgs]
    assert wanted_att_val == att_val


def is_self_consistent(cfg: Config, p: Optional[Config]=None, n: Optional[str]=None):
    parent = object.__getattribute__(cfg, '__parent')
    key = object.__getattribute__(cfg, '__key')

    if p is None:
        assert parent is None
        assert key is None
    else:
        assert parent is p
        assert n == key

    for k, v in cfg.items():
        if cfg.is_subtree(v, cfg):
            is_self_consistent(v, cfg, k)


def test_init_with_config(init_with_config_config):
    """检查从Config初始化：能正确转枝赋叶，并找到root。检查配置树自洽性。"""
    leaves = [1, {'a': 2, 'b': 3}, 4, Dict({'c': 5, 'd': 6}), 7, Dict({'e': 8, 'f': 9}), 'init_with_config']

    assert len(list(init_with_config_config.branches)) == 4
    for b in init_with_config_config.branches:
        assert type(b) is Cfg2Tune

    assert len(list(init_with_config_config.leaves)) == len(leaves)
    for l1, l2 in zip(init_with_config_config.leaves, leaves):
        assert l1 == l2
        assert type(l1) is type(l2)

    is_self_consistent(init_with_config_config)


def test_init_with_cfg2tune(init_with_cfg2tune_config):
    """检查从Config初始化：能正确拷枝赋叶，并找到root。检查配置树自洽性。"""
    leaves = [1, Config({'a': 2, 'b': 3}), 4, Dict({'c': 5, 'd': 6}), 7, {'e': 8, 'f': 9}, 'init_with_cfg2tune']

    assert len(list(init_with_cfg2tune_config.branches)) == 4
    for b in init_with_cfg2tune_config.branches:
        assert type(b) is Cfg2Tune

    assert len(list(init_with_cfg2tune_config.leaves)) == len(leaves)
    for l1, l2 in zip(init_with_cfg2tune_config.leaves, leaves):
        assert l1 == l2
        assert type(l1) is type(l2)

    is_self_consistent(init_with_cfg2tune_config)


def test_init_with_update(init_with_update_config):
    """检查从Config初始化：能正确拷枝赋叶，正确更新，并找到root。检查配置树自洽性。"""
    cfg_wanted = Cfg2Tune()

    cfg_wanted.foo0.foo1.a = 1
    cfg_wanted.foo0.foo1.b = {'aa': 2, 'bb': 3}
    cfg_wanted.foo0.foo3.l = 'hh'
    cfg_wanted.foo0.i = 6.8

    cfg_wanted.foo2 = ('c', 'd')

    cfg_wanted.c = 7
    cfg_wanted.d.ee = 8
    cfg_wanted.d.ff = 9
    cfg_wanted.cc = 7
    cfg_wanted.e.f = {10, 'gg'}
    cfg_wanted.e.g = 11

    cfg_wanted.rslt_dir = 'init_with_update'

    # * 从dict角度，检查内容相同。
    assert init_with_update_config == cfg_wanted

    # * 检查枝干和结构相同。
    assert len(list(init_with_update_config.branches)) == len(list(cfg_wanted.branches))
    for b1, b2 in zip(init_with_update_config.branches, cfg_wanted.branches):
        assert b1 == b2
        assert type(b1) is type(b2)

    # * 检查叶子和结构相同。
    assert len(list(init_with_update_config.leaves)) == len(list(cfg_wanted.leaves))
    for l1, l2 in zip(init_with_update_config.leaves, cfg_wanted.leaves):
        assert l1 == l2
        assert type(l1) is type(l2)

    is_self_consistent(init_with_update_config)


def test_init_with_param_lazy():
    """检查从带有ParamLazy的Cfg2Tune初始化。"""
    with pytest.raises(RuntimeError, match="Cfg2Tune can't init by leaf with type"):
        Cfg2Tune('configs/easy_config/cfg.py')


def test_freeze(easy_config):
    """检查冻结与解冻功能。"""
    easy_config.freeze()

    with pytest.raises(RuntimeError, match="is frozen"):
        easy_config.c = 3

    with pytest.raises(RuntimeError, match="is frozen"):
        easy_config.foo0.a_max = 3

    with pytest.raises(RuntimeError, match="is frozen"):
        easy_config.foo2.a = 3


def test_cfg_tuned_subtree(easy_config):
    """检查配置树类型能否正确转换。"""
    cfgs = easy_config.get_cfgs()

    for cfg in cfgs:
        assert type(cfg) is Config
        assert len(list(cfg.branches)) == 3
        assert len(list(cfg.leaves)) == 6
