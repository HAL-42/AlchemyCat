#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2024/7/10 4:16
@File    : test_yacs_yaml.py
@Software: PyCharm
@Desc    : 
"""
import sys
from pathlib import Path

import pytest

from alchemy_cat.py_tools import Config, load_config, open_config, Cfg2Tune, Param2Tune

YAML_CFG = 'py_tools/config/tests/migrate/yaml.yaml'
YACS_CFG = 'py_tools/config/tests/migrate/yacs_cfg.py'
YACS_GET_CFG = 'py_tools/config/tests/migrate/yacs_get.py'
MM_CFG = 'py_tools/config/tests/migrate/mm_configs/deeplabv3plus/deeplabv3plus_r50-d8_4xb2-40k_cityscapes-512x1024.py'
AC_CFG = 'py_tools/config/tests/migrate/ac_cfg.py'

sys.path = ['', 'py_tools/config/tests'] + sys.path


def is_yaml() -> bool:
    try:
        import yaml
        return True
    except ImportError:
        return False


def is_yacs() -> bool:
    try:
        from yacs.config import CfgNode
        return True
    except ImportError:
        return False


def is_mm() -> bool:
    try:
        from mmengine import Config
        return True
    except ImportError:
        return False


@pytest.fixture(scope='module')
def dump_py():
    dump_py = Path('/tmp/test_alchemy_cat_config/dump.py')
    dump_py.parent.mkdir(parents=True, exist_ok=True)
    yield dump_py
    # dump_py.unlink(missing_ok=True)


@pytest.fixture(scope='module')
def dump_yaml():
    dump_yaml = Path('/tmp/test_alchemy_cat_config/dump.yaml')
    dump_yaml.parent.mkdir(parents=True, exist_ok=True)
    yield dump_yaml
    # dump_yaml.unlink(missing_ok=True)


@pytest.fixture(scope='function')
def mm_config2tune():
    cfg = Cfg2Tune(caps=MM_CFG)
    cfg.rslt_dir = 'xxx'
    cfg.model.backbone.depth = Param2Tune([50, 101])
    cfg.train_cfg.max_iters = Param2Tune([10_000, 20_000])
    yield cfg


@pytest.mark.skipif(not is_yaml(), reason="yaml is not installed")
def test_open_yaml():
    import yaml

    loaded_cfg = load_config(YAML_CFG, create_rslt_dir=False).unfreeze()
    del loaded_cfg.rslt_dir

    init_cfg = Config(YAML_CFG)

    caps_cfg = Config(caps=YAML_CFG)
    caps_cfg.rslt_dir = 'xxx'
    caps_cfg.parse(create_rslt_dir=False).unfreeze()
    del caps_cfg.rslt_dir

    with open(YAML_CFG, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    assert yaml_cfg == loaded_cfg == init_cfg == caps_cfg


@pytest.mark.skipif(not is_yacs(), reason="yacs is not installed")
def test_open_yacs_cfg():
    loaded_cfg = load_config(YACS_CFG, create_rslt_dir=False).unfreeze()
    del loaded_cfg.rslt_dir

    init_cfg = Config(YACS_CFG)

    caps_cfg = Config(caps=YACS_CFG)
    caps_cfg.rslt_dir = 'xxx'
    caps_cfg.parse(create_rslt_dir=False).unfreeze()
    del caps_cfg.rslt_dir

    from migrate.yacs_cfg import cfg as yacs_cfg

    assert yacs_cfg == loaded_cfg == init_cfg == caps_cfg


@pytest.mark.skipif(not is_yacs(), reason="yacs is not installed")
def test_open_yacs_get_cfg():
    loaded_cfg = load_config(YACS_GET_CFG, create_rslt_dir=False).unfreeze()
    del loaded_cfg.rslt_dir

    init_cfg = Config(YACS_GET_CFG)

    caps_cfg = Config(caps=YACS_GET_CFG)
    caps_cfg.rslt_dir = 'xxx'
    caps_cfg.parse(create_rslt_dir=False).unfreeze()
    del caps_cfg.rslt_dir

    from migrate.yacs_get import get_cfg_defaults

    assert get_cfg_defaults() == loaded_cfg == init_cfg == caps_cfg


@pytest.mark.skipif(not is_mm(), reason="mmengine is not installed")
def test_open_mm():
    loaded_cfg = load_config(MM_CFG, create_rslt_dir=False).unfreeze()
    del loaded_cfg.rslt_dir

    init_cfg = Config(MM_CFG)

    caps_cfg = Config(caps=MM_CFG)
    caps_cfg.rslt_dir = 'xxx'
    caps_cfg.parse(create_rslt_dir=False).unfreeze()
    del caps_cfg.rslt_dir

    from mmengine import Config as MMConfig
    mm_cfg = MMConfig.fromfile(MM_CFG)._cfg_dict

    assert mm_cfg == loaded_cfg == init_cfg == caps_cfg


def test_to_cfg(dump_py: Path):
    x_cfg = open_config(AC_CFG)[0]

    Config.from_x_to_y(AC_CFG, dump_py, y_type='alchemy-cat')

    y_cfg = open_config(str(dump_py))[0]

    assert x_cfg == y_cfg


@pytest.mark.skipif(not is_yaml(), reason="yaml is not installed")
def test_to_yaml(dump_yaml: Path):
    x_cfg = open_config(AC_CFG)[0]

    Config.from_x_to_y(AC_CFG, dump_yaml, y_type='yaml')

    y_cfg = open_config(str(dump_yaml))[0]

    assert x_cfg == y_cfg


@pytest.mark.skipif(not is_mm(), reason="mmengine is not installed")
def test_to_mm(dump_py: Path):
    x_cfg = open_config(AC_CFG)[0]

    Config.from_x_to_y(AC_CFG, dump_py, y_type='mmcv')

    y_cfg = open_config(str(dump_py))[0]

    assert x_cfg == y_cfg


@pytest.mark.skipif(not is_mm(), reason="mmengine is not installed")
def test_tune_mm(mm_config2tune: Cfg2Tune, dump_py: Path):
    from mmengine import Config as MMConfig

    mm_cfg = MMConfig.fromfile(MM_CFG)

    cfgs = list(mm_config2tune.get_cfgs())

    cfgs[0].save_mmcv(dump_py)
    mm_cfg0 = MMConfig.fromfile(str(dump_py))
    del mm_cfg0._cfg_dict['rslt_dir']
    assert mm_cfg0.model.backbone.depth == 50
    assert mm_cfg0.train_cfg.max_iters == 10_000
    mm_cfg0.model.backbone.depth = 50
    mm_cfg0.train_cfg.max_iters = 40_000
    assert mm_cfg0._cfg_dict == mm_cfg._cfg_dict

    cfgs[1].save_mmcv(dump_py)
    mm_cfg1 = MMConfig.fromfile(str(dump_py))
    del mm_cfg1._cfg_dict['rslt_dir']
    assert mm_cfg1.model.backbone.depth == 50
    assert mm_cfg1.train_cfg.max_iters == 20_000
    mm_cfg1.model.backbone.depth = 50
    mm_cfg1.train_cfg.max_iters = 40_000
    assert mm_cfg1._cfg_dict == mm_cfg._cfg_dict

    cfgs[2].save_mmcv(dump_py)
    mm_cfg2 = MMConfig.fromfile(str(dump_py))
    del mm_cfg2._cfg_dict['rslt_dir']
    assert mm_cfg2.model.backbone.depth == 101
    assert mm_cfg2.train_cfg.max_iters == 10_000
    mm_cfg2.model.backbone.depth = 50
    mm_cfg2.train_cfg.max_iters = 40_000
    assert mm_cfg2._cfg_dict == mm_cfg._cfg_dict

    cfgs[3].save_mmcv(dump_py)
    mm_cfg3 = MMConfig.fromfile(str(dump_py))
    del mm_cfg3._cfg_dict['rslt_dir']
    assert mm_cfg3.model.backbone.depth == 101
    assert mm_cfg3.train_cfg.max_iters == 20_000
    mm_cfg3.model.backbone.depth = 50
    mm_cfg3.train_cfg.max_iters = 40_000
    assert mm_cfg3._cfg_dict == mm_cfg._cfg_dict
