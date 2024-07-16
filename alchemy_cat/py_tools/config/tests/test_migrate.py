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

from alchemy_cat.py_tools import Config, load_config, open_config, Cfg2TuneRunner

YAML_CFG = 'py_tools/config/tests/migrate/yaml.yaml'
YACS_CFG = 'py_tools/config/tests/migrate/yacs_cfg.py'
YACS_GET_CFG = 'py_tools/config/tests/migrate/yacs_get.py'
MM_CFG = 'py_tools/config/tests/migrate/mm_configs/deeplabv3plus/deeplabv3plus_r50-d8_4xb4-40k_voc12aug-512x512.py'
MM_CFG2TUNE = 'py_tools/config/tests/migrate/mm_configs/d3p/tune_bs,iter/cfg.py'
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
def mm_cfg2tune_cfgs():
    runner = Cfg2TuneRunner(MM_CFG2TUNE,
                            config_root='py_tools/config/tests/migrate/mm_configs',
                            experiment_root='/tmp/mm_exp')
    yield list(runner.cfg2tune.get_cfgs())


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
def test_tune_mm(mm_cfg2tune_cfgs, dump_py: Path):
    from mmengine import Config as MMConfig

    mm_cfg = MMConfig.fromfile(MM_CFG)

    def compare_with_origin(mm_cfg_n):
        del mm_cfg_n._cfg_dict['rslt_dir']
        del mm_cfg_n._cfg_dict['work_dir']

        mm_cfg_n.model.auxiliary_head.loss_decode.loss_weight = 0.4
        mm_cfg_n.train_cfg.max_iters = 40_000
        mm_cfg_n.train_dataloader.batch_size = 4
        mm_cfg_n.optim_wrapper.optimizer.lr = 0.01
        mm_cfg_n.param_scheduler[0]['end'] = 40_000

        assert mm_cfg_n._cfg_dict == mm_cfg._cfg_dict

    mm_cfg2tune_cfgs[0].save_mmcv(dump_py)
    mm_cfg0 = MMConfig.fromfile(str(dump_py))
    assert mm_cfg0.work_dir == '/tmp/mm_exp/d3p/tune_bs,iter/max_iters=20000,batch_size=8'
    assert mm_cfg0.rslt_dir == mm_cfg0.work_dir
    assert mm_cfg0.model.auxiliary_head.loss_decode.loss_weight == 0.2
    assert mm_cfg0.train_cfg.max_iters == 20_000
    assert mm_cfg0.train_dataloader.batch_size == 8
    assert mm_cfg0.param_scheduler[0]['end'] == 20_000
    assert mm_cfg0.optim_wrapper.optimizer.lr == pytest.approx(0.01)
    compare_with_origin(mm_cfg0)

    mm_cfg2tune_cfgs[1].save_mmcv(dump_py)
    mm_cfg1 = MMConfig.fromfile(str(dump_py))
    assert mm_cfg1.work_dir == '/tmp/mm_exp/d3p/tune_bs,iter/max_iters=20000,batch_size=16'
    assert mm_cfg1.rslt_dir == mm_cfg1.work_dir
    assert mm_cfg1.model.auxiliary_head.loss_decode.loss_weight == 0.2
    assert mm_cfg1.train_cfg.max_iters == 20_000
    assert mm_cfg1.train_dataloader.batch_size == 16
    assert mm_cfg1.param_scheduler[0]['end'] == 20_000
    assert mm_cfg1.optim_wrapper.optimizer.lr == pytest.approx(0.02)
    compare_with_origin(mm_cfg1)

    mm_cfg2tune_cfgs[2].save_mmcv(dump_py)
    mm_cfg2 = MMConfig.fromfile(str(dump_py))
    assert mm_cfg2.work_dir == '/tmp/mm_exp/d3p/tune_bs,iter/max_iters=40000,batch_size=8'
    assert mm_cfg2.rslt_dir == mm_cfg2.work_dir
    assert mm_cfg2.model.auxiliary_head.loss_decode.loss_weight == 0.2
    assert mm_cfg2.train_cfg.max_iters == 40_000
    assert mm_cfg2.train_dataloader.batch_size == 8
    assert mm_cfg2.param_scheduler[0]['end'] == 40_000
    assert mm_cfg2.optim_wrapper.optimizer.lr == pytest.approx(0.01)
    compare_with_origin(mm_cfg2)

    mm_cfg2tune_cfgs[3].save_mmcv(dump_py)
    mm_cfg3 = MMConfig.fromfile(str(dump_py))
    assert mm_cfg3.work_dir == '/tmp/mm_exp/d3p/tune_bs,iter/max_iters=40000,batch_size=16'
    assert mm_cfg3.rslt_dir == mm_cfg3.work_dir
    assert mm_cfg3.model.auxiliary_head.loss_decode.loss_weight == 0.2
    assert mm_cfg3.train_cfg.max_iters == 40_000
    assert mm_cfg3.train_dataloader.batch_size == 16
    assert mm_cfg3.param_scheduler[0]['end'] == 40_000
    assert mm_cfg3.optim_wrapper.optimizer.lr == pytest.approx(0.02)
    compare_with_origin(mm_cfg3)
