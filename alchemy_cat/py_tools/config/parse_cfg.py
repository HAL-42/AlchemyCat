#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2022/5/24 14:37
@File    : parse_cfg.py
@Software: PyCharm
@Desc    : 
"""
import os
import warnings
from os import path as osp
from pathlib import Path
from typing import Any, Union

from .open_cfg import open_config
from .py_cfg import Config, ItemLazy

__all__ = ['auto_rslt_dir', 'parse_config', 'load_config']


def _check_emtpy_value(val: Any, memo: str='base.'):
    """Recursively detect empty val in dict"""
    warnings.warn("COM feature is deprecated", DeprecationWarning)
    if (not val) and not isinstance(val, (int, bool)):  # 若val被判否，且不是布尔（False）或int（0）类型，则是一个空Value。
        warnings.warn(f"{memo[:-1]} is a empty val: {val}")
    elif isinstance(val, dict):
        for k, v in val.items():
            _check_emtpy_value(v, memo + str(k) + '.')  # noqa
    else:
        pass


def auto_rslt_dir(file: str, experiments_root: str='', config_root: str='./configs', trunc_cwd: bool=False) -> str:
    """根据配置文件的__file__属性，自动返回配置文件的rslt_dir。

    Args:
        file: 配置文件的__file__属性。
        experiments_root: 实验目录的根目录。
        config_root: 相对工作目录，路径之于配置目录者。
        trunc_cwd: 是否去掉当前工作目录。

    Returns:
        配置文件的rslt_dir，默认是__file__去掉配置目录和配置文件后的路径。
    """
    if not trunc_cwd:
        rslt_dir = osp.dirname(osp.relpath(file, config_root))
    else:
        rslt_dir = str(Path(file).relative_to(Path.cwd()).relative_to(config_root).parent)

    if experiments_root != '':  # 修正rslt_dir。
        rslt_dir = osp.join(experiments_root, rslt_dir)

    return rslt_dir


def _process_yaml_config(config: dict, experiments_root: str, create_rslt_dir: bool=True):
    assert config.get('EXP_DIR') is None and config.get('TRAIN_DIR') is None and config.get('TEST_DIR') is None
    EXP_ID = config.get('EXP_ID')
    if EXP_ID is None:
        raise ValueError("CONFIG.EXP_ID must be defined")
    EXP_DIR = osp.join(experiments_root, str(EXP_ID))
    if create_rslt_dir:
        os.makedirs(EXP_DIR, exist_ok=True)
    config['EXP_DIR'] = EXP_DIR
    rslt_dir = EXP_DIR

    TRAIN_ID = config.get('TRAIN_ID')
    if TRAIN_ID is not None:
        TRAIN_DIR = osp.join(EXP_DIR, 'trains', str(TRAIN_ID))
        if create_rslt_dir:
            os.makedirs(TRAIN_DIR, exist_ok=True)
        rslt_dir = TRAIN_DIR
    else:
        TRAIN_DIR = EXP_DIR
    config['TRAIN_DIR'] = TRAIN_DIR

    TEST_ID = config.get('TEST_ID')
    if TEST_ID is not None:
        if TRAIN_ID is not None:
            TEST_DIR = osp.join(TRAIN_DIR, 'tests', str(TEST_ID))
        else:
            TEST_DIR = osp.join(EXP_DIR, 'tests', str(TEST_ID))
        if create_rslt_dir:
            os.makedirs(TEST_DIR, exist_ok=True)
        rslt_dir = TEST_DIR
    else:
        TEST_DIR = EXP_DIR
    config['TEST_DIR'] = TEST_DIR
    config['rslt_dir'] = rslt_dir
    if 'RAND_SEED' in config:
        config['rand_seed'] = config['RAND_SEED']
    return config


def _process_py_config(config: dict, config_path: Union[str, dict], experiments_root: str, config_root: str,
                       create_rslt_dir: bool=True):
    if isinstance(config, Config):
        # TODO 支持下级cfg删除上级cfg项。
        config.update_at_parser()
        # NOTE 1）此时归并COM，令后续IL计算、用户对COM无感。先update再归并，确保config能够接触、修改各层的所有COM树。
        # NOTE 2）归并可能令叶子间共享容器、IL函数（COM_会更新多个并形项）。故要确保
        # NOTE      a. 不直接修改叶子容器，而是覆盖叶子。
        # NOTE      b. IL函数无记忆，即多次调用，结果相同。
        # config.reduce_COM()

    # -* 首先判断rslt_dir能否获取。
    if not isinstance(config_path, dict):
        rslt_dir_available = True
    elif ('rslt_dir' in config) and config['rslt_dir'] is not ...:
        rslt_dir_available = True
    else:
        rslt_dir_available = False

    # -* 若rslt_dir可获取，总是获取。
    if rslt_dir_available:
        if config.get('rslt_dir', ...) is ...:  # 计算rslt_dir。
            config['rslt_dir'] = auto_rslt_dir(config_path, config_root=config_root, experiments_root=experiments_root)

        if create_rslt_dir:  # 创建rslt_dir。
            os.makedirs(config['rslt_dir'], exist_ok=True)
    # -* 若rslt_dir不可获取，但需要创建rslt_dir，则报错。
    elif create_rslt_dir:
        raise RuntimeError("When create_rslt_dir is True, should set cfg.rslt_dir or open config from file.")
    # -* 若rslt_dir不可获取，但也不需要创建rslt_dir，则不做任何处理。
    else:
        pass

    return config


def parse_config(config_path: Union[str, dict], experiments_root: str='', config_root: str='./configs',
                 create_rslt_dir: bool=True) -> Config:
    """Parse config from config path.

    This function will read yaml config from config path then create experiments dirs according to config file.

    Args:
        config_path: Path of yaml config.
        experiments_root: Root dictionary for experiments.
        config_root: Root dictionary of configs, for auto_rslt_dir only.
        create_rslt_dir: Whether to create rslt_dir.

    Returns:
        YAML config in Addict format
    """
    # NOTE experiment_root 默认为''，即不修正实验目录。此时该参数无任何影响。

    # * Open config, get config dict.
    config, is_py = open_config(config_path, is_yaml=False)  # NOTE config为dict类型是保证的。

    # * Create experiment dirs according to CONFIG
    if not is_py:
        config = _process_yaml_config(config, experiments_root, create_rslt_dir)
    else:
        config = _process_py_config(config, config_path, experiments_root,
                                    config_root=config_root,
                                    create_rslt_dir=create_rslt_dir)

    # * Check empty value and return
    # _check_emtpy_value(config, memo='cfg.')  # NOTE 似乎无此必要。
    return Config.from_dict(config)


def load_config(config_path: Union[str, dict], experiments_root: str='', config_root: str='./configs',
                create_rslt_dir: bool=True) -> Config:
    """Parse config then compute lazy items and freeze.

    Args:
        config_path: See `parse_config`.
        experiments_root: See `parse_config`.
        config_root: See `parse_config`.
        create_rslt_dir: See `parse_config`.

    Returns:
        Frozen config with lazy items computed.
    """
    if isinstance(config_path, Config):
        config_path.unfreeze()  # 如果是load，接下来需要parse和compute_item_lazy，故解冻。反正最后总是会freeze。
    config = parse_config(config_path, experiments_root, config_root, create_rslt_dir)
    config = ItemLazy.compute_item_lazy(config)
    config.freeze()
    return config
