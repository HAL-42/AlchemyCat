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
from typing import Any

import os
from os import path as osp
import warnings

from .open_cfg import open_config
from .py_cfg import Config

__all__ = ['auto_rslt_dir', 'parse_config']


def _check_emtpy_value(val: Any, memo: str='base.'):
    """Recursively detect empty val in dict"""
    if (not val) and not isinstance(val, (int, bool)):  # 若val被判否，且不是布尔（False）或int（0）类型，则是一个空Value。
        warnings.warn(f"{memo[:-1]} is a empty val: {val}")
    elif isinstance(val, dict):
        for k, v in val.items():
            _check_emtpy_value(v, memo + k + '.')
    else:
        pass


def auto_rslt_dir(file: str, config_dir: str='./configs') -> str:
    """根据配置文件的__file__属性，自动返回配置文件的rslt_dir。

    Args:
        file: 配置文件的__file__属性。
        config_dir: 相对工作目录，路径之于配置目录者。

    Returns:
        配置文件的rslt_dir，默认是__file__去掉配置目录和配置文件后的路径。
    """
    return osp.dirname(osp.relpath(file, config_dir))


def _process_yaml_config(config: dict, experiments_root: str):
    assert config.get('EXP_DIR') is None and config.get('TRAIN_DIR') is None and config.get('TEST_DIR') is None
    EXP_ID = config.get('EXP_ID')
    if EXP_ID is None:
        raise ValueError("CONFIG.EXP_ID must be defined")
    EXP_DIR = osp.join(experiments_root, str(EXP_ID))
    os.makedirs(EXP_DIR, exist_ok=True)
    config['EXP_DIR'] = EXP_DIR
    rslt_dir = EXP_DIR

    TRAIN_ID = config.get('TRAIN_ID')
    if TRAIN_ID is not None:
        TRAIN_DIR = osp.join(EXP_DIR, 'trains', str(TRAIN_ID))
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
        os.makedirs(TEST_DIR, exist_ok=True)
        rslt_dir = TEST_DIR
    else:
        TEST_DIR = EXP_DIR
    config['TEST_DIR'] = TEST_DIR
    config['rslt_dir'] = rslt_dir
    if 'RAND_SEED' in config:
        config['rand_seed'] = config['RAND_SEED']
    return config


def _process_py_config(config: dict, config_path: str, experiments_root: str,):
    if not config.get('rslt_dir'):
        raise RuntimeError(f"config should indicate result save dir at config['rslt_dir'] = {config.get('rslt_dir')}")

    if config['rslt_dir'] is ...:
        config['rslt_dir'] = auto_rslt_dir(config_path)

    config['rslt_dir'] = osp.join(experiments_root, config['rslt_dir'])
    os.makedirs(config['rslt_dir'], exist_ok=True)

    return config


def parse_config(config_path: str, experiments_root: str) -> Config:
    """Parse config from config path.

    This function will read yaml config from config path then create experiments dirs according to config file.

    Args:
        config_path: Path of yaml config.
        experiments_root: Root dictionary for experiments.

    Returns:
        YAML config in Addict format
    """
    # * Open config, get config dict.
    config, is_py = open_config(config_path, is_yaml=False)
    if config is None:
        raise RuntimeError(f"Failed to parse config at {config_path}")

    # * Create experiment dirs according to CONFIG
    if not is_py:
        config = _process_yaml_config(config, experiments_root)
    else:
        config = _process_py_config(config, config_path, experiments_root)

    # * Check empty value and return
    _check_emtpy_value(config, memo='config.')
    return Config.from_dict(config)
