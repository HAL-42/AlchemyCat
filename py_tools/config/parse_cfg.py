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
from typing import Any

from .open_cfg import open_config
from .py_cfg import Config

__all__ = ['auto_rslt_dir', 'parse_config']


def _check_emtpy_value(val: Any, memo: str='base.'):
    """Recursively detect empty val in dict"""
    if (not val) and not isinstance(val, (int, bool)):  # 若val被判否，且不是布尔（False）或int（0）类型，则是一个空Value。
        warnings.warn(f"{memo[:-1]} is a empty val: {val}")
    elif isinstance(val, dict):
        for k, v in val.items():
            _check_emtpy_value(v, memo + str(k) + '.')
    else:
        pass


def auto_rslt_dir(file: str, config_root: str='./configs', trunc_cwd: bool=False) -> str:
    """根据配置文件的__file__属性，自动返回配置文件的rslt_dir。

    Args:
        file: 配置文件的__file__属性。
        config_root: 相对工作目录，路径之于配置目录者。
        trunc_cwd: 是否去掉当前工作目录。

    Returns:
        配置文件的rslt_dir，默认是__file__去掉配置目录和配置文件后的路径。
    """
    if not trunc_cwd:
        return osp.dirname(osp.relpath(file, config_root))
    else:
        return str(Path(file).relative_to(Path.cwd()).relative_to(config_root).parent)


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


def _process_py_config(config: dict, config_path: str, experiments_root: str, config_root: str,
                       create_rslt_dir: bool=True):
    if isinstance(config, Config):
        # TODO 支持下级cfg删除上级cfg项。
        config.update_at_parser()
        # NOTE 1）此时归并COM，令后续IL计算、用户对COM无感。先update再归并，确保config能够接触、修改各层的所有COM树。
        # NOTE 2）归并可能令叶子间共享容器、IL函数（COM_会更新多个并形项）。故要确保
        # NOTE      a. 不直接修改叶子容器，而是覆盖叶子。
        # NOTE      b. IL函数无记忆，即多次调用，结果相同。
        config.reduce_COM()

    if not config.get('rslt_dir'):
        raise RuntimeError(f"config should indicate result save dir at config['rslt_dir'] = {config.get('rslt_dir')}")

    if config['rslt_dir'] is ...:
        config['rslt_dir'] = auto_rslt_dir(config_path, config_root)

    config['rslt_dir'] = osp.join(experiments_root, config['rslt_dir'])
    if create_rslt_dir:
        os.makedirs(config['rslt_dir'], exist_ok=True)

    return config


def parse_config(config_path: str | dict, experiments_root: str=None, config_root: str='./configs',
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
    if create_rslt_dir and experiments_root is None:
        # NOTE 如果需要创建rslt_dir，那么experiments_root必须提供。
        # NOTE 相反，若之后experiments_root为None，肯定不需要创建rslt_dir，此时experiment_root只用于修正原来的rslt_dir。
        # NOTE 因此，不妨在这种情况下，将experiments_root视作''。
        raise ValueError("experiments_root must be provided when create_rslt_dir is True")

    # * Open config, get config dict.
    config, is_py = open_config(config_path, is_yaml=False)
    if config is None:
        raise RuntimeError(f"Failed to parse config at {config_path}")
    if isinstance(config_path, dict):
        # NOTE 当config_path是dict时，auto_rslt_dir不可为...——因为无法获取真正的config_path。
        # NOTE 此时，需要确保config['rslt_dir']已经被赋值。
        if not config.get('rslt_dir'):
            raise RuntimeError(f"config should indicate result save dir at "
                               f"config['rslt_dir'] = {config.get('rslt_dir')}")
        if config['rslt_dir'] is ...:
            raise RuntimeError(f"When config_path is dict, config['rslt_dir'] should not be ...")

    # * Create experiment dirs according to CONFIG
    if not is_py:
        config = _process_yaml_config(config, '' if experiments_root is None else experiments_root, create_rslt_dir)
    else:
        config = _process_py_config(config, config_path, '' if experiments_root is None else experiments_root,
                                    config_root=config_root,
                                    create_rslt_dir=create_rslt_dir)

    # * Check empty value and return
    _check_emtpy_value(config, memo='config.')
    return Config.from_dict(config)
