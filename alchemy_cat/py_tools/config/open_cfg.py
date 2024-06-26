#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2022/5/24 20:05
@File    : open_cfg.py
@Software: PyCharm
@Desc    : 
"""
import glob
import json
import pickle
from os import path as osp
from typing import Tuple

import yaml
from yamlinclude import YamlIncludeConstructor

from ..load_module import load_module_from_py

__all__ = ['open_config']


def open_config(config_path: str | dict, is_yaml: bool = False) -> Tuple[dict, bool]:
    """根据文件路径载入配置树所在模块，并读取名为config的配置树。

    Args:
        config_path (str): 配置树模块路径。
        is_yaml (bool): 过时的无效参数。

    Returns: 配置树模块中名为config的配置树。
    """
    # * is_yaml参数已经过时。
    if is_yaml:
        raise DeprecationWarning("is_yaml parameter is deprecated. ")

    # * 若config_path是dict，则直接返回。
    if isinstance(config_path, dict):
        return config_path, True

    # * Get extension
    name, ext = osp.splitext(config_path)
    # ** auto ext
    if ext == '':
        cfg_path_found = glob.glob(f'{name}.*', recursive=False)
        assert len(cfg_path_found) == 1, f"Found {len(cfg_path_found)} config files with {name=}: {cfg_path_found}"
        config_path = cfg_path_found[0]
        _, ext = osp.splitext(config_path)

    # * Check for existence
    if (ext != '.py') and (not osp.isfile(config_path)):  # py文件找不到，可能是从其他路径导入。
        raise RuntimeError(f"No config file found at {config_path}")

    is_py = False
    # * Read yml/yaml config
    if ext == '.yml' or ext == '.yaml':
        # * Read and set config
        config_dir, _ = osp.split(config_path)
        YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, base_dir=config_dir)
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    elif ext == '.py':
        is_py = True
        # * Return config
        cfg_module = load_module_from_py(config_path)
        if hasattr(cfg_module, 'config'):
            config: dict = cfg_module.config
            if hasattr(cfg_module, 'cfg') and (cfg_module.cfg is not cfg_module.config):
                raise RuntimeError(f"Both 'config' and 'cfg' are defined in {cfg_module.__path__} but not the same.")
        elif hasattr(cfg_module, 'cfg'):
            config: dict = cfg_module.cfg
        else:
            raise ValueError(f"config module {cfg_module.__path__} must have 'config' or 'cfg' defined.")

    elif ext == '.json':
        is_py = True
        with open(config_path, 'r') as json_f:
            config = json.load(json_f)
    elif ext == ".pkl":
        is_py = True
        with open(config_path, 'rb') as pkl_f:
            config: dict = pickle.load(pkl_f)
    else:
        raise ValueError(f"config_path = {config_path} must be python, json, pkl, yml or yaml file.")
    return config, is_py
