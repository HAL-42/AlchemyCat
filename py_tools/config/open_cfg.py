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
from typing import Tuple

from os import path as osp
import json
import pickle

import yaml
from yamlinclude import YamlIncludeConstructor

from ..load_module import load_module_from_py

__all__ = ['open_config']


def open_config(config_path: str, is_yaml: bool = False) -> Tuple[dict, bool]:
    """根据文件路径载入配置树所在模块，并读取名为config的配置树。

    Args:
        config_path (str): 配置树模块路径。
        is_yaml (bool): 过时的无效参数。

    Returns: 配置树模块中名为config的配置树。
    """
    # * is_yaml参数已经过时。
    if is_yaml:
        raise DeprecationWarning("is_yaml parameter is deprecated. ")

    # * Get extension
    name, ext = osp.splitext(config_path)

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
        config: dict = load_module_from_py(config_path).config
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
