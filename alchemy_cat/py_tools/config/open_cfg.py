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
from pathlib import Path
from typing import Tuple, Union

try:
    from yacs.config import CfgNode, _assert_with_logging, _valid_type, _VALID_TYPES  # noqa
    YACS_AVAILABLE = True
except ImportError:
    YACS_AVAILABLE = False

try:
    from mmengine import Config as MMConfig  # noqa
    MM_AVAILABLE = True
except ImportError:
    MM_AVAILABLE = False

from ..load_module import load_module_from_py

__all__ = ['open_config']


def open_config(config_path: Union[str, dict], is_yaml: bool = False) -> Tuple[dict, bool]:
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

    is_py = True
    # * Read yml/yaml config
    if ext == '.yml' or ext == '.yaml':
        # * Read and set config
        import yaml
        config = yaml.safe_load(Path(config_path).read_text())
    elif ext == '.py':
        cfg_module = load_module_from_py(config_path)
        if hasattr(cfg_module, 'config'):
            config: dict = cfg_module.config
            if hasattr(cfg_module, 'cfg') and (cfg_module.cfg is not cfg_module.config):
                raise RuntimeError(f"Both 'config' and 'cfg' are defined in {cfg_module.__path__} but not the same.")
        elif hasattr(cfg_module, 'cfg'):
            config: dict = cfg_module.cfg
        elif hasattr(cfg_module, 'get_cfg_defaults'):  # yacs兼容
            config: dict = cfg_module.get_cfg_defaults()
        elif hasattr(cfg_module, '_C'):  # yacs兼容
            config: dict = getattr(cfg_module, '_C')
        elif MM_AVAILABLE:
            mm_config: MMConfig = MMConfig.fromfile(config_path)
            if hasattr(mm_config, 'to_dict'):
                config: dict = mm_config.to_dict()
            else:
                config: dict = object.__getattribute__(mm_config, '_cfg_dict').to_dict()
        else:
            raise ValueError(f"config module {cfg_module.__path__} must have 'config'/'cfg' defined, "
                             f"or be a yacs or mmcv config.")

        # -* 如果是yacs配置，则转为dict。
        if YACS_AVAILABLE and isinstance(config, CfgNode):
            def convert_to_dict(cfg_node, key_list):
                if not isinstance(cfg_node, CfgNode):
                    _assert_with_logging(
                        _valid_type(cfg_node),
                        "Key {} with value {} is not a valid type; valid types: {}".format(
                            ".".join(key_list), type(cfg_node), _VALID_TYPES
                        ),
                    )
                    return cfg_node
                else:
                    cfg_dict = dict(cfg_node)
                    for k, v in cfg_dict.items():
                        cfg_dict[k] = convert_to_dict(v, key_list + [k])
                    return cfg_dict

            config = convert_to_dict(config, [])
    elif ext == '.json':
        with open(config_path, 'r') as json_f:
            config = json.load(json_f)
    elif ext == ".pkl":
        with open(config_path, 'rb') as pkl_f:
            config: dict = pickle.load(pkl_f)
    else:
        raise ValueError(f"config_path = {config_path} must be python, json, pkl, yml or yaml file.")

    if not isinstance(config, dict):
        raise ValueError(f"The opened {config_path} should be a dict, but got {type(config)}")

    return config, is_py
