#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: env.py
@time: 2020/1/16 4:46
@desc:
"""
import os
import os.path as osp
import sys
import torch
from addict import Dict
from typing import Union, Optional, Tuple

import yaml
from yamlinclude import YamlIncludeConstructor

from alchemy_cat.py_tools.random import set_rand_seed


def get_device(is_cuda: bool=True, verbosity: bool=True) -> torch.device:
    """Get default device

    Args:
        is_cuda (bool): If false, always use cpu as device
        verbosity (bool): If True, show current devices

    Returns: Default device

    """
    is_cuda = is_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")
    if verbosity:
        if is_cuda:
            print("Device:")
        for i in range(torch.cuda.device_count()):
            print("    {}:".format(i), torch.cuda.get_device_name(i))
        else:
            print("Device: CPU")
    return device


def welcome():
    print("\033[32m###############o(*≧▽≦)ツ Alchemy Cat is Awesome φ(≧ω≦*)♪###############\033[0m")


def open_config(config_path: str, is_yaml: bool=False) -> Union[Dict, dict]:
    """Open yaml config

    Args:
        config_path (str): yaml config path
        is_yaml (bool): If True, return yaml dict rather than addict.Dict

    Returns: CONFIG
    """
    with open("configs/train.yaml", 'r') as f:
        yaml_config = yaml.load(f, Loader=yaml.FullLoader)
    return Dict(yaml_config)


def init_env(is_cuda: bool=True, is_benchmark: bool=False, is_train: bool=True, config_path: Optional[str]=None,
             experiments_root: str = "experiment", fix_random: bool=False, verbosity: bool=True) \
            -> Union[Tuple[torch.device, Dict], torch.device]:
    """Init torch training environment

    Args:
        is_cuda (bool): If False, always use CPU
        is_benchmark (bool): If True, set torch.backends.cudnn.benchmark = True
        is_train (bool): If False, disable grad
        config_path (Optional[str]): The path of yaml config
        experiments_root (str): The path where experiments result are stored
        fix_random (bool): If True, fix random of torch, numpy, python's random module
        verbosity (bool): If True, print detailed info

    Returns: Default device if config_path is not given, rather (Default Device, CONFIG)
    """
    # * Show system info
    if verbosity:
        welcome()
        print(f"Current working dir is {os.getcwd()}")
        print(f"Current python environment path is\n{sys.path}")

    # * Get device
    device = get_device(is_cuda, verbosity)

    # * Set benchmark
    torch.backends.cudnn.benchmark = is_benchmark
    if verbosity:
        print(f"torch.backends.cudnn.benchmark = {is_benchmark}")

    # * If test, disable grad
    torch.set_grad_enabled(is_train)
    if verbosity:
        print(f"torch.set_grad_enabled({is_train})")

    # * Read config and set config include
    if config_path is not None:
        config_dir, _ = osp.split(config_path)
        YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, base_dir=config_dir)
        CONFIG = open_config(config_path, is_yaml=True)
    else:
        CONFIG = None

    # * Create experiment dirs according to CONFIG
    if config_path is not None:
        assert CONFIG is not None
        assert CONFIG.get('EXP_DIR') is None and CONFIG.get('TRAIN_DIR') is None and CONFIG.get('TEST_DIR') is None

        EXP_ID = CONFIG.get('EXP_ID')
        if EXP_ID is None:
            raise ValueError("CONFIG.EXP_ID must be defined")
        EXP_DIR = osp.join(experiments_root, str(EXP_ID))
        os.makedirs(EXP_DIR, exist_ok=True)
        CONFIG['EXP_DIR'] = EXP_DIR

        TRAIN_ID = CONFIG.get('TRAIN_ID')
        if TRAIN_ID is not None:
            TRAIN_DIR = osp.join(EXP_DIR, 'trains', str(TRAIN_ID))
            os.makedirs(TRAIN_DIR, exist_ok=True)
        else:
            TRAIN_DIR = EXP_DIR
        CONFIG['TRAIN_DIR'] = TRAIN_DIR

        TEST_ID = CONFIG.get('TEST_ID')
        if TEST_ID is not None:
            TEST_DIR = osp.join(EXP_DIR, 'tests', str(TEST_ID))
            os.makedirs(TEST_DIR, exist_ok=True)
        else:
            TEST_DIR = EXP_DIR
        CONFIG['TEST_DIR'] = TEST_DIR

        if verbosity:
            print(f"EXP_DIR={EXP_DIR}",
                  f"TRAIN_DIR={TRAIN_DIR}",
                  f"TEST_DIR={TEST_DIR}", sep="\n")

    if fix_random:
        if config_path is None:
            raise ValueError(
                "Rand seed need to be generated according to CONFIG.EXP_ID, CONFIG.TRAIN_ID, CONFIG.TEST_ID")
        # * Get str hash seed, then set rand seed
        def ID2str(ID):
            return str(ID) if ID is not None else ""

        seed = ID2str(EXP_ID) + ID2str(TRAIN_ID) + ID2str(TEST_ID)
        set_rand_seed(seed)
        if verbosity:
            print(f"Set rand seed with hash({seed})")

    # * Return
    if config_path is not None:
        return device, Dict(CONFIG)
    else:
        return device