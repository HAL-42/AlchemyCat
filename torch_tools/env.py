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
import cv2
from pprint import pprint

import yaml
from yamlinclude import YamlIncludeConstructor

from alchemy_cat.py_tools import set_rand_seed, Logger

__all__ = ["get_device", "open_config", "init_env"]


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
    with open(config_path, 'r') as f:
        yaml_config = yaml.load(f, Loader=yaml.FullLoader)

    if is_yaml:
        return yaml_config
    else:
        return Dict(yaml_config)


def parse_config(config_path: str, experiments_root: str):
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
            if TRAIN_ID is not None:
                TEST_DIR = osp.join(TRAIN_DIR, 'tests', str(TEST_ID))
            else:
                TEST_DIR = osp.join(EXP_DIR, 'tests', str(TEST_ID))
            os.makedirs(TEST_DIR, exist_ok=True)
        else:
            TEST_DIR = EXP_DIR
        CONFIG['TEST_DIR'] = TEST_DIR

        return Dict(CONFIG)


def init_env(is_cuda: bool=True, is_benchmark: bool=False, is_train: bool=True, config_path: Optional[str]=None,
             experiments_root: str = "experiment", rand_seed: bool=False,
             cv2_num_threads: int=0, verbosity: bool=True, log_stdout: bool=False) \
        -> Union[Tuple[torch.device, Dict], torch.device]:
    """Init torch training environment

    Args:
        is_cuda (bool): If False, always use CPU
        is_benchmark (bool): If True, set torch.backends.cudnn.benchmark = True
        is_train (bool): If False, disable grad
        config_path (Optional[str]): The path of yaml config
        experiments_root (str): The path where experiments result are stored
        rand_seed : If not True, fix random of torch, numpy, python's random module from CONFIG.RAND_SEED
        cv2_num_threads (int): Set cv2 threads num by cv2.setNumThreads(cv2_num_threads)
        verbosity (bool): If True, print detailed info
        log_stdout (bool): If True, the stdout will be logged to corresponding experiment dir

    Returns: Default device if config_path is not given, rather (Default Device, CONFIG)
    """
    # * Read CONFIG
    CONFIG = parse_config(config_path, experiments_root)

    # * Log stdout
    if log_stdout:
        if 'TEST_ID' in CONFIG:
            stdout_log_file = osp.join(CONFIG.TEST_DIR, 'stdout.log')
        elif 'TRAIN_ID' in CONFIG:
            stdout_log_file = osp.join(CONFIG.TRAIN_DIR, 'stdout.log')
        else:
            stdout_log_file = osp.join(CONFIG.EXP_ID, 'stdout.log')

        Logger(stdout_log_file, real_time=True)

        if verbosity:
            print(f"Log stdout at {stdout_log_file}")

    # * Welcome & Show system info
    if verbosity:
        welcome()
        print(f"Current working dir is {os.getcwd()}")
        print(f"Current python environment path is\n{sys.path}")
        print("\n")

    if verbosity:
            print("\033[32m------------------------------- CONFIG -------------------------------\033[0m")
            pprint(dict(CONFIG))
            print("\033[32m----------------------------- CONFIG END -----------------------------\033[0m")
            print("\n")

    if verbosity:
        print("\033[32m-------------------------------- INIT --------------------------------\033[0m")

    # * Get device
    device = get_device(is_cuda, verbosity)

    # * Set benchmark
    torch.backends.cudnn.benchmark = is_benchmark
    if verbosity:
        print(f"torch.backends.cudnn.benchmark = {is_benchmark}")

    # * Set cv2 threads num
    cv2.setNumThreads(cv2_num_threads)
    if verbosity:
        print(f"cv2.setNumThreads({cv2_num_threads})")

    # * If test, disable grad
    torch.set_grad_enabled(is_train)
    if verbosity:
        print(f"torch.set_grad_enabled({is_train})")

    # * Set rand seed
    if rand_seed:
        if 'RAND_SEED' not in CONFIG:
            raise ValueError(f"CONFIG did't have key RAND_SEED")
        set_rand_seed(CONFIG.RAND_SEED)
        if verbosity:
            print(f"Set rand seed {CONFIG.RAND_SEED}")

    if verbosity:
        print("\033[32m------------------------------ INIT END ------------------------------\033[0m")
        print("\n")

    # * Return
    if config_path is not None:
        return device, CONFIG
    else:
        return device
