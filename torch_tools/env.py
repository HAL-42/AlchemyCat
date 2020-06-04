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
import warnings
import torch
from addict import Dict
from typing import Union, Optional, Tuple
import cv2
from pprint import pprint

import yaml
from yamlinclude import YamlIncludeConstructor

from alchemy_cat.py_tools import set_rand_seed, Logger
from alchemy_cat.py_tools import get_process_info

__all__ = ["get_device", "open_config", "init_env", "parse_config"]


def _check_emtpy_value(val, memo='base.'):
    """Recursively detect empty val in dict"""
    if (not val) and (val is not False) and (val is not 0):
        warnings.warn(f"{memo[:-1]} is a empty val: {val}")
    elif isinstance(val, dict):
        for k, v in val.items():
            _check_emtpy_value(v, memo + k + '.')
    else:
        pass


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
    """Parse config from config path.

    This function will read yaml config from config path then create EXP_DIR, TRAIN_DIR, TEST_DIR according to config
    file.

    Args:
        config_path: Path of yaml config.
        experiments_root: Root dictionary for experiments.

    Returns:
        YAML config in Addict format
    """
    if not osp.isfile(config_path):
        raise RuntimeError(f"No config file found at {config_path}")

    # * Read and set config
    config_dir, _ = osp.split(config_path)
    YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, base_dir=config_dir)

    config = open_config(config_path, is_yaml=True)
    if config is None:
        raise RuntimeError(f"Failed to parse config at {config_path}")

    # * Create experiment dirs according to CONFIG
    assert config.get('EXP_DIR') is None and config.get('TRAIN_DIR') is None and config.get('TEST_DIR') is None
    EXP_ID = config.get('EXP_ID')
    if EXP_ID is None:
        raise ValueError("CONFIG.EXP_ID must be defined")
    EXP_DIR = osp.join(experiments_root, str(EXP_ID))
    os.makedirs(EXP_DIR, exist_ok=True)
    config['EXP_DIR'] = EXP_DIR
    TRAIN_ID = config.get('TRAIN_ID')
    if TRAIN_ID is not None:
        TRAIN_DIR = osp.join(EXP_DIR, 'trains', str(TRAIN_ID))
        os.makedirs(TRAIN_DIR, exist_ok=True)
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
    else:
        TEST_DIR = EXP_DIR
    config['TEST_DIR'] = TEST_DIR
    _check_emtpy_value(config, memo='config.')
    return Dict(config)


def init_env(is_cuda: bool=True, is_benchmark: bool=False, is_train: bool=True, config_path: Optional[str]=None,
             experiments_root: str="experiment", rand_seed: Union[bool, str, int]=False,
             cv2_num_threads: int=0, verbosity: bool=True, log_stdout: Union[bool, str]=False) \
        -> Tuple[torch.device, Optional[Dict]]:
    """Init torch training environment

    Args:
        is_cuda (bool): If False, always use CPU
        is_benchmark (bool): If True, set torch.backends.cudnn.benchmark = True
        is_train (bool): If False, disable grad
        config_path (Optional[str]): The path of yaml config
        experiments_root (str): The path where experiments result are stored
        rand_seed (Union[bool, str, int]) : If True, fix random of torch, numpy, python's random module from
            config.RAND_SEED. If False, don't fix random. If rand_seed is int or str, fix random according to
            rand_seed. (Default: False)
        cv2_num_threads (int): Set cv2 threads num by cv2.setNumThreads(cv2_num_threads)
        verbosity (bool): If True, print detailed info
        log_stdout (Union[bool, str]): If True, the stdout will be logged to corresponding experiment dir. If False, the
            stdout will not be logged. If log_stdout is str, it will be recognized as a path and stdout will be logged
            to that path. (Default: False)

    Returns: Default device and config (If config_path is None, config is None)
    """
    # * Read CONFIG
    config = parse_config(config_path, experiments_root) if config_path is not None else None

    def set_stdout_log_file_from_config():
        if config is not None:
            if 'TEST_ID' in config:
                stdout_log_file = osp.join(config.TEST_DIR, 'stdout.log')
            elif 'TRAIN_ID' in config:
                stdout_log_file = osp.join(config.TRAIN_DIR, 'stdout.log')
            else:
                stdout_log_file = osp.join(config.EXP_ID, 'stdout.log')
        else:
            raise ValueError(f"Can't set stdout log file according to config for config_path is not given.")

        return stdout_log_file

    # * Log stdout
    if isinstance(log_stdout, bool):
        stdout_log_file = set_stdout_log_file_from_config() if log_stdout else None
    elif isinstance(log_stdout, str):
        stdout_log_file = log_stdout
    else:
        raise ValueError(f"log_stdout: {log_stdout} should be bool or path str")

    if stdout_log_file is not None:
        Logger(stdout_log_file, real_time=True)
        if verbosity:
            print(f"Log stdout at {stdout_log_file}")

    # * Welcome & Show system info
    if verbosity:
        welcome()
        print(f"Current working dir is {os.getcwd()}")
        print(f"Current python environment path is\n{sys.path}")
        print(f"Current Process Info: ")
        pprint(get_process_info())
        print("\n")

    if verbosity and config is not None:
        print("\033[32m------------------------------- CONFIG -------------------------------\033[0m")
        pprint(dict(config))
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

    def get_rand_seed_from_config():
        if 'RAND_SEED' not in config:
            raise ValueError(f"CONFIG did't have key RAND_SEED")
        return config.RAND_SEED

    # * Set rand seed
    if isinstance(rand_seed, bool):
        rand_seed_ = get_rand_seed_from_config() if rand_seed else None
    elif isinstance(rand_seed, int) or isinstance(rand_seed, str):
        rand_seed_ = rand_seed
    else:
        raise ValueError(f"rand_seed should be bool, int or str.")

    if rand_seed_ is not None:
        set_rand_seed(rand_seed_)
        if verbosity:
            print(f"Set rand seed {rand_seed_}")

    # * End of init
    if verbosity:
        print("\033[32m------------------------------ INIT END ------------------------------\033[0m")
        print("\n")

    # * Return
    return device, config
