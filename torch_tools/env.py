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
import json
import os
import os.path as osp
import pickle
import sys
import warnings
from importlib.util import spec_from_file_location, module_from_spec
from pprint import pprint
from typing import Union, Optional, Tuple

import cv2
import torch
import torch.distributed as dist
import yaml
from addict import Dict
from alchemy_cat.py_tools import get_process_info, get_local_time_str
from alchemy_cat.py_tools import set_rand_seed, Logger
from yamlinclude import YamlIncludeConstructor

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


def get_device(is_cuda: bool = True, cuda_id: int = 0, verbosity: bool = True) -> torch.device:
    """Get default device

    Args:
        is_cuda (bool): If false, always use cpu as device
        cuda_id (int): Default cuda device id.
        verbosity (bool): If True, show current devices

    Returns: Default device

    """
    if is_cuda:
        if not torch.cuda.is_available():
            raise ValueError("is_cuda = True while no cuda device is available.")
        if cuda_id >= torch.cuda.device_count():
            raise ValueError(f"cuda_id must < device_count = {torch.cuda.device_count()}")
        if cuda_id < 0:
            raise ValueError(f"cuda_id = {cuda_id} must > 0")

    device = torch.device("cuda:" + str(cuda_id) if is_cuda else "cpu")

    if verbosity:
        if is_cuda:
            print("Device:")
            for i in range(torch.cuda.device_count()):
                msg = f"    {i}: {torch.cuda.get_device_name(i)}"
                if i == cuda_id:
                    msg += ", used as default device."
                print(msg)
        else:
            print("Device: CPU")
    return device


def welcome():
    print("\033[32m###############o(*≧▽≦)ツ Alchemy Cat is Awesome φ(≧ω≦*)♪###############\033[0m")


def open_config(config_path: str, is_yaml: bool = False) -> Tuple[Union[Dict, dict], bool]:
    """Open yaml config

    Args:
        config_path (str): yaml config path
        is_yaml (bool): If True, return yaml dict rather than addict.Dict

    Returns: CONFIG
    """
    # * Check for existence
    if not osp.isfile(config_path):
        raise RuntimeError(f"No config file found at {config_path}")

    # * Get extension
    name, ext = osp.splitext(config_path)

    is_py = False
    # * Read yml/yaml config
    if ext == '.yml' or ext == '.yaml':
        # * Read and set config
        config_dir, _ = osp.split(config_path)
        YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, base_dir=config_dir)
        with open(config_path, 'r') as f:
            yaml_config = yaml.load(f, Loader=yaml.FullLoader)

        if is_yaml:
            config = yaml_config
        else:
            config = Dict(yaml_config)
    elif ext == '.py':
        is_py = True

        # # * Get import name
        # def get_import_name(name):
        #     name = ntpath.normpath(name)
        #     return '.'.join(name.split(ntpath.sep))

        # * Return config
        spec = spec_from_file_location("foo", config_path)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        config: Dict = module.config
        # config: Dict = import_module(get_import_name(name)).config
    elif ext == '.json':
        is_py = True
        with open(config_path, 'r') as json_f:
            config = Dict(json.load(json_f))
    elif ext == ".pkl":
        is_py = True
        with open(config_path, 'rb') as pkl_f:
            config = Dict(pickle.load(pkl_f))
    else:
        raise ValueError(f"config_path = {config_path} must be python, json, pkl, yml or yaml file.")
    return config, is_py


def _process_yaml_config(config: Dict, experiments_root: str):
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
    config.rslt_dir = rslt_dir
    if 'RAND_SEED' in config:
        config.rand_seed = config.RAND_SEED
    return config


def _process_py_config(config: Dict, experiments_root: str):
    if not config.rslt_dir:
        raise RuntimeError(f"config should indicate result save dir at config.rslt_dir = {config.rslt_dir}")
    config.rslt_dir = osp.join(experiments_root, config.rslt_dir)
    os.makedirs(config.rslt_dir, exist_ok=True)

    return config


def parse_config(config_path: str, experiments_root: str) -> Dict:
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
        config = _process_py_config(config, experiments_root)

    # * Check empty value and return
    _check_emtpy_value(config, memo='config.')
    return config


def init_env(is_cuda: Union[bool, int] = True, is_benchmark: bool = False, is_train: bool = True,
             config_path: Optional[str] = None,
             experiments_root: str = "experiment", rand_seed: Union[bool, str, int] = False,
             cv2_num_threads: int = -1, verbosity: bool = True, log_stdout: Union[bool, str] = False,
             local_rank: Optional[int] = None, silence_non_master_rank: Optional[bool] = False) \
        -> Tuple[torch.device, Optional[Dict]]:
    """Init torch training environment

    Args:
        is_cuda (Optional(bool, int)): If False, always use CPU. If Ture, use GPU and set GPU:0 as default device. If
            int, set GPU:i as default device.
        is_benchmark (bool): If True, set torch.backends.cudnn.benchmark = True
        is_train (bool): If False, disable grad
        config_path (Optional[str]): The path of yaml config
        experiments_root (str): The path where experiments result are stored
        rand_seed (Union[bool, str, int]) : If True, fix random of torch, numpy, python's random module from
            config.RAND_SEED. If False, don't fix random. If rand_seed is int or str, fix random according to
            rand_seed. (Default: False)
        cv2_num_threads (int): Set cv2 threads num by cv2.setNumThreads(cv2_num_threads). If < 0,
            don't set. (Default: -1)
        verbosity (bool): If True, print detailed info
        log_stdout (Union[bool, str]): If True, the stdout will be logged to corresponding experiment dir. If False, the
            stdout will not be logged. If log_stdout is str, it will be recognized as a path and stdout will be logged
            to that path. (Default: False)
        local_rank (Optional[int]): If not None, init distributed parallel env with rank = local_rank with
            init_method = "env://".  Default device will also bu set as "cuda:local_rank" .Make sure environment
            is pre-set.
        silence_non_master_rank (bool): If True, non master rank's (rank > 0) print will be silenced. (Default: False)

    Returns: Default device and config (If config_path is None, config is None)
    """
    # * Set distributed, verbosity is delayed.
    if local_rank is not None:
        # ** Check distributed env
        if not dist.is_available():
            raise ValueError(f"local_rank = {local_rank} while torch.distributed is not available")
        if not torch.cuda.is_available():
            raise ValueError(f"init_env only support cuda device distributed with nccl backend. "
                             f"local_rank = {local_rank} while torch.cuda is not available")

        # ** Set cuda device id
        if is_cuda is False:
            raise ValueError(f"When set local rank, cuda is needed. However, is_cuda = {is_cuda}")
        if isinstance(is_cuda, int) and (is_cuda != local_rank):
            raise ValueError(f"local_rank = {local_rank} must equal to is_cuda = {is_cuda}")
        is_cuda = local_rank

        # ** set device & init_process_group
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl',
            init_method="env://",
            rank=local_rank
        )

    # * Read CONFIG, verbosity is delayed.
    config = parse_config(config_path, experiments_root) if config_path is not None else None

    def get_stdout_log_dir_from_config():
        stdout_log_dir = osp.join(config.rslt_dir, 'stdout')

        return stdout_log_dir

    def get_stdout_log_file(stdout_log_dir):
        if local_rank is not None:
            prefix = '.' if dist.get_rank() > 0 else ''
            file_name = prefix + '-'.join(['stdout', f"rank{dist.get_rank()}", get_local_time_str()]) + '.log'
        else:
            file_name = '-'.join(['stdout', get_local_time_str(for_file_name=True)]) + '.log'
        return osp.join(stdout_log_dir, file_name)

    # * Log stdout
    # ** Get log file
    if isinstance(log_stdout, bool):
        stdout_log_file = get_stdout_log_file(get_stdout_log_dir_from_config()) if log_stdout else None
    elif isinstance(log_stdout, str):
        stdout_log_file = get_stdout_log_file(log_stdout)
    else:
        raise ValueError(f"log_stdout: {log_stdout} should be bool or path str")
    # ** Set logger
    if stdout_log_file is not None:
        if local_rank is not None:
            silence = silence_non_master_rank and (dist.get_rank() > 0)
            Logger(stdout_log_file, real_time=True, silence=silence)
            if verbosity:
                print(f"Log stdout at {stdout_log_file}. Silence = {silence}")
        else:
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

    # * Print distributed delaying verbosity
    if verbosity and (local_rank is not None):
        print(f"Using torch.distributed. Current cuda device id is set to local_rank = {local_rank}. \n"
              f"    Progress Group Rank: {dist.get_rank()}\n"
              f"    World Size: {dist.get_world_size()}\n"
              f"    Local Rank: {local_rank}")

    # * Print config's delaying verbosity
    if verbosity and config is not None:
        print("\033[32m------------------------------- CONFIG -------------------------------\033[0m")
        pprint(dict(config))
        print("\033[32m----------------------------- CONFIG END -----------------------------\033[0m")
        print("\n")

    if verbosity:
        print("\033[32m-------------------------------- INIT --------------------------------\033[0m")

    # *  Get device
    if isinstance(is_cuda, bool):
        device = get_device(is_cuda, cuda_id=0, verbosity=verbosity)
    elif isinstance(is_cuda, int):
        device = get_device(is_cuda=True, cuda_id=is_cuda, verbosity=verbosity)
    else:
        raise ValueError(f"Parameter is_cuda = {is_cuda} must be str or int")

    # * Set benchmark
    torch.backends.cudnn.benchmark = is_benchmark
    if verbosity:
        print(f"torch.backends.cudnn.benchmark = {is_benchmark}")

    # * Set cv2 threads num
    if cv2_num_threads >= 0:
        cv2.setNumThreads(cv2_num_threads)
        if verbosity:
            print(f"cv2.setNumThreads({cv2_num_threads})")

    # * If test, disable grad
    torch.set_grad_enabled(is_train)
    if verbosity:
        print(f"torch.set_grad_enabled({is_train})")

    def get_rand_seed_from_config():
        if 'rand_seed' not in config:
            raise ValueError(f"CONFIG did't have key rand_seed")
        return config.rand_seed

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
