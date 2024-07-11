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
from pprint import pprint
from typing import Union, Optional, Tuple

import cv2
import torch
import torch.distributed as dist

from ..dl_config import parse_config, Config, ItemLazy
from ..py_tools import get_process_info, get_local_time_str, set_rand_seed, Logger, meow, init_loguru

__all__ = ['get_device', 'init_env']


def get_device(is_cuda: bool=True, cuda_id: int=0, verbosity: bool=True) -> torch.device:
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
            print("设备列表:")
            for i in range(torch.cuda.device_count()):
                msg = f"    {i}: {torch.cuda.get_device_name(i)}"
                if i == cuda_id:
                    msg += "，作为默认设备。"
                print(msg)
        else:
            print("设备列表: CPU")
        print("\n")
    return device


def welcome():
    print("\033[32m##################o(*≧▽≦)ツ 圣火昭昭 圣火耀耀 凡我弟子 喵喵喵喵 φ(≧ω≦*)♪##################\033[0m",
          end="\n\n")


def init_env(*, is_cuda: Union[bool, int] = True, is_benchmark: bool = False, is_train: bool = True,
             config_path: str | dict = None, config_root: str='./configs',
             experiments_root: str = "experiment", rand_seed: Union[bool, str, int] = False,
             cv2_num_threads: int = -1,
             verbosity: bool = True, log_stdout: Union[bool, str] = False, loguru_ini: bool | dict=True,
             local_rank: Optional[int] = None, silence_non_master_rank: Optional[bool] = False,
             reproducibility: Optional[bool] = False,
             is_debug: bool=False) \
        -> Tuple[torch.device, Optional[Config]]:
    """Init torch training environment

    Args:
        is_cuda (Optional(bool, int)): If False, always use CPU. If Ture, use GPU and set GPU:0 as default device. If
            int, set GPU:i as default device.
        is_benchmark (bool): If True, set torch.backends.cudnn.benchmark = True
        is_train (bool): If False, disable grad
        config_path (str | Config): The path to config
        config_root: Root dictionary of configs, for auto_rslt_dir only.
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
        loguru_ini (dict): init loguru with loguru_ini. (Default: True)
        local_rank (Optional[int]): If not None, init distributed parallel env with rank = local_rank with
            init_method = "env://".  Default device will also be set as "cuda:local_rank" .Make sure environment
            is pre-set.
        silence_non_master_rank (bool): If True, non-master rank's (rank > 0) print will be silenced. (Default: False)
        reproducibility (bool): If True, do everything possible to make the pytorch program reproducible.
            (Default: False)
        is_debug (bool): If True, meow.is_debug will be set to True. (Default: False)

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
        if (isinstance(is_cuda, int) and not isinstance(is_cuda, bool)) and (is_cuda != local_rank):  # bool是int的子类。
            raise ValueError(f"local_rank = {local_rank} must equal to is_cuda = {is_cuda}")

        if isinstance(is_cuda, bool):
            is_cuda = local_rank  # 若is_cuda为bool，则将其设置为local_rank，即默认使用local_rank作为cuda_id。

        # ** set device & init_process_group
        dist.init_process_group(
            backend='nccl',
            init_method="env://",
            rank=local_rank
        )

    # * Read CONFIG, verbosity is delayed.
    config = parse_config(config_path, experiments_root, config_root) if config_path is not None else None
    if meow.cfg is not None:
        raise RuntimeError(f"meow.cfg = {meow.cfg} should be None before init_env")
    meow.cfg = config  # 新瓶装旧酒，旧酒望新瓶。置酒猫头上，配置从我游。

    def get_stdout_log_dir_from_config():
        return osp.join(config.rslt_dir, 'stdout')

    def get_stdout_log_file(stdout_log_dir):
        if local_rank is not None:
            prefix = '.' if dist.get_rank() > 0 else ''
            file_name = prefix + '-'.join(['stdout', f"rank{dist.get_rank()}",
                                           get_local_time_str(for_file_name=True)]) + '.log'
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
            Logger(stdout_log_file, real_time=False, silence=silence)
            if verbosity:
                print(f"标准输出重定向到：{stdout_log_file}，Silence = {silence}。", end="\n\n")
        else:
            Logger(stdout_log_file, real_time=False)
            if verbosity:
                print(f"标准输出重定向到：{stdout_log_file}。", end="\n\n")

    # -* loguru初始化。
    if loguru_ini:
        init_loguru(**({} if isinstance(loguru_ini, bool) else loguru_ini))

    # * Welcome & Show system info
    if verbosity:
        welcome()
        print(f"当前工作目录为：{os.getcwd()}")
        print(f"当前python搜索路径为：")
        pprint(sys.path)
        print(f"当前进程信息有：")
        pprint(get_process_info(), sort_dicts=False)
        print("\n")

    if verbosity:
        print("\033[32m-------------------------------------- 初始化开始 --------------------------------------\033[0m",
              end="\n\n")

    # * Print distributed delaying verbosity
    if verbosity and (local_rank is not None):
        print(f"已初始化torch分布式环境，默认CUDA设备ID被设置为：local_rank = {local_rank}。 \n"
              f"    Progress Group Rank: {dist.get_rank()}\n"
              f"    World Size: {dist.get_world_size()}\n"
              f"    Local Rank: {local_rank}", end="\n\n")

    # *  Get device
    if isinstance(is_cuda, bool):
        device = get_device(is_cuda, cuda_id=0, verbosity=verbosity)
    elif isinstance(is_cuda, int):
        device = get_device(is_cuda=True, cuda_id=is_cuda, verbosity=verbosity)
    else:
        raise ValueError(f"Parameter is_cuda = {is_cuda} must be str or int")
    if device != torch.device('cpu'):
        torch.cuda.set_device(device)

    # * Set benchmark
    torch.backends.cudnn.benchmark = is_benchmark
    if verbosity:
        print(f"torch.backends.cudnn.benchmark = {is_benchmark}", end="\n\n")

    # * Set cv2 threads num
    if cv2_num_threads >= 0:
        cv2.setNumThreads(cv2_num_threads)
        if verbosity:
            print(f"cv2.setNumThreads({cv2_num_threads})", end="\n\n")

    # * If test, disable grad
    torch.set_grad_enabled(is_train)
    if verbosity:
        print(f"torch.set_grad_enabled({is_train})", end="\n\n")

    def get_rand_seed_from_config():
        if 'rand_seed' not in config:
            raise ValueError(f"CONFIG didn't have key rand_seed")
        return config.rand_seed

    # * Set rand seed
    # ** 若rand_seed不是False，则解析得到rand_seed_ori，反之rand_seed_ori为None。
    if isinstance(rand_seed, bool):
        rand_seed_ori = get_rand_seed_from_config() if rand_seed else None
    elif isinstance(rand_seed, int) or isinstance(rand_seed, str):
        rand_seed_ori = rand_seed
    else:
        raise ValueError(f"rand_seed should be bool, int or str.")

    # ** 若rand_seed_ori不是None（需要设置随机种子点），且分布式训练，则给每个rank设置不同随机种子偏置。反之偏置为None。
    if rand_seed_ori is not None and local_rank is not None:
        # 偏置乘以1000，使不同rank的种子点，即使向后递延（如Dataloader的worker，种子点为s、s+1、···）也不会发生冲突。
        rand_seed_bias = local_rank * 1000 if isinstance(rand_seed_ori, int) else str(local_rank * 1000)
    else:
        rand_seed_bias = None

    # ** 若rand_seed_ori不为None，则设置随机种子点。若rand_seed_bias不为None，则要给种子点加上偏置。
    if rand_seed_ori is not None:
        rand_seed_final = rand_seed_ori if rand_seed_bias is None else rand_seed_ori + rand_seed_bias
        set_rand_seed(rand_seed_final)
        meow.rand_seed_final = rand_seed_final
        if verbosity:
            print(f"设置随机种子为：{rand_seed_final}" +
                  (f" = {rand_seed_ori} + {rand_seed_bias}" if rand_seed_bias is not None else ''), end="\n\n")

    # * Set reproducibility
    if reproducibility:
        # ** 检查是否设置了随机种子点。若没有设置（rand_seed为False，报错）。
        if isinstance(rand_seed, bool) and not rand_seed:
            raise ValueError(f"rand_seed must be set if reproducibility required. ")
        # ** 检查是否设置了benchmark。若设置，则与复现要求冲突，报错。
        if is_benchmark:
            raise ValueError(f"is_benchmark must be False if reproducibility required. ")
        # ** 针对不同版本pytorch，设置其尽量使用确定性算法。
        if hasattr(torch, 'set_deterministic'):
            torch.set_deterministic(True)
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        if verbosity:
            print("设置torch.backends.cudnn.deterministic = True，torch.set_deterministic(True)或"
                  "torch.use_deterministic_algorithms(True)被调用，使算法可复现。", end="\n\n")

    # * Set is_debug
    meow.is_debug = is_debug
    if verbosity:
        print(f"meow.is_debug = {is_debug}", end="\n\n")

    # * End of init
    if verbosity:
        print("\033[32m-------------------------------------- 初始化结束 --------------------------------------\033[0m",
              end="\n\n")

    # * Compute config's ItemLazy.
    if config is not None:
        print("\033[32m-------------------------------------- 计算惰性项 --------------------------------------\033[0m")
        config = ItemLazy.compute_item_lazy(config)
        config.freeze()
        print("\033[32m-------------------------------------- 惰性项算毕 --------------------------------------\033[0m",
              end="\n\n")

    # * Print config's delaying verbosity
    if verbosity and config is not None:
        print("\033[32m-------------------------------------- 配置项开始 --------------------------------------\033[0m")
        print(config)
        print("\033[32m-------------------------------------- 配置项结束 --------------------------------------\033[0m",
              end="\n\n")

    # * Return
    return device, config
