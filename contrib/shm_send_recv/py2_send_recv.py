#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2021/7/30 21:22
@File    : py2_send_recv.py
@Software: PyCharm
@Desc    :
"""
import os
import os.path as osp
import pickle
import glob
from time import sleep

import numpy as np

__all__ = ['client_send', 'server_send', 'client_recv', 'server_recv']


kBufferDirPrefix = 'Temp' if os.name == 'nt' else osp.join('/dev', 'shm')
kFileName = '0'

kSleepTime = .0001  # 0.1ms

kClientBufferDirSuffix = 'client_buffer'
kServerBufferDirSuffix = 'server_buffer'

kLockFile = osp.join(kBufferDirPrefix, 'server.lock')


class TimeoutError(OSError):
    pass


def _makedirs(name, exist_ok=False):
    if not osp.isdir(name):
        os.makedirs(name)
    elif not exist_ok:
        raise OSError("%s already exits." % name)
    else:
        pass


def _save_obj_sent(obj_file, obj):
    if isinstance(obj, np.ndarray):
        obj_file = obj_file + '.npy'
        np.save(obj_file, obj)
    else:
        obj_file = obj_file + '.pkl'
        with open(obj_file, 'wb') as pkl_f:
            pickle.dump(obj, pkl_f, protocol=2)

    return obj_file


def _load_obj_recv(obj_file):
    file_type = osp.splitext(obj_file)[-1]
    if file_type == '.npy':
        obj = np.load(obj_file)
    elif file_type == '.pkl':
        with open(obj_file, 'rb') as pkl_f:
            obj = pickle.load(pkl_f)
    else:
        raise RuntimeError("Receive unsupported file type %s" % file_type)

    return obj


def shm_send(obj, buffer_dir_suffix, time_out=-1):
    """Send object to buffer in share memory.

    Args:
        obj: object to be send.
        buffer_dir_suffix: buffer dir name.
        time_out: Max time to wait, 0.1ms as unit. If time_out < 0, then can wait for any longer. (Default: -1)

    Returns: Object send path.
    """
    buffer_dir = osp.join(kBufferDirPrefix, buffer_dir_suffix)
    _makedirs(buffer_dir, exist_ok=True)

    signal_file = osp.join(buffer_dir, kFileName + '.signal')
    obj_file = osp.join(buffer_dir, kFileName)

    wait_times = 0
    while osp.isfile(signal_file):
        sleep(kSleepTime)
        wait_times += 1
        if 0 <= time_out < wait_times:
            raise TimeoutError("Send option has waited for %d times" % wait_times)

    _save_obj_sent(obj_file, obj)
    with open(signal_file, mode='w'):
        pass

    return obj_file


def shm_recv(buffer_dir_suffix, time_out=-1):
    """Receive object from buffer in share memory.

    Args:
        buffer_dir_suffix: buffer dir name.
        time_out: Max time to wait, 0.1ms as unit. If time_out < 0, then can wait for any longer. (Default: -1)

    Returns: Received object.
    """
    buffer_dir = osp.join(kBufferDirPrefix, buffer_dir_suffix)
    signal_file = osp.join(buffer_dir, kFileName + '.signal')

    wait_times = 0
    while not osp.isfile(signal_file):
        sleep(kSleepTime)
        wait_times += 1
        if 0 <= time_out < wait_times:
            raise TimeoutError("Recv option has waited for %d times" % wait_times)

    ls = glob.glob(osp.join(buffer_dir, kFileName) + '.*')
    ls.remove(signal_file)
    assert len(ls) == 1
    obj_file = ls[0]

    obj = _load_obj_recv(obj_file)

    os.remove(obj_file)
    os.remove(signal_file)

    return obj


def client_send(obj, time_out=-1):
    """Send object to buffer in client buffer.

    Args:
        obj: object to be send.
        time_out: Max time to wait, 0.1ms as unit. If time_out < 0, then can wait for any longer. (Default: -1)

    Returns: Img send path.
    """
    return shm_send(obj, buffer_dir_suffix=kClientBufferDirSuffix, time_out=time_out)


def server_send(obj, time_out=-1):
    """Send object to buffer in server buffer.

    Args:
        obj: object to be send.
        time_out: Max time to wait, 0.1ms as unit. If time_out < 0, then can wait for any longer. (Default: -1)

    Returns: Img send path.
    """
    return shm_send(obj, buffer_dir_suffix=kServerBufferDirSuffix, time_out=time_out)


def client_recv(time_out=-1):
    """Receive object from server buffer.

    Args:
        time_out: Max time to wait, 0.1ms as unit. If time_out < 0, then can wait for any longer. (Default: -1)

    Returns: Received object.
    """
    return shm_recv(kServerBufferDirSuffix, time_out=time_out)


def server_recv(time_out=-1):
    """Receive object from client buffer.

    Args:
        time_out: Max time to wait, 0.1ms as unit. If time_out < 0, then can wait for any longer. (Default: -1)

    Returns: Received object.
    """
    return shm_recv(kClientBufferDirSuffix, time_out=time_out)
