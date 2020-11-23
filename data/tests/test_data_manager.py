#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: test_data_manager.py
@time: 2020/3/21 13:31
@desc:
"""
import pytest

import numpy as np
import sys
import torch

from alchemy_cat.contrib.voc import VOCAug, VOCTrainAuger
from alchemy_cat.data import DataManager, Subset
from alchemy_cat.py_tools.random import set_rand_seed
from alchemy_cat.data.tests.voc2012_auger_with_sub_graph import VOCTrainAuger as VOCTrainAugerSubGraph


sub_voc_aug = Subset(VOCAug(split='val'), list(range(50)))
sub_voc_aug.mean_bgr = VOCAug.mean_bgr
sub_voc_aug.ignore_label = VOCAug.ignore_label

voc_auger = VOCTrainAuger(sub_voc_aug, slim=True,
                          scale_factors=(0.5, 1.0), is_multi_mirror=True, is_color_jitter=True)
voc_auger_sub_graph = VOCTrainAugerSubGraph(sub_voc_aug, slim=True,
                                            scale_factors=(0.5, 1.0), is_multi_mirror=True, is_color_jitter=True)

if sys.platform == 'win32':
    voc_data_manager = DataManager(dataset=sub_voc_aug, data_auger=voc_auger, log_dir='./Temp/test_data_manager',
                                   batch_size=10, shuffle=True)
else:
    voc_data_manager = DataManager(dataset=sub_voc_aug, data_auger=voc_auger, log_dir='./Temp/test_data_manager',
                                   batch_size=10, shuffle=True, num_workers=4)

if sys.platform == 'win32':
    voc_data_manager_sub_graph = DataManager(dataset=sub_voc_aug, data_auger=voc_auger_sub_graph,
                                             log_dir='./Temp/test_data_manager/test_sub_graph',
                                             batch_size=10, shuffle=True)
else:
    voc_data_manager_sub_graph = DataManager(dataset=sub_voc_aug, data_auger=voc_auger_sub_graph,
                                             log_dir='./Temp/test_data_manager/test_sub_graph',
                                             batch_size=10, shuffle=True, num_workers=4)


def setup_function(func):
    set_rand_seed(0)
    print(f"-----------setup function: set_rand_seed(0)-----------")


@pytest.fixture(scope='module')
def data_manager():
    return voc_data_manager


@pytest.fixture(scope='function', params=[voc_data_manager, voc_data_manager_sub_graph], ids=['no_sub', 'with_sub'])
def data_manager_with_sub_graph(request):
    return request.param


def test_recover_and_recreate(data_manager_with_sub_graph):
    """Test Recover and Recreate

        * Run 15 iter with 'start_epoch', record their imgs
        * Move to iter0. Run 30 iter with 'next_batch', test whether the first 150 iter have the same img. Record imgs.
        * Restart at iter15 with 'start_epoch', run 15 iter. Test whether they are the same to recorded img.
        * Move to iter15 with and run 15 iter in recreate mode with 'next_batch'. Test whether the first 5 iter are
          the same to the record img while the last not.

    Args:
        data_manager: voc data manager with 1 epoch == 10 iteration
    """
    record_imgs = []

    epoch_iter = data_manager_with_sub_graph.start_epoch()
    for i in range(15):
        try:
            batch = next(epoch_iter)
        except StopIteration:
            epoch_iter = data_manager_with_sub_graph.start_epoch()
            batch = next(epoch_iter)
        img_ids, imgs, labels = batch
        imgs = imgs.numpy()

        record_imgs.append(imgs)

    data_manager_with_sub_graph.move_epoch_to(0, 0)
    for i in range(30):
        batch = data_manager_with_sub_graph.next_batch()
        img_ids, imgs, labels = batch
        imgs = imgs.numpy()

        if i < 15:
            assert np.all(record_imgs[i] == imgs)
        else:
            record_imgs.append(imgs)

    epoch_iter = data_manager_with_sub_graph.start_epoch(15)
    for i in range(15, 30):
        try:
            batch = next(epoch_iter)
        except StopIteration:
            epoch_iter = data_manager_with_sub_graph.start_epoch()
            batch = next(epoch_iter)
        img_ids, imgs, labels = batch
        imgs = imgs.numpy()

        assert np.all(record_imgs[i] == imgs)

    data_manager_with_sub_graph.move_epoch_to(1, 5)
    for i in range(15, 30):
        batch = data_manager_with_sub_graph.next_batch(recreate=True)
        img_ids, imgs, labels = batch
        imgs = imgs.numpy()

        if i < 20:
            assert np.all(record_imgs[i] == imgs)
        else:
            assert np.any(record_imgs[i] != imgs)


def test_batches2indices(data_manager):
    data_manager.move_epoch_to(0, 0)
    batches = data_manager.epoch_batches
    indices = data_manager.batches2indices(batches)

    flatten_batches = np.array(batches).ravel().tolist()
    assert indices == flatten_batches
    assert data_manager.epoch_indices == indices
    indices.sort()
    assert indices == list(range(100))


def test_batch_num(data_manager):
    assert len(data_manager) == 10
    assert data_manager.batch_num == 10


def test_backtrace(data_manager):
    ret = data_manager.backtrace(0, epoch_loc=0)

    data_manager.move_epoch_to(0, 0)
    batch = data_manager.next_batch(recreate=True)
    img_ids, imgs, labels = batch
    img = imgs.numpy()[0]

    assert np.all(ret['auger_output'][1] == img)
    print(ret)


@pytest.mark.skipif('sys.platform == "win32"')
def test_worker_init_fn():
    def worker_ini_fn(id):
        print(f"WorkerID: {id} with torch seed: {torch.initial_seed()}")

    data_manager = DataManager(dataset=sub_voc_aug, data_auger=voc_auger, log_dir='./Temp/test_data_manager',
                                   batch_size=10, shuffle=True, num_workers=4, worker_init_fn=worker_ini_fn)

    epoch_iter = data_manager.start_epoch()
    for i in range(15):
        try:
            batch = next(epoch_iter)
        except StopIteration:
            epoch_iter = data_manager.start_epoch()
            batch = next(epoch_iter)
