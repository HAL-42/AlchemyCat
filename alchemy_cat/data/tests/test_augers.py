#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: test_augers.py
@time: 2020/2/5 2:24
@desc:
"""
import pytest

import numpy as np
from colorama import Fore, Style
import matplotlib.pyplot as plt
import cv2

from alchemy_cat.contrib.voc import VOCAug, label_map2color_map
from alchemy_cat.data import DataAuger
from alchemy_cat.data.plugins.augers import RandMirror, MultiMirror, RandColorJitter, RandUpDown, MultiUpDown, \
    int_img2float32_img, centralize, pad_img_label, RandCrop, FiveCrop, RandScale, MultiScale
from alchemy_cat.dag import Node
from alchemy_cat.acplot.figure_wall import RowFigureWall, ColumnFigureWall
from alchemy_cat.py_tools import set_numpy_rand_seed
from alchemy_cat.alg import find_nearest_odd_size, find_nearest_even_size


def plot_img_label_pair(img_pair, label_pair=None, img_num=None):
    img_wall = RowFigureWall(img_pair, space_width=20, pad_location='center', color_channel_order='BGR')
    label_wall = RowFigureWall(label_pair, space_width=20, pad_location='center', color_channel_order='RGB') \
        if label_pair is not None else None

    show_wall = ColumnFigureWall([img_wall, label_wall], space_width=20, color_channel_order='BGR') \
        if label_wall is not None else img_wall
    show_wall.plot(num=img_num)
    plt.show()


class BaseAuger(DataAuger):
    def build_graph(self):
        @self.graph.register(inputs=['example'], outputs=['img_id', 'img', 'label'])
        def split_example(example):
            return example


@pytest.fixture(scope="module")
def voc_dataset():
    return VOCAug(split='val')


def setup_function(func):
    set_numpy_rand_seed(0)
    print(f"-----------setup function: set_numpy_rand_seed(0)-----------")


def test_rand_mirror(voc_dataset):
    """Test rand mirror with following steps:
        1. Create a rand_mirror auger on voc val dataset
        2. Test is all img is mirrored accroding to it's rand_seed
        3. Test is rand distribution is random and balanced
        4. Visualization

    Args:
        voc_dataset: VOCAug val dataset
    """
    print(Fore.LIGHTYELLOW_EX + "===========Test Rand Mirror==========" + Style.RESET_ALL)

    class RandMirrorAuger(BaseAuger):

        def build_graph(self):
            super(RandMirrorAuger, self).build_graph()

            self.graph.add_node(RandMirror, inputs=['img', 'label'], outputs=['mirrored_img', 'mirrored_label'])

    auger = RandMirrorAuger(voc_dataset, verbosity=0, slim=True)
    mirror_node: Node = auger.rand_nodes[0]

    mirrored, not_mirrored = 0, 0
    for index, outputs in enumerate(auger):
        mirrored_img, mirrored_label = outputs
        _, img, label = voc_dataset[index]

        rand_seed = auger.rand_seeds[mirror_node.id]

        assert np.all(mirrored_img[:, ::rand_seed] == img), f"Current index={index}, rand_seed={rand_seed}"
        assert np.all(mirrored_label[:, ::rand_seed] == label), f"Current index={index}, rand_seed={rand_seed}"

        if index % 100 == 0:
            print(Fore.BLUE + f"{index // 100}: rand_seed = {rand_seed}\n" + Style.RESET_ALL)
            img_pair = [img, mirrored_img]
            label_pair = map(label_map2color_map, [label, mirrored_label])

            plot_img_label_pair(img_pair, label_pair, img_num=f"ID: {_}; index: {index}")

        if rand_seed == 1:
            not_mirrored += 1
        elif rand_seed == -1:
            mirrored += 1
        else:
            raise ValueError(f"rand_seed={rand_seed} is not 1 or -1")

    assert mirrored + not_mirrored == len(voc_dataset)
    assert mirrored / len(voc_dataset) == pytest.approx(0.5, abs=0.04)
    print(Fore.LIGHTYELLOW_EX + f"mirrored: {mirrored}; not_mirrored: {not_mirrored}" + Style.RESET_ALL)


def test_weighted_rand_mirror(voc_dataset):
    """Test weighted rand mirror with following steps:
        1. Create a rand_mirror auger on voc val dataset
        2. Test is all img is mirrored accroding to it's rand_seed
        3. Test is rand distribution is random and weighted
        4. Visualization

    Args:
        voc_dataset: VOCAug val dataset
    """
    print(Fore.LIGHTYELLOW_EX + "===========Test Weighted Rand Mirror==========" + Style.RESET_ALL)

    class WeightedRandMirror(RandMirror):
        weight = [90, 10]

    class WeightedRandMirrorAuger(BaseAuger):

        def build_graph(self):
            super(WeightedRandMirrorAuger, self).build_graph()

            self.graph.add_node(WeightedRandMirror,
                                inputs=['img', 'label'], outputs=['mirrored_img', 'mirrored_label'])

    auger = WeightedRandMirrorAuger(voc_dataset, verbosity=0, slim=True)
    mirror_node: Node = auger.rand_nodes[0]

    mirrored, not_mirrored = 0, 0
    for index, outputs in enumerate(auger):
        mirrored_img, mirrored_label = outputs
        _, img, label = voc_dataset[index]

        rand_seed = auger.rand_seeds[mirror_node.id]

        assert np.all(mirrored_img[:, ::rand_seed] == img), f"Current index={index}, rand_seed={rand_seed}"
        assert np.all(mirrored_label[:, ::rand_seed] == label), f"Current index={index}, rand_seed={rand_seed}"

        if index % 100 == 0:
            print(Fore.BLUE + f"{index // 100}: rand_seed = {rand_seed}\n" + Style.RESET_ALL)
            img_pair = [img, mirrored_img]
            label_pair = map(label_map2color_map, [label, mirrored_label])

            plot_img_label_pair(img_pair, label_pair, img_num=f"ID: {_}; index: {index}")

        if rand_seed == 1:
            not_mirrored += 1
        elif rand_seed == -1:
            mirrored += 1
        else:
            raise ValueError(f"rand_seed={rand_seed} is not 1 or -1")

    assert mirrored + not_mirrored == len(voc_dataset)
    assert not_mirrored / len(voc_dataset) == pytest.approx(0.9, abs=0.04)
    print(Fore.LIGHTYELLOW_EX + f"mirrored: {mirrored}; not_mirrored: {not_mirrored}" + Style.RESET_ALL)


def test_rand_log(voc_dataset):
    """Test rand mirror with rand log following steps:
        1. Create two rand_mirror auger on voc val dataset
        2. Step0: Test is all img is mirrored accroding to it's rand_seed
        3. Step0: Test is rand distribution is random and balanced
        4. Step1: Test is all img is mirrored accroding to it's rand_seed
        5. Step1: Test is rand distribution is random and balanced
        6. Test is Step0's random is the same to step1's random
        7. Visualization

    Args:
        voc_dataset: VOCAug val dataset
    """
    print(Fore.LIGHTYELLOW_EX + "===========Test Rand Log==========" + Style.RESET_ALL)

    class RandMirrorAuger(BaseAuger):

        def build_graph(self):
            super(RandMirrorAuger, self).build_graph()

            self.graph.add_node(RandMirror, inputs=['img', 'label'], outputs=['mirrored_img', 'mirrored_label'])

    auger0 = RandMirrorAuger(voc_dataset, verbosity=0, slim=True, rand_seed_log='./Temp/rand_log')
    mirror_node0: Node = auger0.rand_nodes[0]

    rand_seed_record0 = []
    for index, outputs in enumerate(auger0):
        mirrored_img, mirrored_label = outputs
        _, img, label = voc_dataset[index]

        rand_seed = auger0.rand_seeds[mirror_node0.id]

        assert np.all(mirrored_img[:, ::rand_seed] == img), f"Current index={index}, rand_seed={rand_seed}"
        assert np.all(mirrored_label[:, ::rand_seed] == label), f"Current index={index}, rand_seed={rand_seed}"

        if index % 300 == 0:
            print(Fore.BLUE + f"{index // 100}: rand_seed = {rand_seed}\n" + Style.RESET_ALL)
            img_pair = [img, mirrored_img]
            label_pair = map(label_map2color_map, [label, mirrored_label])

            plot_img_label_pair(img_pair, label_pair, img_num=f"ID: {_}; index: {index}")

        rand_seed_record0.append(rand_seed)

    assert len(rand_seed_record0) == len(voc_dataset)
    assert rand_seed_record0.count(1) / len(voc_dataset) == pytest.approx(0.5, abs=0.04)

    auger1 = RandMirrorAuger(voc_dataset, verbosity=0, slim=True, rand_seed_log='./Temp/rand_log')
    mirror_node1: Node = auger1.rand_nodes[0]

    rand_seed_record1 = []
    for index, outputs in enumerate(auger1):
        mirrored_img, mirrored_label = outputs
        _, img, label = voc_dataset[index]

        rand_seed = auger1.rand_seeds[mirror_node1.id]

        assert np.all(mirrored_img[:, ::rand_seed] == img), f"Current index={index}, rand_seed={rand_seed}"
        assert np.all(mirrored_label[:, ::rand_seed] == label), f"Current index={index}, rand_seed={rand_seed}"

        if index % 300 == 0:
            print(Fore.BLUE + f"{index // 100}: rand_seed = {rand_seed}\n" + Style.RESET_ALL)
            img_pair = [img, mirrored_img]
            label_pair = map(label_map2color_map, [label, mirrored_label])

            plot_img_label_pair(img_pair, label_pair, img_num=f"ID: {_}; index: {index}")

        rand_seed_record1.append(rand_seed)

    assert len(rand_seed_record1) == len(voc_dataset)
    assert rand_seed_record1.count(1) / len(voc_dataset) == pytest.approx(0.5, abs=0.04)

    assert rand_seed_record0 == rand_seed_record1


def test_multi_mirror(voc_dataset):
    """Test multi mirror with following steps:
        1. Create a multi mirror auger on voc test dataset
        2. Test whether all img is mirrored accroding to it's output_index
        3. Test whether output_index is met with dataset index and auger index
        4. Test whether the total number is right
        5. Visualization

    Args:
        voc_dataset: VOCAug val dataset
    """
    print(Fore.LIGHTYELLOW_EX + "===========Test Multi Mirror==========" + Style.RESET_ALL)

    class MultiMirrorAuger(BaseAuger):

        def build_graph(self):
            super(MultiMirrorAuger, self).build_graph()

            self.graph.add_node(MultiMirror, inputs=['img', 'label'], outputs=['mirrored_img', 'mirrored_label'])

    auger = MultiMirrorAuger(voc_dataset, verbosity=0, slim=True)
    multi_node: Node = auger.multi_nodes[0]

    mirrored, not_mirrored = 0, 0
    for auger_index, outputs in enumerate(auger):
        mirrored_img, mirrored_label = outputs

        dataset_index = auger_index // 2
        _, img, label = voc_dataset[dataset_index]

        output_index = multi_node._fct.output_index
        assert output_index == auger_index - 2 * dataset_index

        mirror = 1 if output_index == 0 else -1
        assert np.all(mirrored_img[:, ::mirror] == img), f"Current index={dataset_index}, mirror={mirror}"
        assert np.all(mirrored_label[:, ::mirror] == label), f"Current dataset_index={dataset_index}, mirror={mirror}"

        if dataset_index % 99 == 0:
            print(Fore.BLUE + f"{dataset_index // 99}: mirror = {mirror}\n" + Style.RESET_ALL)
            img_pair = [img, mirrored_img]
            label_pair = map(label_map2color_map, [label, mirrored_label])

            plot_img_label_pair(img_pair, label_pair)

        if mirror == 1:
            not_mirrored += 1
        elif mirror == -1:
            mirrored += 1
        else:
            raise ValueError(f"mirror={mirror} is not 1 or -1")

    assert mirrored + not_mirrored == 2 * len(voc_dataset)
    assert mirrored == not_mirrored
    print(Fore.LIGHTYELLOW_EX + f"mirrored: {mirrored}; not_mirrored: {not_mirrored}" + Style.RESET_ALL)


def test_rand_updown(voc_dataset):
    """Test rand mirror with following steps:
        1. Create a rand updown auger on voc val dataset
        2. Test is all img is updown accroding to it's rand_seed
        3. Test is rand distribution is random and balanced
        4. Visualization

    Args:
        voc_dataset: VOCAug val dataset
    """
    print(Fore.LIGHTYELLOW_EX + "===========Test Rand UpDown==========" + Style.RESET_ALL)

    class RandUpDownAuger(BaseAuger):

        def build_graph(self):
            super(RandUpDownAuger, self).build_graph()

            self.graph.add_node(RandUpDown, inputs=['img', 'label'], outputs=['updown_img', 'updown_label'])

    auger = RandUpDownAuger(voc_dataset, verbosity=0, slim=True)
    updown_node: Node = auger.rand_nodes[0]

    updown, not_updown = 0, 0
    for index, outputs in enumerate(auger):
        updown_img, updown_label = outputs
        _, img, label = voc_dataset[index]

        rand_seed = auger.rand_seeds[updown_node.id]

        assert np.all(updown_img[::rand_seed] == img), f"Current index={index}, rand_seed={rand_seed}"
        assert np.all(updown_label[::rand_seed] == label), f"Current index={index}, rand_seed={rand_seed}"

        if index % 100 == 0:
            print(Fore.BLUE + f"{index // 100}: rand_seed = {rand_seed}\n" + Style.RESET_ALL)
            img_pair = [img, updown_img]
            label_pair = map(label_map2color_map, [label, updown_label])

            plot_img_label_pair(img_pair, label_pair)

        if rand_seed == 1:
            not_updown += 1
        elif rand_seed == -1:
            updown += 1
        else:
            raise ValueError(f"rand_seed={rand_seed} is not 1 or -1")

    assert updown + not_updown == len(voc_dataset)
    assert updown / len(voc_dataset) == pytest.approx(0.5, abs=0.04)
    print(Fore.LIGHTYELLOW_EX + f"updown: {updown}; not_updown: {not_updown}" + Style.RESET_ALL)


def test_multi_updown(voc_dataset):
    """Test multi mirror with following steps:
        1. Create a multi updown auger on voc test dataset
        2. Test whether all img is updown accroding to it's output_index
        3. Test whether output_index is met with dataset index and auger index
        4. Test whether the total number is right
        5. Visualization

    Args:
        voc_dataset: VOCAug val dataset
    """
    print(Fore.LIGHTYELLOW_EX + "===========Test Multi UpDown==========" + Style.RESET_ALL)

    class MultiUpDownAuger(BaseAuger):

        def build_graph(self):
            super(MultiUpDownAuger, self).build_graph()

            self.graph.add_node(MultiUpDown, inputs=['img', 'label'], outputs=['updown_img', 'updown_label'])

    auger = MultiUpDownAuger(voc_dataset, verbosity=0, slim=True)
    multi_node: Node = auger.multi_nodes[0]

    updown_count, not_updown_count = 0, 0
    for auger_index, outputs in enumerate(auger):
        updown_img, updown_label = outputs

        dataset_index = auger_index // 2
        _, img, label = voc_dataset[dataset_index]

        output_index = multi_node._fct.output_index
        assert output_index == auger_index - 2 * dataset_index

        updown = 1 if output_index == 0 else -1
        assert np.all(updown_img[::updown, ...] == img), f"Current index={dataset_index}, updown={updown}"
        assert np.all(updown_label[::updown, ...] == label), f"Current dataset_index={dataset_index}, updown={updown}"

        if dataset_index % 99 == 0:
            print(Fore.BLUE + f"{dataset_index // 99}: updown = {updown}\n" + Style.RESET_ALL)
            img_pair = [img, updown_img]
            label_pair = map(label_map2color_map, [label, updown_label])

            plot_img_label_pair(img_pair, label_pair)

        if updown == 1:
            not_updown_count += 1
        elif updown == -1:
            updown_count += 1
        else:
            raise ValueError(f"mirror={updown} is not 1 or -1")

    assert updown_count + not_updown_count == 2 * len(voc_dataset)
    assert updown_count == not_updown_count
    print(Fore.LIGHTYELLOW_EX + f"updown: {updown_count}; not_updown: {not_updown_count}" + Style.RESET_ALL)


def test_rand_color_jitter(voc_dataset):
    """Test RandColorJitter
        1. Create RandColorJitter Auger
        2. Reverse the jitter process accroding to rand seed, then compare to the origin img.
            - Except ignore pixel, warning if approx equal (abs=5) pixel ratio (also called reversible ratio) > 0.8
            - Except ignore pixel, macro mean reversible ratio > 0.8
        3. Test the whether the distribution is random and balanced
        4. Make sure the distortion ration is not too high under default parameter
            - Warning if distortion ratio >= 0.7 for single img.
            - Macro mean distortion ratio < 0.2
        5. Total output number should be correct
        6. Visualization

    Args:
        voc_dataset: VOCAug val dataset
    """
    print(Fore.LIGHTYELLOW_EX + "===========Test Color Jitter==========" + Style.RESET_ALL)

    class RandColorJitterAuger(BaseAuger):

        def build_graph(self):
            super(RandColorJitterAuger, self).build_graph()

            self.graph.add_node(RandColorJitter, inputs=['img'], outputs=['jitter_img'])

    auger = RandColorJitterAuger(voc_dataset, verbosity=0, slim=True)
    jitter_node: Node = auger.rand_nodes[0]

    def dejitter_bright(delta_bright, img, ignore):
        if delta_bright > 0:
            ignore |= np.any((img == 255), axis=2)
        else:
            ignore |= np.any((img == 0), axis=2)
        return cv2.convertScaleAbs(img, alpha=1, beta=(- delta_bright))

    def dejitter_contract(mul_contract, img, ignore):
        if mul_contract > 1:
            ignore |= np.any((img == 255), axis=2)
        return cv2.convertScaleAbs(img, alpha=1 / mul_contract, beta=0)

    def dejitter_saturate(mul_saturate, img, ignore):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if mul_saturate > 1:
            ignore |= np.any((img == 255), axis=2)
        img[:, :, 1] = cv2.convertScaleAbs(img[:, :, 1], alpha=(1 / mul_saturate), beta=0)
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    def dejitter_hue(delta_hue, img, ignore):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
        if delta_hue > 0:
            ignore |= np.any((img == 255), axis=2)
        if delta_hue < 0:
            ignore |= np.any((img == 0), axis=2)
        img[:, :, 0] = cv2.convertScaleAbs(img[:, :, 0], alpha=1, beta=(- delta_hue))
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR_FULL)

    sum_order = np.zeros(4, dtype=np.int64)
    sum_activated = np.zeros(4, dtype=np.int64)

    distortion_ratios = []
    reversible_ratios = []
    for index, jitter_img in enumerate(auger):
        _, img, _ = voc_dataset[index]

        rand_seed = auger.rand_seeds[jitter_node.id]
        jitter_order = rand_seed['jitter_order']
        sum_order += np.array(jitter_order, dtype=np.int64)

        # * Dejitter
        distortion = np.zeros(img.shape[:2], dtype=np.bool_)
        dejitter_img = jitter_img
        delta_bright, mul_contract, mul_saturate, delta_hue = \
            map(lambda k: rand_seed.get(k), ['delta_bright', 'mul_contract', 'mul_saturate', 'delta_hue'])

        for order in jitter_order[::-1]:
            if order == 0 and delta_bright is not None:
                sum_activated[order] += 1
                dejitter_img = dejitter_bright(delta_bright, dejitter_img, distortion)
            elif order == 1 and mul_contract is not None:
                sum_activated[order] += 1
                dejitter_img = dejitter_contract(mul_contract, dejitter_img, distortion)
            elif order == 2 and mul_saturate is not None:
                sum_activated[order] += 1
                dejitter_img = dejitter_saturate(mul_saturate, dejitter_img, distortion)
            elif order == 3 and delta_hue is not None:
                sum_activated[order] += 1
                dejitter_img = dejitter_hue(delta_hue, dejitter_img, distortion)

        # * Visualization
        def vis(problem):
            print(Fore.LIGHTYELLOW_EX + f"{index} [{problem}]: rand_seed = {rand_seed}" + Style.RESET_ALL)
            plot_img_label_pair([img, jitter_img, dejitter_img])

        # * Test distortion ratio
        distortion_ratio = np.sum(distortion) / distortion.size
        if distortion_ratio >= 0.7:
            vis(f"distortion_ratio = {distortion_ratio} >= 0.7")
        # assert distortion_ratio < 0.7
        distortion_ratios.append(distortion_ratio)

        # * Test reversible ratio
        reversible_without_distortion = np.all(np.isclose(dejitter_img, img, atol=5.0, rtol=0), axis=2) & (~distortion)
        reversible_ratio = np.sum(reversible_without_distortion) / (distortion.size - np.sum(distortion))
        if reversible_ratio <= 0.65:
            vis(f"reversible_ratio = {reversible_ratio} <= 0.65")
        # assert reversible_ratio > 0.65
        reversible_ratios.append(reversible_ratio)

    # * Test total num
    assert len(distortion_ratios) == len(voc_dataset)

    # * Test whether distribution is random and balanced
    assert sum_activated / len(voc_dataset) == pytest.approx(0.5, abs=0.04)
    assert sum_order / len(voc_dataset) == pytest.approx(1.5, abs=0.04)

    # * Test macro mean distortion and reversible ratio
    assert np.mean(distortion_ratios) < 0.2
    assert np.nanmean(reversible_ratios) > 0.8


def test_int_img2float32_img(voc_dataset):
    """Test int img converting to float32 img
        1. Test output img is float32
        2. Test output img has the same value as int img

    Args:
        voc_dataset: VOC Aug val dataset
    """
    _, img, _ = voc_dataset[0]

    float_img = int_img2float32_img(img)
    assert float_img.dtype == np.float32
    assert np.all(float_img.astype(np.uint8) == img)


def test_centralize(voc_dataset):
    """Test int img converting to float32 img
        1. centralized img could be decentralized correctly accroding to mean and std

    Args:
        voc_dataset: VOC Aug val dataset
    """
    _, img, _ = voc_dataset[0]

    float_img = int_img2float32_img(img)
    centralized_img = centralize(float_img, voc_dataset.mean_bgr, std=(2.2, 3.5, 4.5))

    decentralized_img = \
        (centralized_img * np.array([2.2, 3.5, 4.5], dtype=np.float32)
         + np.array(voc_dataset.mean_bgr, dtype=np.float32))

    assert np.all(decentralized_img.astype(np.uint8) == pytest.approx(img, abs=1.0))


def test_pad_img_label(voc_dataset):
    """Test pad img and label. The padded size, location and pad value should be correct under following circumstances:
        1. pad size is larger or smaller than img size at H or W
        2. img pad value is number(int, float) or scalar
        3. img pad to size is int or Iterable(H, W)
        4. with or without pad aligner
        5. with or without label
        6. Different pad locations
    Args:
        voc_dataset: VOC Aug val dataset
    """
    print(Fore.LIGHTYELLOW_EX + "===========Test Pad Img Label==========" + Style.RESET_ALL)

    _, img, label = voc_dataset[0]

    def _check_padded_img_label(img, padded_img, padded_h, padded_w, padded_val: np.ndarray,
                                label=None, padded_label=None):
        # * img, label share the same size
        if label is not None:
            assert img.shape[:2] == label.shape
            assert padded_img.shape[:2] == padded_label.shape

        # * padded size is correct
        assert padded_img.shape[0] == padded_h
        assert padded_img.shape[1] == padded_w

        # * left-top part is the same
        assert np.all(padded_img[:img.shape[0], :img.shape[1]] == img)
        if label is not None:
            assert np.all(padded_label[:label.shape[0], :label.shape[1]] == label)

        # * padded value is correct
        img_pad_right = np.zeros(padded_img.shape[:2], dtype=np.bool_)
        img_pad_right[:img.shape[0], :img.shape[1]] = True
        img_pad_right |= np.all(np.isclose(padded_img, padded_val), axis=2)
        assert np.all(img_pad_right)

        if label is not None:
            label_pad_right = np.zeros(padded_label.shape[:2], dtype=np.bool_)
            label_pad_right[:label.shape[0], :label.shape[1]] = True
            label_pad_right |= padded_label == 255
            assert np.all(label_pad_right)

    # * Test diff pad size
    for delta_h in [-10, 10]:
        for delta_w in [-10, 10]:
            pad_img_to = (img.shape[0] + delta_h, img.shape[1] + delta_w)
            img_pad_val = np.array([1, 2, 3])
            padded_img, padded_label = pad_img_label(img, label, pad_img_to, img_pad_val=img_pad_val)
            padded_h = pad_img_to[0] if delta_h > 0 else img.shape[0]
            padded_w = pad_img_to[1] if delta_w > 0 else img.shape[1]
            _check_padded_img_label(img, padded_img, padded_h, padded_w, img_pad_val, label, padded_label)

    # * Test pad int to int size with aligner
    padded_img, padded_label = pad_img_label(img, label, pad_img_to=511, img_pad_val=0,
                                             pad_aligner=find_nearest_odd_size)
    _check_padded_img_label(img, padded_img, 513, 513, np.array([0, 0, 0]), label, padded_label)

    padded_img, padded_label = pad_img_label(img, label, pad_img_to=511, img_pad_val=0,
                                             pad_aligner=[find_nearest_odd_size, find_nearest_even_size])
    _check_padded_img_label(img, padded_img, 513, 512, np.array([0, 0, 0]), label, padded_label)

    # * Test pad float without label
    float_img = int_img2float32_img(img)
    padded_img = pad_img_label(float_img, pad_img_to=513, img_pad_val=np.array([2.3, 3.3, 6.6]))
    assert padded_img.dtype is np.dtype('float32')
    _check_padded_img_label(float_img, padded_img, 513, 513, np.array([2.3, 3.3, 6.6]))

    # * Test Different pad location
    padded_img, padded_label = pad_img_label(img, label, pad_img_to=513, img_pad_val=128, pad_location='left-top')
    assert np.all(padded_img[-img.shape[0]:, -img.shape[1]:] == img)
    assert np.all(padded_label[-img.shape[0]:, -img.shape[1]:] == label)

    padded_img, padded_label = pad_img_label(img, label, pad_img_to=513, img_pad_val=128, pad_location='left-bottom')
    assert np.all(padded_img[:img.shape[0], -img.shape[1]:] == img)
    assert np.all(padded_label[:label.shape[0], -label.shape[1]:] == label)

    padded_img, padded_label = pad_img_label(img, label, pad_img_to=513, img_pad_val=128, pad_location='right-top')
    assert np.all(padded_img[-img.shape[0]:, :img.shape[1]] == img)
    assert np.all(padded_label[-label.shape[0]:, :label.shape[1]] == label)

    padded_img, padded_label = pad_img_label(img, label, pad_img_to=513, img_pad_val=128, pad_location='center')
    pad_h = 513 - img.shape[0]
    pad_w = 513 - img.shape[1]
    assert np.all(padded_img[pad_h // 2:-((pad_h + 1) // 2), pad_w // 2:-((pad_w + 1) // 2)] == img)
    assert np.all(padded_label[pad_h // 2:-((pad_h + 1) // 2), pad_w // 2:-((pad_w + 1) // 2)] == label)


def test_rand_crop(voc_dataset):
    """Test RandCrop
        1. Create RandCrop Auger
        2. Test whether the cropped img and label corresponding to the rand seed
        3. Test error raised when crop size > img size
        4. Test the rand seed distribution is random and balanced
        5. Visualization

    Args:
        voc_dataset: VOCAug val dataset
    """
    print(Fore.LIGHTYELLOW_EX + "===========Test Rand Crop==========" + Style.RESET_ALL)

    kPadSize = 513
    kCropSize = (321, 257)
    kErrorCropSize = (323, 258)

    kLoopTimes = 10000

    class RandCropAuger(BaseAuger):

        def build_graph(self):
            super(RandCropAuger, self).build_graph()
            self.graph.add_node(pad_img_label, inputs=['img', 'label'], outputs=['padded_img', 'padded_label'],
                                kwargs={'pad_img_to': kPadSize, 'img_pad_val': 128})

            self.graph.add_node(RandCrop, inputs=['padded_img', 'padded_label'],
                                outputs=['cropped_img', 'cropped_label'], init={'crop_size': kCropSize})

    auger = RandCropAuger(voc_dataset, verbosity=0, slim=True)
    rand_node: Node = auger.rand_nodes[0]

    def _check_cropped_img_label(img, cropped_img, offset_h, offset_w, label=None, cropped_label=None):
        # * img, label share the same size
        if label is not None:
            assert img.shape[:2] == label.shape
            assert cropped_img.shape[:2] == cropped_label.shape

        # * padded size is correct
        assert cropped_img.shape[0] == kCropSize[0]
        assert cropped_img.shape[1] == kCropSize[1]

        # * cropped part is the same
        assert np.all(img[offset_h: offset_h + kCropSize[0], offset_w: offset_w + kCropSize[1]] == cropped_img)
        if label is not None:
            assert np.all(label[offset_h: offset_h + kCropSize[0], offset_w: offset_w + kCropSize[1]] == cropped_label)

    _, img, label = voc_dataset[0]
    padded_img, padded_label = pad_img_label(img, label, kPadSize, img_pad_val=128)

    offset_h_sum, offset_w_sum = np.zeros(padded_img.shape[0] - kCropSize[0] + 1, dtype=np.int64), \
                                 np.zeros(padded_img.shape[1] - kCropSize[1] + 1, dtype=np.int64)
    for i in range(kLoopTimes):
        cropped_img, cropped_label = auger[0]
        offset_h, offset_w = auger.rand_seeds[rand_node.id]

        offset_h_sum[offset_h] += 1
        offset_w_sum[offset_w] += 1

        _check_cropped_img_label(padded_img, cropped_img, offset_h, offset_w, padded_label, cropped_label)

        if i % 500 == 0:
            print(Fore.LIGHTYELLOW_EX + f"{i}: offset_h = {offset_h}; offset_w = {offset_w}" + Style.RESET_ALL)
            img_pair = [padded_img, cropped_img]
            label_pair = map(label_map2color_map, [padded_label, cropped_label])
            plot_img_label_pair(img_pair, label_pair)

    assert offset_h_sum == pytest.approx(kLoopTimes / (kPadSize - kCropSize[0] + 1), rel=0.5)
    assert offset_w_sum == pytest.approx(kLoopTimes / (kPadSize - kCropSize[1] + 1), rel=0.5)

    auger.graph.add_node(RandCrop, inputs=['cropped_img'], outputs=['wrong_cropped_img'],
                         init={'crop_size': kErrorCropSize})
    with pytest.raises(ValueError, match=f"img_h {kCropSize[0]} must >= crop_h {kErrorCropSize[0]}; " \
                     f"img_w {kCropSize[1]} must >= crop_w {kErrorCropSize[1]}"):
        _ = auger[0]


def test_five_crop(voc_dataset):
    """Test Five Crop with following steps:
        1. Create a FiveCrop auger on voc test dataset
        2. Test whether all img is cropped accroding to it's output_index
        3. Test whether output_index is met with dataset index and auger index
        4. Test whether the total number is right
        5. Visualization

    Args:
        voc_dataset: VOCAug val dataset
    """
    print(Fore.LIGHTYELLOW_EX + "===========Test Rand Crop==========" + Style.RESET_ALL)

    kCropH, kCropW = 513, 321
    kCropSize = (kCropH, kCropW)

    class FiveCropAuger(BaseAuger):

        def build_graph(self):
            super(FiveCropAuger, self).build_graph()

            self.graph.add_node(pad_img_label, inputs=['img', 'label'], outputs=['padded_img', 'padded_label'],
                                kwargs={'pad_img_to': kCropSize, 'img_pad_val': (173, 68, 142),
                                        'pad_location': 'center'})

            self.graph.add_node(FiveCrop, inputs=['padded_img', 'padded_label'],
                                outputs=['cropped_img', 'cropped_label'], init={'crop_size': kCropSize})

    auger = FiveCropAuger(voc_dataset, verbosity=0, slim=True)
    multi_node: Node = auger.multi_nodes[0]

    index_num = np.zeros(5, dtype=np.int64)
    for auger_index, outputs in enumerate(auger):
        dataset_index = auger_index // 5

        cropped_img, cropped_label = outputs
        padded_img, padded_label = auger.graph.data['padded_img'], auger.graph.data['padded_label']
        _, img ,label = voc_dataset[dataset_index]

        output_index = multi_node._fct.output_index
        assert output_index == auger_index % 5

        if output_index == 0:
            assert np.all(cropped_img == padded_img[:kCropH, :kCropW])
            assert np.all(cropped_label == padded_label[:kCropH, :kCropW])
        elif output_index == 1:
            assert np.all(cropped_img == padded_img[:kCropH, -kCropW:])
            assert np.all(cropped_label == padded_label[:kCropH, -kCropW:])
        elif output_index == 2:
            assert np.all(cropped_img == padded_img[-kCropH:, :kCropW])
            assert np.all(cropped_label == padded_label[-kCropH:, :kCropW])
        elif output_index == 3:
            assert np.all(cropped_img == padded_img[-kCropH:, -kCropW:])
            assert np.all(cropped_label == padded_label[-kCropH:, -kCropW:])
        elif output_index == 4:
            head_H = (padded_img.shape[0] - kCropH) // 2
            tail_H = -((padded_img.shape[0] - kCropH + 1) // 2) if head_H != 0 else kCropH
            head_W = (padded_img.shape[1] - kCropW) // 2
            tail_W = -((padded_img.shape[1] - kCropW + 1) // 2) if head_W != 0 else kCropW

            assert np.all(cropped_img == padded_img[head_H:tail_H, head_W:tail_W])
            assert np.all(cropped_label == padded_label[head_H:tail_H, head_W:tail_W])
        else:
            raise ValueError(f"output_index{output_index} not in [0, 1, 2, 3, 4]")

        if dataset_index % 100 == 0:
            print(Fore.BLUE + f"{dataset_index // 100}: output_index = {output_index}\n" + Style.RESET_ALL)
            img_row = [img, padded_img, cropped_img]
            label_row = map(label_map2color_map, [label, padded_label, cropped_label])

            plot_img_label_pair(img_row, label_row)

        index_num[output_index] += 1

    assert np.all(index_num == len(voc_dataset))
    print(Fore.LIGHTYELLOW_EX + f"index_num: {index_num}" + Style.RESET_ALL)


def test_rand_scale(voc_dataset):
    """Test rand scale with following steps:
        1. Create a rand_scale auger on voc val dataset
        2. Test is all img is scale accroding to it's rand_seed
            a) The scaled size is correct
            b) Img and label can be roughly recovered by rescale
        3. Test is rand distribution is random and balanced
        4. Visualization

    Args:
        voc_dataset: VOCAug val dataset
    """
    print(Fore.LIGHTYELLOW_EX + "===========Test Rand Scale==========" + Style.RESET_ALL)

    kScaleFactors = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]

    class RandScaleAuger(BaseAuger):

        def build_graph(self):
            super(RandScaleAuger, self).build_graph()

            self.graph.add_node(RandScale, inputs=['img', 'label'], outputs=['scaled_img', 'scaled_label'],
                                init={'scale_factors': kScaleFactors,
                                      'aligner': [lambda x: x + 1, lambda x: x - 1]})

    auger = RandScaleAuger(voc_dataset, verbosity=0, slim=True)
    rand_node: Node = auger.rand_nodes[0]

    def vis():
        img_pair = [img, scaled_img, recovered_img]
        label_pair = map(label_map2color_map, [label, scaled_label, recovered_label])
        plot_img_label_pair(img_pair, label_pair)

    scale_num = np.zeros(len(kScaleFactors), dtype=np.int64)
    img_recover_ratios, label_recover_ratios = [], []
    for index, outputs in enumerate(auger):
        scaled_img, scaled_label = outputs
        assert scaled_img.shape[:2] == scaled_label.shape

        _, img, label = voc_dataset[index]
        img: np.ndarray; label: np.ndarray

        rand_seed = auger.rand_seeds[rand_node.id]
        scale_num[kScaleFactors.index(rand_seed)] += 1

        assert int(img.shape[0] * rand_seed) + 1 == scaled_img.shape[0]
        assert int(img.shape[1] * rand_seed) - 1 == scaled_img.shape[1]

        recovered_img = cv2.resize(scaled_img, img.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)
        img_recover_mask = np.isclose(recovered_img, img, rtol=0.1, atol=5.0)
        img_recover_ratios.append(np.sum(img_recover_mask) / img.size)

        recovered_label = cv2.resize(scaled_label, label.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        label_recover_mask = recovered_label == label
        label_recover_ratios.append(np.sum(label_recover_mask) / label.size)

        if img_recover_ratios[-1] < 0.7:
            print(Fore.LIGHTRED_EX +
                  f"img recovery fail. {index}: rand_seed = {rand_seed}\n"
                  + Style.RESET_ALL)
            vis()

        if label_recover_ratios[-1] < 0.7:
            print(Fore.LIGHTRED_EX +
                  f"label recovery fail. {index}: rand_seed = {rand_seed}\n"
                  + Style.RESET_ALL)
            vis()

        if index % 200 == 0:
            print(Fore.LIGHTRED_EX +
                  f"{index // 100}: rand_seed = {rand_seed}\n"
                  + Style.RESET_ALL)
            vis()

    assert np.sum(scale_num) == len(voc_dataset)
    assert scale_num / len(voc_dataset) == pytest.approx(1 / len(kScaleFactors), abs=0.1)

    macro_img_recover_ratio = np.mean(img_recover_ratios)
    macro_label_recover_ratio = np.mean(label_recover_ratios)

    assert macro_img_recover_ratio > 0.7
    assert macro_label_recover_ratio > 0.7

    print(Fore.LIGHTYELLOW_EX +
          f"scale_num = {scale_num}; "
          f"macro_img_recover_ratio = {macro_img_recover_ratio}; "
          f"macro_label_recover_ratio = {macro_label_recover_ratio}"
          + Style.RESET_ALL)


def test_multi_scale(voc_dataset):
    """Test multi scale with following steps:
        1. Create a multi scale auger on voc val dataset
        2. Test whether all img is scaled accroding to it's output_index
            a) The scaled size is correct
            b) Img and label can be roughly recovered by rescale
        3. Test whether output_index is met with dataset index and auger index
        4. Test whether the total number is right
        5. Visualization

    Args:
        voc_dataset: VOCAug val dataset
    """
    print(Fore.LIGHTYELLOW_EX + "===========Test Multi Scale==========" + Style.RESET_ALL)

    kScaleFactors = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]

    class MultiScaleAuger(BaseAuger):

        def build_graph(self):
            super(MultiScaleAuger, self).build_graph()

            self.graph.add_node(MultiScale, inputs=['img', 'label'], outputs=['scaled_img', 'scaled_label'],
                                init={'scale_factors': kScaleFactors,
                                      'aligner': [lambda x: x + 1, lambda x: x - 1], 'align_corner': True})

    auger = MultiScaleAuger(voc_dataset, verbosity=0, slim=True)
    multi_node: Node = auger.multi_nodes[0]

    scale_factor_count = [0] * len(kScaleFactors)
    for auger_index, outputs in enumerate(auger):
        scaled_img, scaled_label = outputs
        assert scaled_label.shape == scaled_img.shape[:2]

        dataset_index = auger_index // len(kScaleFactors)
        _, img, label = voc_dataset[dataset_index]

        output_index = multi_node._fct.output_index
        assert output_index == auger_index - len(kScaleFactors) * dataset_index

        scale_factor = multi_node._fct.scale_factors[output_index]
        assert int(img.shape[0] * scale_factor) + 1 == scaled_img.shape[0]
        assert int(img.shape[1] * scale_factor) - 1 == scaled_img.shape[1]

        if dataset_index % 100 == 0:
            print(Fore.BLUE + f"{dataset_index // 100}: scaled_factor = {scale_factor}\n" + Style.RESET_ALL)
            img_pair = [img, scaled_img]
            label_pair = map(label_map2color_map, [label, scaled_label])

            plot_img_label_pair(img_pair, label_pair)

        scale_factor_count[kScaleFactors.index(scale_factor)] += 1

    assert np.sum(scale_factor_count) == len(kScaleFactors) * len(voc_dataset)
    assert scale_factor_count == [len(voc_dataset)] * len(kScaleFactors)
    print(Fore.LIGHTYELLOW_EX + f"scale_factor_count = {scale_factor_count}" + Style.RESET_ALL)
