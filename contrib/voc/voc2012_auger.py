#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: voc2012_auger.py
@time: 2020/3/21 13:41
@desc:
"""
from typing import Union, Iterable

import numpy as np
import matplotlib.pyplot as plt

from alchemy_cat.data import DataAuger
from alchemy_cat.acplot import HWC2CHW, CHW2HWC
from alchemy_cat.data.plugins.augers import RandColorJitter, RandMirror, RandScale, RandCrop, pad_img_label, \
    int_img2float32_img, MultiMirror, centralize
from alchemy_cat.acplot.figure_wall import RowFigureWall, ColumnFigureWall
from alchemy_cat.contrib.voc.voc2012seg import VOC, VOCAug, label_map2color_map
from alchemy_cat.contrib.voc.utils import VOC_CLASSES

__all__ = ['VOCTrainAuger', 'VOCClsTrainAuger', 'VOCTestAuger', 'VOCClsTestAuger', 'attach_cls', 'collect_example',
           'lb2cls_lb']


def lb2cls_lb(label: np.ndarray) -> np.ndarray:
    return (np.bincount(label.ravel(), minlength=VOC.ignore_label + 1) != 0).astype(np.uint8)[:len(VOC_CLASSES)]


def attach_cls(example):
    img_id, img, label = example
    cls_in_label = lb2cls_lb(label)
    return img_id, img, label, cls_in_label


def collect_example(img_id, img, label):
    return img_id, img.copy(), label.copy() # Given continuous arr


class _VOCBaseAuger(DataAuger):
    def build_graph(self):
        @self.graph.register(inputs=['example'], outputs=['img_id', 'img', 'label'])
        def split_example(example):
            return example


class VOCTrainAuger(_VOCBaseAuger):
    """DataAuger for VOC Dataset when training

    The input img can be augered by random color jittering, random scaling, random crop and mirroring.
    """

    def __init__(self, dataset: Union[VOC, VOCAug],
                 verbosity: int=0, pool_size: int=0, slim: bool=False, rand_seed_log: str=None,
                 is_multi_mirror: bool=False, is_color_jitter: bool=False,
                 scale_factors: Iterable=(1.0), crop_size: Union[int, Iterable[int]]=321):
        """DataAuger for VOC Dataset when training

        Args:
            dataset: VOC or VOCAug dataset
            verbosity: See alchemy_cat.data.DataAuger
            pool_size: See alchemy_cat.data.DataAuger
            slim: See alchemy_cat.data.DataAuger
            rand_seed_log: See alchemy_cat.data.DataAuger
            is_multi_mirror: If False, img and label will be random mirrored, else both mirrored and not mirrored
                img/label will be given in augmentation.(Default: False)
            is_color_jitter: If True, use color jitter the input img. (Default: False)
            scale_factors: Scale factors for random scale. eg. [0.5, 1, 1.5] means the img (and label) will be
                rand scale with factor 0.5, 1.0 or 1.5(Default: (1.0))
            crop_size: Crop size. If size is int, the crop_height=value, crop_width=value. Else will be parsed as
                crop_height=list(value)[0], crop_width=list(value)[1]

        See Also:
            alchemy_cat.data.DataAuger
        """
        self.is_multi_mirror = is_multi_mirror
        self.is_color_jitter = is_color_jitter
        self.scale_factors = scale_factors
        self.crop_size = crop_size

        super(VOCTrainAuger, self).__init__(dataset, verbosity, pool_size, slim, rand_seed_log)

    def build_graph(self):
        super(VOCTrainAuger, self).build_graph()

        # * Color jittering
        rand_color_jitter = RandColorJitter() if self.is_color_jitter else RandColorJitter(jitter_prob=0.0)
        self.graph.add_node(rand_color_jitter, inputs=['img'], outputs=['jitter_img'])

        # * Rand scale
        rand_scale = RandScale(self.scale_factors)
        self.graph.add_node(rand_scale, inputs=['jitter_img', 'label'], outputs=['scaled_img', 'scaled_label'])

        self.graph.add_node(int_img2float32_img, inputs=['scaled_img'], outputs=['float_img'])

        self.graph.add_node(centralize, inputs=['float_img', {'mean': self.dataset.mean_bgr}],
                            outputs=['centralized_img'])

        self.graph.add_node(pad_img_label,
                            inputs=['centralized_img', 'scaled_label', {'pad_img_to': self.crop_size}],
                            kwargs={'pad_location': 'center', 'ignore_label': self.dataset.ignore_label},
                            outputs=['padded_img', 'padded_label'])

        # * Rand crop
        rand_crop = RandCrop(self.crop_size)
        self.graph.add_node(rand_crop, inputs=['padded_img', 'padded_label'], outputs=['cropped_img', 'cropped_label'])

        # * Mirror
        mirror = RandMirror() if not self.is_multi_mirror else MultiMirror()
        self.graph.add_node(mirror, inputs=['cropped_img', 'cropped_label'], outputs=['mirrored_img', 'mirrored_label'])

        self.graph.add_node(HWC2CHW, inputs=['mirrored_img'], outputs=['CHW_img'])

        self.graph.add_node(collect_example,
                            inputs=['img_id', 'CHW_img', 'mirrored_label'], outputs=['VOCTrainAuger_output'])


class VOCClsTrainAuger(VOCTrainAuger):
    """VOCTrain Auger with class in label

    The output of auger will be (img_id, img, label, cls_in_label). The cls_in_label is an binary vector with length
    class_num, where 1 means class exits in label, else does not exit.
    """
    def build_graph(self):
        super(VOCClsTrainAuger, self).build_graph()

        self.graph.add_node(attach_cls, inputs=['VOCTrainAuger_output'], outputs=['VOCTrainAugerCls_output'])


class VOCTestAuger(_VOCBaseAuger):
    """DataAuger for VOC Dataset when training"""

    def __init__(self, dataset: Union[VOC, VOCAug],
                 verbosity: int=0, pool_size: int=0):
        """DataAuger for VOC Dataset when Test. The img will be centralized and transposed to (C, H, W)

        Args:
            dataset: VOC or VOCAug dataset
            verbosity: See alchemy_cat.data.DataAuger
            pool_size: See alchemy_cat.data.DataAuger
        """
        super(VOCTestAuger, self).__init__(dataset, verbosity, pool_size, slim=True, rand_seed_log=None)

    def build_graph(self):
        super(VOCTestAuger, self).build_graph()

        self.graph.add_node(int_img2float32_img, inputs=['img'], outputs=['float_img'])

        self.graph.add_node(centralize, inputs=['float_img', {'mean': self.dataset.mean_bgr}],
                            outputs=['centralized_img'])

        self.graph.add_node(HWC2CHW, inputs=['centralized_img'], outputs=['CHW_img'])

        self.graph.add_node(collect_example, inputs=['img_id', 'CHW_img', 'label'], outputs=['VOCTestAuger_output'])


class VOCClsTestAuger(VOCTestAuger):
    """VOCTest Auger with class in label

    The output of auger will be (img_id, img, label, cls_in_label). The cls_in_label is an binary vector with length
    class_num, where 1 means class exits in label, else does not exit.
    """
    def build_graph(self):
        super(VOCClsTestAuger, self).build_graph()

        self.graph.add_node(attach_cls, inputs=['VOCTestAuger_output'], outputs=['VOCTestAugerCls_output'])


if __name__ == '__main__':
    voc_aug = VOCAug(split='val')
    voc_aug_auger = VOCClsTrainAuger(voc_aug,
                                     slim=True,
                                     is_multi_mirror=False, is_color_jitter=True, scale_factors=[0.5, 1.0, 1.5])

    mirror_node = None
    for node in voc_aug_auger.rand_nodes:
        if isinstance(node._fct, RandMirror):
            mirror_node = node

    mirror_count = [0] * 2
    scale_factor_count = [0] * 3
    for index, (origin_example, augered_example) in enumerate(zip(voc_aug, voc_aug_auger)):
        origin_img_id, origin_img, origin_label = origin_example
        augered_img_id, augered_img, augered_label, cls_in_label = augered_example

        assert origin_img_id == augered_img_id

        # * Test rand mirror
        if mirror_node._fct.rand_seed == 1:
            mirror_count[0] += 1
        elif mirror_node._fct.rand_seed == -1:
            mirror_count[1] += 1
        else:
            raise ValueError()

        # * Test rand scale
        if int(origin_img.shape[0] * 0.5) == voc_aug_auger.graph.data['scaled_img'].shape[0]:
            scale_factor_count[0] += 1
        elif int(origin_img.shape[0] * 1.0) == voc_aug_auger.graph.data['scaled_img'].shape[0]:
            scale_factor_count[1] += 1
        elif int(origin_img.shape[0] * 1.5) == voc_aug_auger.graph.data['scaled_img'].shape[0]:
            scale_factor_count[2] += 1
        else:
            raise ValueError()

        # * Test Crop Size
        assert augered_img.shape[1:] == (321, ) * 2
        assert augered_img.shape[1:] == augered_label.shape

        # * Test cls_in_label
        assert len(cls_in_label) == 21

        # * Recover img
        if index % 100 == 0:
            recovered_img = (CHW2HWC(augered_img) + np.array(VOC.mean_bgr)).astype(np.uint8)

            img_wall = RowFigureWall([origin_img, recovered_img],
                                     space_width=20, pad_location='center', color_channel_order='BGR')
            label_wall = RowFigureWall(map(label_map2color_map, [origin_label, augered_label]),
                                       space_width=20, pad_location='center', color_channel_order='RGB')

            show_wall = ColumnFigureWall([img_wall, label_wall], space_width=20)
            show_wall.plot()
            plt.show()

            print([VOC_CLASSES[cls_idx] for cls_idx in np.nonzero(cls_in_label)[0]])

    voc_aug_auger = VOCClsTestAuger(voc_aug)

    for index, (origin_example, augered_example) in enumerate(zip(voc_aug, voc_aug_auger)):
        origin_img_id, origin_img, origin_label = origin_example
        augered_img_id, augered_img, augered_label, cls_in_label = augered_example

        assert origin_img_id == augered_img_id

        # * Test cls_in_label
        assert len(cls_in_label) == 21

        # * Recover img
        recovered_img = (CHW2HWC(augered_img) + np.array(VOC.mean_bgr)).astype(np.uint8)

        if index % 100 == 0:

            img_wall = RowFigureWall([origin_img, recovered_img],
                                     space_width=20, pad_location='center', color_channel_order='BGR')
            label_wall = RowFigureWall(map(label_map2color_map, [origin_label, augered_label]),
                                       space_width=20, pad_location='center', color_channel_order='RGB')

            show_wall = ColumnFigureWall([img_wall, label_wall], space_width=20)
            show_wall.plot()
            plt.show()

            print([VOC_CLASSES[cls_idx] for cls_idx in np.nonzero(cls_in_label)[0]])

        assert np.all(recovered_img == origin_img)
