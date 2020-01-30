#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: auger.py
@time: 2020/1/13 9:43
@desc:
"""
from typing import Optional, Union, Iterable, Callable
import numpy as np
import cv2

from alchemy_cat.data.data_auger import RandMap, MultiMap
from alchemy_cat.py_tools import Compose, Lambda, is_intarr



class RandMirror(RandMap):

    rand_seeds = [1, -1]

    def forward(self, img: np.ndarray, label: Optional[np.ndarray]=None) -> Union[np.ndarray, tuple]:
        """ Random mirror image and it's label(if exits)

        Args:
            img (np.ndarray): Img(H, W, C) to be mirrored
            label (Optional[np.ndarray]): Label(H, W) to be mirrored

        Returns: Img and label(if exits) random mirrored
        """
        if label is None:
            return img[:, ::self.rand_seed, :]
        else:
            return img[:, ::self.rand_seed, :], label[:, ::self.rand_seed]


class MultiMirror(MultiMap):

    output_num = 2

    def forward(self, img: np.ndarray, label: Optional[np.ndarray]=None) -> Union[np.ndarray, tuple]:
        """Multi output of mirrored img and label(if exits).
        output_index = 0: Don't mirror
        output_index = 1: mirror

        Args:
            img (np.ndarray): Image(H, W, C) to be mirrored
            label (Optional(np.ndarray)): Label(H, W) to be mirrored

        Returns: Img and label(if exits) random mirrored
        """
        mirror = 1 if self.output_index == 0 else -1

        if label is None:
            return img[:, ::mirror, :]
        else:
            return img[:, ::mirror, :], label[:, ::mirror]


class RandUpDown(RandMap):

    rand_seeds = [1, -1]

    def forward(self, img: np.ndarray, label: Optional[np.ndarray]=None) -> Union[np.ndarray, tuple]:
        """ Random Upside Down image and it's label(if exits)

        Args:
            img (np.ndarray): Img(H, W, C) to be upside down
            label (Optional[np.ndarray]): Label(H, W) to be upside down

        Returns: Img and label(if exits) random upside down
        """
        if label is None:
            return img[::self.rand_seed, :, :]
        else:
            return img[::self.rand_seed, :, :], label[::self.rand_seed, :]


class MultiUpDown(MultiMap):

    output_num = 2

    def forward(self, img: np.ndarray, label: Optional[np.ndarray]=None) -> Union[np.ndarray, tuple]:
        """Multi output of upside down img and label(if exits).
        output_index = 0: Don't upside down
        output_index = 1: upside down

        Args:
            img (np.ndarray): Image(H, W, C) to be upside down
            label (Optional(np.ndarray)): Label(H, W) to be upside down

        Returns: Img and label(if exits) random upside down
        """
        up_down = 1 if self.output_index == 0 else -1

        if label is None:
            return img[::up_down, :, :]
        else:
            return img[::up_down, :, :], label[::up_down, :]


class RandColorJitter(RandMap):

    def __init__(self, max_delta_bright: int=32, range_mul_contract: tuple=(0.5, 1.5),
                 range_mul_saturate: tuple=(0.5, 1.5), max_delta_hue: int=18,
                 jitter_prob: Union[Iterable[float], float]=0.5):
        """Random color jitter for image

        Args:
            max_delta_bright (int): max delta of bright
            range_mul_contract (tuple): range of multiplier of contract
            range_mul_saturate (tuple): range of multiplier of saturate
            max_delta_hue (int): max delta of hue
            jitter_prob (Union[list, tuple, float]): probs of each jitter step (jitter bright, jitter contract, jitter saturate, jitter hue) implemented
        """
        super(RandColorJitter, self).__init__()

        if max_delta_bright <= 0:
            raise ValueError(f"max_delta_bright={max_delta_bright} should be larger than 0")
        self.max_delta_bright = max_delta_bright

        if range_mul_contract[0] <= 0 or range_mul_contract[0] <= range_mul_contract[1]:
            raise ValueError(
                f"range_mul_contract={range_mul_contract}'s lower bound should larger than 0, lower than upper bound")
        self.range_mul_contract = range_mul_contract

        if range_mul_saturate[0] <= 0 or range_mul_saturate[0] <= range_mul_saturate[1]:
            raise ValueError(
                f"range_mul_saturate={range_mul_saturate}'s lower bound should larger than 0, lower than upper bound")
        self.range_mul_saturate = range_mul_saturate

        if max_delta_hue <= 0:
            raise ValueError(f"max_delta_hue={max_delta_hue} should be larger than 0")
        self.max_delta_hue = max_delta_hue

        jps = [jitter_prob] * 4 if isinstance(jitter_prob, float) else list(jitter_prob)
        if len(jps) != 4:
            raise ValueError(f"If the jitter_prob is not float, then "
                             f"you must give each step's probability(Length of jitter_prob {jitter_prob} must be 4).")
        for jp in jps:
            if jp < 0 or jp > 1:
                raise ValueError(f"jitter probs={jps} should >=0 and <=1")

        self.jitter_prob = jps

    def generate_rand_seed(self, *fwd_args, **fwd_kwargs):
        rand_seed = {}

        is_jitter_bright = np.random.random() < self.jitter_prob[0]
        rand_seed['delta_bright'] = \
            np.random.uniform(-1 * self.max_delta_bright, self.max_delta_bright) if is_jitter_bright else None

        is_jitter_contract = np.random.random() < self.jitter_prob[1]
        rand_seed['mul_contract'] = \
            np.random.uniform(*self.range_mul_contract) if is_jitter_contract else None

        is_jitter_saturate = np.random.random() < self.jitter_prob[2]
        rand_seed['mul_saturate'] = \
            np.random.uniform(*self.range_mul_saturate) if is_jitter_saturate else None

        is_jitter_hue = np.random.random() < self.jitter_prob[3]
        rand_seed['delta_hue'] = \
            np.random.uniform(-1 * self.max_delta_hue, self.max_delta_hue) if is_jitter_hue else None

        jitter_order = list(range(len(rand_seed)))
        np.random.shuffle(jitter_order)
        rand_seed['jitter_order'] = jitter_order

        return rand_seed

    def jitter_bright(self, img:np.ndarray):
        delta_bright = self.rand_seed.get('delta_bright')
        if delta_bright is not None:
            return cv2.convertScaleAbs(img, alpha=1, beta=delta_bright)
        return img

    def jitter_contract(self, img:np.ndarray):
        mul_contract = self.rand_seed.get('mul_contract')
        if mul_contract is not None:
            return cv2.convertScaleAbs(img, alpha=mul_contract, beta=0)
        return img

    def jitter_saturate(self, img:np.ndarray):
        mul_saturate = self.rand_seed.get('mul_saturate')
        if mul_saturate is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 1] = cv2.convertScaleAbs(img[:, :, 1], alpha=mul_saturate, beta=0)
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    def jitter_hue(self, img:np.ndarray):
        delta_hue = self.rand_seed.get('delta_hue')
        if delta_hue is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 0] = cv2.convertScaleAbs(img[:, :, 0], alpha=1, beta=delta_hue)
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    def forward(self, img: np.ndarray):
        if not is_intarr(img):
            raise ValueError('Image should be int array')

        jitters = [Lambda(self.jitter_bright), Lambda(self.jitter_contract), Lambda(self.jitter_saturate),
                   Lambda(self.jitter_hue)]
        ordered_jitters = Compose([jitters[i] for i in self.rand_seed['jitter_order']])

        return ordered_jitters(img)


class RandScale(RandMap):

    def __init__(self, scale_factors: Iterable, aligner: Optional[Callable]=None):
        """Auger to rand rescale the input img and corresponding label

        Args:
            scale_factors: scale_factors: scale factors. eg. [0.5, 1, 1.5] means the img (and label) will be
            rand scale with factor 0.5, 1.0 or 1.5
            aligner: If not None, the scaled_size calculated by scale_factor * img_size will be fix by
            aligner(scaled_size)
        """
        super(RandScale, self).__init__()

        self.rand_seeds = []
        for factor in scale_factors:
            factor = float(factor)
            if factor <= 0:
                raise ValueError(f"scale factors {scale_factors} must all larger than 0")
            self.rand_seeds.append(factor)

        if aligner is None:
            self.aligner = lambda x: x
        else:
            self.aligner = aligner

    def forward(self, img: np.ndarray, label: Optional[np.ndarray]=None):
        if label is not None:
            if img.shape[-2:] != label.shape[-2:]:
                raise ValueError(f"RandScale's img size {img.shape[-2:]} should be equal to label{img.shape[-2:]} size")

        scaled_factor = self.rand_seed

        scaled_h, scaled_w = \
            self.aligner(int(scaled_factor * img.shape[-2])), self.aligner(int(scaled_factor * img.shape[-1]))

        scaled_img = cv2.resize(img, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
        if label is not None:
            scaled_label = cv2.resize(label, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)

        if label is not None:
            return scaled_img, scaled_label
        else:
            return scaled_img


class MultiScale(MultiMap):







