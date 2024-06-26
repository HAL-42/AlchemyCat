#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: augers.py
@time: 2020/1/13 9:43
@desc:
"""
from typing import Optional, Union, Iterable, Callable, Tuple, Literal
import numpy as np
import cv2
from collections import abc
from scipy import ndimage as nd
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from alchemy_cat.data.data_auger import RandMap, MultiMap
from alchemy_cat.py_tools import Compose, Lambda
from alchemy_cat.py_tools.type import is_int, is_intarr, is_floatarr, tolist
from alchemy_cat.alg import size2HW, color2scalar
from alchemy_cat.acplot.shuffle_ch import RGB2BGR, BGR2RGB

__all__ = ['RandMirror', 'MultiMirror', 'RandUpDown', 'MultiUpDown', 'RandColorJitter', 'scale_img_label',
           'RandScale', 'MultiScale', 'pad_img_label', 'int_img2float32_img', 'centralize', 'RandCrop', 'FiveCrop',
           'RandRangeScale', 'identical', 'pack_identical', 'arr2PIL', 'PIL2arr', 'RandomResizeCrop']


class RandMirror(RandMap):
    rand_seeds = [1, -1]

    def forward(self, img: np.ndarray, label: Optional[np.ndarray] = None) -> Union[np.ndarray, tuple]:
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

    def forward(self, img: np.ndarray, label: Optional[np.ndarray] = None) -> Union[np.ndarray, tuple]:
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

    def forward(self, img: np.ndarray, label: Optional[np.ndarray] = None) -> Union[np.ndarray, tuple]:
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

    def forward(self, img: np.ndarray, label: Optional[np.ndarray] = None) -> Union[np.ndarray, tuple]:
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

    def __init__(self, max_delta_bright: int = 25, range_mul_contract: tuple = (0.6, 1.4),
                 range_mul_saturate: tuple = (0.5, 1.5), max_delta_hue: int = 25,
                 jitter_prob: Union[Iterable[float], float] = 0.5):
        """Random color jitter for image

        Args:
            max_delta_bright (int): max delta of bright
            range_mul_contract (tuple): range of multiplier of contract
            range_mul_saturate (tuple): range of multiplier of saturate
            max_delta_hue (int): max delta of hue
            jitter_prob (Union[list, tuple, float]): probs of each jitter step (jitter bright, jitter contract,
                jitter saturate, jitter hue) implemented
        """
        super(RandColorJitter, self).__init__()

        if max_delta_bright <= 0:
            raise ValueError(f"max_delta_bright={max_delta_bright} should be larger than 0")
        self.max_delta_bright = max_delta_bright

        if range_mul_contract[0] <= 0 or range_mul_contract[0] >= range_mul_contract[1]:
            raise ValueError(
                f"range_mul_contract={range_mul_contract}'s lower bound should larger than 0, lower than upper bound")
        self.range_mul_contract = range_mul_contract

        if range_mul_saturate[0] <= 0 or range_mul_saturate[0] >= range_mul_saturate[1]:
            raise ValueError(
                f"range_mul_saturate={range_mul_saturate}'s lower bound should larger than 0, lower than upper bound")
        self.range_mul_saturate = range_mul_saturate

        if max_delta_hue <= 0:
            raise ValueError(f"max_delta_hue={max_delta_hue} should be larger than 0")
        self.max_delta_hue = max_delta_hue

        jps = [jitter_prob] * 4 if isinstance(jitter_prob, float) else tolist(jitter_prob)
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

    def jitter_bright(self, img: np.ndarray):
        delta_bright = self.rand_seed.get('delta_bright')
        if delta_bright is not None:
            return cv2.convertScaleAbs(img, alpha=1, beta=delta_bright)
        return img

    def jitter_contract(self, img: np.ndarray):
        mul_contract = self.rand_seed.get('mul_contract')
        if mul_contract is not None:
            return cv2.convertScaleAbs(img, alpha=mul_contract, beta=0)
        return img

    def jitter_saturate(self, img: np.ndarray):
        mul_saturate = self.rand_seed.get('mul_saturate')
        if mul_saturate is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 1] = cv2.convertScaleAbs(img[:, :, 1], alpha=mul_saturate, beta=0)
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    def jitter_hue(self, img: np.ndarray):
        delta_hue = self.rand_seed.get('delta_hue')
        if delta_hue is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
            img[:, :, 0] = cv2.convertScaleAbs(img[:, :, 0], alpha=1, beta=delta_hue)
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR_FULL)
        return img

    def forward(self, img: np.ndarray) -> np.ndarray:
        """Color jitter img according to rand seed

        Args:
            img (np.ndarray):  img to be jitter

        Returns: img after jittering
        """
        if not is_intarr(img):
            raise ValueError('Image should be int array')

        jitters = [Lambda(self.jitter_bright), Lambda(self.jitter_contract), Lambda(self.jitter_saturate),
                   Lambda(self.jitter_hue)]
        ordered_jitters = Compose([jitters[i] for i in self.rand_seed['jitter_order']])

        return ordered_jitters(img)


def _check_img_size_equal_label_size(img, label):
    if label is not None and img.shape[:2] != label.shape:
        raise ValueError(f"img size {img.shape[:2]} should be equal to label{label.shape} size")


def scale_img_label(scale_factor: float | tuple[float, float] | int | tuple[int, int],
                    img: np.ndarray, label: Optional[np.ndarray] = None,
                    aligner: Union[Callable[[int], int], Iterable[Callable[[int], int]]] = lambda x: x,
                    align_corner: bool=False, PIL_mode: int | None=None
                    ) -> Union[np.ndarray, Tuple[np.ndarray]]:
    """Scale img and label accroding to scaled factor

    Args:
        scale_factor: scaled factor
        img: img to be scaled
        label: If not none, label to be scaled
        aligner: The scaled_size calculated by scale_factor * img_size will be fix by aligner(scaled_size). If
            Iterable, then first and second aligners separately used to align H and W.
        align_corner: If true, use ndimage's zoom to resize img and label, which can align corner. Else use cv2's
            resize to scaled img and label, which will align center. (Default: False)
        PIL_mode: 若不为None，使用指定的PIL插值模式采样图片。

    Returns: Scaled img and label(If exits)
    """
    _check_img_size_equal_label_size(img, label)

    if PIL_mode is not None and align_corner:
        raise ValueError("使用PIL采样时，align_corner必须是False。")

    if isinstance(aligner, abc.Callable):
        aligner_h, aligner_w = aligner, aligner
    elif isinstance(aligner, abc.Iterable):
        aligner_h, aligner_w = aligner
    else:
        raise ValueError(f"pad_aligner {aligner} must be Callable or Iterable[Callable]")

    match scale_factor:
        case float(s):
            assert s > 0, f"{scale_factor=} 应当 > 0."
            scaled_h, scaled_w = aligner_h(round(s * img.shape[0])), aligner_w(round(s * img.shape[1]))
        case (float(s_h), float(s_w)):
            assert s_h > 0 and s_w > 0, f"{scale_factor=} 应当 > 0."
            scaled_h, scaled_w = aligner_h(round(s_h * img.shape[0])), aligner_w(round(s_w * img.shape[1]))
        case (int() | (int(), int())) as size:
            scaled_h, scaled_w = size2HW(size)
            scaled_h, scaled_w = aligner_h(scaled_h), aligner_w(scaled_w)
        case _:
            raise ValueError(f"不支持的{scale_factor=}。")

    if PIL_mode is not None:
        pil = arr2PIL(img, order='RGB')  # 缩放时，RGB通道顺序无所谓。
        scaled_pil = pil.resize((scaled_w, scaled_h), resample=PIL_mode)
        scaled_img = PIL2arr(scaled_pil, order='RGB')
    elif not align_corner:
        scaled_img = cv2.resize(img, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
    else:
        scaled_img = nd.zoom(img, (scaled_h / img.shape[0], scaled_w / img.shape[1], 1.0), order=1)

    if label is not None:
        if PIL_mode is not None:
            pil = arr2PIL(label)
            scaled_pil = pil.resize((scaled_w, scaled_h), resample=Image.NEAREST)
            scaled_label = PIL2arr(scaled_pil)
        elif not align_corner:
            scaled_label = cv2.resize(label, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)
        else:
            scaled_label = nd.zoom(label, (scaled_h / label.shape[0], scaled_w / label.shape[1]), order=0)
        return scaled_img, scaled_label
    else:
        return scaled_img


class RandScale(RandMap):

    def __init__(self, scale_factors: Iterable[float | tuple[float, float] | int | tuple[int, int]],
                 aligner: Union[Callable[[int], int], Iterable[Callable[[int], int]]] = lambda x: x,
                 align_corner: bool=False, PIL_mode: int | None=None):
        """Auger to rand rescale the input img and corresponding label

        Args:
            scale_factors: scale_factors: scale factors. eg. [0.5, 1, 1.5] means the img (and label) will be
            rand scale with factor 0.5, 1.0 or 1.5
            aligner: The scaled_size calculated by scale_factor * img_size will be fix by aligner(scaled_size). If
                Iterable, then first and second aligners separately used to align H and W.
            align_corner: If true, use ndimage's zoom to resize img and label, which can align corner. Else use cv2's
                resize to scaled img and label, which will align center. (Default: False)
            PIL_mode: 若不为None，使用指定的PIL插值模式采样图片。
        """
        super(RandScale, self).__init__()

        # self.rand_seeds = []
        # for factor in scale_factors:
        #     factor = float(factor)
        #     if factor <= 0:
        #         raise ValueError(f"scale factors {scale_factors} must all larger than 0")
        #     self.rand_seeds.append(factor)
        self.rand_seeds = list(scale_factors)

        self.aligner = aligner
        self.align_corner = align_corner
        self.PIL_mode = PIL_mode

    def forward(self, img: np.ndarray, label: Optional[np.ndarray] = None) -> Union[np.ndarray, Tuple[np.ndarray]]:
        """Rand Scale img according to rand seed

        Args:
            img (np.ndarray): img with shape (H, W, C)
            label (Optional[np.ndarray]): label with shape (H, W)

        Returns: Scaled img and label(if exit)
        """
        scale_factor = self.rand_seed

        return scale_img_label(scale_factor, img, label, self.aligner,
                               align_corner=self.align_corner, PIL_mode=self.PIL_mode)


class MultiScale(MultiMap):
    def __init__(self, scale_factors: Iterable,
                 aligner: Union[Callable[[int], int], Iterable[Callable[[int], int]]] = lambda x: x,
                 align_corner: bool=False, PIL_mode: int | None=None):
        """Auger to Multi rescale the input img and corresponding label

        Args:
            scale_factors: scale_factors: scale factors. eg. [0.5, 1, 1.5] means the img (and label) will be
            rand scale with factor 0.5, 1.0 or 1.5
            aligner: The scaled_size calculated by scale_factor * img_size will be fix by aligner(scaled_size). If
                Iterable, then first and second aligners separately used to align H and W.
            align_corner: If true, use ndimage's zoom to resize img and label, which can align corner. Else use cv2's
                resize to scaled img and label, which will align center. (Default: False)
            PIL_mode: 若不为None，使用指定的PIL插值模式采样图片。
        """
        super(MultiScale, self).__init__()

        self.scale_factors = []
        for factor in scale_factors:
            factor = float(factor)
            if factor <= 0:
                raise ValueError(f"scale factors {scale_factors} must all larger than 0")
            self.scale_factors.append(factor)

        self.output_num = len(self.scale_factors)

        self.aligner = aligner
        self.align_corner = align_corner
        self.PIL_mode = PIL_mode

    def forward(self, img: np.ndarray, label: Optional[np.ndarray] = None) -> Union[np.ndarray, Tuple[np.ndarray]]:
        """Rand Scale img according to rand seed

        Args:
            img (np.ndarray): img with shape (H, W, C)
            label (Optional[np.ndarray]): label with shape (H, W)

        Returns: Scaled img and label(if exit)
        """
        scale_factor = self.scale_factors[self.output_index]

        return scale_img_label(scale_factor, img, label, self.aligner,
                               align_corner=self.align_corner, PIL_mode=self.PIL_mode)


def pad_img_label(img: np.ndarray, label: Optional[np.ndarray] = None, pad_img_to: Union[Iterable, int] = 0,
                  pad_aligner: Union[Callable[[int], int], Iterable[Callable[[int], int]]] = lambda x: x,
                  img_pad_val: Union[int, float, Iterable] = 0.0, ignore_label: int = 255,
                  pad_location: Union[str, int] = 'right-bottom',
                  with_mask: bool=False) \
        -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray,
                                                                  np.ndarray, Tuple[int, int, int, int]]]:
    """Pad img to size pad_aligner(max(img_origin_size, pad_img_to))

    Args:
        img (np.ndarray): img to be padded
        label (np.ndarray): label to be padded
        pad_img_to (Union[None, Iterable, int]): img pad size. If value is int, the img_pad_to will be parsed as
            H=value, W=value. Else will be parsed as H=list(value)[0], W=list(value)[1]
        pad_aligner (Union[Callable, Iterable[Callable]]): Final pad size will be refined by callable aligner. If
            Iterable, then first and second aligners separately used to align H and W.
        img_pad_val: (Union[int, float, Iterable]): If value is int or float, return (value, value, value),
            if value is Iterable with 3 element, return totuple(value), else raise error
        ignore_label (int): value to pad the label.
        pad_location (Union[str, int]): Indicate pad location. Can be 'left-top'/0, 'right-top'/1, 'left-bottom'/2
            'right-bottom'/3, 'center'/4.
        with_mask: If True, return mask of padded area and (i, j, h, w) of img in padded img.

    Returns: Padded img and label(if given)
    """
    # * Check img size = label size
    _check_img_size_equal_label_size(img, label)

    # * Check ignore label value
    if not is_int(ignore_label):
        raise ValueError(f"ignore_label{ignore_label} should be int")
    else:
        ignore_label = int(ignore_label)

    # * Get img pad scalar. If img is int array, then pad val can't be float.
    img_pad_scalar = color2scalar(img_pad_val)
    if is_intarr(img) and is_floatarr(np.array(img_pad_val)):
        raise ValueError("Input img is int array, while img_pad_val has float. Which may cause unexpected "
                         "behaviour when padding. Using int_img2float32img before padding int img with float value.")

    # * Get pad aligner
    if isinstance(pad_aligner, abc.Callable):
        pad_aligner_h, pad_aligner_w = pad_aligner, pad_aligner
    elif isinstance(pad_aligner, abc.Iterable):
        pad_aligner_h, pad_aligner_w = pad_aligner
    else:
        raise ValueError(f"pad_aligner {pad_aligner} must be Callable or Iterable[Callable]")

    # * Get pad_h, pad_w
    pad_to_h, pad_to_w = size2HW(pad_img_to)
    img_h, img_w = size2HW(img.shape[:2])
    padded_h, padded_w = pad_aligner_h(max(pad_to_h, img_h)), pad_aligner_w(max(pad_to_w, img_w))
    pad_h, pad_w = padded_h - img_h, padded_w - img_w

    if pad_h < 0 or pad_w < 0:
        raise ValueError("pad aligner's return size must larger than input size")

    # * Get right/left/bottom/top pad size
    if isinstance(pad_location, str):
        pad_location_index_dict = {'left-top': 0, 'right-top': 1, 'left-bottom': 2, 'right-bottom': 3, 'center': 4}
        pad_location_index = pad_location_index_dict.get(pad_location)
        if pad_location is None:
            raise ValueError(f"pad_location should be in {list(pad_location_index_dict.keys())}")
    else:
        pad_location_index = pad_location
        if pad_location_index not in [0, 1, 2, 3, 4]:
            raise ValueError(f"If pad location is given by int, then it should be in range 0~4")

    pad_left, pad_right, pad_top, pad_bottom = 0, 0, 0, 0
    if pad_location_index & 1:
        pad_right = pad_w
    else:
        pad_left = pad_w
    if pad_location_index & 2:
        pad_bottom = pad_h
    else:
        pad_top = pad_h
    if pad_location_index & 4:
        pad_top, pad_bottom = pad_h // 2, (pad_h + 1) // 2  # floor(pad_h/2), ceil(pad_h/2)
        pad_left, pad_right = pad_w // 2, (pad_w + 1) // 2

    ret = []

    img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right,
                             borderType=cv2.BORDER_CONSTANT, value=img_pad_scalar)
    ret.append(img)

    if label is not None:
        label = cv2.copyMakeBorder(label, pad_top, pad_bottom, pad_left, pad_right,
                                   borderType=cv2.BORDER_CONSTANT, value=ignore_label)
        ret.append(label)

    if with_mask:
        mask = np.ones_like(img[:, :, 0], dtype=bool)
        mask[pad_top:pad_top + img_h, pad_left:pad_left + img_w] = False
        ret.append(mask)
        ret.append(np.asarray((pad_top, pad_left, img_h, img_w)))  # (i, j, h, w)

    return ret[0] if len(ret) == 1 else tuple(ret)


def int_img2float32_img(img: np.ndarray, scale: bool=False) -> np.ndarray:
    """Convert int img to float32 img

    Args:
        img (np.ndarray): int numpy img
        scale (bool): If True, img will be scale to range [0.0, 1.0]. (Default: False)

    Returns: float32 img
    """
    if not is_intarr(img):
        raise ValueError(f"img {img} is supposed to be an int arr")
    img = img.astype(np.float32)  # Return copy
    if scale:
        img /= 255  # Can be inplace
    return img


def centralize(img: np.ndarray, mean: Union[int, float, Iterable], std: Union[int, float, Iterable, None] = None):
    """Centralize img by minus the mean and divide the std

    Args:
        img: float img with size (H, W, C) to be centralized
        mean: mean value to be minus
        std: If not None, std value to be divided after img is minuted mean value

    Returns: Centralized img
    """
    if not is_floatarr(img):
        raise ValueError(f"img to be centralized should be float array")

    mean_scalar, std_scalar = np.array(color2scalar(mean), dtype=np.float32), \
        np.array(color2scalar(std), dtype=np.float32) if std is not None else None

    img = img - mean  # Return copy
    if std_scalar is not None:
        img /= std  # Inplace

    return img


def _crop_img_label(img, label, offset_h, offset_w, crop_h, crop_w) -> Union[np.ndarray, Tuple[np.ndarray]]:
    cropped_img = img[offset_h:offset_h + crop_h, offset_w:offset_w + crop_w, :]
    if label is None:
        return cropped_img
    else:
        cropped_label = label[offset_h:offset_h + crop_h, offset_w:offset_w + crop_w]
        return cropped_img, cropped_label


class RandCrop(RandMap):

    def __init__(self, crop_size: Union[int, Iterable[int]]):
        """Random crop img to crop_size

        Args:
            crop_size: Crop size. If size is int, the crop_height=value, crop_width=value. Else will be parsed as
                crop_height=list(value)[0], crop_width=list(value)[1]
        """
        super(RandCrop, self).__init__()

        self.crop_h, self.crop_w = size2HW(crop_size)

    def generate_rand_seed(self, img: np.ndarray, label: Optional[np.ndarray] = None):
        img_h, img_w = img.shape[:2]

        if img_h < self.crop_h or img_w < self.crop_w:
            raise ValueError(f"img_h {img_h} must >= crop_h {self.crop_h}; img_w {img_w} must >= crop_w {self.crop_w}")

        offset_h = np.random.randint(0, img_h - self.crop_h + 1)
        offset_w = np.random.randint(0, img_w - self.crop_w + 1)

        return int(offset_h), int(offset_w)

    def forward(self, img: np.ndarray, label: Optional[np.ndarray] = None) -> Union[np.ndarray, Tuple[np.ndarray]]:
        """Generate rand cropped img and label accroding to rand seed

        Args:
            img: img to be cropped
            label: If not None, means label to be cropped

        Returns: Cropped img and label(if not None)
        """
        _check_img_size_equal_label_size(img, label)

        img_h, img_w = img.shape[:2]
        offset_h, offset_w = self.rand_seed
        if offset_h + self.crop_h > img_h or offset_w + self.crop_w > img_w:
            raise ValueError(f"Crop size out of range. offset_h + crop_h = {offset_h + self.crop_h}, "
                             f"offset_w + crop_w = {offset_w + self.crop_w}, while img has shape {(img_h, img_w)}")

        return _crop_img_label(img, label, offset_h, offset_w, self.crop_h, self.crop_w)


class FiveCrop(MultiMap):
    output_num = 5

    def __init__(self, crop_size: Union[int, Iterable[int]]):
        """Five crop img to crop_size, with index 0-4 meaning left-top, right-top, left-bottom, right-bottom, center

        Args:
            crop_size: Crop size. If size is int, the crop_height=value, crop_width=value. Else will be parsed as
                crop_height=list(value)[0], crop_width=list(value)[1]
        """
        super(FiveCrop, self).__init__()

        self.crop_h, self.crop_w = size2HW(crop_size)

    def forward(self, img: np.ndarray, label: Optional[np.ndarray] = None) -> Union[np.ndarray, Tuple[np.ndarray]]:
        """Return cropped img and label according to output_index

        Args:
            img: img to be cropped
            label: If not None, label to be cropped

        Returns: Cropped img and label(If exits)
        """
        _check_img_size_equal_label_size(img, label)

        img_h, img_w = img.shape[:2]
        if img_h < self.crop_h or img_w < self.crop_w:
            raise ValueError(f"img_h {img_h} must >= crop_h {self.crop_h}; img_w {img_w} must >= crop_w {self.crop_w}")

        bias_h, bias_w = img_h - self.crop_h, img_w - self.crop_w
        offset_h, offset_w = 0, 0

        if self.output_index & 1:
            offset_w += bias_w
        if self.output_index & 2:
            offset_h += bias_h
        if self.output_index & 4:
            offset_h, offset_w = bias_h // 2, bias_w // 2

        return _crop_img_label(img, label, offset_h, offset_w, self.crop_h, self.crop_w)


class RandRangeScale(RandMap):

    def __init__(self, low_size: int, high_size: int, short_thresh: int,
                 aligner: Union[Callable[[int], int], Iterable[Callable[[int], int]]] = lambda x: x,
                 align_corner: bool=True, PIL_mode: int | None=None):
        """在保证短边大于阈值的情况下，尝试将长边缩放到随机尺寸。

        Args:
            low_size: 长边随机缩放的最短尺寸。
            high_size: 长边随机缩放的最大尺寸。
            short_thresh: 短边缩放后的最小尺寸。
            align_corner: 缩放时是否对齐角点。
        """
        super(RandRangeScale, self).__init__()

        self.low_size = low_size
        self.high_size = high_size
        self.short_thresh = short_thresh
        self.aligner = aligner
        self.align_corner = align_corner
        self.PIL_mode = PIL_mode

    def generate_rand_seed(self, img: np.ndarray, label: Optional[np.ndarray] = None) -> float:
        # * 先计算一个“目标尺寸”。
        object_size = self.low_size + np.random.randint(self.high_size - self.low_size + 1)

        img_h, img_w = img.shape[:2]

        long_size = img_h if img_h > img_w else img_w
        short_size = img_h if img_h < img_w else img_w
        # * 若长边达到目标尺寸的同时，短边大于short thresh，则让长边缩放到目标尺寸。否则将短边缩放到阈值尺寸。
        scale_ratio = max(object_size / long_size, self.short_thresh / short_size)
        return scale_ratio

    def forward(self, img: np.ndarray, label: Optional[np.ndarray] = None) -> Union[np.ndarray, Tuple[np.ndarray]]:
        return scale_img_label(self.rand_seed, img, label,
                               aligner=self.aligner, align_corner=self.align_corner, PIL_mode=self.PIL_mode)


class RandomResizeCrop(RandMap):
    """随机裁剪图像。"""

    def __init__(
            self,
            size: Union[int, Iterable[int]],
            scale: tuple[float, float]=(0.08, 1.0),
            ratio: tuple[float, float]=(3.0 / 4.0, 4.0 / 3.0),
            interpolation=Image.BILINEAR):
        """封装torchvision的RandomResizedCrop，实现对image和label的同步增强。

        Args:
            size: 裁剪后的尺寸。
            scale: 随机缩放的范围。
            ratio: 随机裁剪的宽高比范围。
            interpolation: 插值方式。
        """
        super(RandomResizeCrop, self).__init__()

        self.transform = T.RandomResizedCrop(size, scale, ratio, interpolation)

        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    def generate_rand_seed(self, img: np.ndarray, label: Optional[np.ndarray] = None) -> tuple[int, int, int, int]:
        transform = self.transform
        pil_img = arr2PIL(img, order='RGB')  # 只用于获取尺寸，通道顺序无所谓。
        return transform.get_params(pil_img, transform.scale, transform.ratio)

    def forward(self, img: np.ndarray, label: Optional[np.ndarray] = None) -> Union[np.ndarray, Tuple[np.ndarray]]:
        transform = self.transform
        i, j, h, w = self.rand_seed

        pil_img = arr2PIL(img, order='RGB')  # 只用于缩放，通道顺序无所谓。
        scaled_img = TF.resized_crop(pil_img, i, j, h, w, transform.size, transform.interpolation)

        if label is not None:
            pil_label = arr2PIL(label)
            scaled_label = TF.resized_crop(pil_label, i, j, h, w, transform.size, Image.NEAREST)
            return PIL2arr(scaled_img, order='RGB'), PIL2arr(scaled_label)
        else:
            return PIL2arr(scaled_img, order='RGB')


def identical(x):
    """原样返回单个参数。"""
    return x


def pack_identical(*x):
    """原样返回多个参数。"""
    return x


def arr2PIL(arr: np.ndarray, order: Literal['RGB', 'BGR']='BGR') -> Image.Image:
    """将numpy数组转为PIL图片。

    Args:
        arr: (H, W, [C])的numpy数组。
        order: 彩图的通道顺序。对单通道图无效。

    Returns:
        PIL图片。
    """
    is_color = arr.ndim == 3
    if is_color and order == 'BGR':
        arr = BGR2RGB(arr)
    return Image.fromarray(arr, mode='RGB' if is_color else 'L')


def PIL2arr(pil: Image.Image, order: Literal['RGB', 'BGR']='BGR') -> np.ndarray:
    """将numpy数组转为PIL图片。

    Args:
        pil: PIL图片。
        order: 彩图的通道顺序。对单通道图无效。

    Returns:
        numpy数组。
    """
    arr = np.asarray(pil)
    if pil.mode == 'L':
        return arr
    elif pil.mode == 'RGB':
        if order == 'RGB':
            return arr
        elif order == 'BGR':
            return RGB2BGR(arr)
        else:
            raise ValueError(f"不支持的{order=}。")
    else:
        raise ValueError(f"不支持的PIL模式{pil.mode=}。")
