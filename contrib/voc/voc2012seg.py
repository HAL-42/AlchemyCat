#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: voc2012seg.py
@time: 2020/1/7 17:16
@desc:
"""
import os.path as osp
import pickle
from math import ceil

import cv2
import imagesize
import numpy as np
from PIL import Image
from addict import Dict
from alchemy_cat.acplot.shuffle_ch import BGR2RGB, RGB2BGR
from alchemy_cat.data.dataset import Dataset, Subset

from .utils import label_map2color_map, VOC_CLASSES

__all__ = ['VOC', 'VOCAug', 'VOCAug2']


class _VOCBase(Dataset):
    """
    PASCAL VOC and VOC Aug Segmentation base dataset
    """
    class_names = VOC_CLASSES
    class_num = len(class_names)
    mean_bgr = [104.008, 116.669, 122.675]
    std_bgr = [57.375, 57.12, 58.395]
    ignore_label = 255

    def __init__(self, root: str="./contrib/datasets", year="2012", split: str="train", PIL_read: bool=False,
                 ret_img_file: bool=False):
        """
        Args:
            root (str): The parent dir of VOC dataset
            year (str): "2012"
            split (str): "train"/"val"/"trainval"
            PIL_read (bool): If True, use PIL to read image, else use cv2
            ret_img_file (bool): If True, return image file path
        """
        self.root = root
        self.year = year
        self.split = split
        self.PIL_read = PIL_read
        self.ret_img_file = ret_img_file
        self.files = []
        self.image_ids = []

        self.image_dir: str | None = None
        self.label_dir: str | None = None

        self._set_files()

    def _set_files(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.files)

    def get_by_img_id(self, img_id: str):
        try:
            index = self.image_ids.index(img_id)
        except ValueError:
            raise RuntimeError(f"Can't find img_id {img_id} in dataset's image_ids list")
        return self[index]


class VOC(_VOCBase):
    """
    PASCAL VOC Segmentation dataset
    """
    def _set_files(self):
        self.root = osp.join(self.root, f"VOC{self.year}")
        self.image_dir = osp.join(self.root, "JPEGImages")
        self.label_dir = osp.join(self.root, "SegmentationClass")

        if self.split in ["train", "trainval", "val", "test"]:
            file_list = osp.join(
                self.root, "ImageSets", "Segmentation", self.split + ".txt"
            )
            file_list = tuple(open(file_list, "r"))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files = file_list
        else:
            raise ValueError("Invalid split name: {}".format(self.split))

        self.image_ids = self.files

    def get_item(self, index):
        image_id = self.files[index]

        # * Load imgs
        image_path = osp.join(self.image_dir, image_id + ".jpg")
        if self.ret_img_file:
            image = image_path
        elif self.PIL_read:
            image = RGB2BGR(np.asarray(Image.open(image_path), dtype=np.uint8))
        else:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # * Load Labels
        if self.split != 'test':
            label_path = osp.join(self.label_dir, image_id + ".png")
            label = np.asarray(Image.open(label_path), dtype=np.uint8)
        else:
            # If no label, then giving a all ignore ground truth
            img_size = imagesize.get(image)[::-1] if isinstance(image, str) else image.shape[:2]
            label = np.zeros(img_size, dtype=np.uint8) + self.ignore_label

        return image_id, image, label


class VOCAug(_VOCBase):
    """
    PASCAL VOC Aug Segmentation dataset
    """
    def __init__(self, root: str = "./contrib/datasets", year="2012", split: str = "train", PIL_read: bool=False,
                 ret_img_file: bool=False):
        """
        Args:
            root (str): The parent dir of VOC dataset
            year (str): "2012"
            split (str): "train"/"val"/"trainval"/"train_aug"/"trainval_aug"
            PIL_read (bool): If True, use PIL to read image, else use cv2
        """
        self.labels = []
        super(VOCAug, self).__init__(root, year, split, PIL_read, ret_img_file)

    def _set_files(self):
        self.root = osp.join(self.root, f"VOC{self.year}")
        self.image_dir = osp.join(self.root, "JPEGImages")
        self.label_dir = osp.join(self.root, "SegmentationClassAug")

        if self.split in ["train", "train_aug", "trainval", "trainval_aug", "val"]:
            file_list = osp.join(
                self.root, "ImageSets", "SegmentationAug", self.split + ".txt"
            )
            file_list = tuple(open(file_list, "r"))
            file_list = [id_.rstrip().split(" ") for id_ in file_list]
            self.files, self.labels = list(zip(*file_list))
        elif self.split == 'test':
            file_list = osp.join(
                self.root, "ImageSets", "SegmentationAug", self.split + ".txt"
            )
            file_list = tuple(open(file_list, "r"))
            self.files = [id_.rstrip() for id_ in file_list]
        else:
            raise ValueError("Invalid split name: {}".format(self.split))

        self.image_ids = [file.split("/")[-1].split(".")[0] for file in self.files]

    def get_item(self, index):
        # * Load image
        image_id = self.files[index].split("/")[-1].split(".")[0]
        image_path = osp.join(self.root, self.files[index][1:])
        if self.ret_img_file:
            image = image_path
        elif self.PIL_read:
            image = RGB2BGR(np.asarray(Image.open(image_path), dtype=np.uint8))
        else:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # * Load Label
        if self.split != 'test':
            label_path = osp.join(self.root, self.labels[index][1:])
            label = np.asarray(Image.open(label_path), dtype=np.uint8)
        else:
            # If no label, then giving a all ignore ground truth
            img_size = imagesize.get(image)[::-1] if isinstance(image, str) else image.shape[:2]
            label = np.zeros(img_size, dtype=np.uint8) + self.ignore_label

        return image_id, image, label


class VOCAug2(VOCAug):
    """在标准VOCAug基础上:

    1. 返回cls_lb。
    2. 支持将lb替换为伪标签。
    3. 将输出打包为字典。
    4. 支持返回RGB img。
    """
    def __init__(self, root: str = "./contrib/datasets", year="2012", split: str = "train",
                 cls_labels_type: str='seg_cls_labels',
                 ps_mask_dir: str=None,
                 rgb_img: bool=False,
                 ret_img_file: bool=False):
        super().__init__(root, year, split, PIL_read=True, ret_img_file=ret_img_file)
        # * 参数检查与记录。
        assert cls_labels_type in ('seg_cls_labels', 'det_cls_labels', 'ignore_diff_cls_labels')
        self.cls_labels_type = cls_labels_type
        # * 读取图像级标签。
        with open(osp.join(self.root, 'third_party', f'{cls_labels_type}.pkl'), 'rb') as pkl_f:
            self.id2cls_labels = pickle.load(pkl_f)
        # * 记录伪真值目录。
        self.ps_mask_dir = ps_mask_dir
        # * 是否返回RGB图像。
        self.rgb_img = rgb_img

    def get_item(self, index: int) -> Dict:
        img_id, img, lb = super().get_item(index)

        if self.rgb_img:
            img = BGR2RGB(img).copy()

        out = Dict()
        out.img_id, out.img, out.lb = img_id, img, lb

        if self.split != 'test':
            out.cls_lb = self.id2cls_labels[img_id]
        else:
            out.cls_lb = np.zeros((self.class_num,), dtype=np.uint8)

        if self.ps_mask_dir is not None:
            out.lb = np.asarray(Image.open(osp.join(self.ps_mask_dir, f'{img_id}.png')), dtype=np.uint8)

        return out

    @classmethod
    def subset(cls,
               split_idx_num: tuple[int, int],
               root: str = "./contrib/datasets", year="2012", split: str = "train",
               cls_labels_type: str='seg_cls_labels',
               ps_mask_dir: str=None,
               rgb_img: bool=False):
        dt = cls(root, year, split, cls_labels_type, ps_mask_dir, rgb_img)

        split_idx, split_num = split_idx_num
        assert split_idx < split_num

        step = ceil(len(dt) / split_num)
        indexes = list(range(split_idx * step, min((split_idx + 1) * step, len(dt))))

        sub_dt = Subset(dt, indexes)
        return sub_dt

    @staticmethod
    def label_map2color_map(label_map: np.ndarray) -> np.ndarray:
        return label_map2color_map(label_map)


if __name__ == "__main__":
    from alchemy_cat.acplot.figure_wall import RectFigureWall, RowFigureWall
    import matplotlib.pyplot as plt

    voc = VOC()
    voc_aug = VOCAug(split="train_aug")

    voc_indexes = np.random.choice(len(voc), size=10, replace=False)
    voc_fig_wall = RectFigureWall([RowFigureWall([BGR2RGB(item[1]), label_map2color_map(item[2])], space_width=20)
                                   for item in voc[voc_indexes]],
                                  row_num=2, col_num=5, space_width=10)
    voc_fig_wall.plot(dpi=600)

    plt.show()

    vocaug_indexes = np.random.choice(len(voc_aug), size=10, replace=False)
    vocaug_fig_wall = RectFigureWall([RowFigureWall([BGR2RGB(item[1]), label_map2color_map(item[2])], space_width=40)
                                      for item in voc_aug[vocaug_indexes]],
                                     row_num=2, col_num=5, space_width=10)
    vocaug_fig_wall.plot(dpi=600)

    plt.show()