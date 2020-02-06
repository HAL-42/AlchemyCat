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
import cv2
import numpy as np
from PIL import Image
import os.path as osp

from alchemy_cat.data.dataset import IterableDataset, Dataset
from alchemy_cat import BGR2RGB
from contrib.voc.utils import label_map2color_map

__all__ = ['VOC', 'VOCAug']


class _VOCBase(IterableDataset):
    """
    PASCAL VOC and VOC Aug Segmentation base dataset
    """
    mean_bgr = [104.008, 116.669, 122.675]
    ignore_label = 255

    def __init__(self, root: str="datasets", year="2012", split: str="train"):
        """
        Args:
            root (str): The parent dir of VOC dataset
            split (str): "train"/"val"/"trainval"
        """
        self.root = root
        self.year = year
        self.split = split
        self.files = []
        self._set_files()

        cv2.setNumThreads(0)

    def _set_files(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.files)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


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

    def get_item(self, index):
        # Set paths
        image_id = self.files[index]
        image_path = osp.join(self.image_dir, image_id + ".jpg")
        label_path = osp.join(self.label_dir, image_id + ".png")
        # Load an image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = np.asarray(Image.open(label_path), dtype=np.uint8)
        return image_id, image, label


class VOCAug(_VOCBase):
    """
    PASCAL VOC Aug Segmentation dataset
    """
    def __init__(self, root: str = "datasets", year="2012", split: str = "train"):
        """
        Args:
            root (str): The parent dir of VOC dataset
            split (str): "train"/"val"/"trainval"/"train_aug"/"trainval_aug"
        """
        self.labels = []
        super(VOCAug, self).__init__(root, year, split)

    def _set_files(self):
        self.root = osp.join(self.root, f"VOC{self.year}")

        if self.split in ["train", "train_aug", "trainval", "trainval_aug", "val"]:
            file_list = osp.join(
                self.root, "ImageSets", "SegmentationAug", self.split + ".txt"
            )
            file_list = tuple(open(file_list, "r"))
            file_list = [id_.rstrip().split(" ") for id_ in file_list]
            self.files, self.labels = list(zip(*file_list))
        else:
            raise ValueError("Invalid split name: {}".format(self.split))

    def get_item(self, index):
        # Set paths
        image_id = self.files[index].split("/")[-1].split(".")[0]
        image_path = osp.join(self.root, self.files[index][1:])
        label_path = osp.join(self.root, self.labels[index][1:])
        # Load an image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = np.asarray(Image.open(label_path), dtype=np.uint8)
        return image_id, image, label


if __name__ == "__main__":
    from alchemy_cat.visualization.figure_wall import RectFigureWall, RowFigureWall
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