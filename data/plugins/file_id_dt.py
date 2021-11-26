#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2021/11/24 22:51
@File    : file_id_dt.py
@Software: PyCharm
@Desc    : 
"""
import os.path as osp

import cv2

from alchemy_cat.data import Dataset
from alchemy_cat.py_tools.find_files import find_files_by_exts

__all__ = ["FileIDDt", "ImgIDDt"]


class FileIDDt(Dataset):
    """Dataset organized with file id"""

    def __init__(self, root_dir: str, file_ext: str):
        """
        Args:
            root_dir (str): root dir.
            file_ext (str): File extension map, can be '.png', '.jpg', etc.
        """
        self.root_dir = root_dir
        self.file_ext = file_ext

        self.files = None
        self.file_ids = None
        self._set_files()

    def _set_files(self):
        self.files = list(find_files_by_exts(self.root_dir, [self.file_ext]))
        self.files.sort()
        self.file_ids = [osp.basename(img_file).split('.')[0] for img_file in self.files]

    def __len__(self):
        return len(self.file_ids)

    def get_item(self, index):
        return self.file_ids[index], self.load_file(self.files[index])

    def load_file(self, file):
        raise NotImplementedError

    def get_by_id(self, file_id: str):
        try:
            index = self.file_ids.index(file_id)
        except ValueError:
            raise RuntimeError(f"Can't find file_id {file_id} in dataset's file_ids list.")
        return self[index]


class ImgIDDt(FileIDDt):
    """Dataset organized with img_id"""

    def __init__(self, root_dir: str, file_ext: str = '.png', cv2_read_flag: int = cv2.IMREAD_GRAYSCALE):
        super().__init__(root_dir, file_ext)
        self.cv2_read_flag = cv2_read_flag

    def load_file(self, file):
        img = cv2.imread(file, self.cv2_read_flag)
        assert img is not None
        return img
