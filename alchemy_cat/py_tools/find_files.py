#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: find_files.py
@time: 2021/9/16 14:06
@desc: 
"""
from typing import List, Callable

import os
import os.path as osp
import re

__all__ = ['IMG_EXTS', 'find_files', 'find_files_by_exts', 'find_files_by_names', 'find_files_by_patterns',
           'find_img_files']


IMG_EXTS = ['.png', '.jpg', '.JPG', '.PNG', '.JPEG', '.jpeg']


def _match_exts(name_ext: str, exts: List[str]) -> bool:
    ext = osp.splitext(name_ext)[-1]
    return ext in exts


def _match_names(name_ext: str, names: List[str]) -> bool:
    name = osp.splitext(name_ext)[0]
    return name in names


def _match_patterns(name_ext: str, patterns: List[str]) -> bool:
    for pattern in patterns:
        if re.search(pattern, name_ext) is not None:
            return True
    return False


def find_files(root: str, is_match: Callable[[str], bool], recursion: bool=True, memo: str= '',
               is_full_path: bool=True) -> str:
    """Find files by function.

    Args:
        root: Root dir.
        is_match: Function receive files name_ext and return bool to indicate whether found.
        recursion: If True, find files recursively. (Default: True)
        memo: For recursion. (Default: '')
        is_full_path: If True return full path of files found. (Default: True)

    Returns:
        Yields find files under root.
    """
    name_exts = os.listdir(osp.join(root, memo))
    for name_ext in name_exts:
        full_path = osp.join(root, memo, name_ext)
        if osp.isdir(full_path) and recursion:
            yield from find_files(root, is_match, recursion, osp.join(memo, name_ext), is_full_path)
        elif osp.isfile(full_path) and is_match(name_ext):
            yield full_path if is_full_path else name_ext
        else:
            pass


def find_files_by_exts(root: str, exts: List[str], recursion: bool=True, memo: str= '', is_full_path: bool=True) -> str:
    """Find files by function.

    Args:
        root: Root dir.
        exts: Extensions wanted.
        recursion: If True, find files recursively. (Default: True)
        memo: For recursion. (Default: '')
        is_full_path: If True return full path of files found. (Default: True)

    Returns:
        Yields find files under root.
    """
    return find_files(root, lambda name_ext: _match_exts(name_ext, exts), recursion, memo, is_full_path)


def find_files_by_names(root: str, names: List[str], recursion: bool=True, memo: str= '',
                        is_full_path: bool=True) -> str:
    """Find files by function.

    Args:
        root: Root dir.
        names: Names wanted.
        recursion: If True, find files recursively. (Default: True)
        memo: For recursion. (Default: '')
        is_full_path: If True return full path of files found. (Default: True)

    Returns:
        Yields find files under root.
    """
    return find_files(root, lambda name_ext: _match_names(name_ext, names), recursion, memo, is_full_path)


def find_files_by_patterns(root: str, patterns: List[str], recursion: bool=True, memo: str= '', 
                           is_full_path: bool=True) -> str:
    """Find files by function.

    Args:
        root: Root dir.
        patterns: Patterns wanted.
        recursion: If True, find files recursively. (Default: True)
        memo: For recursion. (Default: '')
        is_full_path: If True return full path of files found. (Default: True)

    Returns:
        Yields find files under root.
    """
    return find_files(root, lambda name_ext: _match_patterns(name_ext, patterns), recursion, memo, is_full_path)


def find_img_files(root: str, recursion: bool=True, is_full_path: bool=True) -> str:
    """Find all image files.

    Args:
        root: Root dir.
        recursion: If True, find files recursively. (Default: True)
        is_full_path: If True return full path of files found. (Default: True)

    Returns:
        Yields find images under root.
    """
    return find_files_by_exts(root, IMG_EXTS, recursion, is_full_path=is_full_path)
