#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: figure_wall.py
@time: 2020/1/7 22:18
@desc:
"""
import numpy as np
from matplotlib import pyplot as plt
from typing import Union, Optional, Iterable
from collections import abc

from alchemy_cat.visualization.utils import stack_figs
from alchemy_cat.py_tools import is_intarr, indent


__all__ = ["SquareFigureWall", "RectFigureWall", "RowFigureWall", "ColumnFigureWall"]


class FigureWall(object):
    """
    A class to operate a figure wall
    """

    def __init__(self, figs: Union[np.ndarray, Iterable[Union[np.ndarray, 'FigureWall']]],
                 is_normalize: bool = False, space_width: int = 1):
        """
        Args:
            figs (Iterable, np.ndarray): Iterable[fig] or figs. fig is supposed to be (H, W, C) and RGB mode.
            is_normalize (bool): If true, the figures will be min-max normalized
            space_width (int): Space width between figs
        """
        if isinstance(figs, list):
            if isinstance(figs[0], np.ndarray):
                self.figs = stack_figs(figs)
            elif isinstance(figs[0], FigureWall):
                self.figs = stack_figs([figure_wall.tiled_figs for figure_wall in figs])
        elif isinstance(figs, np.ndarray):
            self.figs = figs
        elif isinstance(figs, abc.Iterable):
            self.__init__(list(figs), is_normalize, space_width)
            return
        else:
            raise TypeError("The figs should be Iterator of (H, W, C) imgs or (N, H, W, C) ndarray")

        if is_normalize:
            self.figs = (self.figs - self.figs.min()) / (self.figs.max() - self.figs.min())

        self.tiled_figs = self._tile_figs(space_width)

        self.space_width = space_width

    def _tile_figs(self, space_width):
        """
        Tiles figs to figure wall.
        Args:
            space_width (int): Space width between figs

        Returns:
            ndarray with shape (H, W, C)
        """
        raise NotImplementedError

    def plot(self, **kwargs) -> plt.Figure:
        """
        Show figure wall.
        Args:
            **kwargs (): kwargs for plt.figure()

        Returns:
            Showed figure
        """
        figure = plt.figure(**kwargs)
        plt.imshow(self.tiled_figs)
        plt.axis('off')

        return figure

    def save(self, img_file: str):
        """
        Save figure wall to file.
        Args:
            img_file (str): Path to save the figure wall

        Returns:
            None
        """
        plt.imsave(img_file, self.tiled_figs)

    def __add__(self, other: 'FigureWall') -> 'FigureWall':
        """
        Args:
            other (FigureWall): A figure wall instance

        Returns:
            A new figure wall with figs from self and other. The type of figure wall will follow the first factor.

        Collect figs from self and other's figs and tile them to a new figure wall.
        """
        cls = type(self)
        figs = list(self.figs) + list(other.figs)
        return cls(figs, space_width=self.space_width)

    def __repr__(self):
        return f"FigureWall <{self.__class__.__name__}:>" + "\n" + \
                indent(f"#figs: {self.figs.shape[0]}") + "\n" + \
                indent(f"space width: {self.space_width}")


class SquareFigureWall(FigureWall):
    """
    Figures will be tiled to an Square
    """

    def _tile_figs(self, space_width):
        n = int(np.ceil(np.sqrt(self.figs.shape[0])))
        padding = (((0, n ** 2 - self.figs.shape[0]),
                    (0, space_width), (0, space_width))  # add some space between filters
                   + ((0, 0),) * (self.figs.ndim - 3))  # don't pad the last dimension (if there is one)
        if is_intarr(self.figs):
            constant_values = 255
        else:
            constant_values = 1.

        tiled_figs = np.pad(self.figs, padding, mode='constant', constant_values=constant_values)  # pad with white

        # tile the filters into an image
        tiled_figs = tiled_figs.reshape((n, n) + tiled_figs.shape[1:]).transpose(
            (0, 2, 1, 3) + tuple(range(4, tiled_figs.ndim + 1)))
        tiled_figs = tiled_figs.reshape((n * tiled_figs.shape[1], n * tiled_figs.shape[3]) + tiled_figs.shape[4:])

        return tiled_figs[:-space_width, :-space_width, ...]  # Delete the padding border


class RectFigureWall(FigureWall):

    def __init__(self, figs: Union[Iterable[Union[np.ndarray, 'FigureWall']], np.ndarray], is_normalize: bool = False,
                 space_width: int = 1, row_num: Optional[int]=None, col_num: Optional[int]=None):
        """Figures will be tiled to an Rectangle

        Args:
            figs (list, np.ndarray): List[fig] or figs. fig is supposed to be (H, W, C) and RGB mode.
            is_normalize (bool): If true, the figures will be min-max normalized
            space_width (int): Space width between figs
            row_num (Optional[int]): row num of rectangle figure wall. If None, it will be calculated by ceil(figure's num / col_num)
            col_num (Optional[int]): cow num of rectangle figure wall. If None, it will be calculated by ceil(figure's num / row_num)
        """
        self.row_num = row_num
        self.col_num = col_num
        super(RectFigureWall, self).__init__(figs, is_normalize, space_width)

    def _tile_figs(self, space_width):
        if self.row_num is None and self.col_num is None:
            raise ValueError(f"row_num={self.row_num} or col_num={self.col_num} should not be int, not None")

        figs_num = self.figs.shape[0]
        if self.row_num is None:
            self.row_num = int(np.ceil(figs_num / self.col_num))
        elif self.col_num is None:
            self.col_num = int(np.ceil(figs_num / self.row_num))

        n = self.row_num * self.col_num

        if n < self.figs.shape[0]:
            raise ValueError(f"row_num * col_num = {n} < self.figs.shape[0]")

        padding = (((0, n - self.figs.shape[0]),
                    (0, space_width), (0, space_width))  # add some space between filters
                   + ((0, 0),) * (self.figs.ndim - 3))  # don't pad the last dimension (if there is one)
        if is_intarr(self.figs):
            constant_values = 255
        else:
            constant_values = 1.

        tiled_figs = np.pad(self.figs, padding, mode='constant', constant_values=constant_values)  # pad with white

        # tile the filters into an image
        tiled_figs = tiled_figs.reshape((self.row_num, self.col_num) + tiled_figs.shape[1:]).transpose(
            (0, 2, 1, 3) + tuple(range(4, tiled_figs.ndim + 1)))
        tiled_figs = tiled_figs.reshape((self.row_num * tiled_figs.shape[1], self.col_num * tiled_figs.shape[3]) + tiled_figs.shape[4:])

        return tiled_figs[:-space_width, :-space_width, ...]  # Delete the padding border


class ColumnFigureWall(FigureWall):
    """
    Figures will be tiled to a Column
    """

    def _tile_figs(self, space_width):
        padding = ( ((0, 0), (0, space_width)) + ((0, 0),) * (self.figs.ndim - 2) )
        if is_intarr(self.figs):
            constant_values = 255
        else:
            constant_values = 1.

        tiled_figs = np.pad(self.figs, padding, mode='constant', constant_values=constant_values)

        return tiled_figs.reshape((tiled_figs.shape[0] * tiled_figs.shape[1],) + tiled_figs.shape[2:])[:-space_width, ...]


class RowFigureWall(FigureWall):
    """
    Figures will be tiled to a Row
    """

    def _tile_figs(self, space_width):
        padding = ( ((0, 0), (0, 0), (0, space_width)) + ((0, 0),) * (self.figs.ndim - 3) )
        if is_intarr(self.figs):
            constant_values = 255
        else:
            constant_values = 1.

        tiled_figs = np.pad(self.figs, padding, mode='constant', constant_values=constant_values)

        tiled_figs = tiled_figs.transpose((1, 0, 2) + tuple(range(3, tiled_figs.ndim)))
        tile_figs = tiled_figs.reshape((tiled_figs.shape[0], tiled_figs.shape[1] * tiled_figs.shape[2]) + tiled_figs.shape[3:])

        return tile_figs[:, :-space_width, ...]


if __name__ == "__main__":
    col_wall = ColumnFigureWall(list(np.random.randn(10, 224, 224, 3)), is_normalize=True, space_width=5)
    col_wall.plot(dpi=300)

    row_wall = RowFigureWall(list(np.random.randn(10, 224, 224, 3)), is_normalize=True, space_width=5)
    row_wall.plot(dpi=300)

    sq_wall = SquareFigureWall(list(np.random.randn(10, 224, 224, 3)), is_normalize=True, space_width=5)
    sq_wall.plot(dpi=300)

    rect_wall = RectFigureWall(list(np.random.randn(10, 224, 224, 3)), is_normalize=True, space_width=5, row_num=2, col_num=5)
    rect_wall.plot(dpi=300)

    rect_wall = RectFigureWall(list(np.random.randn(9, 224, 224, 3)), is_normalize=True, space_width=5, row_num=2)
    rect_wall.plot(dpi=300)

    rect_wall = RectFigureWall(list(np.random.randn(9, 224, 224, 3)), is_normalize=True, space_width=5, col_num=4)
    rect_wall.plot(dpi=300)

    wall_wall = RowFigureWall([col_wall, row_wall, sq_wall], space_width=10)
    wall_wall.plot(dpi=300)

    add_wall = sq_wall + col_wall + row_wall
    add_wall.plot(dpi=300)

    plt.show()