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
import os
import copy
import numpy as np
from matplotlib import pyplot as plt
from typing import Union, Optional, Iterable
from collections import abc

from alchemy_cat.acplot.utils import stack_figs
from alchemy_cat.acplot.shuffle_ch import BGR2RGB, RGB2BGR
from alchemy_cat.py_tools.type import is_intarr
from alchemy_cat.py_tools.str_formatters import indent
from alchemy_cat.alg import color2scalar


__all__ = ["SquareFigureWall", "RectFigureWall", "RowFigureWall", "ColumnFigureWall"]


KDefaultPadValue = (127, 140, 141) # BGR Default pad value


class FigureWall(object):
    """A class to operate a figure wall"""

    def __init__(self, figs: Union[np.ndarray, Iterable[Union[np.ndarray, 'FigureWall']]],
                 is_normalize: bool = False, space_width: int = 1,
                 img_pad_val: Union[int, float, Iterable] = None,
                 pad_location: Union[str, int]='right-bottom', color_channel_order: str='RGB'):
        """A class to operate a figure wall

        Args:
            figs (Iterable, np.ndarray): Iterable[fig] or figs((N, H, W, C) ndarray). fig is supposed to be (H, W, C)
                ndarray or a figure wall which will be seen as it's tiled_figs.
            is_normalize (bool): (Deprecated)If true, the figures will be min-max normalized.
            space_width (int): Space width between figs
            img_pad_val: (Union[int, float, Iterable]): Img pad value when stack imgs.If value is int or float, return
                (value, value, value), if value is Iterable with 3 element, return totuple(value), else raise error.
                (Default: BGR(127, 140, 141))
            pad_location (Union[str, int]): Img pad location when stack imgs. Indicate pad location. Can be
                'left-top'/0, 'right-top'/1, 'left-bottom'/2, 'right-bottom'/3, 'center'/4.
            color_channel_order (str): Indicate color channel order. If 'BGR', then fig of figs should be BGR ndarray. If fig is a
                FigureWall object, fig.BGR() will be called to make figure_wall's color channel order meet with
                indicated order. Same to the 'RGB'.
        """
        if color_channel_order not in ('BGR', 'RGB'):
            raise ValueError(f"color_channel_order {color_channel_order} must be \"BGR\" or \"RGB\"")
        color_mode_converter = (lambda x: x.BGR()) if color_channel_order == 'BGR' else (lambda x: x.RGB())

        if img_pad_val is None:
            img_pad_val = KDefaultPadValue if color_channel_order == 'BGR' else KDefaultPadValue[::-1]
        else:
            img_pad_val = color2scalar(img_pad_val)

        if isinstance(figs, list):
            figs_to_stack = []
            for fig in figs:
                if isinstance(fig, np.ndarray):
                    figs_to_stack.append(fig)
                elif isinstance(fig, FigureWall):
                    figs_to_stack.append(color_mode_converter(fig).tiled_figs)
                else:
                    raise ValueError(f"The fig {fig} should be ndarray or FigureWall")

            self.figs = stack_figs(figs_to_stack, img_pad_val, pad_location)
        elif isinstance(figs, np.ndarray):
            self.figs = figs
        elif isinstance(figs, abc.Iterable):
            self.__init__(list(figs), is_normalize, space_width, img_pad_val, pad_location, color_channel_order)
            return
        else:
            raise ValueError("The figs should be Iterator of (H, W, C) imgs or (N, H, W, C) ndarray")

        self.color_channel_order = color_channel_order
        self.space_width = space_width
        self.is_normalize = is_normalize
        self.img_pad_val = img_pad_val
        self.pad_location = pad_location

        if is_normalize:
            self.figs = (self.figs - self.figs.min()) / (self.figs.max() - self.figs.min())

        self.tiled_figs = self._tile_figs(self.space_width)


    def _tile_figs(self, space_width) -> np.ndarray:
        """Tiles figs to figure wall.
        Args:
            space_width (int): Space width between figs

        Returns:
            ndarray with shape (H, W, C)
        """
        raise NotImplementedError

    def plot(self, **kwargs) -> plt.Figure:
        """Plot figure wall with pyplot.
        Args:
            **kwargs (): kwargs for plt.figure()

        Returns:
            Shown figure
        """
        tiled_figs = self.tiled_figs if self.color_channel_order == 'RGB' else BGR2RGB(self.tiled_figs)

        figure = plt.figure(**kwargs)
        plt.imshow(tiled_figs)
        plt.axis('off')

        return figure

    def save(self, img_file: str, **kwargs):
        """Save figure wall to file

        Args:
            img_file (str): Path to save the figure wall
            kwargs (dict): key-word arguments for plt.imsave

        Returns:
            None
        """
        tiled_figs = self.tiled_figs if self.color_channel_order == 'RGB' else BGR2RGB(self.tiled_figs)

        os.makedirs(os.path.split(img_file)[0], exist_ok=True)

        plt.imsave(img_file, tiled_figs, **kwargs)

    def __add__(self, other: 'FigureWall') -> 'FigureWall':
        """Collect figs from self and other's figs and tile them to a new figure wall.

        Args:
            other (FigureWall): A figure wall instance

        Returns:
            A new figure wall with figs from self and other. The type and attribute of figure wall
                will follow the first factor.
        """
        if not isinstance(other, FigureWall):
            raise ValueError(f"Adder {other} <{type(other)}> should be FigureWall")

        other = other.BGR() if self.color_channel_order == 'BGR' else other.RGB()
        figs = list(self.figs) + list(other.figs)
        return type(self)(figs, self.is_normalize, self.space_width, self.img_pad_val, self.pad_location,
                          self.color_channel_order)

    def __repr__(self):
        return f"FigureWall <{self.__class__.__name__}:>" + "\n" + \
                indent(f"#figs: {self.figs.shape[0]}") + "\n" + \
                indent(f"space width: {self.space_width}")

    def BGR(self, inplace_if_converted: bool=False) -> 'FigureWall':
        """Convert to a BGR figure wall

        If figure wall is already an BGR figure wall, then return self. Else return a copy of self which is converted
        to BGR mode when inplace_if_concerted=False, or just convert figure wall self to BGR mode then return it's self
        when inplace_if_concerted=True

        See Also:
            FigureWall.RGB

        Args:
            inplace_if_converted: When figure wall is in RGB mode, then convert a copy of self to BGR mode to return if
                inplace_if_converted=False, else convert self to BGR mode then return self if replace_if_converted=True.
                (Default: False)

        Returns:
            A BGR figure wall
        """
        if self.color_channel_order == 'BGR':
            return self

        if inplace_if_converted:
            ret = self
        else:
            ret = copy.deepcopy(self)

        ret.color_channel_order = 'BGR'
        ret.figs = RGB2BGR(ret.figs)
        ret.tiled_figs = RGB2BGR(ret.tiled_figs)
        ret.img_pad_val = ret.img_pad_val[::-1]

        return ret

    def RGB(self, inplace_if_converted: bool=False) -> 'FigureWall':
        """Convert to a RGB figure wall

        If figure wall is already an RGB figure wall, then return self. Else return a copy of self which is converted
        to RGB mode when inplace_if_concerted=False, or just convert figure wall self to RGB mode then return it's self
        when inplace_if_concerted=True

        See Also:
            FigureWall.BGR

        Args:
            inplace_if_converted: When figure wall is in BGR mode, then convert a copy of self to RGB mode to return if
                inplace_if_converted=False, else convert self to RGB mode then return self if replace_if_converted=True.
                (Default: False)

        Returns:
            A RGB figure wall
        """
        if self.color_channel_order == 'RGB':
            return self

        if inplace_if_converted:
            ret = self
        else:
            ret = copy.deepcopy(self)

        ret.color_channel_order = 'RGB'
        ret.figs = BGR2RGB(ret.figs)
        ret.tiled_figs = BGR2RGB(ret.tiled_figs)
        ret.img_pad_val = ret.img_pad_val[::-1]

        return ret


class SquareFigureWall(FigureWall):
    """Figures will be tiled to an Square"""

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
    """Figures will be tiled to a Rectangle"""

    def __init__(self, figs: Union[Iterable[Union[np.ndarray, 'FigureWall']], np.ndarray], is_normalize: bool = False,
                 space_width: int = 1, row_num: Optional[int]=None, col_num: Optional[int]=None,
                 img_pad_val: Union[int, float, Iterable] = None,
                 pad_location: Union[str, int]='right-bottom', color_channel_order: str='RGB'):
        """Figures will be tiled to an Rectangle

        Args:
            figs (list, np.ndarray): List[fig] or figs. fig is supposed to be (H, W, C) and RGB mode.
            is_normalize (bool): If true, the figures will be min-max normalized
            space_width (int): Space width between figs
            row_num (Optional[int]): row num of rectangle figure wall. If None, it will be calculated by
                ceil(figure's num / col_num)
            col_num (Optional[int]): cow num of rectangle figure wall. If None, it will be calculated by
                ceil(figure's num / row_num)
            img_pad_val: Same to the param of FigureWall
            pad_location (Union[str, int]): Same to the param of FigureWall
            color_channel_order (str): Same to the param of FigureWall

        See Also:
            FigureWall
        """
        self.row_num = row_num
        self.col_num = col_num
        super(RectFigureWall, self).__init__(figs, is_normalize, space_width,
                                             img_pad_val, pad_location, color_channel_order)

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
    """Figures will be tiled to a Column"""

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
    from alchemy_cat.contrib.voc import VOCAug
    voc_dataset = VOCAug()
    img_wall = RowFigureWall([img for _, img, label in voc_dataset[:3]], pad_location='center')
    img_wall.plot(dpi=600)

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