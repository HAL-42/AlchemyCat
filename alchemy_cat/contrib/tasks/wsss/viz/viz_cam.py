#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2021/5/8 22:18
@File    : viz_cam.py
@Software: PyCharm
@Desc    : 
"""
from typing import List

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from alchemy_cat.contrib.voc import label_map2color_map

from alchemy_cat.acplot.subplots_row_col import square

__all__ = ["viz_cam"]


def viz_cam(fig: plt.Figure,
            img_id: str, img: np.ndarray, label: np.ndarray,
            cls_in_label: np.ndarray, cam: np.ndarray, cls_names: List[str],
            gamma: float = 1.0, blend_alpha: float = 0.5,
            get_row_col: callable=square, color_map: callable=label_map2color_map) -> plt.Figure:
    """ 可视化CAM。作图结果为：图像+label，图像+CAM0，图像+CAM1，······。

    Args:
        fig: plt的Figure对象。
        img_id: 图像的编号。
        img: 图像数组。
        label: 标签数组。
        cls_in_label: 长度为类别数的0,1数组，表示那些类别为正类。正类名字会被标红。
        cam: 类别激活图数组。
        cls_names: 类别名字。
        gamma: 若不为1，则对CAM做gamma矫正，但此时无法画出色带。
        blend_alpha: 叠加图像时的透明度。
        get_row_col: 函数，返回作图的行列数。
        color_map: 函数，输入单通道的数值标签，输出3通道的彩色标签。

    Returns:
        plt的Figure对象。
    """
    assert cam.shape[1:] == img.shape[:2]
    assert cam.shape[0] + 1 == len(cls_names)
    assert len(cls_names) == len(cls_in_label)

    # * Get grid size
    row_num, col_num = get_row_col(len(cls_names))

    # * Plot ref img
    ax: plt.Axes = fig.add_subplot(row_num, col_num, 1)

    ax.imshow(img)
    ax.imshow(color_map(label.copy()), alpha=blend_alpha)
    ax.set_title(img_id, fontsize='smaller')
    ax.axis("off")

    # * Show CAM
    for fore_cls, cls_name in enumerate(cls_names[1:]):
        # * Show CAM with bottom img.
        ax = fig.add_subplot(row_num, col_num, fore_cls + 2)
        ax.imshow(img)

        # * Get CAM
        cls_cam = cam[fore_cls, ...].astype(np.float32)

        if gamma == 1.0:
            mappable = ax.imshow(cls_cam, cmap=plt.get_cmap('jet'), alpha=blend_alpha)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2.5%", pad=0.05)
            cax.tick_params(labelsize=3, length=2, width=0.5, pad=0.5)

            fig.colorbar(mappable=mappable, cax=cax)
        else:
            cls_viz_cam = ((cls_cam - cls_cam.min()) / (cls_cam.max() - cls_cam.min())) ** gamma
            ax.imshow(cls_viz_cam, cmap=plt.get_cmap('jet'), alpha=blend_alpha)

        # * Set CAM Title
        font_dict = dict(color='red', fontweight='bold') if cls_in_label[fore_cls + 1] else dict()
        ax.set_title(cls_name, fontsize='smaller', **font_dict)
        ax.axis("off")

    fig.tight_layout(pad=0.2, h_pad=0.1, w_pad=0.2)
    return fig
