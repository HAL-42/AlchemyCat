#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2021/6/16 17:10
@File    : viz_seed.py
@Software: PyCharm
@Desc    : 
"""
import matplotlib.pyplot as plt
import numpy as np

from alchemy_cat.contrib.voc import label_map2color_map

__all__ = ['viz_seed']


def viz_seed(fig: plt.Figure, img_id: str, bottom_img: np.ndarray, label: np.ndarray, seed: np.ndarray,
             blend_alpha: float=0.7, color_map: callable=label_map2color_map):
    """ 可视化种子点。作图结果为：图像+label，图像+种子点，种子点。

    Args:
        fig: plt的Figure对象。
        img_id: 图像的编号。
        bottom_img: 图像数组。
        label: 标签数组。
        seed: 种子点数组。
        blend_alpha: 叠加图像时的透明度。
        color_map: 函数，输入单通道的数值标签，输出3通道的彩色标签。

    Returns:
        plt的Figure对象。
    """
    # * Show bottom_img + gt
    ax: plt.Axes = fig.add_subplot(3, 1, 1)
    ax.imshow(bottom_img)
    ax.imshow(color_map(label), alpha=blend_alpha)
    ax.set_title(img_id, fontsize='smaller')
    ax.axis("off")

    # * Show bottom_img + seg_mask
    ax: plt.Axes = fig.add_subplot(3, 1, 2)
    ax.imshow(bottom_img)
    ax.imshow(color_map(seed), alpha=blend_alpha)
    ax.axis("off")

    # * Show seg_mask only
    ax: plt.Axes = fig.add_subplot(3, 1, 3)
    ax.imshow(color_map(seed))
    ax.axis("off")

    fig.tight_layout(pad=0.2, h_pad=0.2, w_pad=0.2)
    return fig
