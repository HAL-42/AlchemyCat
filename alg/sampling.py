#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2022/5/3 14:09
@File    : samp.py
@Software: PyCharm
@Desc    : 采样工具包。
"""
from typing import Union, Optional, List

import math

import torch

__all__ = ['shuffle_cat', 'samp_idx_with_1d_mask', 'samp_idxes_with_2d_mask', 'top_k_on_mask', 'samp_on_2d_mask']


def _get_rank_argsort_argsort(val: torch.Tensor, reverse: bool=False, dim: int=1) -> torch.Tensor:
    return val.argsort(dim=dim, descending=reverse).argsort(dim=dim)


def _get_rank_argsort_scatter(val: torch.Tensor, reverse: bool=False) -> torch.Tensor:
    idx_sorted = val.argsort(dim=1, descending=reverse)
    rank_to_scatter = torch.arange(val.shape[1], dtype=torch.int32, device=val.device)[None, :].expand(val.shape)
    rank = torch.zeros_like(rank_to_scatter)
    return rank.scatter_(1, idx_sorted, rank_to_scatter)


def _get_rank_with_split(val: torch.Tensor, reverse: bool=False, split_size: int=2000) -> torch.Tensor:
    rank = torch.empty_like(val, dtype=torch.int32)
    for i in range(math.ceil(val.shape[0] / split_size)):
        rank[split_size * i:min(split_size * (i + 1), rank.shape[0]), :] = \
            _get_rank_argsort_scatter(val[split_size * i:min(split_size * (i + 1), rank.shape[0]), :], reverse)
    return rank


def shuffle_cat(tensor: torch.Tensor, shuffled_len: int=None, shuffle_all: bool=False,
                g: Optional[torch.Generator]=None) -> torch.Tensor:
    """将张量沿着0维多次洗牌后拼接到shuffled_len长度。

    Args:
        tensor: 被洗牌、拼接的的(N, ...)张量。
        shuffled_len: 洗牌、拼接后的长度。若为None，则为tensor.shape[0]。
        shuffle_all: 是否要完全洗牌，即于洗牌、拼接后的张量上，再做一次整体洗牌。
        g: torch的随机序列生成器。注意若不为None，则要和tensor在同一设备上。

    Returns:
        洗牌、拼接后的(shuffled_len, ...)张量。
    """
    if shuffled_len is None:
        shuffled_len = tensor.shape[0]

    weights = torch.ones((math.ceil(shuffled_len / tensor.shape[0]), tensor.shape[0]),
                         dtype=torch.float32, device=tensor.device)
    shuffled_idxes = torch.multinomial(weights, tensor.shape[0], replacement=False, generator=g)
    shuffled_idxes = shuffled_idxes.view(-1)[:shuffled_len]

    if shuffle_all:
        shuffled_idxes = shuffle_cat(shuffled_idxes, shuffled_idxes.shape[0], shuffle_all=False, g=g)

    return tensor[shuffled_idxes]


def samp_idx_with_1d_mask(mask: torch.Tensor, samp_num: int, resamp_lim: int=1,
                          g: Optional[torch.Generator] = None) -> torch.Tensor:
    """在1维掩码为真处，尝试采样samp_num个索引。若True个数小于samp_num，则尝试多轮采样采足samp_num个索引。
    但每个索引被重采样的次数不得超过resamp_lim次。

    Args:
        mask: 1维布尔掩码。
        samp_num: 尝试采样多少个索引。
        resamp_lim: 每个索引至多比重采样几次。
        g: torch的随机序列生成器。注意若不为None，则要和mask在同一设备上。

    Returns:
        采样后的1维索引。
    """
    assert mask.ndim == 1
    # * 获取mask为True位置的索引，即可采索引。
    idx: torch.Tensor = torch.nonzero(mask, as_tuple=True)[0]
    if idx.shape[0] == 0:
        return idx
    # * 计算（最大）实采数。
    max_final_samp_num = idx.shape[0] * resamp_lim
    final_samp_num = min(samp_num, max_final_samp_num)
    # * 将可采索引洗牌拼接到洗牌拼接到实采数长度并返回。
    return shuffle_cat(idx, final_samp_num, shuffle_all=False, g=g)


def samp_idxes_with_2d_mask(mask: torch.Tensor, samp_nums: Union[int, torch.Tensor], resamp_lim: int=1,
                            g: Optional[torch.Generator] = None) -> List[torch.Tensor]:
    """对2维掩码的每一行i，做samp_idx_with_1d_mask(mask[i], samp_nums[i], ...)。返回List[每行采出索引]

    Args:
        mask: 2维布尔掩码。
        samp_nums: 尝试采样多少个索引。若为int，则每行尝试采样samp_nums个索引；若为Tensor，则行i尝试采样sump_nums[i]个索引。
        resamp_lim: 每个索引至多比重采样几次。
        g: torch的随机序列生成器。注意若不为None，则要和mask在同一设备上。

    Returns:
        List[各行的采出索引]。
    """
    assert mask.ndim == 2
    if torch.is_tensor(samp_nums):
        assert samp_nums.shape[0] == mask.shape[0]

    samped_idxes = []
    # * 对mask的每一行，做带mask索引采样。
    for r in range(mask.shape[0]):
        samp_num = samp_nums[r].item() if torch.is_tensor(samp_nums) else samp_nums
        samped_idxes.append(samp_idx_with_1d_mask(mask[r], samp_num, resamp_lim, g))
    return samped_idxes


def _rank_with_mask(val: torch.Tensor, mask: torch.Tensor, reverse: bool=False) -> torch.Tensor:
    """在mask前景位置，沿着dim，得到val的排名。mask背景位置的排名总是为-1。

    Args:
        val: 被排序的值。
        mask: 掩码，mask背景位置的值不参与排名。
        reverse: 若为True，则从大到小排序，反之从小大大排序。

    Returns:
        val同尺寸张量，在mask前景位置，表示val沿着dim维的排名，背景位置总是为-1。
    """
    assert mask.dtype == torch.bool
    assert torch.is_floating_point(val)
    assert val.shape == mask.shape
    assert val.device == mask.device
    # * 得到排序值。若不reverse（从小到大排），排序值的背景部分为inf，反之为-inf。
    sort_val = val.clone()
    sort_val[~mask] = float('inf') if not reverse else -float('inf')
    # * 根据排序值，得到排名。
    rank = _get_rank_with_split(sort_val, reverse=reverse)
    # * 背景部分排名设为-1，返回。
    rank[~mask] = -1
    return rank


def top_k_on_mask(val: torch.Tensor, mask: torch.Tensor, k: int, reverse: bool=False) -> torch.Tensor:
    """沿着指定维度，将val前景（由mask指定）的top-k设为True。若前景不足k个，则所有前景均为True。

    Args:
        val: 用来寻找top-k的值。
        mask: 掩码，指定val的前背景。
        k: top-k的k值。
        reverse: 若为True，寻找最大的top-k，反之寻找最小的top-k。

    Returns:
        val同尺寸张量，沿着dim维度，top-k前景为True，反之为False。
    """
    rank = _rank_with_mask(val, mask, reverse)
    return (rank >= 0) & (rank < k)


def samp_on_2d_mask(mask: torch.Tensor, samp_nums: Union[int, torch.Tensor],
                    g: Optional[torch.Generator] = None) -> torch.Tensor:
    """沿着2维掩码的第二维，尝试采样samp_nums个前景设为True。若前景数小于采样数，则保留所有前景。

    Args:
        mask: 2维掩码。
        samp_nums: 尝试采样多少个前景。若为int，则每行尝试采样samp_nums个索引；若为Tensor，则行i尝试采样sump_nums[i]个索引。
        g: torch的随机序列生成器。注意若不为None，则要和mask在同一设备上。

    Returns:
        mask同尺度张量，采出前景为True，其余为False。
    """
    assert mask.ndim == 2
    if torch.is_tensor(samp_nums):
        assert samp_nums.shape[0] == mask.shape[0]
    # * 得到mask每行元素得到一个不重复的随机值。
    weights = torch.ones_like(mask, dtype=torch.float32)
    rand_val = torch.multinomial(weights, mask.shape[1], replacement=False, generator=g).to(torch.float32)
    # * 根据随机值排序，mask的前景得到一个随机排名。
    rand_rank = _rank_with_mask(rand_val, mask)
    # * 返回随机排名小于采样数的前景。
    if torch.is_tensor(samp_nums):
        return (rand_rank >= 0) & (rand_rank < samp_nums[:, None])
    else:
        return (rand_rank >= 0) & (rand_rank < samp_nums)
