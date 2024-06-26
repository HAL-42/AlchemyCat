#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: dataset.py
@time: 2019/12/7 23:46
@desc: A dataset lib. Quite similar to torch.util.dataset.
"""
import bisect
from typing import Union, Sequence

from numpy.random import permutation
import numpy as np
import torch
from torch.utils import data as torch_data

from alchemy_cat.alg import accumulate
from alchemy_cat.py_tools import indent


__all__ = ["Dataset", "TensorDataset", "Subset", "random_split"]


class Dataset(object):
    def __getitem__(self, index: Union[slice, list, np.ndarray, torch.Tensor, int]):
        if isinstance(index, slice):
            # Get the start, stop, and step from the slice, return an iterator
            return (self[ii] for ii in range(*index.indices(len(self))))
        if isinstance(index, list) or isinstance(index, np.ndarray) or isinstance(index, torch.Tensor):
            return [self[ii] for ii in index]
        elif isinstance(int(index), int):
            index = int(index)
            if index < 0:  # Handle negative indices
                index += len(self)
            if index < 0 or index >= len(self):
                raise IndexError("The index (%d) is out of range" % index)
            return self.get_item(index)  # Get the data from elsewhere
        else:
            raise TypeError("Invalid argument type")

    def __add__(self, other):
        return ConcatDataset([self, other])

    def get_item(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __repr__(self):
        return f"Dataset <{self.__class__.__name__}>:\n" + \
               indent("#data: {}".format(self.__len__()))

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class TensorDataset(Dataset):
    r"""Dataset wrapping tensors.
    Each sample will be retrieved by indexing tensors along the first dimension.
    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, *tensors):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors

    def get_item(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)


class ConcatDataset(Dataset):
    r"""Dataset as a concatenation of multiple datasets.
    This class is useful to assemble different existing datasets.
    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        for d in self.datasets:
            assert len(d) >= 0
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def get_item(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]


class Subset(Dataset):
    r"""
    Subset of a dataset at specified indices.
    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset: Union[Dataset, torch_data.Dataset], indices: Sequence[int]):
        self.dataset = dataset
        self.indices = indices

    def get_item(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def random_split(dataset: Union[Dataset, torch_data.Dataset], lengths):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.
    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
    """
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = permutation(sum(lengths)).tolist()
    return [Subset(dataset, indices[offset - length:offset]) for offset, length in zip(accumulate(lengths), lengths)]
