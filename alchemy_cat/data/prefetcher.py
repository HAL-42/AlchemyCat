#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: prefetcher.py
@time: 2020/3/20 21:22
@desc: See also https://github.com/NVIDIA/apex/issues/304
"""
from typing import Iterator
from collections import abc
import warnings
import torch

from alchemy_cat.py_tools import indent


class Prefetcher(object):
    """Wrapper for iter(data_loader).

    Return an prefetcher iter which won't block the current stream when load data.
    """

    def __init__(self, iterator: Iterator):
        """Return an prefetcher iter which won't block the current stream when load data.

        Args:
            iterator: iter(data_loader)
        """
        if not torch.cuda.is_available():
            raise RuntimeError(f"Prefetcher can only work when cuda is available")

        if not isinstance(iterator, abc.Iterator):
            raise ValueError(f"iter f{iterator} is supposed to be an iterator")

        self.iter = iterator
        self.feed_stream = torch.cuda.Stream()
        self.batch = None

        self.preload()

    def preload(self):
        try:
            self.batch = next(self.iter)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.feed_stream):
            def prefetch(batch):
                if isinstance(batch, torch.Tensor):
                        return batch.cuda(non_blocking=True)
                elif isinstance(batch, list):
                    return [prefetch(elem) for elem in batch]
                else:
                    return batch

            self.batch = prefetch(self.batch)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.feed_stream)
        batch = self.batch

        if batch is None:
            raise StopIteration

        def record(batch):
            if isinstance(batch, torch.Tensor):
                # Make sure tensors created by current stream won't occupied the memory of tensors from other steam
                batch.record_stream(torch.cuda.current_stream())
            elif isinstance(batch, list):
                for elem in batch:
                    record(elem)
            else:
                pass

        record(batch)

        self.preload()
        return batch

    def __iter__(self):
        warnings.warn(f"Prefetcher is already an iterator.")
        return self

    def __repr__(self):
        return f"Prefetcher: {self.__class__.__name__}\n" \
                + indent(f"iter: {self.iter}")