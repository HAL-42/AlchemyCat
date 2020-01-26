#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: data_auger.py
@time: 2020/1/12 1:48
@desc:
"""
import numpy as np
from inspect import signature
import lmdb
import pickle
from functools import reduce

from alchemy_cat.dag import Graph
from alchemy_cat.data import Dataset


class RandMap(object):

    rand_seeds = None # Can be overloaded. The static rand seeds where rand seed is selected.
    weight = None     # Can be overloaded. The static weight to choose the rand seeds

    def __init__(self):
        self._rand_seed = None
        self._is_rand_seed_set = False

        # Check whether the arguments of self.forward is the same to the self.generate_rand_seed
        def get_param_names(func):
            return list(signature(func).parameters.keys())

        if get_param_names(self.generate_rand_seed) != get_param_names(self.__call__) and \
            get_param_names(self.generate_rand_seed) != get_param_names(self.forward):
            raise RuntimeError(f"{type(self).__name__}'s self.generate_rand_seed's args is not "
                               f"equal to the args of self.__call__ or self.forward")

    def __len__(self):
        return len(self.rand_seeds)

    def generate_rand_seed(self, *fwd_args, **fwd_kwargs):
        """Generate rand seed which will be used in forward.
        Can be overloaded to dynamically generate the rand seed according to the forward() inputs.
        But the arguments of this function should be exactly the same to the forward() function.

        Returns:
            rand_seed: Generated rand seed
        """
        prob = np.array(self.weight, dtype=np.float32) / np.sum(self.weight)
        rand_seed_index = np.random.choice(len(self), p=prob)
        return self.rand_seeds[rand_seed_index]

    @property
    def rand_seed(self):
        if self._rand_seed is None:
            raise AttributeError(f"Object of {type(self).__name__} still don't have rand_seed yet")
        else:
            return self._rand_seed

    @rand_seed.setter
    def rand_seed(self, value):
        if self.rand_seeds is not None and value not in self.rand_seeds:
            raise KeyError(f"rand_seed value f{value} to be set is not in self.rand_seeds={self.rand_seeds}")

        self._rand_seed = value
        self._is_rand_seed_set = True

    def forward(self, *fwd_args, **fwd_kwargs):
        """Function which you wish to call when the functor is called. Need to be overloaded
        It should be implemented according to rand_seed
        """
        raise NotImplementedError

    def __call__(self, *fwd_args, **fwd_kwargs):
        if not self._is_rand_seed_set:
            self.rand_seed = self.generate_rand_seed(*fwd_args, **fwd_kwargs)

        ret = self.forward(*fwd_args, **fwd_kwargs)
        self._is_rand_seed_set = False
        return ret


class MultiMap(object):

    output_num = None

    def __init__(self):
        self._output_index = 0

    def __len__(self):
        return self.output_num

    @property
    def output_index(self):
        return self._output_index

    @output_index.setter
    def output_index(self, value):
        if value < 0:
            value += len(self)

        if value <0 or value >= len(self):
            raise ValueError(f"output_index value {value} to be set is out of range [{0}, {self.output_num})")

        self._output_index = value

    def forward(self, *fwd_args, **fwd_kwargs):
        """Function which you wish to call when the functor is called. Need to be overloaded.
        It should be implemented according to output_index
        """
        raise NotImplementedError

    def __call__(self, *fwd_args, **fwd_kwargs):
        return self.forward(*fwd_args, **fwd_kwargs)


class DataAuger(object):

    def __init__(self, dataset: Dataset, verbosity=0, rand_seed_log = None):
        self._dataset = dataset
        self.rand_seed_log = rand_seed_log

        self.graph = Graph(verbosity=verbosity)
        self.ordered_nodes = self.graph.ordered_nodes
        self.multi_nodes = [node for node in self.ordered_nodes if isinstance(node._fct, MultiMap)]
        self.rand_nodes = [node for node in self.ordered_nodes if isinstance(node._fct, RandMap)]

        self.multi_factors = self._get_multi_factors()
        self.divide_factors = self._get_divide_factors()

    def build_graph(self):
        """Build self.graph here. Need to be overloaded"""
        raise NotImplementedError

    def _get_multi_factors(self):
        multi_factors = [len(self._dataset)]

        for node in self.multi_nodes:
            multi_factors.append(len(node._fct))

        return multi_factors

    def _get_divide_factors(self):
        divide_factors = self.multi_factors[::-1][:-1]
        divide_factors = reduce(lambda x, y: x * y, divide_factors)
        return divide_factors[::-1] + [1]

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, value):
        self._dataset = value
        self.multi_factors = self._get_multi_factors()

    @property
    def multi_factor(self):
        return reduce(lambda x, y: x * y, self.multi_factors[1:])

    def __len__(self):
        return reduce(lambda x, y: x * y, self.multi_factors)

    def load_indices(self, idx):
        dataset_idx = idx // self.divide_factors[0]
        idx %= self.divide_factors[0]

        for multi_node, divide_factor in zip(self.multi_nodes, self.divide_factors[1:]):
            multi_node._fct.output_index = idx // divide_factor
            idx %= divide_factor

        return dataset_idx

    def calculate_indices(self, idx):
        dataset_idx = idx // self.divide_factors[0]
        idx %= self.divide_factors[0]

        node_indices = {}
        for multi_node, divide_factor in zip(self.multi_nodes, self.divide_factors[1:]):
            node_indices[multi_node._id] = idx // divide_factor
            idx %= divide_factor

        return dataset_idx, node_indices

    @property
    def rand_seeds(self):
        rand_seeds = {}

        for node in self.rand_nodes:
            rand_seeds[node._id] = node._fct.rand_seed
        return rand_seeds

    def load_rand_seeds(self, rand_seeds):
        for node in self.rand_nodes:
            node._fct.rand_seed = rand_seeds[node._id]

    def __getitem__(self, idx):
        example = self._dataset[self.load_indices(idx)]

        if self.rand_seed_log is not None:
            env = lmdb.open(self.rand_seed_log.encode(), meminit=False)
            with env.begin() as txn:
                rand_seeds = txn.get(str(idx).encode())
            if rand_seeds is not None:
                rand_seeds = pickle.loads(rand_seeds)
                self.load_rand_seeds(rand_seeds)

        ret = self.graph.calculate(data={'example': example})

        if self.rand_seed_log is not None and rand_seeds is None:
            with env.begin(write=True) as txn:
                txn.put(str(idx).encode(), pickle.dumps(self.rand_seeds))
            env.close()

        return ret