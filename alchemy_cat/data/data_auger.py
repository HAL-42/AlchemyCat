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
from typing import Union, Optional
import numpy as np
from inspect import signature
import lmdb
import pickle
from functools import reduce
from torch.utils import data as torch_data

from alchemy_cat.dag import Graph
from alchemy_cat.data import Dataset
from alchemy_cat.py_tools.str_formatters import indent
from alchemy_cat.py_tools.type import is_int
from alchemy_cat.alg import accumulate

__all__ = ["RandMap", "MultiMap", "DataAuger"]


class RandMap(object):
    rand_seeds = None  # Can be overloaded. The static rand seeds where rand seed is selected.
    weight = None  # Can be overloaded. The static weight to choose the rand seeds

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

        Returns: Generated rand seed
        """
        prob = None
        if self.weight is not None:
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

    def __repr__(self):
        return f"RandMap: {self.__class__.__name__}\n" \
                + f"    _rand_seed: {self._rand_seed}\n" \
                + f"    _is_rand_seed_set: {self._is_rand_seed_set}"


class MultiMap(object):
    output_num = None

    def __init__(self):
        self._output_index = 0

    def __len__(self):
        if not is_int(self.output_num):
            raise ValueError(f"{self.__class__.__name__}'s output_num {self.output_num} is not an int value")
        return int(self.output_num)

    @property
    def output_index(self):
        return self._output_index

    @output_index.setter
    def output_index(self, value):
        if not is_int(value):
            raise ValueError(f"output_index {value} should be int")

        if value < 0:
            value += len(self)

        if value < 0 or value >= len(self):
            raise ValueError(f"output_index value {value} to be set is out of range [{0}, {self.output_num})")

        self._output_index = value

    def forward(self, *fwd_args, **fwd_kwargs):
        """Function which you wish to call when the functor is called. Need to be overloaded.
        It should be implemented according to output_index
        """
        raise NotImplementedError

    def __call__(self, *fwd_args, **fwd_kwargs):
        return self.forward(*fwd_args, **fwd_kwargs)

    def __repr__(self):
        return f"MultiMap: {self.__class__.__name__}\n" \
                + f"    output_num: {self.output_num}\n" \
                + f"    _output_index: {self._output_index}"


class DataAuger(Dataset):

    def __init__(self, dataset: Union[Dataset, torch_data.Dataset], verbosity: int = 0, pool_size: int = 0,
                 slim: bool = False, rand_seed_log: str = None):
        """Given dataset and argument it with a calculate graph.

        Args:
            dataset: dataset to be augmented
            verbosity: Calculate graph's verbosity. 0: No log output; 1: Only give graph level log;
                >1: Give graph and node level log.
            pool_size: Calculate graph's pool size. 0 means don't use parallel.
            slim: If True, calculate graph use copy rather than deepcopy when setting value of Node's input.
            rand_seed_log: lmdb database where rand seeds saved
        """
        self._dataset = dataset

        self._rand_seed_log: Optional[str] = None
        self.lmdb_env: Optional[lmdb.Environment] = None
        self.rand_seed_log = rand_seed_log

        self.graph = Graph(verbosity=verbosity, pool_size=pool_size, slim=slim)
        self.build_graph()

        self.deep_prefix_id_ordered_nodes = self.graph.deep_prefix_id_ordered_nodes()

        multi_prefix_id_nodes = [(prefix_id_node) for prefix_id_node in self.deep_prefix_id_ordered_nodes
                                 if isinstance(prefix_id_node[1]._fct, MultiMap)]
        self.multi_nodes_prefix_id, self.multi_nodes = zip(*multi_prefix_id_nodes) if multi_prefix_id_nodes \
            else ([], [])

        rand_prefix_id_nodes = [(prefix_id_node) for prefix_id_node in self.deep_prefix_id_ordered_nodes
                                if isinstance(prefix_id_node[1]._fct, RandMap)]
        self.rand_nodes_prefix_id, self.rand_nodes = zip(*rand_prefix_id_nodes) if rand_prefix_id_nodes else \
            ([], [])

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
        divide_factors = list(accumulate(divide_factors, lambda x, y: x * y))
        return divide_factors[::-1] + [1]

    @property
    def dataset(self):
        """Get auger's dataset"""
        return self._dataset

    @dataset.setter
    def dataset(self, value):
        """Reset auger's dataset"""
        self._dataset = value
        self.multi_factors = self._get_multi_factors()

    @property
    def multi_factor(self):
        """Return the number of one example were auger to"""
        return reduce(lambda x, y: x * y, self.multi_factors[1:])

    def __len__(self):
        """Return the size of augmented dataset"""
        return reduce(lambda x, y: x * y, self.multi_factors)

    def load_indices(self, idx):
        """Set every multi node's idx and return dataset's idx with auger's idx"""
        dataset_idx = idx // self.divide_factors[0]
        idx %= self.divide_factors[0]

        for multi_node, divide_factor in zip(self.multi_nodes, self.divide_factors[1:]):
            multi_node._fct.output_index = idx // divide_factor
            idx %= divide_factor

        return dataset_idx

    def calculate_indices(self, idx):
        """Return dataset's idx and every multi node's idx with auger's idx"""
        dataset_idx = idx // self.divide_factors[0]
        idx %= self.divide_factors[0]

        node_indices = {}
        for prefix_id, divide_factor in zip(self.multi_nodes_prefix_id, self.divide_factors[1:]):
            node_indices[prefix_id] = idx // divide_factor
            idx %= divide_factor

        return dataset_idx, node_indices

    @property
    def rand_seeds(self):
        """Return rand seed of every rand node"""
        rand_seeds = {}

        for prefix_id, node in zip(self.rand_nodes_prefix_id ,self.rand_nodes):
            rand_seeds[prefix_id] = node._fct.rand_seed
        return rand_seeds

    def load_rand_seeds(self, rand_seeds):
        """Load rand seed from rand_seeds"""
        for prefix_id, node in zip(self.rand_nodes_prefix_id ,self.rand_nodes):
            node._fct.rand_seed = rand_seeds[prefix_id]

    @property
    def rand_seed_log(self):
        return self._rand_seed_log

    @rand_seed_log.setter
    def rand_seed_log(self, rand_seed_log):
        if (not isinstance(rand_seed_log, str)) and (rand_seed_log is not None):
            raise ValueError(f"rand_seed_log = {rand_seed_log} must be str or None")

        self._rand_seed_log = rand_seed_log

        if self.lmdb_env is not None:
            self.lmdb_env.close()
            self.lmdb_env = None

        if rand_seed_log is not None:
            self.lmdb_env = lmdb.open(rand_seed_log, meminit=False, map_size=2147483648, max_spare_txns=64,
                                      sync=False, metasync=False, lock=True)

    def __del__(self):
        self.rand_seed_log = None

    def get_item(self, idx):
        example = self._dataset[self.load_indices(idx)]

        if self.lmdb_env is not None:
            with self.lmdb_env.begin() as txn:
                rand_seeds = txn.get(str(idx).encode())
            if rand_seeds is not None:
                rand_seeds = pickle.loads(rand_seeds)
                self.load_rand_seeds(rand_seeds)

        ret = self.graph.calculate(data={'example': example})

        if self.lmdb_env is not None and rand_seeds is None:
            with self.lmdb_env.begin(write=True) as txn:
                txn.put(str(idx).encode(), pickle.dumps(self.rand_seeds))

        return ret

    def __repr__(self):
        return f"DataAuger <{self.__class__.__name__}>:\n" \
                + indent(f"graph: {self.graph}") + "\n" \
                + indent(f"dataset: {self.dataset}") + "\n" \
                + indent(f"#DataAuger: {len(self)}")
