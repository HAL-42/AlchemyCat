#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2021/5/6 19:33
@File    : cfg2tune.py
@Software: PyCharm
@Desc    : 
"""
import json
import os
import pickle
import warnings
from collections import OrderedDict
from importlib.util import spec_from_file_location, module_from_spec
from typing import Iterable, Callable, Any, List
import os.path as osp

from addict import Dict

__all__ = ["Param2Tune", "Cfg2Tune"]


class Param2Tune(object):

    def __init__(self, optional_values: Iterable, subject_to: Callable[[Any], bool] = lambda x: True):
        """A Parameter need to be tuned

        Args:
            optional_values: Parameters optional value.
            subject_to: Filter used to check whether the parameter value is legal.
        """
        self.optional_val = optional_values
        self.subject_to = subject_to

        self._cur_val = None  # Current Param Value

    @property
    def cur_val(self):
        if self._cur_val is None:
            raise RuntimeError(f"The current value of param {self} is None. Check if the former param call later param"
                               f"in it's subject_to.")
        return self._cur_val

    def reset(self):
        self._cur_val = None

    def __iter__(self):
        for opt_val in self.optional_val:
            if self.subject_to(opt_val):
                self._cur_val = opt_val
                yield opt_val

    def __repr__(self):
        return f"{self.__class__} with optional_val = {list(self.optional_val)}"


class Cfg2Tune(Dict):
    """Config to be tuned with parameters to be tuned."""

    def __init__(self, **kwargs):
        super(Cfg2Tune, self).__init__(**kwargs)
        object.__setattr__(self, '_params2tune', OrderedDict())
        object.__setattr__(self, '_root', None)

    def update(self, *args, **kwargs):
        warnings.warn(f"{self.__class__} can't recursively track Param2Tune in the dict updated.")
        super(Cfg2Tune, self).update(*args, **kwargs)

    def __setitem__(self, name, value):
        parent = object.__getattribute__(self, '__parent')
        key = object.__getattribute__(self, '__key')

        root = object.__getattribute__(self, "_root")
        if parent is not None and root is None:
            parent[key] = self

        if root is None:
            root = self
            while object.__getattribute__(root, '__parent') is not None:
                root = object.__getattribute__(root, '__parent')
            object.__setattr__(self, '_root', root)

        if isinstance(value, Param2Tune):
            params2tune = object.__getattribute__(root, "_params2tune")
            if name in params2tune:
                raise RuntimeError(f"name = {name} for param2tune repeated.")

            params2tune[name] = value

        dict.__setitem__(self, name, value)

    @staticmethod
    def dfs_params2tune(params2tune: List[Param2Tune]):
        cur_param = params2tune[0]

        def reset_later_params(params: List[Param2Tune]):
            for param in params[1:]:
                param.reset()

        reset_later_params(params2tune)
        for _ in cur_param:
            if len(params2tune) == 1:
                yield
            else:
                yield from Cfg2Tune.dfs_params2tune(params2tune[1:])
                reset_later_params(params2tune)

    @property
    def cfg_tuned(self) -> Dict:
        other = Dict()

        for k, v in self.items():
            if isinstance(v, type(self)):
                other[k] = v.cfg_tuned
            elif isinstance(v, Param2Tune):
                other[k] = v.cur_val
            else:
                other[k] = v

        if object.__getattribute__(self, "_root") is self:
            assert 'rslt_dir' in self
            rslt_dir_suffix = ",".join([f"{name}=" + str(param.cur_val).splitlines()[0][:10]
                                        for name, param in object.__getattribute__(self, "_params2tune").items()])
            other.rslt_dir = osp.join(self.rslt_dir, rslt_dir_suffix)

        return other

    def get_cfgs(self):
        params2tune = object.__getattribute__(self, "_params2tune")

        if len(params2tune) == 0:
            raise RuntimeError(f"The {self} must have more than 1 Param2Tune")

        for _ in self.dfs_params2tune(list(params2tune.values())):
            yield self.cfg_tuned

    def dump_cfgs(self, cfg_dir='configs') -> List[str]:
        cfg_files = []
        for cfg in self.get_cfgs():
            cfg_save_dir = osp.join(cfg_dir, cfg.rslt_dir)
            os.makedirs(cfg_save_dir, exist_ok=True)

            try:
                cfg_file = osp.join(cfg_save_dir, 'cfg.json')
                with open(cfg_file, 'w') as json_f:
                    json.dump(cfg.to_dict(), json_f)
            finally:
                cfg_file = osp.join(cfg_save_dir, 'cfg.pkl')
                with open(cfg_file, 'wb') as pkl_f:
                    pickle.dump(cfg.to_dict(), pkl_f)
            cfg_files.append(cfg_file)

        return cfg_files

    @staticmethod
    def load_cfg2tune(cfg2tune_py: str) -> 'Cfg2Tune':
        spec = spec_from_file_location("foo", cfg2tune_py)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.config
