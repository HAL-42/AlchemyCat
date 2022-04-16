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
from importlib import import_module
from importlib.util import spec_from_file_location, module_from_spec
from typing import Iterable, Callable, Any, List
import os.path as osp
import traceback

from addict import Dict

from .utils import param_val2str

__all__ = ["Param2Tune", "ParamLazy", "Cfg2Tune"]


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


class ParamLazy(object):

    def __init__(self, func: Callable[[Dict], Any]):
        """Cfg2Tune's ParamLazy value will be computed after Cfg2Tune is transferred to cfg_tuned."""
        self.func = func

    def __call__(self, cfg: Dict) -> Any:
        return self.func(cfg)

    @staticmethod
    def compute_param_lazy(cfg_tuned: Dict) -> Dict:
        def compute(cfg: Dict):
            for k, v in cfg.items():
                if isinstance(v, Dict):
                    compute(v)
                elif isinstance(v, ParamLazy):
                    cfg[k] = v(cfg_tuned)
                else:
                    pass
        compute(cfg_tuned)
        return cfg_tuned


class Cfg2Tune(Dict):
    """Config to be tuned with parameters to be tuned."""

    def __init__(self, **kwargs):
        super(Cfg2Tune, self).__init__(**kwargs)
        object.__setattr__(self, '_params2tune', OrderedDict())
        object.__setattr__(self, '_root', None)

    def update(self, *args, **kwargs):
        """Cfg2Tune只允许从{}开始生成，而应当避免update。"""
        warnings.warn(f"{self.__class__} can't recursively track Param2Tune in the dict updated.")
        super(Cfg2Tune, self).update(*args, **kwargs)

    def __setitem__(self, name, value):
        """重载Addict的__setitem__"""
        '''
        setitem时，Cfg2Tune有三种状态：
        1. 用户刚刚用Cfg2Tune()生成，此时__parent，__key，_root均为None。此时需要设置root。
        2. 刚刚用__missing__方法生成。此时__parent，__key存在，_root为None。此时需要设置root，并将自身加入到父Cfg2Tune中。
        3. 用户刚用Cfg2Tune()生成，但已经（1）过。此时__parent，__key为None，但_root为self。
        4. 用__missing__方法生成，但已经（2）过。此时__parent，__key，_root均存在。
        '''
        parent = object.__getattribute__(self, '__parent')
        key = object.__getattribute__(self, '__key')
        root = object.__getattribute__(self, "_root")

        # * 若Cfg2Tune刚刚由__missing__得到，则将其加入到父Cfg2Tune中。
        if parent is not None and root is None:
            parent[key] = self

        # * 若还没有设置root，则找到root。
        if root is None:
            # ** 如果是根Dict，则root是self，否则根据parent一路找上去。
            root = self
            while object.__getattribute__(root, '__parent') is not None:
                root = object.__getattribute__(root, '__parent')
            object.__setattr__(self, '_root', root)

        # 若为待调参数，则将待调参数注册到root的_params2tune有序字典里。
        if isinstance(value, Param2Tune):
            params2tune = object.__getattribute__(root, "_params2tune")
            if name in params2tune:
                raise RuntimeError(f"name = {name} for param2tune repeated.")

            params2tune[name] = value

        dict.__setitem__(self, name, value)

    @staticmethod
    def dfs_params2tune(params2tune: List[Param2Tune]):
        """协程每找到一个新的参数组合，yield一次。"""
        cur_param = params2tune[0]

        def reset_later_params(params: List[Param2Tune]):
            for param in params[1:]:
                param.reset()

        reset_later_params(params2tune)
        for _ in cur_param:
            # 若只剩下一个参数，则每找到一个新参数值，就yield。
            if len(params2tune) == 1:
                yield
            # 否则遍历参数组中，第一个参数的所有可选值。对剩余参数，每找到一个新参数组合，就yield一次。
            else:
                yield from Cfg2Tune.dfs_params2tune(params2tune[1:])
                # 剩余参数遍历过所有组合后，重置为初始状态。
                reset_later_params(params2tune)

    def _cfg_tuned(self) -> Dict:
        other = Dict()

        for k, v in self.items():
            if isinstance(v, type(self)):
                other[k] = v._cfg_tuned()
            elif isinstance(v, Param2Tune):
                other[k] = v.cur_val
            else:
                other[k] = v

        if object.__getattribute__(self, "_root") is self:
            assert 'rslt_dir' in self

            rslt_dir_suffix = ",".join([f"{name}=" + param_val2str(param.cur_val)
                                        for name, param in object.__getattribute__(self, "_params2tune").items()])
            other.rslt_dir = osp.join(self.rslt_dir, rslt_dir_suffix)

        return other

    @property
    def cfg_tuned(self) -> Dict:
        """根据当前待调参数的值，返回一个Addict配置。"""
        return ParamLazy.compute_param_lazy(self._cfg_tuned())

    def get_cfgs(self):
        """遍历待调参数的所有可能组合，对每个组合，返回其对应的配置。"""
        params2tune = object.__getattribute__(self, "_params2tune")

        if len(params2tune) == 0:
            raise RuntimeError(f"The {self} must have more than 1 Param2Tune")

        for _ in self.dfs_params2tune(list(params2tune.values())):
            yield self.cfg_tuned

    def dump_cfgs(self, cfg_dir='configs') -> List[str]:
        """遍历待调参数的所有组合，将每个组合对应的配置，以json（如果可以）和pkl格式保存到cfg_dir下。"""
        cfg_files = []
        for cfg in self.get_cfgs():
            cfg_save_dir = osp.join(cfg_dir, cfg.rslt_dir)
            os.makedirs(cfg_save_dir, exist_ok=True)

            try:
                cfg_file = osp.join(cfg_save_dir, 'cfg.json')
                with open(cfg_file, 'w') as json_f:
                    json.dump(cfg.to_dict(), json_f)
            except Exception:
                pass
            finally:
                cfg_file = osp.join(cfg_save_dir, 'cfg.pkl')
                with open(cfg_file, 'wb') as pkl_f:
                    pickle.dump(cfg.to_dict(), pkl_f)
            cfg_files.append(cfg_file)

        return cfg_files

    @staticmethod
    def load_cfg2tune(cfg2tune_py: str) -> 'Cfg2Tune':
        """导入、运行含有config = Cfg2Tune()的py文件，返回该config。"""
        '''
        若config的值，或者其Param2Tune的可选值中含有函数（也就是说cfg_tuned值含有函数），如何确保函数被正确序列化与反序列化？
        - 若函数为外部导入，譬如“from *** import func”，则确保load时，能正确执行“from *** import func”即可。
        - 若函数在config中定义，则保证：
            1）cfg2tune_py按照下面cfg2tune_import_path被import。
            2）load是也能import cfg2tune_import_path。
        注意，subject_to函数、ParamLazy函数不会被pickle，二者只在生成Addict时被执行。
        '''
        try:
            cfg2tune_import_path = '.'.join(osp.normpath(osp.splitext(cfg2tune_py)[0]).lstrip(osp.sep).split(osp.sep))
            module = import_module(cfg2tune_import_path)
        except Exception:
            print(traceback.format_exc())
            print(f"未能用import_module导入{cfg2tune_py},尝试直接执行文件。")
            spec = spec_from_file_location("foo", cfg2tune_py)
            module = module_from_spec(spec)
            spec.loader.exec_module(module)
        return module.config
