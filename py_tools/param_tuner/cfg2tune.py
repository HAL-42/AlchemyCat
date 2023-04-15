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
from typing import Iterable, Callable, Any, List, Generator

import json
import os
import pickle
from collections import OrderedDict
import os.path as osp

from .utils import param_val2str
from ..load_module import load_module_from_py
from ..config import Config, is_subtree, auto_rslt_dir

__all__ = ["Param2Tune", "ParamLazy", "PL", "Cfg2Tune"]


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

    def __init__(self, func: Callable[[Config], Any]):
        """Cfg2Tune's ParamLazy value will be computed after Cfg2Tune is transferred to cfg_tuned."""
        self.func = func

    def __call__(self, cfg: Config) -> Any:
        return self.func(cfg)

    @staticmethod
    def compute_param_lazy(cfg_tuned: Config) -> Config:
        def compute(cfg: Config):
            for k, v in cfg.items():
                if is_subtree(v, cfg):
                    compute(v)
                elif isinstance(v, ParamLazy):
                    cfg[k] = v(cfg_tuned)
                else:
                    pass
        compute(cfg_tuned)
        return cfg_tuned


PL = ParamLazy


class Cfg2Tune(Config):
    """Config to be tuned with parameters to be tuned."""

    def __init__(self, *cfgs, cfgs_update_at_parser: tuple=(), **kwargs):
        """支持从其他其他配置树模块路径或配置树dict初始化。所有配置树会被逐个dict_update到当前配置树上。

        Args:
            *cfgs: List[配置树所在模块|配置树]
            **kwargs: 传递给Dict，不应该使用。
        """
        object.__setattr__(self, '_params2tune', OrderedDict())
        object.__setattr__(self, '_root', None)
        super(Cfg2Tune, self).__init__(*cfgs, cfgs_update_at_parser=cfgs_update_at_parser, **kwargs)

        for leaf in self.leaves:
            if isinstance(leaf, Param2Tune):
                raise RuntimeError(f"Cfg2Tune can't init by leaf with type = f{type(leaf)}")

        assert '_cfgs_update_at_parser' not in self  # _cfgs_update_at_parser务必在顶层定义，不可来自导入。

    def _mount2parent(self):
        parent = object.__getattribute__(self, '__parent')
        key = object.__getattribute__(self, '__key')
        # root指示了是否挂载过，凡是执行过_mount2parent，root一定不为None，反之root一定为None。
        root = root_ = object.__getattribute__(self, "_root")

        # * 若还没有设置root，则找到root。
        if root is None:  # 执行了_mount2parent，root一定不为None。
            # ** 如果是根Dict，则root是self，否则根据parent一路找上去。
            root = self
            while object.__getattribute__(root, '__parent') is not None:
                root = object.__getattribute__(root, '__parent')
            object.__setattr__(self, '_root', root)  # 除了init，只有_mount2parent才会设置root。

        # * 若Cfg2Tune刚刚由__missing__得到，则将其加入到父Cfg2Tune中。
        if parent is not None and root_ is None:  # 若已经挂载过了，就不要重复挂载。
            parent[key] = self

    def __setitem__(self, name, value):
        """重载Addict的__setitem__"""
        '''
        setitem时，Cfg2Tune有以下状态：
        1. 用户刚刚用Cfg2Tune()生成，此时__parent，__key，_root均为None。此时需要设置root。
        2. 刚刚用__missing__方法生成。此时__parent，__key存在，_root为None。此时需要设置root，并将自身加入到父Cfg2Tune中。
        3. 用户刚用Cfg2Tune()生成，但已经（1）/set_whole过。此时__parent，__key为None，但_root为self。
        4. 用__missing__方法生成，但已经（2）/set_whole过。此时__parent，__key，_root均存在。
        
        若被设置的value为Cfg2Tune，则value有三种状态：
        1. value由__missing__生成的子节点，已经找到了正确的root，且内容为空。此时只要正常setitem即可。我们不考虑该value被
           再次挂载。
        2. value是由Cfg2Tune()生成的根节点，根节点root为自己或None（若为空配置树）。
        3. value是update时，从原配置树上摘下来的子树，该子树的root和parent均不在新树上。
        4. self的子树，重新用新的名字绑定。
        2、3共性是value的root、parent、key不匹配，4则key不匹配。子树内部都是自洽的（P2T在根节点上，parent，key正确）。
        对2、3、4，需要调整子树再挂载，即重设value的root、根节点的parent，合并P2T，确保新树自洽。
        '''
        if self.is_frozen():  # 调用addict setitem前，检查frozen，若是，则完全禁止setitem。
            raise RuntimeError(f"{self.__class__} is frozen. ")

        self._mount2parent()
        root = object.__getattribute__(self, "_root")

        # * 若值为待调参数，则将待调参数注册到root的_params2tune有序字典里。
        if isinstance(value, Param2Tune):
            root_params2tune = object.__getattribute__(root, "_params2tune")
            if name in root_params2tune:
                raise RuntimeError(f"name = {name} for param2tune repeated.")

            root_params2tune[name] = value

        # * 若值不是__missing__生成的子节点，则将该待调配置树的所有节点设置正确的root，根节点设置正确的parent、key，
        # * 并合并其params2tune。
        if isinstance(value, Cfg2Tune) and \
                (object.__getattribute__(value, "_root") is not root or
                 object.__getattribute__(value, "__parent") is not self or
                 object.__getattribute__(value, "__key") != name):
            for b in value.branches:
                object.__setattr__(b, '_root', root)

            object.__setattr__(value, '__parent', self)  # 只需重设根节点的parent、key即可。
            object.__setattr__(value, '__key', name)

            root_params2tune = object.__getattribute__(root, "_params2tune")
            value_params2tune = object.__getattribute__(value, "_params2tune")
            for value_param_name, value_param in value_params2tune.items():
                if value_param_name in root_params2tune:
                    raise RuntimeError(f"value_param_name = {value_param_name} for param2tune repeated.")
                root_params2tune[value_param_name] = value_param
            value_params2tune.clear()

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

    @property
    def cfgs_update_at_parser(self):
        """
        Cfg2Tune支持在顶层（参见__init__，不可来自导入），以k-v对形式指定_cfgs_update_at_parser。如此可支持对
        _cfgs_update_at_parser的调参。

        当cfgs_update_at_parser来自外置k-v对时，则不应该在初始化时，不应再次指定_cfgs_update_at_parser。

        Cfg2Tune本身不会parse，所以其_cfgs_update_at_parser只用于传递给子配置。注意子配置不支持k-v对形式的
        _cfgs_update_at_parser。

        Config不会支持k-v对形式的_cfgs_update_at_parser。更新时，k-v对形式的_cfgs_update_at_parser会被覆盖（尽管不影响
        最终结果，但会造成混乱）。
        """
        if '_cfgs_update_at_parser' in self:
            assert object.__getattribute__(self, '_cfgs_update_at_parser') == ()
            if isinstance(cfgs_update_at_parser := self['_cfgs_update_at_parser'], Param2Tune):
                return cfgs_update_at_parser.cur_val
            else:
                return cfgs_update_at_parser
        else:
            return object.__getattribute__(self, '_cfgs_update_at_parser')

    def _cfg_tuned(self) -> Config:
        other = Config()

        self.copy_cfg_attr_to(other)  # 将依赖、是否不可分等属性拷贝给other。
        # cfgs_update_at_parser可能来自字段，需要单独解析。
        object.__setattr__(other, '_cfgs_update_at_parser', self.cfgs_update_at_parser)

        for k, v in self.items():
            if k == '_cfgs_update_at_parser':
                continue  # k-v对形式的_cfgs_update_at_parser总是以attr形式传递给子配置。

            if is_subtree(v, self):
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
    def cfg_tuned(self) -> Config:
        """根据当前待调参数的值，返回一个Addict配置。"""
        return ParamLazy.compute_param_lazy(self._cfg_tuned())

    def get_cfgs(self) -> Generator[Config, None, None]:
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
                    pickle.dump(cfg, pkl_f)
            cfg_files.append(cfg_file)

        return cfg_files

    @staticmethod
    def load_cfg2tune(cfg2tune_py: str, config_root: str='./configs') -> 'Cfg2Tune':
        """导入、运行含有config = Cfg2Tune()的py文件，返回该config。"""
        '''
        若config的值，或者其Param2Tune的可选值中含有函数（也就是说cfg_tuned值含有函数），如何确保函数被正确序列化与反序列化？
        - 若函数为外部导入，譬如“from *** import func”，则确保load时，能正确执行“from *** import func”即可。
        - 若函数在config中定义，则保证：
            1）cfg2tune_py按照下面cfg2tune_import_path被import。
            2）load是也能import cfg2tune_import_path。
        注意，subject_to函数、ParamLazy函数不会被pickle，二者只在生成Addict时被执行。
        '''
        cfg2tune = load_module_from_py(cfg2tune_py).config

        if not cfg2tune.rslt_dir:
            raise RuntimeError(f"cfg2tune should indicate result save dir at {cfg2tune.rslt_dir=}")

        if cfg2tune.rslt_dir is ...:
            cfg2tune.rslt_dir = auto_rslt_dir(cfg2tune_py, config_root)

        return cfg2tune

    def __getstate__(self):
        # TODO 令Cfg2Tune支持pickle：当前Cfg2Tune不支持pickle——在“恢复高祖数据”时，__setitem__会在没有__init__的情况下
        #      被调用。__setitem__会尝试获取root、parent、key等不存在的属性（其实之后就会恢复），并报错。
        #      解决办法：setitem时判断是否在unpickle（没有init过），若时，则执行dict的setitem。
        raise NotImplementedError(f"{self.__class__} currently not support pickle. ")
