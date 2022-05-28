#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2022/5/24 14:37
@File    : py_cfg.py
@Software: PyCharm
@Desc    : 
"""
from typing import Callable, Any, List, TypeVar, Type, Union, Generator

import warnings

from addict import Dict

from .open_cfg import open_config

__all__ = ['is_subtree', 'ItemLazy', 'IL', 'Config']

T_Config = TypeVar('T_Config', bound='Config')
T_dict = TypeVar('T_dict', bound=dict)


def is_subtree(tree: Any, root: dict) -> bool:
    """判断tree是否是root的子树。"""
    return type(tree) is type(root)


class ItemLazy(object):

    def __init__(self, func: Callable[[dict], Any]):
        """配置树的惰性叶子。"""
        self.func = func

    def __call__(self, cfg: dict) -> Any:
        return self.func(cfg)

    @staticmethod
    def compute_item_lazy(config: T_dict) -> T_dict:
        def compute(cfg: dict):
            for k, v in cfg.items():
                if is_subtree(v, cfg):
                    compute(v)
                elif isinstance(v, ItemLazy):
                    cfg[k] = v(config)
                else:
                    pass
        compute(config)
        return config


IL = ItemLazy


class Config(Dict):
    """配置字典。继承自Dict，从类型上和Addict区分，以便分离配置项和配置树。"""

    def __init__(self, *cfgs: List[Union[str, dict]], **kwargs):
        """支持从其他其他配置树模块路径或配置树dict初始化。所有配置树会被逐个dict_update到当前配置树上。

        Args:
            *cfgs: List[配置树所在模块|配置树]
            **kwargs: 传递给Dict，不应该使用。
        """
        super().__init__(**kwargs)

        # * 遍历配置树，更新到当前配置树上。
        for i, cfg in enumerate(cfgs):
            # * 若给出配置树模块路径，则根据路径打开配置树。
            if isinstance(cfg, str):
                cfg, _ = open_config(cfg)
            # * 检查配置树是dict。
            if not isinstance(cfg, dict):
                raise ValueError(f"{i}'th item = {cfg} in cfgs should be (opened as) dict.")
            # * 将配置树更新到本配置树上。
            self.dict_update(cfg)

    def branch_copy(self: T_Config) -> T_Config:
        """拷贝枝干（Config及其子Config），直接赋值叶子（Config的所有值）。

        Returns:
            拷贝后的Config，枝干用当前类别重建，叶子直接赋值自self。
        """
        ret = self.__class__()  # 建立一棵本类新树。
        for k, v in self.items():
            if is_subtree(v, self):
                ret[k] = v.branch_copy()  # 若v是子树，拷枝赋叶后赋给新树。
            else:
                ret[k] = v  # 若v是叶子，直接赋给新树。
        return ret

    @classmethod
    def from_dict(cls: Type[T_Config], other: dict) -> T_Config:
        """接收一个dict类型（或其子类）的配置树，若不是当前类别，则拷贝配置树，转为当前类型；反之直接返回。

        Returns:
            若other不是当前类别，回转换后的Config，枝干用当前类别重建，叶子直接赋值自other；反之直接返回自身。
        """
        if type(other) is cls:
            return other

        ret = cls()  # 建立一棵本类新树。
        for k, v in other.items():
            if is_subtree(v, other):
                ret[k] = cls.from_dict(v)  # 若v是other类别的子树，则将子树转为当前类别子树再赋给新树。
            else:
                ret[k] = v  # v是叶子，则直接赋值新树。
        return ret

    def dict_update(self, other: dict):
        """接收一个dict类型（或其子类）的配置树，更新到当前配置树上。

        应当确保，此后对本配置树枝干的修改，不会同步到原配置树上。因此，应当拷/转枝赋叶原配置树后，再做更新。

        Args:
            other: dict类型的配置树。

        Returns:
            更新后的配置树。
        """
        if type(other) is not type(self):  # 若other不是当前类型，则转枝赋叶，得当前类型的配置树。
            other = self.from_dict(other)
        else:  # 若other是当前类型，则拷枝赋叶。
            other = other.branch_copy()

        for k, v in other.items():
            # 若人有我有，且都是子树，则子树更新子树。
            if (k in self) and is_subtree(other[k], other) and is_subtree(self[k], self):
                self[k].dict_update(v)
            else:  # 反之（人有我有，但至少一方不是子树；或人有我无），则直接更新。
                self[k] = v

    def update(self, *args, **kwargs):
        """默认update会产生意想不到的结果——如dict也被update。尽量避免使用"""
        warnings.warn(f"{self.__class__} update may give unexpected results. Try use addict_update.")
        super().update(*args, **kwargs)

    def freeze(self, shouldFreeze=True):
        object.__setattr__(self, '__frozen', shouldFreeze)
        for k, v in self.items():
            if is_subtree(v, self):  # 将所有子树也冻结。
                v.freeze(shouldFreeze)

    def is_frozen(self) -> bool:
        try:
            is_frozen = object.__getattribute__(self, '__frozen')
        except AttributeError:
            return False
        return is_frozen

    def __setitem__(self, name, value):
        if self.is_frozen():  # 调用addict setitem前，检查frozen，若是，则完全禁止setitem。
            raise RuntimeError(f"{self.__class__} is frozen. ")
        dict.__setitem__(self, name, value)
        try:
            p = object.__getattribute__(self, '__parent')
            key = object.__getattribute__(self, '__key')
        except AttributeError:
            p = None
            key = None
        if p is not None:
            p[key] = self
            object.__delattr__(self, '__parent')
            object.__delattr__(self, '__key')

    def __missing__(self, name):  # 覆盖原有__missing__，freeze后任然允许返回一个空字典，只是不能赋值。
        return self.__class__(__parent=self, __key=name)

    def __getnewargs__(self):  # 覆盖原有__getnewargs__，无需返回任何值。
        return tuple()

    def __getstate__(self):  # 覆盖原有__getstate__，无需设置状态。
        return self.__dict__  # dict的优先级高于getattr。

    def __setstate__(self, state):  # 覆盖原有__setstate__，无需设置状态。
        self.__dict__.update(state)

    def to_dict(self):  # 覆盖原有to_dict，只要是Dict及其子类，总是做to_dict。
        base = {}
        for key, value in self.items():
            if isinstance(value, Dict):
                base[key] = value.to_dict()
            elif isinstance(value, (list, tuple)):
                base[key] = type(value)(
                    item.to_dict() if isinstance(item, Dict) else
                    item for item in value)
            else:
                base[key] = value
        return base

    @property
    def leaves(self) -> Generator:
        for k, v in self.items():
            if is_subtree(v, self):
                yield from v.leaves  # 若v是子树，则抛出子树的叶子。
            else:
                yield v  # 若v是叶子，则直接抛出叶子。

    @property
    def branches(self: T_Config) -> Generator[T_Config, None, None]:
        for k, v in self.items():
            if is_subtree(v, self):
                yield from v.branches  # 若v是子树，则抛出子树的叶子。
        yield self
