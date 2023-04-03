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
from typing import Callable, Any, TypeVar, Type, Generator, Tuple

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

    def __init__(self, func: Callable, priority: int=1):
        """配置树的惰性叶子。"""
        self.func = func
        self.priority = priority

    def __call__(self, cfg: dict) -> Any:
        return self.func(cfg)

    @staticmethod
    def compute_item_lazy(config: T_dict) -> T_dict:
        def dfs_find_item_lazy(cfg: dict) -> Tuple[dict, str, int]:
            for k, v in cfg.items():
                if is_subtree(v, cfg):
                    yield from dfs_find_item_lazy(v)
                elif isinstance(v, ItemLazy):
                    yield cfg, k, v.priority
                else:
                    pass

        item_lazy_dic_keys = sorted(dfs_find_item_lazy(config), key=lambda x: x[2])

        if len(item_lazy_dic_keys) > 0:
            for item_lazy_dic, item_lazy_key, _ in item_lazy_dic_keys:
                item_lazy_dic[item_lazy_key] = item_lazy_dic[item_lazy_key](config)
            return ItemLazy.compute_item_lazy(config)
        else:
            return config


IL = ItemLazy


class Config(Dict):
    """配置字典。继承自Dict，从类型上和Addict区分，以便分离配置项和配置树。"""

    def __init__(self, *cfgs, cfgs_update_at_parser: tuple=(), **kwargs):
        """支持从其他其他配置树模块路径或配置树dict初始化。所有配置树会被逐个dict_update到当前配置树上。

        Args:
            *cfgs: List[配置树所在模块|配置树]
            cfgs_update_at_parser: 解析时用于更新的基配置。
            **kwargs: 传递给Dict，不应该使用。
        """
        # * 初始化。
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
            # * 将配置树的parser时配置更新上来，且优先级更高。
            if cfg_dep := object.__getattribute__(cfg, '_cfgs_update_at_parser'):
                # A --解析时--> B; D, B --解析时--> C 含义明确，即加载时DFS，从祖先开始增量更新到C，优先级为B、A、D。
                # A --解析时--> B; D, B --加载时--> C 若D与A冲突，则D中键值会阻塞A，优先级为B、D、A，这是我们不希望的。
                # 因此，最好不要混用解析时和加载时依赖。
                warnings.warn(f"{cfg=}存在解析时依赖{cfg_dep=}。\n"
                              f"因此，该配置也应该作为当前配置的解析时依赖。否则并列的加载时依赖，可能干扰cfg_dep更新cfg。")
                cfgs_update_at_parser = cfg_dep + cfgs_update_at_parser

        # * 记录基配置。
        object.__setattr__(self, '_cfgs_update_at_init', cfgs)
        object.__setattr__(self, '_cfgs_update_at_parser', cfgs_update_at_parser)

    def update_at_parser(self):
        # * 获取解析式基配置。
        cfgs_update_at_parser = object.__getattribute__(self, '_cfgs_update_at_parser')
        # * 若无需更新，跳过。
        if not cfgs_update_at_parser:
            return
        # * 保存主配置。
        main_cfg = self.branch_copy()
        # * 清空当前配置树。
        self.clear()
        # * 逐个读取基配置，并更新到当前配置中。
        for i, base_cfg in enumerate(cfgs_update_at_parser):
            # * 若给出配置树模块路径，则根据路径打开配置树。
            if isinstance(base_cfg, str):
                base_cfg, _ = open_config(base_cfg)
            # * 基配置也进行解析时更新。
            if isinstance(base_cfg, Config):
                base_cfg.update_at_parser()
            # * 检查配置树是dict。
            if not isinstance(base_cfg, dict):
                raise ValueError(f"{i}'th item = {base_cfg} in self._cfgs_update_at_parser should be "
                                 f"(opened as) dict.")
            # * 将基配置树更新到本配置树上。
            self.dict_update(base_cfg)
        # * 将主配置树更新回当前配置树。
        self.dict_update(main_cfg)

    @property
    def subtrees_wt_COM(self: T_Config) -> list[T_Config]:
        return [subtree for subtree in self.branches if 'COM_' in subtree]

    def _check_COM(self: T_Config):
        """检查COM树是否合规。由于reduce_COM会逐步删除子COM树，可能影响安全检查。故检查放在reduce_COM之前。"""
        # NOTE COM树，可以有子COM树。因为子树搜索为DFS后序，故子树COM树总是先于父COM树处理。
        # NOTE 而COM并列项如果有子COM树，则无法确定和当前COM项的执行顺序，非常危险，要警告。
        for subtree_wt_COM in self.subtrees_wt_COM:
            # * COM项应该是子树。
            if not is_subtree(subtree_wt_COM.COM_, self):
                raise RuntimeError(f"COM树{subtree_wt_COM}的COM项{subtree_wt_COM.COM_}应当为子树。")
            # * COM并列项不含COM子树。
            COM_parallel_vals = [v for k, v in subtree_wt_COM.items() if (k != 'COM_') and (is_subtree(v, self))]
            for par_val in COM_parallel_vals:
                par_val: T_Config
                if par_val.subtrees_wt_COM:
                    warnings.warn(f"COM树{subtree_wt_COM}的COM并列项{par_val}存在子COM树。无法确定该树和当前"
                                  f"COM树的执行顺序，reduce_COM行为不可预测。")

    def reduce_COM(self: T_Config):
        """归并掉所有COM项。"""
        self._check_COM()  # 合规检查。
        for subtree_wt_COM in self.subtrees_wt_COM:
            # * 获得COM项和COM并列项。
            COM_val = subtree_wt_COM.COM_
            COM_parallel_vals = [v for k, v in subtree_wt_COM.items() if (k != 'COM_') and (is_subtree(v, self))]
            # * 将COM项增量更新到并列项上。
            for par_val in COM_parallel_vals:
                par_val: T_Config
                par_val.dict_update(COM_val, incremental=True)
            # * 删除COM项。
            del subtree_wt_COM['COM_']

    def branch_copy(self: T_Config) -> T_Config:
        """拷贝枝干（Config及其子Config），直接赋值叶子（Config的所有值）。

        Returns:
            拷贝后的Config，枝干用当前类别重建，叶子直接赋值自self。
        """
        ret = self.__class__()  # 建立一棵本类新树。
        # 先拷贝依赖，避免拷贝后丢失依赖。
        object.__setattr__(ret, '_cfgs_update_at_init', object.__getattribute__(self, '_cfgs_update_at_init'))
        object.__setattr__(ret, '_cfgs_update_at_parser', object.__getattribute__(self, '_cfgs_update_at_parser'))

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

    def dict_update(self, other: dict, incremental: bool=False):
        """接收一个dict类型（或其子类）的配置树，更新到当前配置树上。

        应当确保，此后对本配置树枝干的修改，不会同步到原配置树上。因此，应当拷/转枝赋叶原配置树后，再做更新。

        Args:
            other: dict类型的配置树。
            incremental: 增量式更新，如果冲突，保留旧树。

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
                self[k].dict_update(v, incremental=incremental)
            else:
                if incremental:  # 若增量式，则人有我无，方才更新。
                    if k not in self:
                        self[k] = v
                    else:
                        pass
                else:  # 人有我有，但至少一方不是子树；或人有我无，则直接更新。
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
        return self

    def unfreeze(self):
        return self.freeze(False)

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

    def set_IL(self, name: str | None=None, priority: int=1):
        """返回装饰器，装饰器将被装饰函数注册为当前配置树的惰性项。

        Args:
            name: 函数名，若为None，则使用被装饰函数的函数名。
            priority: 惰性项的优先级。

        Returns:
            装饰器。
        """

        def decorator(func):
            assert callable(func), f"IL must be callable, but got {func}"

            if name is not None:
                self[name] = IL(func, priority=priority)
            else:
                self[func.__name__] = IL(func, priority=priority)
            return func

        return decorator

    def set_func(self, name: str | None=None):
        """返回装饰器，装饰器将被装饰函数注册为当前配置树的项目。

        Args:
            name: 函数名，若为None，则使用被装饰函数的函数名。

        Returns:
            装饰器。
        """

        def decorator(func):
            assert callable(func), f"IL must be callable, but got {func}"

            if name is not None:
                self[name] = func
            else:
                self[func.__name__] = func
            return func

        return decorator
