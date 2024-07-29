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
import copy
import os
import pickle
import pprint
import sys
import warnings
from keyword import iskeyword
from pathlib import Path
from typing import Callable, Any, TypeVar, Type, Generator, final, cast, Iterable, Union, ClassVar, Literal

from colorama import Fore, Style

if sys.version_info >= (3, 11):
    from typing import Self, Never, TypeAlias
else:  # 兼容Python<3.11。
    Self = TypeVar('Self', bound='ADict')
    from typing import NoReturn as Never
    TypeAlias = Any

try:
    from addict import Dict
    DICT_AVAILABLE = True
except ImportError:
    Dict = dict
    DICT_AVAILABLE = False

from .open_cfg import open_config

__all__ = ['ItemLazy', 'IL', 'Config', 'ADict', 'DEP']

T = TypeVar("T")
T_dict = TypeVar('T_dict', bound=dict)
T_ADict = TypeVar('T_ADict', bound='ADict')
T_Config = TypeVar('T_Config', bound='Config')
T_IL_Func: TypeAlias = Callable[['Config'], Any]


def is_valid_variable_name(name: str) -> bool:
    return name.isidentifier() and not iskeyword(name)


def prefix2name(prefix: str) -> str:
    return prefix[:-1]  # '' -> '', 'xxx.' -> 'xxx'


def name2prefix(name: str) -> str:
    return f'{name}.' if name else name  # 'xxx' -> 'xxx.', '' -> ''


def append_prefix(prefix: str, suffix: str, still_prefix: bool=True) -> str:
    if is_valid_variable_name(suffix):
        name = f'{prefix}{suffix}'
    else:
        name = f"{prefix2name(prefix)}['{suffix}']"

    return name2prefix(name) if still_prefix else name


class ItemLazy(object):

    def __init__(self, func: T_IL_Func, priority: int=1, rel: bool=True):
        """配置惰性项。

        Args:
            func: 惰性项的计算函数。
            priority: 惰性项优先级，越小越优先。
            rel: 是否使用相对配置根。
        """
        self.func = func
        self.priority = priority
        self._level: Union[int, float, None] = None

        if not rel:  # 如果使用绝对根，则level为无穷。
            self._level = float('inf')

    @property
    def level(self) -> Union[int, float]:
        return self._level

    @level.setter
    def level(self, value: int) -> None:
        assert self._level is None
        self._level = value

    def __call__(self, cfg: 'Config') -> Any:
        return self.func(cfg)

    @classmethod
    def ordered_item_lazy(cls, config: T_Config) -> list[tuple[T_Config, Any, int]]:
        return sorted(((c, k, l.priority) for c, k, l in config.ckl if isinstance(l, cls)), key=lambda x: x[2])

    @classmethod
    def compute_item_lazy(cls, config: T_Config) -> T_Config:
        item_lazy_dic_keys = cls.ordered_item_lazy(config)

        if len(item_lazy_dic_keys) > 0:
            for item_lazy_dic, item_lazy_key, _ in item_lazy_dic_keys:
                item_lazy: cls = item_lazy_dic[item_lazy_key]
                root, _ = item_lazy_dic.find_root(item_lazy.level)
                item_lazy_dic[item_lazy_key] = item_lazy(root)
            return cls.compute_item_lazy(config)
        else:
            return config

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.func}, priority={self.priority}, level={self.level})"


DEP = IL = ItemLazy


class ADict(Dict):
    """继承自addict.Dict，用作数据容器。修复了addict.Dict的诸多问题。"""

    _go_in_list: ClassVar[bool] = True  # 构造时递归进入list。

    # -* 构造与反构造。

    def __init__(self, *args, **kwargs):
        dict.__init__(self)

        # -* 预设__parent、__key、__frozen、_whole。
        self.set_attribute('__parent', kwargs.pop('__parent', None))
        self.set_attribute('__key', kwargs.pop('__key', None))
        self.set_attribute('__frozen', False)
        self.set_attribute('_whole', False)

        # -* CHANGE 与标准字典一样，只允许一个arg。
        if len(args) > 1:
            raise TypeError(f"{self.__class__} expected at most 1 arguments, got {len(args)}")
        elif len(args) == 1:
            arg = args[0]
            if isinstance(arg, dict):
                dic = arg
            else:
                dic = dict(arg)
            self.from_dict(dic, ret=self, memo=None, always_copy=True, deepcopy=False)
        else:
            pass

        # CHANGE kwargs不再硬update，而是先变类拷贝为ADict，再update到当前ADict上。
        if kwargs:
            self.dict_update(self.from_dict(kwargs, ret=None, memo=None, always_copy=True, deepcopy=False))

    @staticmethod
    def is_subtree(tree: Any, root: dict) -> bool:
        """判断tree是否是root的子树。"""
        return isinstance(tree, dict)

    def copy_cfg_attr_to(self, other: 'ADict') -> None:
        if isinstance(other, ADict):
            other.set_attribute('_whole', self.get_attribute('_whole'))

    @classmethod
    def from_dict(cls: Type[T_ADict],
                  other: dict, ret: T_ADict=None, memo: dict=None,
                  always_copy: bool=False, deepcopy: bool=False) -> T_ADict:
        # -* 参数检查。
        if not isinstance(other, dict):
            raise TypeError(f"{cls.__name__}.from_dict expected dict as other, got {type(other)}")

        if ret is not None:  # ret不为None只用于初始化的时候，此时总是做浅拷贝，且因为是从头开始，memo应当为None。
            assert always_copy and (memo is None) and (not deepcopy), "Param 'ret' can only be used by __init__."

        # -* 参数预处理。
        if memo is None:
            memo = {}

        if deepcopy:  # deepcopy时，总是拷贝。
            always_copy = True

        # -* 若不是always_copy，说明目的只是变类。若other是cls的实例，则可以不做拷贝，直接返回。
        if (not always_copy) and (type(other) is cls):  # 此时deepcopy=False, ret=None。
            assert len(memo) == 0  # 若memo不为空，要么是处于deepcopy状态，要么是处于_from_dict_hook后的栈，都不是变类用法。
            return cast(T_ADict, other)

        # -* 开启拷贝。
        if id(other) in memo:  # 若other已经被拷贝过，则直接返回拷贝。
            return memo[id(other)]

        if ret is None:  # 构造新树。
            ret = cls()
        else:  # 新树由__init__提供。
            assert type(ret) is cls and len(ret) == 0, f"{ret=} should be an empty {cls=}."
        memo[id(other)] = ret  # 将新容器加入memo，处理循环引用。

        if isinstance(other, ADict):  # 如果是ADict及其子类，则也要做属性拷贝。
            other.copy_cfg_attr_to(ret)

        for k, v in other.items():
            ret[k] = cls._from_dict_hook(v, other, memo=memo, deepcopy=deepcopy)

        return ret

    @classmethod
    def _from_dict_hook(cls, item: Any, other: dict, memo: dict, deepcopy: bool) -> Any:
        if id(item) in memo:  # 若item已经被拷贝过，则直接返回拷贝。
            ret = memo[id(item)]
        elif cls.is_subtree(item, other):
            ret = cls.from_dict(item, ret=None, memo=memo, always_copy=True, deepcopy=deepcopy)
        elif cls._go_in_list and isinstance(item, list):
            ret = []
            memo[id(item)] = ret
            ret.extend(cls._from_dict_hook(elem, other, memo=memo, deepcopy=deepcopy) for elem in item)
        else:
            ret = copy.deepcopy(item, memo) if deepcopy else item

        return ret

    def to_dict(self, memo: dict=None) -> dict:
        # -* 参数预处理。
        if memo is None:
            memo = {}

        # -* 开启拷贝。
        if id(self) in memo:  # 若other已经被拷贝过，则直接返回拷贝。
            return memo[id(self)]

        ret = {}  # 构造新树。

        memo[id(self)] = ret  # 将新容器加入memo，防止循环引用。

        for k, v in self.items():
            ret[k] = self._to_dict_hook(v, memo=memo)
        return ret

    @classmethod
    def _to_dict_hook(cls, item: Any, memo: dict) -> Any:
        if id(item) in memo:  # 若item已经被拷贝过，则直接返回拷贝。
            ret = memo[id(item)]
        elif isinstance(item, ADict):
            ret = item.to_dict(memo=memo)
        elif DICT_AVAILABLE and isinstance(item, Dict):
            ret = item.to_dict()  # 兼容addict.Dict，但memo无法传递。
        elif cls._go_in_list and isinstance(item, list):  # 保持对称，若from_dict时list被视为普通值，则to_dict时也应该如此。
            ret = []
            memo[id(item)] = ret
            ret.extend(cls._to_dict_hook(elem, memo=memo) for elem in item)
        else:
            ret = item

        return ret

    # -* 序列化与反序列化。

    def __getnewargs__(self) -> tuple:  # CHANGE 无需返回任何值。
        return tuple()

    def __getstate__(self) -> Any:  # CHANGE 返回self.__dict__，而非自身作为state。
        if sys.version_info >= (3, 11):
            return object.__getstate__(self)
        else:
            return vars(self)

    def __setstate__(self, state: dict) -> None:  # CHANGE: 恢复object默认的__setstate__。
        vars(self).update(state)  # 用户自定义的类，哪怕没有__init__过，其实例总是有__dict__。

    # -* 属性操作。

    def get_attribute(self, name: str, /) -> Any:
        return object.__getattribute__(self, name)

    def get_attribute_default(self, name: str, default: Any=None, /) -> Any:
        try:
            return self.get_attribute(name)
        except AttributeError:
            return default

    def __getattr__(self, item: str, /) -> Any:
        return self[item]

    def __setattr__(self, name: str, value: Any, /) -> None:
        if hasattr(self.__class__, name):
            raise AttributeError(f"{self.__class__}'s class attribute '{name}' is read-only.")
        elif name in vars(self):  # CHANGE 实例属性也不可冲突。
            raise AttributeError(f"{self.__class__}'s instance attribute '{name}' is read-only.")
        else:
            self[name] = value

    def __delattr__(self, name, /) -> None:
        del self[name]

    def set_attribute(self, name: str, value: Any, /) -> None:
        object.__setattr__(self, name, value)

    def del_attribute(self, name: str, /) -> None:
        object.__delattr__(self, name)

    # -* 字典操作。

    def _mount2parent(self) -> None:
        if (p := self.get_attribute('__parent')) is not None:
            p[self.get_attribute('__key')] = self

            self.set_attribute('__parent', None)  # CHANGE pk不删除，而是置为None。
            self.set_attribute('__key', None)

    def __setitem__(self, key: Any, value: Any, /) -> None:
        if not vars(self):  # CHANGE 若实例属性字典为空，说明处于pickle.loads的恢复内生数据阶段，可走捷径直接dict setitem。
            dict.__setitem__(self, key, value)
            return

        self._setitem(key, value)

    def _setitem(self, key: Any, value: Any, /) -> None:
        if self.is_frozen:  # CHANGE 调用addict setitem前，检查frozen，若是，则完全禁止setitem。
            raise RuntimeError(f"{self.__class__} is frozen. ")

        self._mount2parent()

        dict.__setitem__(self, key, value)

    def __missing__(self, key: Any, /) -> Self:
        if self.is_frozen:
            raise KeyError(f"{key} not found in frozen {self.__class__}.")
        return self.__class__(__parent=self, __key=key)

    def __delitem__(self, key, /) -> None:
        if self.is_frozen:
            raise RuntimeError(f"{self.__class__} is frozen. ")

        dict.__delitem__(self, key)

    # -* 拷贝与更新。

    def branch_copy(self) -> Self:
        return self.from_dict(self, ret=None, memo=None, always_copy=True, deepcopy=False)

    def copy(self) -> Self:
        return self.branch_copy()

    def __copy__(self) -> Self:
        return self.branch_copy()

    def deepcopy(self) -> Self:
        return self.from_dict(self, ret=None, memo=None, always_copy=True, deepcopy=True)

    def __deepcopy__(self, memo: dict) -> Self:
        return self.from_dict(self, ret=None, memo=memo, always_copy=True, deepcopy=True)

    @classmethod
    def _dict_update(cls, dic: dict, other: dict, incremental: bool=False, copy_other: bool=True) -> None:
        # -* 参数检查。
        if not isinstance(dic, dict):
            raise TypeError(f"{cls.__name__}._dict_update expected dict as dic, got {type(dic)}")

        if not isinstance(other, dict):
            raise TypeError(f"{cls.__name__}._dict_update expected dict as other, got {type(other)}")

        # NOTE 可以证明，对ADict的子树判定（dict子类-dict子类），拷贝前互为子树 ⇔ 拷贝后互为子树，且均为ADict类。
        # NOTE 可以证明，对Config的子树判定（D-D），拷贝前互为子树 ⇒ 拷贝后互为子树，且均为Config类，
        # NOTE 但是 拷贝后互为子树，且均为Config类 不能推出 拷贝前互为子树。
        # NOTE 总而言之，可以认为拷贝能保留子树关系，且子树有is_whole性质。但Config的拷贝可能会建立新的子树关系。
        other: ADict = (cls.from_dict(other, ret=None, memo=None, always_copy=True, deepcopy=False)
                        if copy_other else
                        other)

        # NOTE dict_update只更新k-v对，不涉及更新属性（依赖、是否不可分）。
        # NOTE 因为除了以下场景，想不到需要更新属性的情况：
        # NOTE 1）更新依赖，但这种情况下，应当用_whole完整覆盖原来的树。本打算继承的部分可以独立出来，作为共享的导入。
        for k, v in tuple(other.items()):
            # NOTE 每对键值都只使用一次，故取出后就可以从other中删除，可以防止Config一颗子树两个爹。
            # NOTE 该做法的风险只发生在other中存在循环引用。目前循环引用下的行为是未定义的，干脆不考虑。
            del other[k]

            # 若人有我有，且都是子树，且人的子树不被视作整体，则子树更新子树。
            # NOTE 若某棵树_whole=True，改树作为value时，视作不可递归的整体。
            if ((k in dic) and
                    cls.is_subtree(v, other) and (not isinstance(v, ADict) or not v.is_whole) and
                    cls.is_subtree(dic[k], dic)):
                cls._dict_update(dic[k], v, incremental=incremental,
                                 copy_other=False)  # 只有根的other需要拷贝，后续递归都是直接更新。
            else:
                if incremental:  # 若增量式，则人有我无，方才更新。
                    if k not in dic:
                        dic[k] = v
                    else:
                        pass
                else:  # 人有我有，但至少一方不是子树；或人有我无，则直接更新。
                    dic[k] = v

    def dict_update(self, other: dict, incremental: bool=False) -> None:
        """接收一个dict类型（或其子类）的配置树，更新到当前配置树上。

        应当确保，此后对本配置树枝干的修改，不会同步到原配置树上。因此，应当拷/转枝赋叶原配置树后，再做更新。

        Args:
            other: dict类型的配置树。
            incremental: 增量式更新，如果冲突，保留旧树。

        Returns:
            更新后的配置树。
        """
        self._dict_update(self, other, incremental=incremental)

    def update(self, *args, **kwargs) -> None:
        if len(args) > 1:
            raise TypeError(f"{self.__class__}.update expected at most 1 arguments, got {len(args)}")
        elif len(args) == 1:
            self.dict_update(args[0])
        else:
            pass

        if kwargs:
            self.dict_update(kwargs)

    # -* 运算。
    def __add__(self, other: Never) -> Never:
        raise NotImplementedError("ADict does not support __add__.")

    def __or__(self, other: dict) -> Self:
        if not isinstance(other, dict):
            return NotImplemented(f"{self.__class__} does not support | with {type(other)}.")
        new = self.branch_copy()
        new.dict_update(other)
        return new

    def __ror__(self, other: dict) -> Self:
        return self.__or__(other)

    def __ior__(self, other: dict) -> Self:
        if not isinstance(other, dict):
            return NotImplemented(f"{self.__class__} does not support |= with {type(other)}.")
        self.dict_update(other)
        return self

    # -* 冻结操作。

    def freeze(self, shouldFreeze=True) -> Self:
        for b in self.branches:
            if isinstance(b, Dict if DICT_AVAILABLE else ADict):  # 将所有子树也冻结。
                object.__setattr__(b, '__frozen', shouldFreeze)
        return self

    def unfreeze(self) -> Self:
        return self.freeze(False)

    @property
    def is_frozen(self) -> bool:
        return self.get_attribute_default('__frozen', False)

    # -* 重写受到影响的字典方法。

    def setdefault(self, key: Any, default: Any=None) -> Any:
        if key in self:
            return self[key]
        else:
            self[key] = default
            return default

    @classmethod
    def fromkeys(cls: Type[T_ADict], iterable: Iterable[T], value=None) -> T_ADict:
        return cls.from_dict(dict.fromkeys(iterable, value), ret=None, memo=None, always_copy=True, deepcopy=False)

    def clear(self) -> None:
        if self.is_frozen:
            raise RuntimeError(f"{self.__class__} is frozen. ")

        dict.clear(self)

    def pop(self, key: Any, default: Any=None) -> Any:
        if self.is_frozen:
            raise RuntimeError(f"{self.__class__} is frozen. ")

        return dict.pop(self, key, default)

    def popitem(self) -> tuple[Any, Any]:
        if self.is_frozen:
            raise RuntimeError(f"{self.__class__} is frozen. ")

        return dict.popitem(self)

    # -* _whole属性相关操作。

    def set_whole(self, is_whole: bool=True) -> Self:
        """将自身设置为不可递归的整体。"""
        self.set_attribute('_whole', is_whole)
        self._mount2parent()  # 挂载到父节点，确保设置总是有效。
        return self

    override = set_whole

    @property
    def is_whole(self) -> bool:
        return self.get_attribute_default('_whole', False)

    def empty_leaf(self) -> Self:
        """转为本类的空且whole的字典，即空叶子。常用于表示无效/无定义项。"""
        self.clear()
        return self.set_whole(True)  # 设置为整体，并挂载到父节点。

    def empty_tree(self) -> Self:
        """转为本类的空的字典，即空子树。"""
        self.clear()
        return self.set_whole(False)

    # -* 浏览。

    @classmethod
    def _named_branches(cls, dic: dict, name: str='',
                        memo: set=None) -> Generator[tuple[str, dict], None, None]:
        if memo is None:
            memo = set()

        if id(dic) in memo:  # 若已经遍历过，截断。
            return
        memo.add(id(dic))  # 该树已经遍历过。

        yield name, dic

        for k, v in dic.items():
            if cls.is_subtree(v, dic):
                subtree_name = append_prefix(name2prefix(name), k, still_prefix=False)
                yield from cls._named_branches(v, subtree_name, memo)  # 若v是子树，则抛出子树的枝条。

    @property
    def named_branches(self) -> Generator[tuple[str, dict], None, None]:
        yield from self._named_branches(self)

    @property
    def branches(self) -> Generator[dict, None, None]:
        yield from (b for _, b in self._named_branches(self))

    @classmethod
    def _named_ckl(cls, dic: dict, prefix: str= '',
                   memo: set=None) -> Generator[tuple[str, tuple[dict, Any, Any]], None, None]:
        if memo is None:
            memo = set()

        if id(dic) in memo:  # 若已经遍历过，截断。
            return
        memo.add(id(dic))  # 该树已经遍历过。

        for k, v in dic.items():
            if cls.is_subtree(v, dic):
                pre = append_prefix(prefix, k, still_prefix=True)
                yield from cls._named_ckl(v, pre, memo)  # 若v是子树，则抛出子树的named_ckl。
            else:
                name = append_prefix(prefix, k, still_prefix=False)
                yield name, (dic, k, v)  # 若v是叶子，则抛出名字、父节点、键、值。

    @property
    def named_ckl(self) -> Generator[tuple[str, tuple[dict, Any, Any]], None, None]:
        yield from self._named_ckl(self)

    @property
    def ckl(self) -> Generator[tuple[dict, Any, Any], None, None]:
        yield from ((c, k, l) for _, (c, k, l) in self.named_ckl)

    @property
    def leaves(self) -> Generator[Any, None, None]:
        yield from (l for _, (_, _, l) in self.named_ckl)

    # -* 快捷方式。

    def set_func(self, name: Union[str, None]=None) -> Callable[[Callable], Callable]:
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

    # -* 打印。

    def to_txt(self, prefix: str='dic.', print_empty_leaf: bool=False, color: bool=False) -> str:
        def cr(n: str, fore: str) -> str:
            if color:
                return f'{fore}{n}{Style.RESET_ALL}'
            else:
                return n

        if len(prefix) == 0:  # 为了可读性，禁止空前缀。
            raise ValueError("prefix should not be empty.")

        if len(prefix) > 0 and not prefix.endswith('.'):  # 前缀应当以'.'结尾。
            prefix = name2prefix(prefix)

        root_name = prefix2name(prefix)
        lines = [f'{cr(root_name, Fore.MAGENTA)} = {self.__class__.__name__}()']

        # NOTE 特殊树在此处构造。
        for name, b in self._named_branches(self, name=root_name):
            # 若为_whole或有caps，则为ADict特殊树。特殊树要打印特殊属性。
            if ((isinstance(b, ADict) and b.is_whole) or
                    (isinstance(b, Config) and b.get_attribute('_cfgs_update_at_parser'))):
                line = f"{cr(name, Fore.MAGENTA)}.override({b.is_whole})"  # 打印ADict的_whole属性，该操作还会确保树的存在。
                # 若为Config且存在依赖，还要打印_cfgs_update_at_parser。
                if isinstance(b, Config) and (caps := b.get_attribute('_cfgs_update_at_parser')):
                    line = line + f".set_attribute('_cfgs_update_at_parser', {caps})"
                lines.append(line)

        # NOTE 所有值和普通非空树在此处构造。
        lines.append('# ------- ↓ LEAVES ↓ ------- #')

        for name, (c, k, l) in self._named_ckl(self, prefix=prefix):
            str_l = pprint.pformat(l)
            if '\n' not in str_l:
                lines.append(f"{cr(name, Fore.CYAN)} = {str_l}")
            else:
                lines.append(f"{cr(name, Fore.CYAN)} = \\ \n{str_l}")

        # NOTE 打印空树。
        if print_empty_leaf:
            lines.append('# ------- ↓ EMPTY LEAVES ↓ ------- #')

            for name, b in self._named_branches(self, name=root_name):
                if (len(b) == 0) and (not (isinstance(b, ADict) and b.is_whole)):
                    lines.append(f"{cr(name, Fore.YELLOW)} = {b.__class__.__name__}()")

        return '\n'.join(lines)


class Config(ADict):
    """配置字典。继承自Dict，从类型上和Addict区分，以便分离配置项和配置树。"""

    _go_in_list: ClassVar[bool] = False  # 构造时不递归进入list。

    # -* Config的初始化与解析。

    def __init__(self, *cfgs: Union[dict, str],
                 cfgs_update_at_parser: Union[tuple[str, ...], str]=(), caps: Union[tuple[str, ...], str]=(), **kwargs):
        """支持从其他其他配置树模块路径或配置树dict初始化。所有配置树会被逐个dict_update到当前配置树上。

        Args:
            *cfgs: List[配置树所在模块|配置树]
            cfgs_update_at_parser: 解析时用于更新的基配置。
            caps: cfgs_update_at_parser的别名。
            **kwargs: 传递给Dict，不应该使用。
        """
        # -* 初始化。
        __mounted = kwargs.pop('__mounted', True)
        super().__init__(**kwargs)  # __parent, __key, __frozen, _whole被设置。
        self.set_attribute('__mounted', __mounted)

        # -* 检查caps。
        if caps:
            assert not cfgs_update_at_parser, f"{caps=}和{cfgs_update_at_parser=}不能同时使用。"
            cfgs_update_at_parser = caps

        # -* 同一cfgs_update_at_parser格式。
        if isinstance(cfgs_update_at_parser, str):
            cfgs_update_at_parser = (cfgs_update_at_parser,)

        # -* 遍历配置树，更新到当前配置树上。
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
            if isinstance(cfg, Config) and (cfg_dep := cfg.get_attribute('_cfgs_update_at_parser')):
                # A --解析时--> B; D, B --解析时--> C 含义明确，即加载时DFS，从祖先开始增量更新到C，优先级为B、A、D。
                # A --解析时--> B; D, B --加载时--> C 若D与A冲突，则D中键值会阻塞A，优先级为B、D、A，这是我们不希望的。
                # 因此，最好不要混用解析时和加载时依赖。
                warnings.warn(f"{cfg=}存在解析时依赖{cfg_dep=}。\n"
                              f"因此，该配置也应该作为当前配置的解析时依赖。否则并列的加载时依赖，可能干扰cfg_dep更新cfg。")
                cfgs_update_at_parser = cfgs_update_at_parser + cfg_dep

        # -* 记录基配置。
        # self.set_attribute('_cfgs_update_at_init', cfgs)  # ! 该值除了debug，根本用不到。如果cfgs不是str而是字典，pickle还会增加额外的负担。
        self.set_attribute('_cfgs_update_at_parser', cfgs_update_at_parser)

    def update_at_parser(self) -> None:
        # -* 获取解析式基配置。
        cfgs_update_at_parser = self.get_attribute('_cfgs_update_at_parser')
        # -* 若无需更新，跳过。
        if not cfgs_update_at_parser:
            return
        # -* 保存主配置。
        main_cfg = self.branch_copy()
        # -* 清空当前配置树。
        self.clear()
        # -* 逐个读取基配置，并更新到当前配置中。
        for i, base_cfg in enumerate(cfgs_update_at_parser):
            # -* 若给出配置树模块路径，则根据路径打开配置树。
            if isinstance(base_cfg, str):
                base_cfg, _ = open_config(base_cfg)
            # -* 基配置也进行解析时更新。
            if isinstance(base_cfg, Config):
                base_cfg.update_at_parser()
            # -* 检查配置树是dict。
            if not isinstance(base_cfg, dict):
                raise ValueError(f"{i}'th item = {base_cfg} in self._cfgs_update_at_parser should be "
                                 f"(opened as) dict.")
            # -* 将基配置树更新到本配置树上。
            self.dict_update(base_cfg)
        # * 将主配置树更新回当前配置树。
        self.dict_update(main_cfg)

    def parse(self, experiments_root: str='', config_root: str= './configs',
              create_rslt_dir: bool=True) -> Self:
        # TODO 避免循环导入。Ugly，更好的办法是将parse_cfg.py中的parse_config放到py_cfg.py中。
        from .parse_cfg import parse_config
        return parse_config(self, experiments_root, config_root, create_rslt_dir)

    def compute_item_lazy(self) -> Self:
        return ItemLazy.compute_item_lazy(self)

    def load(self, experiments_root: str='', config_root: str='./configs',
             create_rslt_dir: bool=True) -> 'Config':
        # TODO 避免循环导入。Ugly，更好的办法是将parse_cfg.py中的load_config放到py_cfg.py中。
        from .parse_cfg import load_config
        return load_config(self, experiments_root, config_root, create_rslt_dir)

    @property
    def subtrees_wt_COM(self: T_ADict) -> list[T_ADict]:
        return [subtree for subtree in self.branches if 'COM_' in subtree]

    def _check_COM(self: T_ADict):
        """检查COM树是否合规。由于reduce_COM会逐步删除子COM树，可能影响安全检查。故检查放在reduce_COM之前。"""
        warnings.warn("COM feature is deprecated", DeprecationWarning)
        # NOTE COM树，可以有子COM树。因为子树搜索为DFS后序，故子树COM树总是先于父COM树处理。
        # NOTE 而COM并列项如果有子COM树，则无法确定和当前COM项的执行顺序，非常危险，要警告。
        for subtree_wt_COM in self.subtrees_wt_COM:
            # * COM项应该是子树。
            if not self.is_subtree(subtree_wt_COM.COM_, self):
                raise RuntimeError(f"COM树{subtree_wt_COM}的COM项{subtree_wt_COM.COM_}应当为子树。")
            # * COM并列项不含COM子树。
            COM_parallel_vals = [v for k, v in subtree_wt_COM.items() if (k != 'COM_') and (self.is_subtree(v, self))]
            for par_val in COM_parallel_vals:
                par_val: T_ADict
                if par_val.subtrees_wt_COM:
                    warnings.warn(f"COM树{subtree_wt_COM}的COM并列项{par_val}存在子COM树。无法确定该树和当前"
                                  f"COM树的执行顺序，reduce_COM行为不可预测。")

    def reduce_COM(self: T_ADict):
        """归并掉所有COM项。"""
        warnings.warn("COM feature is deprecated", DeprecationWarning)
        self._check_COM()  # 合规检查。
        for subtree_wt_COM in self.subtrees_wt_COM:
            # * 获得COM项和COM并列项。
            COM_val = subtree_wt_COM.COM_
            COM_parallel_vals = [v for k, v in subtree_wt_COM.items() if (k != 'COM_') and (self.is_subtree(v, self))]
            # * 将COM项增量更新到并列项上。
            for par_val in COM_parallel_vals:
                par_val: T_ADict
                # NOTE 当_COM树本身为_whole时：1）cfg_update时，作为不可递归树覆盖；2）更新并列树时，行为不变（此时_COM树
                # NOTE 作为根节点树，_whole无效。
                # NOTE 当_COM树的子树为_whole时：1）cfg_update时，作为不可递归树覆盖；2）更新并列树时, 作为不可递归树覆盖。
                # NOTE 该行为可能与预期不符（我们希望更新并列树时，不要覆盖），此时应当寻找其他workaround，如让_COM树，而非
                # NOTE _COM树的子树为_whole。
                par_val.dict_update(COM_val, incremental=True)
            # * 删除COM项。
            del subtree_wt_COM['COM_']

    # -* 重载子树定义和属性拷贝。

    @staticmethod
    def is_subtree(tree: Any, root: dict) -> bool:
        """判断tree是否是root的子树。"""
        return type(tree) is type(root)

    @final  # 不应该重载该方法。所有Config子类应当由相同的copy_cfg_attr_to，否则from_dict无法在不同Config子类间正确运行。
    def copy_cfg_attr_to(self, other: ADict) -> None:
        """将self的cfg属性(依赖、是否不可分)拷贝到other上。"""
        super().copy_cfg_attr_to(other)  # 拷贝_whole属性。
        if isinstance(other, Config):
            # other.set_attribute('_cfgs_update_at_init', self.get_attribute('_cfgs_update_at_init'))
            other.set_attribute('_cfgs_update_at_parser', self.get_attribute('_cfgs_update_at_parser'))

    # -* 双向树结构。

    @property
    def _is_mounted(self) -> bool:
        # 若没有_mounted，则处于pickle的new而未init阶段，此时无需担心挂载问题。
        return self.get_attribute_default('__mounted', True)

    def _mount2parent(self) -> None:
        if not self._is_mounted:
            self.get_attribute('__parent')[self.get_attribute('__key')] = self
            self.set_attribute('__mounted', True)

    def demount(self) -> None:
        assert self._is_mounted
        self.set_attribute('__parent', None)
        self.set_attribute('__key', None)

    def find_root(self, level: Union[int, float]=float('inf')) -> tuple[Self, int]:
        """寻找树根。"""
        root = self
        count = level
        cur_level = 0

        while count:
            new_root = root.get_attribute('__parent')
            if new_root is None:
                if count == float('inf'):  # level为inf时，穷尽所有父节点寻找根。
                    break
                else:
                    raise RuntimeError(f"Can only find root at max level={cur_level}, but got level={level}.")
            else:
                root = new_root

            count -= 1
            cur_level += 1

        return root, cur_level

    def _setitem(self, key: Any, value: Any, /) -> None:
        super()._setitem(key, value)  # frozen检查、挂载、dict.__setitem__。
        value = self[key]  # 重新获取value，不能保证value在设置后不变。

        # -* 确保子树自洽。
        if self.is_subtree(value, self):
            value: Config
            # -* 对missing情况，检查parent、key正确，防止一子二父。
            if not value._is_mounted:
                assert value.get_attribute('__parent') is self
                assert value.get_attribute('__key') == key  # key是不可变对象，故用==比较。
            # -* 非missing情况，对已经有parent的子树，替换为它的拷贝，防止一子二父。
            elif (value.get_attribute('__parent') is not None) or (value.get_attribute('__key') is not None):
                self[key] = value.branch_copy()
                value = self[key]
                # TODO 此处有一个隐藏BUG：当该value含有DEP时，改DEP会被赋值过来。若该DEP是rel模式，则level很可能错乱。
                # TODO 解决方案1：重算level和lambda：太复杂，做不到。
                # TODO 解决方案2：拷贝过来的IL作为“分身”，不直接计算，等“真身”计算完后赋值：“真身”若来自其他树，可能不计算，不可信。
                # TODO 解决方案3：安全检查，赋值过来的树里，如果有rel的DEP，如果其level超过了赋值树的顶层，则报错。
            # -* 根子树挂载。
            else:
                value.set_attribute('__parent', self)
                value.set_attribute('__key', key)

        # -* 对IL，设置其层级。
        if isinstance(value, ItemLazy) and value.level is None:
            # 如果IL已经有level，要么是inf，说明是绝对根，不用记录level。要么是已经被设置，当前是branch_copy之类的，反正不用再设置。
            _, value.level = self.find_root(level=float('inf'))

    def __missing__(self, key: Any, /) -> Self:  # CHANGE frozen也返回空字典，反正setitem一定会触发报错。
        return self.__class__(__parent=self, __key=key, __mounted=False)

    def __delitem__(self, key, /) -> None:
        if self.is_frozen:
            raise RuntimeError(f"{self.__class__} is frozen. ")

        if self.is_subtree(v := self[key], self):
            v.demount()
        dict.__delitem__(self, key)

    def clear(self) -> None:
        if self.is_frozen:
            raise RuntimeError(f"{self.__class__} is frozen. ")

        for v in self.values():
            if self.is_subtree(v, self):
                v.demount()
        dict.clear(self)

    def pop(self, key: Any, default: Any=None) -> Any:
        if self.is_frozen:
            raise RuntimeError(f"{self.__class__} is frozen. ")

        if key not in self:
            return default

        ret = dict.pop(self, key)
        if self.is_subtree(ret, self):
            ret.demount()

        return ret

    def popitem(self) -> tuple[Any, Any]:
        if self.is_frozen:
            raise RuntimeError(f"{self.__class__} is frozen. ")

        k, v = dict.popitem(self)
        if self.is_subtree(v, self):
            v.demount()
        return k, v

    # -* 快捷方式。

    def set_IL(self, name: Union[str, None]=None, priority: int=1, rel: bool = True) \
            -> Callable[[T_IL_Func], T_IL_Func]:
        """返回装饰器，装饰器将被装饰函数注册为当前配置树的惰性项。

        Args:
            name: 函数名，若为None，则使用被装饰函数的函数名。
            priority: 惰性项的优先级。
            rel: 是否是相对优先级。

        Returns:
            装饰器。
        """

        def decorator(func):
            assert callable(func), f"IL must be callable, but got {func}"

            if name is not None:
                self[name] = IL(func, priority=priority, rel=rel)
            else:
                self[func.__name__] = IL(func, priority=priority, rel=rel)

            return func

        return decorator

    set_DEP = set_IL

    # -* 对Config，使用to_txt字符化。

    def to_txt(self, prefix: str='cfg.', print_empty_leaf: bool=False, color: bool=False) -> str:
        return super().to_txt(prefix=prefix, print_empty_leaf=print_empty_leaf, color=color)

    def __str__(self) -> str:
        return self.to_txt(color=True)

    def save_py(self, file: Union[str, os.PathLike]) -> Path:
        (file := Path(file)).parent.mkdir(parents=True, exist_ok=True)

        cfg_str = f"""
# -*- THIS CONFIG FILE IS AUTO-GENERATED BY AlchemyCat -*-
from alchemy_cat.dl_config import {type(self).__name__}

{self.to_txt(prefix='cfg.', print_empty_leaf=True)}
"""

        file.write_text(cfg_str)

        return file

    def save_pkl(self, file: Union[str, os.PathLike], save_copy: bool=True) -> Path:
        (file := Path(file)).parent.mkdir(parents=True, exist_ok=True)

        file.write_bytes(pickle.dumps(self.branch_copy() if save_copy else self))

        return file

    # -* 迁移。

    def _save_cfg_for_other_cfg_system(self) -> dict:
        # -* 分离。
        saved_cfg = self.branch_copy()

        # -* 假设该配置拥有rslt_dir，则会经历：update_at_parser -> compute_item_lazy -> freeze
        saved_cfg = saved_cfg.load(experiments_root='', create_rslt_dir=False)

        # -* 转为普通dict返回。
        return saved_cfg.to_dict()

    def save_yaml(self, file: Union[str, os.PathLike]) -> Path:
        import yaml

        (file := Path(file)).parent.mkdir(parents=True, exist_ok=True)

        file.write_text(yaml.dump(self._save_cfg_for_other_cfg_system()))

        return file

    def save_mmcv(self, file: Union[str, os.PathLike]) -> Path:
        from mmengine import Config as MMConfig

        (file := Path(file)).parent.mkdir(parents=True, exist_ok=True)

        # AlchemyCat的`rslt_dir`等价于mmcv的`work_dir`。若AlchemyCat有此意，而mmcv未定义，则补充定义。
        if ('rslt_dir' in self) and ('work_dir' not in self):
            self['work_dir'] = self['rslt_dir']

        mm_config = MMConfig(self._save_cfg_for_other_cfg_system())
        mm_config.dump(file)

        return file

    @classmethod
    def from_x_to_y(cls: type[T_Config], x: Union[str, os.PathLike], y: Union[str, os.PathLike],
                    y_type: Literal['yaml', 'mmcv', 'alchemy-cat']='alchemy-cat') -> T_Config:
        """Convert a Config, yaml, yacs or MMCV-Config file to a desired file type.

        Args:
            x: A Config, yaml, yacs or MMCV-Config file.
            y: The target file path.
            y_type: The target file type. Choices are ['yaml', 'mmcv', 'alchemy-cat'].

        Returns:
            `Config` object.
        """
        # -* 读取dict_cfg。
        dict_cfg, _ = open_config(str(x))
        # -* 转为Config。
        cfg = cls.from_dict(dict_cfg, ret=None, memo=None, always_copy=True, deepcopy=False)
        # -* 保存。
        if y_type == 'yaml':
            cfg.save_yaml(y)
        elif y_type == 'mmcv':
            cfg.save_mmcv(y)
        elif y_type == 'alchemy-cat':
            cfg.save_py(y)
        else:
            raise ValueError(f"{y_type=} is not supported.")

        return cfg

    def merge_from_file(self, cfg_filename: Union[str, os.PathLike]):  # yacs兼容。
        self.unfreeze()
        self.dict_update(open_config(str(cfg_filename))[0])
        self.freeze()
