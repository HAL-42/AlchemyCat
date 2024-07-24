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
import os
import os.path as osp
import pickle
from typing import Iterable, Callable, Any, Generator, cast, Union

from .utils import name_param_val, norm_param_name
from ..config import Config, auto_rslt_dir, ItemLazy, open_config

__all__ = ["Param2Tune", "ParamLazy", "PL", "Cfg2Tune", "P_DEP"]

kExplicitCapsKey = 'cfgs_update_at_parser'


class Param2Tune(object):

    def __init__(self, optional_values: Iterable, subject_to: Callable[[Any], bool]=lambda x: True,
                 optional_value_names: Iterable[str]=None, priority: int=None):
        """A Parameter need to be tuned

        Args:
            optional_values: Parameters optional value.
            subject_to: Filter used to check whether the parameter value is legal.
            optional_value_names: Name of optional values.
        """
        self.optional_val = tuple(optional_values)
        if optional_value_names is not None:
            self.optional_val_name = tuple(norm_param_name(name) for name in optional_value_names)
            assert len(self.optional_val) == len(self.optional_val_name)
        else:
            self.optional_val_name = tuple(name_param_val(val) for val in self.optional_val)
        self.subject_to = subject_to

        self._cur_val = None  # Current Param Value
        self._cur_val_name = None

        self._priority = priority

    @property
    def is_priority_set(self) -> bool:
        return self._priority is not None

    @property
    def priority(self) -> int:
        assert self._priority is not None
        return self._priority

    @priority.setter
    def priority(self, priority: int) -> None:
        assert self._priority is None
        self._priority = priority

    @property
    def cur_val(self) -> Any:
        if self._cur_val is None:
            raise RuntimeError(f"The current value of param {self} is None. Check if the former param call later param"
                               f"in it's subject_to.")
        return self._cur_val

    @property
    def cur_val_name(self) -> str:
        if self._cur_val_name is None:
            raise RuntimeError(f"The current value name of param {self} is None. Check if the former param call later "
                               f"param in it's subject_to.")
        return self._cur_val_name

    def reset(self) -> None:
        self._cur_val = None
        self._cur_val_name = None

    def __iter__(self) -> Generator[tuple[Any, str], None, None]:
        for opt_val, opt_val_name in zip(self.optional_val, self.optional_val_name):
            if self.subject_to(opt_val):
                self._cur_val = opt_val
                self._cur_val_name = opt_val_name
                yield opt_val, opt_val_name
        self.reset()  # 遍历结束后自动复位。

    def __repr__(self) -> str:
        return f"{self.__class__} with priority={self._priority}: \n" \
               f"optional_val={self.optional_val}, optional_val_name={self.optional_val_name}, \n" \
               f"cur_val={self._cur_val}, cur_val_name={self._cur_val_name}"


class ParamLazy(ItemLazy):
    pass


P_DEP = PL = ParamLazy


class Cfg2Tune(Config):
    """Config to be tuned with parameters to be tuned."""

    def __init__(self, *cfgs: Union[dict, str],
                 cfgs_update_at_parser: Union[tuple[str, ...], str]=(), caps: Union[tuple[str, ...], str]=(), **kwargs):
        """支持从其他其他配置树模块路径或配置树dict初始化。所有配置树会被逐个dict_update到当前配置树上。

        Args:
            *cfgs: List[配置树所在模块|配置树]
            cfgs_update_at_parser: 解析时用于更新的基配置。
            caps: cfgs_update_at_parser的别名。
            **kwargs: 传递给Dict，不应该使用。
        """
        super(Cfg2Tune, self).__init__(*cfgs, cfgs_update_at_parser=cfgs_update_at_parser, caps=caps, **kwargs)

        for leaf in self.leaves:  # 禁止从集成中获取待调参数，可以简化逻辑、令配置简单易懂。
            if isinstance(leaf, Param2Tune):
                raise RuntimeError(f"Cfg2Tune can't init by leaf with type = f{type(leaf)}")

        assert kExplicitCapsKey not in self  # cfgs_update_at_parser若定义则必用于调参，故务必在顶层定义，不可来自导入。

    @property
    def ordered_params2tune(self) -> dict[str, Param2Tune]:
        # 按照优先级排序p2t，未被设置优先级的p2t，优先级默认为最高。
        ret = dict(sorted_params2tune := sorted(((k, l) for c, k, l in self.ckl if isinstance(l, Param2Tune)),
                                                key=lambda x: x[1].priority if x[1].is_priority_set else float('-inf')))
        if len(sorted_params2tune) != len(ret):  # 找到重复键并报错。
            keys, dup_key = set(), None
            for k, _ in sorted_params2tune:
                if k in keys:
                    dup_key = k
                    break
                keys.add(k)
            raise RuntimeError(f"key = {dup_key} for param2tune repeated.")

        return ret

    def _setitem(self, key: Any, value: Any, /) -> None:
        super()._setitem(key, value)  # frozen检查、挂载、dict.__setitem__、子树自洽、IL设置。
        value = self[key]  # 重新获取value，不能保证value在设置后不变。

        # -* 若值为待调参数，检查重复。
        if isinstance(value, Param2Tune):
            root_params2tune = self.find_root()[0].ordered_params2tune  # 若p2t的key重复，会在这里报错。

            if not value.is_priority_set:  # 待调参数只要挂上了Cfg2Tune，（若没有手动设置）必被设置待调的优先级。
                cur_max_priority_p2t = next(reversed(root_params2tune.values()))
                if cur_max_priority_p2t is value:
                    assert len(root_params2tune) == 1
                    value.priority = 0
                else:
                    value.priority = cur_max_priority_p2t.priority + 1

    @staticmethod
    def dfs_params2tune(params2tune: list[Param2Tune], is_root: bool=True):
        """协程每找到一个新的参数组合，yield一次。"""
        assert len(params2tune) > 0, "The params2tune must have more than 1 Param2Tune."

        if is_root:  # 如果为外部调用，先手动复位一次。
            for param in params2tune:
                param.reset()

        cur_param = params2tune[0]

        for _ in cur_param:
            # 若只剩下一个参数，则每找到一个新参数值，就yield。
            if len(params2tune) == 1:
                yield
            # 否则遍历参数组中，第一个参数的所有可选值。对剩余参数，每找到一个新参数组合，就yield一次。
            else:
                yield from Cfg2Tune.dfs_params2tune(params2tune[1:], is_root=False)

    @property
    def param_combs(self) -> list[dict[str, tuple[Any, str]]]:
        params2tune = self.ordered_params2tune
        assert len(params2tune) > 0, "The Cfg2Tune must have more than 1 Param2Tune."

        # * 得到所有参数组合[{param1: (val1, val1_name), ...}, {param1: (val2, val2_name), ...}, ...]
        param_combs = []
        for _ in self.dfs_params2tune(list(params2tune.values()), is_root=True):
            param_combs.append({k: (v.cur_val, v.cur_val_name) for k, v in params2tune.items()})
        assert len(param_combs) > 0, "The param_combs must have more than 1 legal param comb."

        return param_combs

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
        if kExplicitCapsKey in self:
            assert self.get_attribute('_cfgs_update_at_parser') == ()
            if isinstance(cfgs_update_at_parser := self[kExplicitCapsKey], Param2Tune):
                return cfgs_update_at_parser.cur_val
            else:
                return cfgs_update_at_parser
        else:
            return self.get_attribute('_cfgs_update_at_parser')

    def _cfg_tuned(self) -> Config:
        """根据当前待调参数的值，返回一个Addict配置。"""
        other = Config()

        self.copy_cfg_attr_to(other)  # 将依赖、是否不可分等属性拷贝给other。
        # cfgs_update_at_parser可能来自字段，需要单独解析。
        other.set_attribute('_cfgs_update_at_parser', self.cfgs_update_at_parser)

        for k, v in self.items():
            if k == kExplicitCapsKey:
                continue  # k-v对形式的_cfgs_update_at_parser总是以attr形式传递给子配置。

            if self.is_subtree(v, self):
                other[k] = v._cfg_tuned()
            elif isinstance(v, Param2Tune):
                other[k] = v.cur_val
            else:
                other[k] = v

        return other

    def cfg_tuned(self, rslt_dir_suffix: str) -> Config:
        """根据当前待调参数的值，返回一个Addict配置。"""
        ret = self._cfg_tuned()
        ret.rslt_dir = osp.join(ret.rslt_dir, rslt_dir_suffix)
        return ParamLazy.compute_item_lazy(ret)

    def get_cfgs(self) -> Generator[Config, None, None]:
        """遍历待调参数的所有可能组合，对每个组合，返回其对应的配置。"""
        assert 'rslt_dir' in self

        params2tune = self.ordered_params2tune

        if len(params2tune) == 0:
            raise RuntimeError(f"The {self} must have more than 1 Param2Tune")

        for _ in self.dfs_params2tune(list(params2tune.values()), is_root=True):
            rslt_dir_suffix = ",".join([f"{name}={param.cur_val_name}" for name, param in params2tune.items()])
            yield self.cfg_tuned(rslt_dir_suffix)

    def dump_cfgs(self, cfg_dir='configs') -> tuple[list[str], list[Config]]:
        """遍历待调参数的所有组合，将每个组合对应的配置，以json（如果可以）和pkl格式保存到cfg_dir下。"""
        cfg_files, cfgs = [], []
        for cfg in self.get_cfgs():
            cfg_save_dir = osp.join(cfg_dir, cfg.rslt_dir)
            os.makedirs(cfg_save_dir, exist_ok=True)

            cfg.save_py(osp.join(cfg_save_dir, 'cfg.log'))

            cfg_file = osp.join(cfg_save_dir, 'cfg.pkl')
            cfg.save_pkl(cfg_file, save_copy=False)

            cfg_files.append(cfg_file)
            cfgs.append(cfg)

        assert len(cfg_files) > 0, "No cfg files saved."

        return cfg_files, cfgs

    @staticmethod
    def load_cfg2tune(cfg2tune_py: str, experiments_root: str='', config_root: str='./configs') -> 'Cfg2Tune':
        """导入、运行含有config = Cfg2Tune()的py文件，返回该config。"""
        '''
        若config的值，或者其Param2Tune的可选值中含有函数（也就是说cfg_tuned值含有函数），如何确保函数被正确序列化与反序列化？
        - 若函数为外部导入，譬如“from *** import func”，则确保load时，能正确执行“from *** import func”即可。
        - 若函数在config中定义，则保证：
            1）cfg2tune_py按照下面cfg2tune_import_path被import。
            2）load是也能import cfg2tune_import_path。
        注意，subject_to函数、ParamLazy函数不会被pickle，二者只在生成Addict时被执行。
        '''
        cfg2tune, _ = cast(tuple[Cfg2Tune, bool], open_config(cfg2tune_py))

        if cfg2tune.get('rslt_dir', ...) is ...:
            cfg2tune['rslt_dir'] = auto_rslt_dir(cfg2tune_py,
                                                 experiments_root=experiments_root, config_root=config_root)

        return cfg2tune
