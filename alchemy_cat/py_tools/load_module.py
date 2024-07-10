#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2022/5/23 22:54
@File    : load_module.py
@Software: PyCharm
@Desc    : 
"""
from types import ModuleType

import os.path as osp
from importlib import import_module
from importlib.util import spec_from_file_location, module_from_spec
# import traceback
import warnings

__all__ = ['load_module_from_py']


def load_module_from_py(py: str, from_file: bool=False) -> ModuleType:
    """给定python文件路径，载入python模块。

    函数首先会尝试将python文件路径转换为'A.B.C'的导入路径，随后用标准的import_module函数导入文件为模块。
    若失败（譬如原路径中含有“.”字符），就尝试直接利用文件查找器+载入器，将python文件载入为模块，并名以“foo”，随后执行文件，绑定
    模块命名空间中对象。

    应当注意，尽量避免进入第二种导入方式。以导入链A import B、C，B、C import D为例：

    1. 文件直接导入不会将模块加入sys.module中，模块导完就释放。若某模块需要导入两次（如上例中D），则D需要独立导入两次。不仅浪费，
    且B、C导入的D中同名对象，并非真正的同一对象（如某些容器、模型），这可能会带来我们不希望的行为。

    2. 文件直接导入所得模块，其中函数被pickle转储时，函数地址（func.__module__）为无意义的foo。故载入时，无法根据地址找到被转储
    的原函数。

    Args:
        py: python文件路径。
        from_file: 是否强制从文件导入。

    Returns:
        载入的python模块。
    """
    module = None

    if not from_file:
        try:
            import_path = '.'.join(osp.normpath(osp.splitext(py)[0]).lstrip(osp.sep).split(osp.sep))
            if '.' == import_path[0]:  # 如果是相对路径。
                # import_path = import_path[1:]  # 相对路径会多一个点。
                raise RuntimeError(f"Relative import is not supported yet. ")
            module = import_module(import_path)
        except Exception:
            # print(traceback.format_exc())
            warnings.warn(f"未能用import_module导入{py},尝试直接执行文件。")

    if module is None:
        spec = spec_from_file_location("foo", py)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)

    return module
