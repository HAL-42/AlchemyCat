#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: quick_init.py
@time: 2019/9/3 1:52
@desc:
"""
from inspect import signature
from collections import OrderedDict


def quick_init(obj, local_var_dict: dict, update_to_obj_dict: bool=False) -> OrderedDict:
    """Quick init the object with all it's __init__ function's params. Use like quick_init(self, locals())

    Args:
        obj (): Object to be initialized
        local_var_dict (): locals() ->
        update_to_obj_dict (): If true, the param dict will be update to object's __dict__

    Returns:
        Dict of __init__ function's param
    """
    # init_dict_keys = obj.__init__.__code__.co_varnames[1:obj.__init__.__code__.co_argcount]
    init_dict_keys = signature(obj.__init__).parameters.keys()

    init_dict = OrderedDict()
    for key in init_dict_keys:
        init_dict[key] = local_var_dict[key]

    # Not Ordered if python version < 3.6
    # init_dict = {key: local_var_dict[key] for key in init_dict_keys}

    if update_to_obj_dict:
        obj.__dict__.update(init_dict)

    return init_dict
