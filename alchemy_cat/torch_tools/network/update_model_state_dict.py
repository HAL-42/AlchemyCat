#!/usr/bin/env src
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: update_model_state_dict.py
@time: 2020/3/23 5:42
@desc:
"""
from typing import Dict, Tuple, List, Union

import re
from colorama import Style, Fore
import torch

import torch.nn as nn

__all__ = ['update_model_state_dict', 'strip_param_name', 'strip_named_params', 'named_param_match_pattern',
           'find_match_miss', 'kParamDictType', 'sub_module_name_of_named_params']

kParamDictType = Union[Dict[str, nn.Parameter], Dict[str, torch.Tensor]]


def update_model_state_dict(model: nn.Module, weight_dict: kParamDictType, verbosity: int = 2):
    """Update model's state_dict with pretrain_dict

        Args:
            model: model to be updated
            weight_dict: pretrain state dict
            verbosity: 0: No info; 1: Show weight loaded; 2: Show weight missed; 3: Show weight Redundant
    """
    model_dict = model.state_dict()
    update_dict = {k: weight_dict[k] for k in model_dict.keys() if k in weight_dict}
    model_dict.update(update_dict)
    model.load_state_dict(model_dict, strict=True)

    if verbosity >= 1:
        for k in sorted(update_dict.keys()):
            print(Fore.GREEN + "[WEIGHT LOADED] " + Style.RESET_ALL
                  + "{:<50s} shape: {}".format(k, update_dict[k].shape))

    if verbosity >= 2:
        missing_dict = {k: model_dict[k] for k in model_dict.keys() if k not in weight_dict}
        for k in sorted(missing_dict.keys()):
            print(Fore.YELLOW + "[WEIGHT MISSED] " + Style.RESET_ALL +
                  "{:<50s} shape: {}".format(k, missing_dict[k].shape))

    if verbosity >= 3:
        redundant_dict = {k: weight_dict[k] for k in weight_dict.keys() if k not in model_dict}
        for k in sorted(redundant_dict.keys()):
            print(Fore.LIGHTCYAN_EX + "[WEIGHT UNUSED] " + Style.RESET_ALL +
                  "{:<50s} shape: {}".format(k, redundant_dict[k].shape))


def strip_param_name(param_name: str) -> str:
    """Input an module's param name, return it's origin name with out parent modules' name

    Args:
        param_name (str): parameter name with it's parent module prefix

    Returns:
        Param's origin name
    """
    splits = param_name.rsplit('.', maxsplit=2)
    return '.'.join(splits[-2:])


def strip_named_params(named_params: kParamDictType) -> Union[Dict[str, nn.Parameter], Dict[str, torch.Tensor]]:
    """Strip all param names in modules' state_dict

    Args:
        named_params: module's state_dict

    Returns:
        A Dict with key of stripped param name and it's origin value.

    See Also:
        strip_param_name
    """
    return {strip_param_name(k): v for k, v in named_params.items()}


def named_param_match_pattern(pattern: str, named_param: kParamDictType,
                              full_match: bool = False, multi_match=True) -> kParamDictType:
    """Find key-value pair in named_param whose key match pattern

    Args:
        pattern: re pattern
        named_param: named parameters
        full_match: If True, the key matched pattern must exactly equal to matched string. (Default: False)
        multi_match: If False, when multi keys match pattern, raise Error. (Default: True)

    Returns:
        Key-value pair whose key match pattern

    Raises:
        RuntimeError:
            More than one key matched
    """
    matched_key_value_pairs = {}
    for key in named_param.keys():
        re_ret = re.search(pattern, key)
        if re_ret is not None:
            if full_match and re_ret.group(0) != key:
                continue
            else:
                matched_key_value_pairs[key] = named_param[key]

    if len(matched_key_value_pairs) > 1 and not multi_match:
        raise RuntimeError(f"More than one parameter {matched_key_value_pairs} can by matched by pattern {pattern}")

    return matched_key_value_pairs


def find_match_miss(model: nn.Module, pretrain_dict: kParamDictType) \
        -> Tuple[Union[Dict[str, nn.Parameter], Dict[str, torch.Tensor]], List[str]]:
    """Find params in pretrain dict with the same suffix name of params of model.

        Args:
            model: torch model
            pretrain_dict: pretrain dict with key: parameter

        Returns:
            matched_dict: keys with corresponding pretrain values of model's state_dict whose suffix name can be found
                in pretrain_dict
            missed_keys: keys of model's state_dict whose suffix name can not be found in pretrain_dict
        """
    matched_dict = dict()
    missed_keys = []
    strip_pretrain_dict = strip_named_params(pretrain_dict)

    for model_param_name in model.state_dict().keys():
        strip_model_param_name = strip_param_name(model_param_name)
        if strip_model_param_name in strip_pretrain_dict:
            matched_dict[model_param_name] = strip_pretrain_dict[strip_model_param_name]
        else:
            missed_keys.append(model_param_name)

    return matched_dict, missed_keys


def sub_module_name_of_named_params(named_params: kParamDictType, module_name_sub_dict: Dict[str, str]) \
    -> Union[Dict[str, nn.Parameter], Dict[str, torch.Tensor]]:
    """Sub named_parameters key's module name part with module_name_sub_dict.

    Args:
        named_params: Key-value pair of param name and param value.
        module_name_sub_dict: Module names' sub dict.

    Returns:
        named parameters whose module name part of it's param name is subbed by module_name_sub_dict.
    """
    sub_named_params = dict()
    for module_param_name, value in named_params.items():
        param_name, module_name = map(lambda inverse_name: inverse_name[::-1],
                                      module_param_name[::-1].split('.', maxsplit=1))
        if module_name not in module_name_sub_dict:
            sub_named_params[module_param_name] = value
        else:
            sub_named_params[module_name_sub_dict[module_name] + '.' + param_name] = value
    return sub_named_params
