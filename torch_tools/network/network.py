#!/usr/bin/env src
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: networks.py
@time: 2020/3/23 22:29
@desc:
"""
from typing import Dict, Tuple, List, Union

import torch
from torch import nn

from alchemy_cat.torch_tools.network.update_model_state_dict import strip_named_params, \
    update_model_state_dict, find_match_miss, kParamDictType

__all__ = ["Network"]


class Network(nn.Module):
    """Torch base networks with default I/O info and initialization"""

    in_batch_size = None
    out_batch_size = None
    in_channel = None
    out_channel = None
    in_height = None
    out_height = None
    in_width = None
    out_width = None

    def initialize(self, pretrain_dict: kParamDictType, verbosity: int=2, **kwargs):
        """Default initialize. Initialize every sub module with type Network using pretrain_dict

        Args:
            pretrain_dict: pretrain state dict
            verbosity: 0: No info; 1: Show weight loaded; 2: Show weight missed; 3: Show weight Redundant. (Default: 2)
        """
        for child in self.children():
            if isinstance(child, Network):
                child.initialize(pretrain_dict, verbosity, **kwargs)

    def find_match_miss(self, pretrain_dict: kParamDictType) \
            -> Tuple[Union[Dict[str, nn.Parameter], Dict[str, torch.Tensor]], List[str]]:
        """Find params in pretrain dict with the same suffix name of params of model.

        Args:
            pretrain_dict: pretrain dict with key: parameter

        Returns:
            matched_dict: keys with corresponding pretrain values of model's state_dict whose suffix name can be found
                in pretrain_dict
            missed_keys: keys of model's state_dict whose suffix name can not be found in pretrain_dict
        """
        return find_match_miss(self, pretrain_dict)

    def update_model_state_dict(self, weight_dict: kParamDictType, verbosity: int=2):
        """Update model's state_dict with pretrain_dict

        Args:
            weight_dict: pretrain state dict
            verbosity: 0: No info; 1: Show weight loaded; 2: Show weight missed; 3: Show weight Redundant
        """
        update_model_state_dict(self, weight_dict, verbosity=verbosity)

    @property
    def strip_named_params(self):
        return strip_named_params(self.state_dict())
