#!/usr/bin/env src
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: e2e.py
@time: 2020/3/23 4:23
@desc:
"""
from typing import Dict, Tuple
from collections import defaultdict
import re

import torch
from torch import nn

from alchemy_cat.contrib.models.wsss.vgg_feature import VGGFeature
from alchemy_cat.contrib.models.wsss.vgg_ASPP import VGGASPP, FCReLUDrop
from alchemy_cat.contrib.models.wsss.pam import PAM

from alchemy_cat.torch_tools import Network
from alchemy_cat.torch_tools import strip_param_name, strip_named_params, update_model_state_dict

__all__ = ['VGGE2ECls', 'VGGE2EPamCollab', 'E2EVGGBackbone', 'E2EClsHead', 'E2EASPPSegHead', 'E2EPAMSegHead']


class E2EVGGBackbone(VGGFeature, Network):
    """VGG Backbone for end to end module for WSSS"""
    def __init__(self):
        super(E2EVGGBackbone, self).__init__(in_ch=3, out_ch=[64, 128, 256, 512, 512],
                                             conv_nums=[2, 2, 3, 3, 3], dilations=[1, 1, 1, 1, 2],
                                             pool_strides=[2, 2, 2, 1, 1], pool_size=[3, 3, 3, 1, 1])


class E2EClsHead(nn.Sequential, Network):
    """Classification head for end to end module for WSSS"""
    def __init__(self, in_ch: int, num_class: int, branch_idx: int):
        super(E2EClsHead, self).__init__()

        self.in_channel = in_ch
        self.out_channel = num_class

        self.add_module(f"e2e_cls_head6_{branch_idx}", FCReLUDrop(in_ch, 1024, kernel_size=3, dilation=1, padding=1,
                                                                  layer_idx=6, branch_idx=branch_idx))
        self.add_module(f"e2e_cls_head7_{branch_idx}", FCReLUDrop(1024, 1024, kernel_size=1, dilation=1, padding=0,
                                                                  layer_idx=7, branch_idx=branch_idx))
        self.add_module(f"fc8_{branch_idx}", nn.Conv2d(1024, num_class, 1, bias=False))

    def initialize(self, pretrain_dict: Dict[str, nn.Parameter], verbosity: int=1, **kwargs):
        vgg_init_from = kwargs.get('vgg_init_from', 'deeplab_v2')

        if vgg_init_from == 'deeplab_v2':
            self.initialize_from_deeplab_v2(pretrain_dict, verbosity)
        elif vgg_init_from == 'deeplab_v1':
            self.initialize_from_deeplab_v1(pretrain_dict, verbosity)
        else:
            raise ValueError(f"kwarg vgg_init_from = {vgg_init_from} must be deeplab_v1 or deeplab_v2.")

    def initialize_from_deeplab_v2(self, pretrain_dict: Dict[str, nn.Parameter], verbosity: int=1):
        strip_pretrain_dict = strip_named_params(pretrain_dict)
        weight_dict = {}

        for model_param_name in self.state_dict().keys():
            strip_model_param_name = strip_param_name(model_param_name)
            if strip_model_param_name in strip_pretrain_dict:
                # Exact Match
                weight_dict[model_param_name] = strip_pretrain_dict[strip_model_param_name]
            else:
                if 'fc8' in model_param_name:
                    # Don't init fc_8
                    continue
                # Fuzzy Match
                fuzzy_strip_model_param_name = re.sub(r'fc\d_\d',
                                                      lambda match: match.group(0)[:-1] + '1', strip_model_param_name)
                weight_dict[model_param_name] = strip_pretrain_dict[fuzzy_strip_model_param_name]

        update_model_state_dict(self, weight_dict, verbosity)

    def initialize_from_deeplab_v1(self, pretrain_dict: Dict[str, nn.Parameter], verbosity: int=1):
        update_dict = {}
        matched_dict, missed_keys = self.find_match_miss(pretrain_dict)
        update_dict.update(matched_dict)

        if len(missed_keys) > 0:
            print("Init cls head from deeplab_v1")
            strip_pretrain_dict = strip_named_params(pretrain_dict)
            for model_param_name in missed_keys:
                strip_model_param_name = strip_param_name(model_param_name)
                if 'fc8' in strip_model_param_name:
                    continue
                else:
                    param_name_in_pretrain = re.sub(r'(fc\d)_\d', lambda match: match.group(1), strip_model_param_name)
                    update_dict[model_param_name] = strip_pretrain_dict[param_name_in_pretrain]

        self.update_model_state_dict(update_dict, verbosity)


class E2EASPPSegHead(VGGASPP, Network):
    """Segmentation head with ASPP for end to end module for WSSS"""
    def __init__(self, in_ch: int, num_class: int):
        super(E2EASPPSegHead, self).__init__(in_ch, num_class,
                                             rates=[6, 12, 18, 24], start_layer_idx=6)


class E2EPAMSegHead(nn.Sequential, Network):
    """Segmentation head with PAM for end to end module for WSSS"""
    def __init__(self, in_ch: int, num_class: int):
        super(E2EPAMSegHead, self).__init__()

        self.in_channel = in_ch
        self.out_channel = num_class

        self.pre_pam = nn.Sequential(
            FCReLUDrop(in_ch, 1024, kernel_size=3, dilation=6, padding=6, layer_idx=6, branch_idx=1),
            FCReLUDrop(1024, 1024, kernel_size=1, dilation=1, padding=0, layer_idx=7, branch_idx=1)
        )

        self.pam = PAM(1024, 256, num_class, layer_idx=8, branch_idx=1)

    def initialize(self, pretrain_dict: Dict[str, nn.Parameter], verbosity: int=2, **kwargs):
        weight_dict = dict()
        matched_dict, missed_keys = self.find_match_miss(pretrain_dict)
        weight_dict.update(matched_dict)

        if len(missed_keys) > 0:
            print("Initial seg head from cls head")
            strip_pretrain_dict = strip_named_params(pretrain_dict)
            for model_param_name in missed_keys:
                if 'fc8' in model_param_name:
                    # Don't init fc_8
                    continue
                if re.search(r'fc\d_\d', model_param_name) is None:
                    continue
                # Fuzzy Match
                strip_model_param_name = strip_param_name(model_param_name)
                fuzzy_strip_model_param_name = re.sub(r'(fc\d_)\d', lambda match: match.group(1) + '0',
                                                      strip_model_param_name)
                weight_dict[model_param_name] = strip_pretrain_dict[fuzzy_strip_model_param_name]

        self.update_model_state_dict(weight_dict, verbosity)


class _E2EBase(Network):
    """Define get_params for all e2e model"""
    def get_param_groups(self) -> dict:
        param_groups = defaultdict(list)
        for name, param in self.named_parameters():
            if 'backbone' in name:
                if 'weight' in name:
                    param_groups['backbone_weight'].append(param)
                elif 'bias' in name:
                    param_groups['backbone_bias'].append(param)
                else:
                    raise RuntimeError(f"UnKnown param {name}: {param}")
            elif 'head' in name:
                if 'weight' in name:
                    param_groups['head_weight'].append(param)
                elif 'bias' in name:
                    param_groups['head_bias'].append(param)
                elif 'gamma' in name:
                    param_groups['head_gamma'].append(param)
                else:
                    raise RuntimeError(f"Unknown param {name}: {param}")
            else:
                raise RuntimeError(f"Unknown param {name}: {param}")

        return dict(param_groups)


class VGGE2ECls(nn.Sequential, _E2EBase, Network):
    """End to end module's classification branch for WSSS"""

    def __init__(self, num_class: int=21):
        super(VGGE2ECls, self).__init__()

        self.in_channel = 3
        self.out_channel = num_class - 1

        self.backbone = VGGFeature(in_ch=3, out_ch=[64, 128, 256, 512, 512],
                                   conv_nums=[2, 2, 3, 3, 3], dilations=[1, 1, 1, 1, 2],
                                   pool_strides=[2, 2, 2, 1, 1], pool_size=[3, 3, 3, 1, 1])
        self.cls_head = E2EClsHead(self.backbone.out_channel, num_class-1, branch_idx=0)


class VGGE2EPamCollab(_E2EBase, Network):
    """Collab End to End model with pad seg head for WSSS"""

    def __init__(self, num_class: int=21):
        super().__init__()

        self.in_channel = 3

        self.vgg_e2e_cls = VGGE2ECls(num_class)

        self.seg_head = E2EPAMSegHead(self.vgg_e2e_cls.backbone.out_channel, num_class)

    def forward(self, img) -> Tuple[torch.Tensor, torch.Tensor]:
        feature_map = self.vgg_e2e_cls.backbone(img)
        cam = self.vgg_e2e_cls.cls_head(feature_map)
        score_map = self.seg_head(feature_map)

        return cam, score_map
