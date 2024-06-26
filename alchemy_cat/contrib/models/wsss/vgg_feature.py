#!/usr/bin/env src
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: vgg_feature.py
@time: 2019/12/12 12:31
@desc: VGG Model. Based on torch VISION
"""
from typing import Dict

import torch
import torch.nn as nn

from alchemy_cat.alg import odd_input_pad_size

from alchemy_cat.torch_tools import update_model_state_dict, strip_named_params, strip_param_name
from alchemy_cat.torch_tools import Network

__all__ = ['ConvReLU', 'VGGLayer', 'VGGFeature']


class ConvReLU(nn.Sequential, Network):

    def __init__(self, in_ch, out_ch, dilation, layer_idx, seq_idx):
        super(ConvReLU, self).__init__()

        self.in_channel = in_ch
        self.out_channel = out_ch

        self.add_module(f"conv{layer_idx}_{seq_idx}",
                        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=dilation,
                                  dilation=dilation))
        self.add_module(f"relu{layer_idx}_{seq_idx}",
                        nn.ReLU(inplace=True))


class VGGLayer(nn.Sequential, Network):

    def __init__(self, in_ch, out_ch, conv_num, dilation, pool_size, pool_stride, layer_idx):
        super(VGGLayer, self).__init__()

        self.in_channel = in_ch
        self.out_channel = out_ch

        for seq_idx in range(1, conv_num+1):
            self.add_module(f"conv_relu_{seq_idx}",
                            ConvReLU(in_ch=in_ch if seq_idx == 1 else out_ch, out_ch=out_ch,
                                     dilation=dilation, layer_idx=layer_idx, seq_idx=seq_idx))

        padding = odd_input_pad_size(pool_size)

        self.add_module(f"pool{layer_idx}",
                        nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride,
                                     padding=padding, ceil_mode=True))


class VGGFeature(nn.Sequential, Network):

    def __init__(self, in_ch, out_ch, conv_nums, dilations, pool_strides, pool_size):
        super(VGGFeature, self).__init__()

        self.in_channel = in_ch
        self.out_channel = out_ch[-1]

        for i, layer_idx in enumerate(range(1, len(out_ch) + 1)):
            self.add_module(f"layer{layer_idx}",
                            VGGLayer(in_ch=in_ch if layer_idx == 1 else out_ch[i - 1],
                                     out_ch=out_ch[i], conv_num=conv_nums[i], dilation=dilations[i],
                                     pool_size=pool_size[i], pool_stride=pool_strides[i], layer_idx=layer_idx))

    def initialize(self, pretrain_dict: Dict[str, nn.Parameter], verbosity: int=1, **kwargs):
        strip_pretrain_dict = strip_named_params(pretrain_dict)
        weight_dict = {}

        for model_param_name in self.state_dict().keys():
            strip_model_param_name = strip_param_name(model_param_name)
            if strip_model_param_name in strip_pretrain_dict:
                weight_dict[model_param_name] = strip_pretrain_dict[strip_model_param_name]

        update_model_state_dict(self, weight_dict, verbosity)


if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    writer=SummaryWriter(log_dir="Temp")

    writer.add_graph(VGGFeature(
        in_ch=3,
        out_ch=[64, 128, 256, 512, 512],
        conv_nums=[2, 2, 3, 3, 3],
        dilations=[1, 1, 1, 1, 2],
        pool_strides=[2, 2, 2, 1, 1],
        pool_size=[3, 3, 3, 1, 1]
    ), input_to_model=torch.randn((10, 3, 321, 321), dtype=torch.float32))

    print("Graph has been writen to the temp dir")
