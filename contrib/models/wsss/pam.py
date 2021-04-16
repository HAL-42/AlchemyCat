#!/usr/bin/env src
# -*- coding:utf-8 -*-
#
# Author:  yaoqi (yaoqi_isee@zju.edu.cn)
# Created Date: 2019-12-25
# Modified By: yaoqi (yaoqi_isee@zju.edu.cn)
# Last Modified: 2019-12-30
# -----
# Copyright (c) 2019 Zhejiang University
import torch
import torch.nn as nn
import torch.nn.functional as F

from alchemy_cat.torch_tools import Network

__all__ = ['PAM']


class PAM(Network):
    def __init__(self, in_channel, mid_channel, num_classes, layer_idx, branch_idx):
        """init function for positional attention module
        
        Parameters
        ----------
        in_channel : int    
            channel of input tensor
        mid_channel : int
            channel of temporal variables
        num_classes : int
            number of classes
        """
        super(PAM, self).__init__()

        self.in_channel = in_channel
        self.out_channel = num_classes

        self.in_dim, self.out_dim = in_channel, in_channel
        self.key_dim, self.query_dim, self.value_dim = mid_channel, mid_channel, in_channel
        self.f_key = nn.Conv2d(self.in_dim, self.key_dim, 1)
        self.f_query = nn.Conv2d(self.in_dim, self.query_dim, 1)

        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
        # self.fc = nn.Conv2d(self.in_dim, num_classes, 3, padding=12, dilation=12)
        self.layer_idx = layer_idx
        self.branch_idx = branch_idx
        self.add_module(f"fc{layer_idx}_{branch_idx}", nn.Conv2d(self.in_dim, num_classes, 1))

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        value = x.view(batch_size, self.value_dim, -1)
        value = value.permute(0, 2, 1)

        query = self.f_query(x).view(batch_size, self.query_dim, -1)
        query = query.permute(0, 2, 1)

        key = self.f_key(x).view(batch_size, self.key_dim, -1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_dim ** -.5) * sim_map
    
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_dim, *x.size()[2:])

        fuse = self.gamma * context + x

        score = getattr(self, f"fc{self.layer_idx}_{self.branch_idx}")(fuse)
        return score
