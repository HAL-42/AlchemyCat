#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2020/11/6 10:26
@File    : intermediate_value_getter.py
@Software: PyCharm
@Desc    : 
"""
from typing import Optional

from torch import nn
import torch

__all__ = ["IntermediateValueGetter"]


class IntermediateValueGetter(object):
    """Get intermediate value of torch module"""
    def __init__(self, module: nn.Module, get_forward: bool=False, get_backward: bool=False):
        """Get intermediate value of torch module

        Args:
            module: Module whose intermediate value need be gotten.
            get_forward: Is get forward input/output (Default: False).
            get_backward: Is get backward grad_input/grad_output (Default: False).
        """
        self.get_forward, self.get_backward = get_forward, get_backward

        self._module = None
        self._input, self._output, self._grad_input, self._grad_output = None, None, None, None

        self._forward_hook_handler: Optional[torch.utils.hooks.RemovableHandle] = None
        self._backward_hook_handler: Optional[torch.utils.hooks.RemovableHandle] = None

        self.module = module

    def _forward_hook_fn(self, _, input, output):
        self._input = input
        self._output = output

    def _backward_hook_fn(self, _, grad_input, grad_output):
        self._grad_input = grad_input
        self._grad_output = grad_output

    def _set_hooks(self, module: nn.Module):
        if self.get_forward:
            self._forward_hook_handler = module.register_forward_hook(self._forward_hook_fn)
        if self.get_backward:
            self._backward_hook_handler = module.register_backward_hook(self._backward_hook_fn)

    @property
    def module(self):
        return self._module

    @module.setter
    def module(self, module):
        self.remove_hooks()
        self._set_hooks(module)
        self._module = module

    def remove_hooks(self):
        if self._forward_hook_handler is not None:
            self._forward_hook_handler.remove()
        if self._backward_hook_handler is not None:
            self._backward_hook_handler.remove()

    def __del__(self):
        self.remove_hooks()

    @property
    def input(self):
        assert self.get_forward
        return self._input

    @property
    def output(self):
        assert self.get_forward
        return self._output

    @property
    def grad_input(self):
        assert self.get_backward
        return self._grad_input

    @property
    def grad_output(self):
        assert self.get_backward
        return self._grad_output

    def __repr__(self):
        return f"IntermediateValueGetter: {self.__class__.__name__}\n" \
                + f"    module: {self.module}\n" \
                + f"    get_forward: {self.get_forward}\n" \
                + f"    get_backward: {self.get_backward}\n" \
                + f"    forward_hook_handler: {self._forward_hook_handler}\n" \
                + f"    backward_hook_handler: {self._backward_hook_handler}"
