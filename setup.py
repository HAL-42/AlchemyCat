#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2021/4/19 22:41
@File    : setup.py
@Software: PyCharm
@Desc    : 
"""
from setuptools import setup, find_packages

setup(
        name='alchemy_cat',
        version='0.0.1',
        description='AlchemyCat alpha version.',
        author='HAL_42',
        author_email='hal_42@zju.edu.cn',
        url='https://github.com/HAL-42/AlchemyCat',
        packages=find_packages(),
        package_data={
                '': ['*.md'],
        },
        zip_safe=True,
)
