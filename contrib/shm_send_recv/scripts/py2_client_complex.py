#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2021/8/2 21:09
@File    : py2_client.py
@Software: PyCharm
@Desc    : 
"""
import numpy as np
import time

from alchemy_cat.contrib.shm_send_recv.py2_send_recv import client_send, client_recv  # 导入client所需函数

kIterTimes = 1000
kImageSize = (512, 1024)

kClientImg = np.ones(kImageSize + (3,), dtype=np.uint8)
kClientArr = np.ones(kImageSize, dtype=np.float32)
kClientObj = list(np.arange(100))

kServerObj = {'a': 1, (3,): 2.2, 'c': 'str', 'd': [1, 2]}

time_start = time.time()

for i in range(kIterTimes):
    client_send(kClientImg)  # 发送图片
    client_send(kClientArr)  # 发送numpy array
    client_send(kClientObj)  # 发送python对象

    obj_recv = client_recv()  # 接收python对象
    assert obj_recv == kServerObj

    arr_recv = client_recv()  # 接收numpy array
    assert arr_recv.shape == kImageSize
    assert np.all(arr_recv == 255.)
    assert arr_recv.dtype == np.float32

    predict = client_recv()  # 接收预测结果
    assert predict.shape == kImageSize
    assert np.all(predict == 255)
    assert predict.dtype == np.uint8

    print('Finish %d/%d' % ((i + 1), kIterTimes))

time_end = time.time()

print('fps =', kIterTimes / (time_end - time_start))
