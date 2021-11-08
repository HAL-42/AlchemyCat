#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2021/8/2 21:09
@File    : py3_server.py
@Software: PyCharm
@Desc    : 
"""
import numpy as np

from alchemy_cat.contrib.shm_send_recv.py3_send_recv import server_send, server_recv, init_server  # 导入server所需函数

kImageSize = (512, 1024)

kClientObj = list(np.arange(100))

kServerObj = {'a': 1, (3,): 2.2, 'c': 'str', 'd': [1, 2]}
kServerArr = np.ones(kImageSize, dtype=np.float32) * 255
kServerPredict = np.ones(kImageSize, dtype=np.uint8) * 255


init_server()  # 初始化server，非必须。可以防止多个server同时运行，且保证server被中断后（譬如被kill）后，buffer能被正确清理。

recv_count = 0
while True:
    img = server_recv()  # 接收图片
    assert img.shape == kImageSize + (3,)
    assert np.all(img == 1)

    arr_recv = server_recv()  # 接收numpy array
    assert arr_recv.shape == kImageSize
    assert np.all(arr_recv == 1.)
    assert arr_recv.dtype == np.float32

    obj_recv = server_recv()  # 接收python对象
    assert obj_recv == kClientObj

    server_send(kServerObj)  # 发送python对象
    server_send(kServerArr)  # 发送numpy array
    server_send(kServerPredict)  # 发送预测结果

    recv_count += 1
    print("Server has received for %d imgs" % recv_count)
