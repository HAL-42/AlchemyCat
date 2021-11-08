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
kServerPredict = np.random.randint(0, 256, size=kImageSize, dtype=np.uint8)


init_server()  # 初始化server，非必须。可以防止多个server同时运行，且保证server被中断后（譬如被kill）后，buffer能被正确清理。

while True:
    img = server_recv()  # 接收图片

    predict = kServerPredict  # 根据输入图片，得到预测结果。这里用常量代替。

    server_send(kServerPredict)  # 发送预测结果
