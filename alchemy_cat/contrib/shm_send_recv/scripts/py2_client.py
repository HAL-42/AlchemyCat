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
kClientImg = np.random.randint(0, 256, size=kImageSize + (3,), dtype=np.uint8)


total_time_start = time.time()

for i in range(kIterTimes):
    sample_time_start = time.time()
    client_send(kClientImg)  # 发送图片
    predict = client_recv()  # 接收预测结果
    sample_time_end = time.time()
    print('sample time consuming = %fs' % (sample_time_end - sample_time_start))

total_time_end = time.time()

print('fps =', kIterTimes / (total_time_end - total_time_start))
