#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2024/7/15 15:02
@File    : mm_cfg2tune.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat import Cfg2Tune, Param2Tune, DEP, P_DEP

# Inherit from standard mmcv config.
cfg = Cfg2Tune(caps='py_tools/config/tests/migrate/mm_configs/'
                    'deeplabv3plus/deeplabv3plus_r50-d8_4xb4-40k_voc12aug-512x512.py')

# Inherit and override
cfg.model.auxiliary_head.loss_decode.loss_weight = 0.2

# Tuning parameters: grid search max_iters and batch_size
cfg.train_cfg.max_iters = Param2Tune([20_000, 40_000])
cfg.train_dataloader.batch_size = Param2Tune([8, 16])

# Dependencies:
# 1) end of param_scheduler increase with max_iters
# 2) learning rate increase with batch_size
cfg.param_scheduler = P_DEP(lambda c: [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=c.train_cfg.max_iters,
        by_epoch=False)
])
cfg.optim_wrapper.optimizer.lr = DEP(lambda c: (c.train_dataloader.batch_size / 8) * 0.01)
