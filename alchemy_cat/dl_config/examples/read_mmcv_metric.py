#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2024/7/15 17:14
@File    : read_mmcv_metric.py
@Software: PyCharm
@Desc    : 
"""
import json
import os
from pathlib import Path

__all__ = ['get_metric']


def get_metric(rslt_dir: str | os.PathLike) -> dict[str, float] | None:
    scalar_jsons = list(Path(rslt_dir).glob('**/scalars.json'))
    if len(scalar_jsons) == 1:
        final_scalar = json.loads(tuple(scalar_jsons[0].open())[-1])
        return {k: final_scalar[k] for k in ('aAcc', 'mIoU', 'mAcc')}
    elif len(scalar_jsons) == 0:
        return None
    else:
        raise RuntimeError(f'Found more than one `scalars.json` in {rslt_dir}')
